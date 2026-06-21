#!/usr/bin/env bash
# codex-review-loop watchdog — read-only. Watches the codex companion job log for
# THIS round and exits with a code describing codex's fate. Performs NO writes
# (safe to allowlist as one command, so the loop runs without permission prompts).
#
#   exit 0  normal end    (companion logged "Turn completed." or a "Final output" line)
#   exit 2  failure        (companion logged "Turn failed.")
#   exit 3  stall / hang    (job-log mtime has not advanced for STALL_SECS — heuristic)
#   exit 4  no codex start / cap reached
#                           (a) no job log within STARTUP_GRACE_SECS (clamped <= WALL_CAP_SECS):
#                               retry-favoring heuristic for a no-start (also trips on an
#                               abnormally slow start or a discovery glitch); the only consequence
#                               is a caller retry. Default 240s is generous so a slow-but-healthy
#                               start is not tripped.
#                           (b) WALL_CAP_SECS elapsed with the log identified but no terminal.
#
# Before exiting it prints, to stdout:
#     LOG=<path>            (empty if no log was identified)
#     OUTCOME=<code> <reason>
# so the caller reads its background output file to learn WHICH log holds the result.
#
# Args (optional, positional): $1=STALL_SECS  $2=WALL_CAP_SECS  $3=STATE_ROOT
#                              $4=STARTUP_GRACE_SECS (no-start fast-fail; default 240)
#                              $5=RUNID (correlation token; enables concurrent sessions)
#
# Identifying THIS round's log — two modes:
#
#   (A) RUNID mode ($5 set): the caller embeds a unique token (RUNID) at the very top of the
#       review prompt; the companion records the prompt opening in the job's ".json" metadata
#       (the "summary" field), so this round's log is the one whose sibling "task-*.json" (or, as
#       a fallback, the ".log" body) contains the RUNID. This identifies the round by CONTENT, not
#       by time or launch order, so concurrent rounds — even two checkouts of a same-named repo —
#       never collide, and the "watchdog-before-agent" ordering constraint is no longer required
#       for correctness (it only helps no-start be detected sooner).
#
#   (B) baseline mode ($5 empty/absent — backward compatible): this round's log is one that did
#       NOT exist at the baseline snapshot; the newest such log wins. A still-running orphan log
#       from a prior or failed round stays in the baseline and is never mistaken for this round's.
#       Correctness depends on the baseline being taken before the companion creates this round's
#       log — ensured by launch ordering (the caller starts this watchdog before the agent; the
#       ~100ms baseline precedes the agent's ~seconds-later job log). A rare mis-order only
#       false-fires no-start, which degrades to a safe retry.
#
# Accepted limitation (baseline mode only; the agent response is the primary result channel, so a
# wrong watchdog signal degrades to a retry, never a wrong fix): two checkouts of a same-named
# repo share the "<repo>-*" prefix, so a truly concurrent codex run in the other checkout could be
# selected. RUNID mode removes this limitation entirely.
set -u

STALL_SECS="${1:-420}"
WALL_CAP_SECS="${2:-1200}"
STATE_ROOT="${3:-${STATE_ROOT:-${HOME:-}/.claude/plugins/data/codex-openai-codex/state}}"
# Default sized as a generous safety margin above a typical companion startup (a few seconds):
# a loaded host can delay this round's log noticeably, so the grace is large enough not to trip a
# slow-but-healthy start, yet small enough to catch a genuine no-start far sooner than the wall cap.
STARTUP_GRACE_SECS="${4:-240}"
# Correlation token for concurrent-safe round identification (empty = backward-compatible baseline
# mode). Restrict to a regex-safe charset (alnum, `_`, `-`; no `.` so the token is an ERE literal
# with no metacharacters) so it can be matched with a bounded pattern and cannot smuggle
# shell/pattern metacharacters; reject anything else fast with a parseable OUTCOME.
RUNID="${5:-}"
case "$RUNID" in
  '') ;;                                  # empty = baseline mode, fine
  *[!A-Za-z0-9_-]*) printf 'LOG=\nOUTCOME=4 bad-arg (invalid RUNID: %s)\n' "$RUNID"; exit 4;;
esac
# Validate the time args as non-negative integers up front. A non-numeric value would make the
# integer comparisons below silently error out (`[ N -ge abc ]`), disabling a backstop and
# reintroducing the very silent hang this watchdog exists to prevent. Fail fast instead with a
# parseable OUTCOME so the caller surfaces the bad arg rather than waiting forever.
for _v in "$STALL_SECS" "$WALL_CAP_SECS" "$STARTUP_GRACE_SECS"; do
  # Reject non-integers. (An empty arg never reaches here: ${x:-default} already substituted a
  # valid default, so empty is safe, not a hang path.)
  case "$_v" in
    ''|*[!0-9]*) printf 'LOG=\nOUTCOME=4 bad-arg (non-integer time value: %s)\n' "$_v"; exit 4;;
  esac
  # Bound the magnitude so the integer comparisons below cannot overflow bash's 64-bit ints and
  # silently disable a backstop. <=18 digits keeps every value under 10^18 < 2^63. (No sane
  # timeout needs more; this is a safety bound, not a tuning limit.)
  [ "${#_v}" -gt 18 ] && { printf 'LOG=\nOUTCOME=4 bad-arg (time value out of range: %s)\n' "$_v"; exit 4; }
done
# Clamp so the no-log deadline is unambiguous: the grace never exceeds the wall cap, so a
# never-identified log always ends as "no-start" at the grace, and the wall cap only ever bounds
# the identified-but-no-terminal case. This removes the otherwise pathological grace > wall-cap
# branch (and the conditional wording it would need).
[ "$STARTUP_GRACE_SECS" -gt "$WALL_CAP_SECS" ] && STARTUP_GRACE_SECS="$WALL_CAP_SECS"
POLL=5

mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null; }
now() { date +%s; }

# Terminal markers, anchored to the companion's own log lines, which begin with a full ISO
# timestamp ("[2026-06-15T03:00:13.785Z] Turn completed."). The strict YYYY-MM-DDT anchor stops
# the same words inside an assistant message, a `Running command: ... "Turn completed"` echo, or
# a short "[2]"/"[note]" bracket line from faking an end.
ts_re='^\[[0-9]{4}-[0-9]{2}-[0-9]{2}T[^]]*\] '
done_re="${ts_re}(Turn completed\.|Final output)"
fail_re="${ts_re}Turn failed\."
report() { printf 'LOG=%s\nOUTCOME=%s %s\n' "$1" "$2" "$3"; }

# Scope discovery to THIS repo's state dir(s). No fall-back to the whole STATE_ROOT: if the
# "<repo>-*" convention ever stops matching, discovery simply finds nothing and the watcher
# degrades to the never-identified exit 4 (no-start at the grace, caller retries) rather than
# risk selecting an unrelated repo's log.
repo=$(basename "$PWD" 2>/dev/null || printf '')
list_logs() {
  local d
  find "$STATE_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${repo}-*" 2>/dev/null \
  | while IFS= read -r d; do
      [ -n "$d" ] && find "$d/jobs" -type f -name 'task-*.log' 2>/dev/null
    done
}

# True if $1 was present at the baseline snapshot (exact line match against the baseline set).
in_baseline() { printf '%s\n' "$baseline" | awk -v k="$1" '$0==k{f=1} END{exit !f}'; }

# True if the log $1 belongs to THIS round, identified by the RUNID embedded at the top of the
# review prompt. The companion saves the prompt opening into the job's sibling "task-*.json"
# ("summary" field), so the ".json" is authoritative when present; only when it is absent do we
# fall back to the ".log" body. The match is the full "REVIEW-RUNID: <RUNID>" marker bounded by a
# non-token character or end of line, so one token is never a substring-match of a longer token
# (e.g. RUNID "abc" must not match "...REVIEW-RUNID: abc1"). RUNID is charset-validated to alnum/
# `_`/`-` above, so it is a literal ERE fragment with no metacharacters.
has_runid() {
  local lg="$1" js="${1%.log}.json" pat
  pat='REVIEW-RUNID:[[:space:]]*'"${RUNID}"'([^A-Za-z0-9_-]|$)'
  if [ -f "$js" ]; then
    grep -Eq -- "$pat" "$js" 2>/dev/null   # .json present -> its verdict is final
    return
  fi
  grep -Eq -- "$pat" "$lg" 2>/dev/null      # .json absent -> fall back to the log body
}

# Snapshot pre-existing logs (paths only). LAUNCH ORDERING (no readiness poll — staying
# prompt-free): the caller starts this watchdog in the background BEFORE the agent. This snapshot
# is a ~100ms find, while the companion's job log appears ~seconds after the agent launches, so
# this round's log is reliably "new since baseline" without any handshake. A rare mis-order only
# false-fires no-start, which degrades to a safe retry — not worth a prompt-generating poll.
baseline=$(list_logs | sort)

# Start the grace/wall clocks right after the baseline snapshot.
start=$(now)

# Pick the newest log for THIS round. RUNID mode: the newest log whose metadata/body contains the
# RUNID (content-based; concurrent-safe; independent of launch order). Baseline mode (RUNID empty):
# the newest log that did not exist at the baseline snapshot.
pick_active() {
  local p cm best="" bestmt=0
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    if [ -n "$RUNID" ]; then
      has_runid "$p" || continue
    else
      in_baseline "$p" && continue
    fi
    cm=$(mtime "$p"); [ -n "$cm" ] || continue
    if [ "$cm" -ge "$bestmt" ]; then bestmt=$cm; best=$p; fi
  done < <(list_logs)
  printf '%s' "$best"
}

# Single loop, two backstops (grace is clamped <= wall cap):
#   - never-identified log: STARTUP_GRACE_SECS -> exit 4 (no-start), checked first.
#   - identified but no terminal marker: WALL_CAP_SECS -> exit 4 (wall-cap).
# Once identified, watch it: failure first, then success, then stall.
log=""
while :; do
  [ -z "$log" ] && log=$(pick_active)
  # No-start fast-fail (retry-favoring HEURISTIC; see header exit 4(a)).
  if [ -z "$log" ] && [ $(( $(now) - start )) -ge "$STARTUP_GRACE_SECS" ]; then
    report "$log" 4 "no-start (no job log within ${STARTUP_GRACE_SECS}s)"
    exit 4
  fi
  if [ $(( $(now) - start )) -ge "$WALL_CAP_SECS" ]; then
    report "$log" 4 "wall-cap"
    exit 4
  fi
  if [ -n "$log" ] && [ -f "$log" ]; then
    if grep -qE "$fail_re" "$log" 2>/dev/null; then report "$log" 2 "turn-failed"; exit 2; fi
    if grep -qE "$done_re" "$log" 2>/dev/null; then report "$log" 0 "completed";  exit 0; fi
    m=$(mtime "$log")
    if [ -n "$m" ] && [ $(( $(now) - m )) -ge "$STALL_SECS" ]; then
      report "$log" 3 "stall"
      exit 3
    fi
  fi
  sleep "$POLL"
done
