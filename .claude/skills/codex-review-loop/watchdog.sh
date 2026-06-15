#!/usr/bin/env bash
# codex-review-loop watchdog — read-only. Watches the codex companion job log for
# THIS round and exits with a code describing codex's fate. Performs NO writes
# (safe to allowlist as one command, so the loop runs without permission prompts).
#
#   exit 0  normal end   (companion logged "Turn completed." or a "Final output" line)
#   exit 2  failure        (companion logged "Turn failed.")
#   exit 3  stall / hang    (job-log mtime has not advanced for STALL_SECS — heuristic)
#   exit 4  cap reached     (WALL_CAP_SECS elapsed; also the path taken when this
#                            round's job log is never identified — we wait the full cap
#                            rather than fail fast, so a healthy run is never killed
#                            early just because its log could not be located)
#
# Before exiting it prints, to stdout:
#     LOG=<path>            (empty if no log was identified)
#     OUTCOME=<code> <reason>
# so the caller reads its background output file to learn WHICH log holds the result.
#
# Args (optional, positional): $1=STALL_SECS  $2=WALL_CAP_SECS  $3=STATE_ROOT
#
# Identifying THIS round's log: discovery is scoped to the CURRENT repo's state
# directory ("<repo-basename>-<hash>/jobs"). This round's log is one that did NOT exist
# at startup (it is new since our snapshot); the newest such log wins. The watchdog
# snapshots within ~100ms of launch, while the companion needs ~1s of node/thread
# startup before it creates the log, so this round's log reliably appears AFTER the
# snapshot. A still-running orphan log from a prior or failed round stays in the
# baseline and is therefore never mistaken for this round's, even while it is being
# written.
#
# Accepted limitation (best-effort safety net; the agent response is the primary
# result channel, so a wrong watchdog signal degrades to a retry, never a wrong fix):
# two checkouts of a same-named repo share the "<repo>-*" prefix, so a truly
# concurrent codex run in the other checkout could be selected. The exact state hash
# is not derivable here, and the scenario is pathological.
set -u

STALL_SECS="${1:-420}"
WALL_CAP_SECS="${2:-1200}"
STATE_ROOT="${3:-${STATE_ROOT:-${HOME:-}/.claude/plugins/data/codex-openai-codex/state}}"
POLL=5

mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null; }
now() { date +%s; }

# Terminal markers, anchored to the companion's own log lines, which begin with a full
# ISO timestamp ("[2026-06-15T03:00:13.785Z] Turn completed."). The strict YYYY-MM-DDT
# anchor stops the same words inside an assistant message, a `Running command: ...
# "Turn completed"` echo, or a short "[2]"/"[note]" bracket line from faking an end.
ts_re='^\[[0-9]{4}-[0-9]{2}-[0-9]{2}T[^]]*\] '
done_re="${ts_re}(Turn completed\.|Final output)"
fail_re="${ts_re}Turn failed\."
report() { printf 'LOG=%s\nOUTCOME=%s %s\n' "$1" "$2" "$3"; }

# Scope discovery to THIS repo's state dir(s). No fall-back to the whole STATE_ROOT:
# if the "<repo>-*" convention ever stops matching, discovery simply finds nothing and
# the watcher degrades to the WALL_CAP timer — which is safe — rather than risk
# selecting an unrelated repo's log.
repo=$(basename "$PWD" 2>/dev/null || printf '')
list_logs() {
  local d
  find "$STATE_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${repo}-*" 2>/dev/null \
  | while IFS= read -r d; do
      [ -n "$d" ] && find "$d/jobs" -type f -name 'task-*.log' 2>/dev/null
    done
}

# True if $1 was present at startup (exact line match against the baseline set).
in_baseline() { printf '%s\n' "$baseline" | awk -v k="$1" '$0==k{f=1} END{exit !f}'; }

start=$(now)

# Snapshot the set of pre-existing logs (paths only).
baseline=$(list_logs | sort)

# Pick the newest log that did not exist at startup (this round's).
pick_active() {
  local p cm best="" bestmt=0
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    in_baseline "$p" && continue
    cm=$(mtime "$p"); [ -n "$cm" ] || continue
    if [ "$cm" -ge "$bestmt" ]; then bestmt=$cm; best=$p; fi
  done < <(list_logs)
  printf '%s' "$best"
}

# Single loop until WALL_CAP. While this round's log is not yet identified, keep trying
# (so a log that only appears after a slow codex start is still picked up at ANY time
# before the cap). Once identified, watch it: failure first, then success, then stall.
# WALL_CAP is the hard guarantee and bounds the never-identified case (log empty -> 4).
log=""
while :; do
  if [ $(( $(now) - start )) -ge "$WALL_CAP_SECS" ]; then
    report "$log" 4 "wall-cap"
    exit 4
  fi
  [ -z "$log" ] && log=$(pick_active)
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
