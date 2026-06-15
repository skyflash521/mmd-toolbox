#!/usr/bin/env python3
"""PreToolUse hook: auto-approve provably read-only Bash commands.

Purpose: keep autonomous runs from stalling on permission prompts for read-only
inspection commands (and read-only pipes of them) that the glob allowlist cannot match.

Design (capability-based allowlist + strict static-syntax constraints):
  * Only bare verbs that have NO write/exec/state-change mode under ANY flag are
    eligible (SAFE). Tools that can write or run programs even while "inspecting" are
    deliberately excluded and fall through to the normal permission flow.
  * The command must be fully STATIC: it is rejected (-> pass-through) if it contains
    any variable/parameter expansion or substitution ("$" or backticks or process
    substitution), any redirect, any variable assignment or env-var prefix, a
    path-qualified command (a verb containing "/", e.g. ./cat), a subshell/group, or
    any segment whose verb is not SAFE. Static-only avoids deferred-semantics bypasses
    (PATH=., S=-s, $VAR-as-flag) and same-named local executables.
  * shlex comment handling is disabled so "echo ok#; rm x" is not silently truncated.

Behavior:
  * Eligible & clean -> "allow": runs with no prompt.
  * Anything else    -> prints nothing, exits 0 (PASS-THROUGH): the normal permission
        flow decides (existing allow rules still auto-allow; otherwise the user is
        prompted). It NEVER auto-denies and never emits "deny".
  * Parse errors / unexpected input also pass through, so a bug here can only ever
    re-introduce a prompt, never wrongly approve a write.
  * The OS-level command sandbox is a separate layer and still applies.

Usage: configured as a PreToolUse hook for the Bash tool. Run with --selftest.
"""
import sys
import json
import re
import shlex

# Bare verbs with no write/exec/state-change capability under any flag. Output-file or
# command-exec capable tools (awk, sed, find, rg, sort, uniq, xxd, jq, git, tee, diff
# [BSD -l runs pr], file [-C writes .mgc], date [-s/positional sets clock], printf
# [-v assigns a shell var], ...) are intentionally excluded; their single read-only
# forms remain covered by the existing glob allowlist via pass-through.
SAFE = {
    "cat", "head", "tail", "ls", "wc", "nl", "grep", "egrep", "fgrep", "cut", "tr",
    "comm", "cmp", "basename", "dirname", "pwd", "echo", "stat", "readlink", "rev",
    "tac", "fold", "column", "seq", "true", "false", "test", "[", "od", "realpath",
}
OPS = {";", "|", "||", "&&", "&", "|&", "\n"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def classify(cmd):
    """Return "allow" if cmd is provably read-only, else "pass" (normal flow)."""
    if not isinstance(cmd, str) or not cmd.strip():
        return "pass"
    # Any expansion/substitution defers or hides semantics -> never auto-approve.
    if "$" in cmd or "`" in cmd or "<(" in cmd or ">(" in cmd:
        return "pass"
    # Drop harmless redirects to /dev/null and stderr<->stdout; any remaining redirect
    # (file write/append, or file-fed input) -> pass through.
    clean = cmd
    for pat in (r"\d*>>?\s*/dev/null\b", r"&>>?\s*/dev/null\b", r"2>&1", r"1>&2"):
        clean = re.sub(pat, " ", clean)
    if re.search(r"[<>]", clean):
        return "pass"
    try:
        segments = []
        for line in clean.split("\n"):
            if not line.strip():
                continue
            lex = shlex.shlex(line, posix=True, punctuation_chars=True)
            lex.commenters = ""  # do not treat '#' as a comment (Bash word-literal)
            tokens = list(lex)
            seg = []
            for t in tokens:
                if t in OPS:
                    if seg:
                        segments.append(seg)
                        seg = []
                else:
                    seg.append(t)
            if seg:
                segments.append(seg)
    except Exception:
        return "pass"
    if not segments:
        return "pass"
    for seg in segments:
        # No variable assignments / env-var prefixes anywhere in a segment.
        if any(ASSIGN.match(t) for t in seg):
            return "pass"
        verb = seg[0]
        if "/" in verb:
            return "pass"  # path-qualified command (./cat, /bin/sh, ...)
        if verb not in SAFE:
            return "pass"
    return "allow"


def main():
    if "--selftest" in sys.argv:
        _selftest()
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # pass-through on bad input
    if data.get("tool_name") != "Bash":
        sys.exit(0)
    cmd = (data.get("tool_input") or {}).get("command")
    if classify(cmd) == "allow":
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }}))
    # else: print nothing -> normal permission flow (allow rules, else prompt).
    sys.exit(0)


def _selftest():
    cases = [
        # read-only, static, bare verbs -> allow
        ("grep -nE 'x' /path/log", "allow"),
        ("grep x f | head -5", "allow"),
        ("cat a | grep b | wc -l", "allow"),
        ("ls -la | wc -l", "allow"),
        ("cat a 2>/dev/null", "allow"),
        ("echo hi; echo bye", "allow"),
        ("tail -n 50 /var/log/x | grep err", "allow"),
        # excluded verbs -> pass
        ("grep x f | sort | uniq -c", "pass"),
        ("awk '{print}' f", "pass"),
        ("sed -n 1p f", "pass"),
        ("find . -name x", "pass"),
        ("git log --oneline | head", "pass"),
        ("jq . f.json", "pass"),
        ("xxd in out", "pass"),
        ("diff a b", "pass"),
        ("file x", "pass"),
        ("date", "pass"),
        ("printf -v PATH .", "pass"),
        # variable / assignment / expansion -> pass
        ("L=/p/x.log; grep -oE 'a' \"$L\"", "pass"),
        ("PATH=.; cat README.md", "pass"),
        ("S=-s; date $S '2030-01-01'", "pass"),
        ("FOO=bar cat x", "pass"),
        ("grep '$' f", "pass"),
        # comment-truncation / path-qualified / danger / syntax -> pass
        ("echo ok#; touch /tmp/pwn", "pass"),
        ("./cat README.md", "pass"),
        ("/bin/cat x", "pass"),
        ("cat x > out.txt", "pass"),
        ("rm -rf /tmp/x", "pass"),
        ("cat $(which rm)", "pass"),
        ("echo ok`rm x`", "pass"),
        ("echo hi; rm x", "pass"),
        ("cat x > /dev/nullfoo", "pass"),
        ("(rm x)", "pass"),
        ("", "pass"),
    ]
    ok = True
    for cmd, want in cases:
        got = classify(cmd)
        flag = "OK  " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"{flag} want={want:5} got={got:5} :: {cmd!r}")
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
