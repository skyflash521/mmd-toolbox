#!/usr/bin/env python3
"""PreToolUse hook: auto-approve the codex-watchdog watchdog.sh launch.

watchdog.sh is a committed, read-only script (find/stat/grep/sleep over the codex
job-state dir), shared by any skill that launches codex:codex-rescue (codex-review-loop,
codex-consult, ...). Its `bash <script> <args>` launch matches the settings allow glob,
but Claude Code can still down-grade an allow-matched `bash <script>` command to an
un-overridable "ask" inside the permission pipeline, so the caller prompts anyway. A
PreToolUse "allow" decision is returned directly as behavior:allow and overrides that,
so this hook approves exactly the single, un-chained watchdog invocation and passes
everything else through to the normal permission flow (it never denies).

Safety: only a lone `bash .claude/skills/codex-watchdog/watchdog.sh ...` with no
chaining/expansion/redirect is approved, so the approval cannot smuggle a second
command. Anything else prints nothing and exits 0 (pass-through), so a bug here can only
re-introduce a prompt, never wrongly approve.

Usage: configured as a Bash PreToolUse hook. Run with --selftest.
"""
import json
import re
import shlex
import sys

WATCHDOG = ".claude/skills/codex-watchdog/watchdog.sh"


def is_watchdog(cmd):
    """True only for a single, un-chained `bash <watchdog.sh> [args...]` command."""
    if not isinstance(cmd, str) or not cmd.strip():
        return False
    # Any chaining/expansion/redirect could append a second command -> never approve.
    if "$" in cmd or "`" in cmd or "<(" in cmd or ">(" in cmd or re.search(r"[<>;|&\n]", cmd):
        return False
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return False
    return len(tokens) >= 2 and tokens[0] == "bash" and tokens[1] == WATCHDOG


def main():
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # pass-through on bad input
    if data.get("tool_name") != "Bash":
        sys.exit(0)
    cmd = (data.get("tool_input") or {}).get("command")
    if is_watchdog(cmd):
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }}))
    sys.exit(0)


def _selftest():
    w = "bash " + WATCHDOG
    cases = [
        (w + ' 420 1200 "" 240 vprv0test9m2', True),  # the reported command
        (w, True),
        (w + " 420 1200 '' 240 tok", True),
        (w + " 420 1200 /some/state 240 tok", True),
        (w + " ; rm -rf x", False),
        (w + " && echo done", False),
        (w + " | grep x", False),
        (w + " > out.txt", False),
        (w + " $(rm x)", False),
        (w + " `rm x`", False),
        ("bash .claude/skills/other.sh", False),
        ("bash /abs/.claude/skills/codex-watchdog/watchdog.sh", False),
        ("cat README.md", False),
        ("", False),
    ]
    ok = True
    for cmd, want in cases:
        got = is_watchdog(cmd)
        if got != want:
            ok = False
            print(f"FAIL want={want} got={got} :: {cmd!r}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
