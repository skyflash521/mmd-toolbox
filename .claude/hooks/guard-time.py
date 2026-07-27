#!/usr/bin/env python3
"""PreToolUse guard for `date`: auto-approve a standalone read-only clock read so the
agent can fetch the current time without a prompt, and deny clock changes.

  * read-only forms (bare, +FORMAT, -u/-R/-I, -d/--date <arg>, ...) -> allow
  * clock-setting forms (-s/--set, or a legacy positional date) -> deny
  * anything compound / with expansion / substitution / redirect -> pass-through
    (exit 0, no output): the normal permission flow / other hooks decide.

This is deliberately the only concern here; bundling extra commands with `date`
falls through to the normal flow rather than being auto-approved.
"""
import json
import re
import shlex
import sys

# date flags that consume the next token as their (read-only) argument.
TAKES_ARG = {"-d", "--date", "-r", "--reference", "-f", "--file"}


def decide(cmd):
    """Return "allow"/"deny" for a standalone static `date`, else None (pass-through)."""
    if not isinstance(cmd, str) or not cmd.strip():
        return None
    # Expansion/substitution/redirect/operator -> not a standalone static command.
    if re.search(r"[$`<>;&|\n(]", cmd):
        return None
    try:
        tokens = shlex.split(cmd, posix=True)
    except Exception:
        return None
    if not tokens or tokens[0] != "date":
        return None

    args = tokens[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        # `set` is date's only state-changing flag. It appears either token-leading
        # (-s, -sVALUE, --set, --set=VALUE, unique abbreviations --s/--se/--s=VALUE) or
        # after boolean short flags in a cluster (-us2030 == -u -s2030). No read-only
        # flag starts with -s/--s, and a read-only short cluster's second char is a
        # value-taking flag (d/f/r/I); reject anything else as a possible clock set.
        if arg.startswith(("-s", "--s")):
            return "deny"
        if arg in TAKES_ARG:
            i += 2  # flag + its separate read-only argument
        elif arg.startswith("+") or arg.startswith("--"):
            i += 1  # +FORMAT or a read-only long flag
        elif arg.startswith("-") and (len(arg) == 2 or (len(arg) > 2 and arg[1] in "dfrI")):
            i += 1  # single short read flag (-u/-R/-I) or -X<value> (-Iseconds, -d20:03)
        else:
            return "deny"  # bare positional or short cluster -> possible clock set
    return "allow"


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    if data.get("tool_name") != "Bash":
        sys.exit(0)

    command = (data.get("tool_input") or {}).get("command")
    decision = decide(command)
    if decision is None:
        sys.exit(0)  # pass-through

    out = {"hookEventName": "PreToolUse", "permissionDecision": decision}
    if decision == "deny":
        out["permissionDecisionReason"] = (
            "date による時計変更は不可。現在時刻は読み取り専用の date を使う。"
            "PowerShellのSet-Date・w32tm・pythonのos/time経由での時刻変更等、"
            "別の手段で同じ変更を回避して実行しないこと。"
        )
    print(json.dumps({"hookSpecificOutput": out}))


if __name__ == "__main__":
    main()
