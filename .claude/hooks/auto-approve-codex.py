#!/usr/bin/env python3
"""PreToolUse hook: auto-approve the single codex-companion runtime invocation.

Why a hook and not an allow-glob: the codex prompt is a multi-line double-quoted
argument. Claude Code splits a Bash command on newlines into subcommands and
requires EACH to match an allow rule, so the inner prompt lines (which do not start
with `node`) never match and the call always prompts. This hook recognizes one
well-formed `node ... codex-companion.mjs ...` call and approves it (same emit
pattern as auto-approve-readonly.py). It never denies; anything unexpected, or a
command that chains a second statement outside the quoted prompt, passes through to
the normal permission flow.
"""
import sys
import json
import re

# `node <script path ending in codex-companion.mjs>` as the leading command.
LEAD = re.compile(r'^\s*node\s+"?[^"\n]*codex-companion\.mjs"?(\s|$)')
# Operators that chain a second command but are inert inside double quotes.
DANGER = re.compile(r'[;&|]|<\(|>\(|[<>]')


def is_codex(cmd):
    if not isinstance(cmd, str) or not LEAD.match(cmd):
        return False
    # Command substitution ($()) and backticks run even inside double quotes, so
    # reject them anywhere in the command.
    if "$" in cmd or "`" in cmd:
        return False
    # The remaining operators are inert inside double quotes: strip the quoted
    # regions, then ensure the unquoted skeleton has none that would smuggle a
    # second command past this approval.
    skeleton = re.sub(r'"(?:[^"\\]|\\.)*"', '', cmd, flags=re.S)
    return DANGER.search(skeleton) is None


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # pass-through on bad input
    if data.get("tool_name") != "Bash":
        sys.exit(0)
    cmd = (data.get("tool_input") or {}).get("command")
    if is_codex(cmd):
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }}))
    sys.exit(0)


if __name__ == "__main__":
    main()
