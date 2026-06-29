#!/usr/bin/env python3
"""PreToolUse hook: deny pytest run with a leading env-var assignment, steer to plain pytest.

`PYTHONPATH=. pytest ...` is redundant -- pyproject.toml の [tool.pytest.ini_options] が
`pythonpath = ["."]` を持つので pytest 自身がルートを sys.path へ足す。さらに先頭の env 代入は
bash 専用構文で allowlist にマッチせず無駄なプロンプトを生む。素の pytest に誘導する。

Regex-level detection is intentional: a false positive only yields a recoverable deny+message,
never a wrongful approval. Run with --selftest.
"""
import json
import re
import sys

REASON = (
    "pytest の前に PYTHONPATH= などの環境変数代入を置かないでください。"
    "pyproject.toml の pythonpath=[\".\"] が既にルートを通すので不要で、"
    "先頭の env 代入は allowlist にマッチせず無駄なプロンプトを生みます。"
    "素の pytest を実行してください（例: .venv/Scripts/python.exe -m pytest <path> -q）。"
)
ENV_PREFIX = re.compile(r"^\s*(?:[A-Za-z_]\w*=\S*\s+)+")
IS_PYTEST = re.compile(r"^[\w./\\-]*pytest\b|^[\w./\\-]*python[\w.]*(?:\.exe)?\s+-m\s+pytest\b")


def classify(command):
    """Return ("deny", reason) or ("pass", None)."""
    if not isinstance(command, str):
        return "pass", None
    m = ENV_PREFIX.match(command)
    if not m:
        return "pass", None
    return ("deny", REASON) if IS_PYTEST.match(command[m.end():]) else ("pass", None)


def main():
    if "--selftest" in sys.argv:
        selftest()
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return
    if data.get("tool_name") != "Bash":
        return
    decision, reason = classify((data.get("tool_input") or {}).get("command"))
    if decision == "deny":
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}))


def selftest():
    cases = [
        ("PYTHONPATH=. .venv/Scripts/python.exe -m pytest lipsync -q", "deny"),
        ("PYTHONPATH=. pytest", "deny"),
        ("PYTHONPATH=.:src pytest x -q", "deny"),
        ("FOO=1 BAR=2 python -m pytest x", "deny"),
        # no env prefix -> pass
        (".venv/Scripts/python.exe -m pytest lipsync -q", "pass"),
        ("pytest x -q", "pass"),
        # env prefix on a non-pytest command -> pass
        ("PYTHONPATH=. python script.py", "pass"),
        ("FOO=1 echo hi", "pass"),
        ("", "pass"),
    ]
    bad = [(c, e, classify(c)[0]) for c, e in cases if classify(c)[0] != e]
    for c, e, a in bad:
        print(f"FAIL expected={e} actual={a}: {c!r}")
    print("ALL PASS" if not bad else "SOME FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
