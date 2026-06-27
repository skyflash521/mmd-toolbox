#!/usr/bin/env python3
"""PreToolUse hook: deny pytest wrapped in shell post-processing, steer to plain pytest.

A bare `pytest ...` / `python -m pytest ...` is allowlisted, but piping (`| tail`), chaining
(`; echo $?`), or file-redirecting (`> out`) it breaks the match and triggers a needless prompt
-- and those wrappers are pointless since `pytest -q` already prints a summary. Deny the wrapped
form with an instruction to run pytest on its own; the harmless `2>&1` / `/dev/null` redirect is
left alone. (Read-only tail/echo pipelines are already auto-approved by auto-approve-readonly.)

Regex-level detection is intentional: a false positive only yields a recoverable deny+message,
never a wrongful approval. Run with --selftest.
"""
import json
import re
import sys

REASON = (
    "pytest をパイプ(|)・コマンド連結(;/&&/||)・ファイルリダイレクト・${PIPESTATUS} と"
    "組み合わせないでください。素の pytest を単体で実行してください"
    "（例: .venv/Scripts/python.exe -m pytest <path> -q）。-q が要約を出すので、"
    "末尾や終了コードだけ見たい場合も後処理を足さず出力をそのまま読みます。"
)
IS_PYTEST = re.compile(r"(?:^|[|;&(`]\s*)[\w./\\-]*pytest\b|\bpython[\w.]*(?:\.exe)?\s+-m\s+pytest\b")
WRAP = re.compile(r"[|;&<>`]|\$\(")


def classify(command):
    """Return ("deny", reason) or ("pass", None)."""
    if not isinstance(command, str) or not IS_PYTEST.search(command):
        return "pass", None
    clean = re.sub(r"2>&1|1>&2|[0-9&]*>>?\s*/dev/null", " ", command)
    return ("deny", REASON) if WRAP.search(clean) else ("pass", None)


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
        ('.venv/Scripts/python.exe -m pytest x -q 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"', "deny"),
        ("pytest x | tail -6", "deny"),
        ("pytest; echo done", "deny"),
        ("pytest && echo ok", "deny"),
        ("python -m pytest x > out.txt", "deny"),
        ("pytest x 2>err.log", "deny"),
        ("out=$(pytest -q); echo \"$out\"", "deny"),
        ("echo $(pytest -q)", "deny"),
        # bare pytest (harmless redirects ok) -> pass
        ("pytest", "pass"),
        ("pytest mocapvmd/tests/test_default_profile.py -q", "pass"),
        (".venv/Scripts/python.exe -m pytest x -q", "pass"),
        ("pytest x 2>&1", "pass"),
        ("pytest x 2>/dev/null", "pass"),
        # not a pytest command -> pass
        ("tail -6 log | head", "pass"),
        ("echo 'run pytest later' | tail", "pass"),
        ("", "pass"),
    ]
    bad = [(c, e, classify(c)[0]) for c, e in cases if classify(c)[0] != e]
    for c, e, a in bad:
        print(f"FAIL expected={e} actual={a}: {c!r}")
    print("ALL PASS" if not bad else "SOME FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
