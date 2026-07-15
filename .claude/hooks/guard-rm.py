#!/usr/bin/env python3
"""PreToolUse hook: deny Bash `rm`(削除でなく trash.py によるごみ箱送りへ誘導する)。

一時文書・未追跡ファイルであっても、削除は取り消せない場合がある(git追跡外・エディタの
ローカル履歴にも残らない等)。rm はコマンド先頭がどこにあっても常に deny し、同じ引数で
trash.py(削除せずOS標準のごみ箱へ送る可逆な代替。settings.json で無プロンプト許可済み)を
使わせる。ユーザー確認を都度挟まず自律進行を止めないまま、誤削除を可逆にする。

Usage: configured as a Bash PreToolUse hook. Run with --selftest.
"""
import json
import shlex
import sys


def has_rm(command):
    """コマンド内のどこかで、セグメント先頭が rm(パス修飾・拡張子形含む)か。"""
    if "<<" in command:  # here-doc 本文は安全に切り出せないので対象外
        return False
    lexer = shlex.shlex(command.replace("\n", "\n;"), posix=True, punctuation_chars=";()<>|&")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    at_head = True
    for token in tokens:
        if token[:1] in ";|&<>(){}":
            at_head = True
            continue
        if at_head and token.replace("\\", "/").rsplit("/", 1)[-1] in ("rm", "rm.exe"):
            return True
        at_head = False
    return False


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return
    if data.get("tool_name") != "Bash":
        return
    command = (data.get("tool_input") or {}).get("command") or ""
    if has_rm(command):
        reason = (
            "rm は常に deny します。同じ引数で python3 .claude/hooks/trash.py <path>... を"
            "使ってください(削除でなくOS標準のごみ箱へ送る可逆な代替です)。"
        )
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}))


def selftest():
    deny_cases = [
        "rm foo.txt",
        "rm -rf /tmp/x",
        "echo hi; rm x",
        "cat a && rm b",
        "rm a | cat",
        "/bin/rm x",
        "rm.exe x",
        "cd t && rm x",
        "echo prep # c\nrm x",
    ]
    pass_cases = [
        "echo rm",
        "echo 'rm x'",
        "cat rm.txt",
        "ls -la",
        "grep rm f",
        "true # rm x",
        "cat <<EOF\nrm x\nEOF",
    ]
    ok = True
    for case in deny_cases:
        if not has_rm(case):
            ok = False
            print("FAIL expected deny:", repr(case))
    for case in pass_cases:
        if has_rm(case):
            ok = False
            print("FAIL expected pass:", repr(case))
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    main()
