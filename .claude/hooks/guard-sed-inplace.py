#!/usr/bin/env python3
"""PreToolUse hook: deny `sed -i`(インプレース編集は Edit ツールで行う)。

Edit/Grep で代替できる sed のインプレース編集を Bash で打つと、書き込みのため許可プロンプトが
出る。それを deny して Edit ツールへ誘導する。コマンド先頭の sed だけを対象にし(引数中の sed は
無視)、値フラグ(-e/-f)の引数・複合コマンド・複数行・行末コメントを踏まえて現実的な sed -i を判定する。

Usage: configured as a Bash PreToolUse hook. Run with --selftest.
"""
import json
import re
import shlex
import sys


def is_inplace_flag(flag):
    """sed のフラグが in-place 編集(-i / -i.bak / -ni / --in-place)か。-e/-f/-l の値部分は除外。"""
    if flag.startswith("--in-place"):
        return True
    if not flag.startswith("-") or flag.startswith("--"):
        return False
    return "i" in re.split("[efl]", flag[1:])[0]


def has_sed_inplace(command):
    """コマンド内のどこかで、コマンド先頭の sed が in-place 編集を行うか。"""
    if "<<" in command:  # here-doc 本文は安全に切り出せないので対象外
        return False
    lexer = shlex.shlex(command.replace("\n", "\n;"), posix=True, punctuation_chars=";()<>|&")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    at_head = True
    is_sed = False
    for token in tokens:
        if token[:1] in ";|&<>(){}":
            at_head, is_sed = True, False
        elif at_head:
            is_sed = token.replace("\\", "/").rsplit("/", 1)[-1] in ("sed", "sed.exe")
            at_head = False
        elif is_sed and is_inplace_flag(token):
            return True
    return False


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return
    if data.get("tool_name") != "Bash":
        return
    command = (data.get("tool_input") or {}).get("command") or ""
    if has_sed_inplace(command):
        reason = (
            "ファイルのインプレース書き換え(sed -i)は Edit ツールで行ってください。"
            "awk -i inplace・perl -i・python -c での読み書き等、別の手段で同じ書き換えを"
            "回避して実行しないこと。"
        )
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}))


def selftest():
    deny_cases = [
        "sed -i 's/a/b/' f",
        "sed -i.bak 's/a/b/' f",
        "sed -ni 'p' f",
        "sed --in-place 's/a/b/' f",
        "cd t && sed -i x f",
        "sed -e 's/a/b/' -i f",
        "echo x\nsed -i 's/a/b/' f",
        "echo prep # c\nsed -i x f",
    ]
    pass_cases = [
        "sed -n '1,5p' f",
        "sed -e 's/time/date/' f",
        "sed -e's/time/date/' f",
        "sed -ffix.sed f",
        "grep sed -i",
        "git grep sed -i",
        "echo sed -i",
        "true # sed -i",
        "sed -n '1,5p' f; grep -i x f",
        'echo "a\nsed -i b"',
        "cat <<EOF\nsed -i x\nEOF",
    ]
    ok = True
    for case in deny_cases:
        if not has_sed_inplace(case):
            ok = False
            print("FAIL expected deny:", repr(case))
    for case in pass_cases:
        if has_sed_inplace(case):
            ok = False
            print("FAIL expected pass:", repr(case))
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    main()
