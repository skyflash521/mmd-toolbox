#!/usr/bin/env python3
"""PreToolUse hook: deny redundant `cd` into the project root.

The cwd of every Bash invocation is already the project root, so navigating back to it with
`cd`/`chdir`/`pushd` is pure noise. This hook denies any command that, in one of its logical
segments, changes directory to the project root (or an equivalent spelling), forcing the agent to
drop the leading `cd` and run the command directly. Navigating into a SUBDIRECTORY, parent, or any
other location is left untouched.

The hook guards the Bash tool, which runs Git Bash (POSIX sh). Only the bash directory-changing
verbs (`cd`, `chdir`, `pushd`) are recognized; PowerShell verbs would not execute here.

A "root" target is recognized by normalizing the argument (drive-letter case, `/` vs `\\`, Git
Bash `/f/...` drive form, trailing slash, `.`/`..` segments, relative-to-root resolution) and
comparing it to CLAUDE_PROJECT_DIR. These spellings all count as root: an absolute path equal to
the root (any case/separator/trailing slash), the Git Bash `/f/...` form, `.`, `./`, `.\\`, and the
literal `$CLAUDE_PROJECT_DIR` / `${CLAUDE_PROJECT_DIR}`. An argument carrying any OTHER
expansion/substitution/home (`$`, `` ` ``, `~`), an empty target (`cd` -> HOME), or `cd -`
(previous dir) is uncertain and passes (deny only when provably the root, to avoid false denials).

The command is tokenized as a whole (so quotes, line continuations, and here-doc bodies keep their
content together rather than being misread as statements) and split into segments on shell control
operators. A leading `cd root` and same-line compositions (`cd root && ...`, `cd root; ...`) are
caught; a redundant `cd` placed on a later line after another command is left to pass (a benign
miss preferred over a false denial, since distinguishing it from quoted/here-doc data needs a full
shell parser). Text after a `#` comment is ignored (bash does not execute it).

Usage: configured as a Bash PreToolUse hook. Run with --selftest.
"""

import json
import os
import posixpath
import re
import shlex
import sys


REASON = (
    "既にプロジェクトルートが cwd です。冗長な cd / Set-Location を外し、"
    "コマンドを直接実行してください（サブディレクトリへの cd は可）。"
)
# Directory-changing verbs (compared case-insensitively, path/extension stripped).
CD_VERBS = {"cd", "chdir", "pushd"}
# Control operators that separate logical segments within a line.
OPS = {";", "|", "||", "&", "&&", "|&"}
# Literal spellings of the project root that survive shlex (no real expansion happens here).
ENV_ROOT_TOKENS = {"$CLAUDE_PROJECT_DIR", "${CLAUDE_PROJECT_DIR}"}


def _tokens(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";()<>|&")
    lexer.whitespace_split = True
    lexer.commenters = "#"  # bash の # コメント(以降行末まで)を無視する
    lexer.escape = ""  # treat backslash literally so Windows paths (F:\...) survive
    return list(lexer)


def _split_segments(tokens):
    """Break a token stream into logical segments on control operators."""
    segments = []
    seg = []
    for token in tokens:
        if token in OPS:
            if seg:
                segments.append(seg)
                seg = []
        else:
            seg.append(token)
    if seg:
        segments.append(seg)
    return segments


def _verb_base(token):
    base = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base[:-4] if base.endswith(".exe") else base


def _is_cd_verb(token):
    return _verb_base(token) in CD_VERBS


def _target_arg(seg):
    """Return (is_cd, target) where target is the first positional directory argument or None.

    Flags (`cd -L`, `cd -P`, `cd --`) take no value and are skipped; the first non-flag token is
    the directory. `cd` with no positional argument targets HOME and yields None. `pushd -n` only
    manipulates the directory stack without changing cwd, so it is not treated as a cd.
    """
    if not seg or not _is_cd_verb(seg[0]):
        return False, None
    args = seg[1:]
    if _verb_base(seg[0]) == "pushd" and "-n" in args:
        return False, None
    for token in args:
        if token.startswith("-") and token != "-":
            continue
        return True, token
    return True, None


def _resolve(path, root):
    """Normalize a path to a comparable lowercase posix form, resolving relatives against root."""
    text = path.replace("\\", "/")
    # Git Bash のドライブ表記 /f/... を f:/... へ正規化(POSIX 風の絶対パスと区別する)。
    drive = re.match(r"^/([A-Za-z])/", text)
    if drive:
        text = drive.group(1) + ":/" + text[3:]
    if re.match(r"^[A-Za-z]:", text) or text.startswith("/"):
        full = text
    else:
        full = root.replace("\\", "/").rstrip("/") + "/" + text
    return posixpath.normpath(full).lower()


def _is_root_arg(arg, root):
    if not arg or arg == "-":
        return False
    if arg in ENV_ROOT_TOKENS:
        return True
    # Any other expansion/substitution/home is uncertain -> not provably root -> pass.
    if "$" in arg or "`" in arg or "~" in arg:
        return False
    return _resolve(arg, root) == _resolve(root, root)


def classify(command, root=None):
    """Return ("deny", reason) or ("pass", None)."""
    if not isinstance(command, str) or not command.strip():
        return "pass", None
    # here-doc(<<)を含むと本文がコマンド列に混ざり、安全に文を切り出せない。構造が不確定
    # なので全体を pass に倒す(誤 deny を避ける。本文中の冗長 cd の取りこぼしは無害)。
    if "<<" in command:
        return "pass", None
    root = root or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    try:
        tokens = _tokens(command)
    except ValueError:
        return "pass", None  # 引用不整合(bash では構文エラー)は不確定 -> pass
    for seg in _split_segments(tokens):
        is_cd, arg = _target_arg(seg)
        if is_cd and _is_root_arg(arg, root):
            return "deny", REASON
    return "pass", None


def main():
    if "--selftest" in sys.argv:
        selftest()
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return
    if data.get("tool_name") != "Bash":
        return
    command = (data.get("tool_input") or {}).get("command")
    decision, reason = classify(command)
    if decision == "deny":
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }))


def selftest():
    root = "f:/Repositories/Skyflash/mmd-toolbox"
    cases = [
        # redundant cd into the project root (various spellings) -> deny
        ("cd f:/Repositories/Skyflash/mmd-toolbox", "deny"),
        ("cd F:\\Repositories\\Skyflash\\mmd-toolbox", "deny"),
        ("cd 'f:/Repositories/Skyflash/mmd-toolbox/'", "deny"),
        ("cd \"f:/Repositories/Skyflash/mmd-toolbox\"", "deny"),
        ("cd f:/Repositories/Skyflash/mmd-toolbox && pytest", "deny"),
        ("cd f:/Repositories/Skyflash/mmd-toolbox;pytest", "deny"),
        ("pytest && cd f:/Repositories/Skyflash/mmd-toolbox", "deny"),
        ("echo && cd .", "deny"),  # echo は成功し && で cd . が実行される(有効・冗長)
        ("cd .", "deny"),
        ("cd ./", "deny"),
        ("cd .\\", "deny"),
        ("cd f:/Repositories/Skyflash/sub/../mmd-toolbox", "deny"),
        ("pushd f:/Repositories/Skyflash/mmd-toolbox", "deny"),
        ("chdir f:/Repositories/Skyflash/mmd-toolbox", "deny"),
        # pushd -n はディレクトリスタック操作のみで cwd を変えない(ルート移動でない) -> pass
        ("pushd -n .", "pass"),
        ("pushd -n f:/Repositories/Skyflash/mmd-toolbox && dirs -v", "pass"),
        ("cd -L .", "deny"),
        ("cd -P -- .", "deny"),
        ("cd $CLAUDE_PROJECT_DIR", "deny"),
        ("cd ${CLAUDE_PROJECT_DIR}", "deny"),
        ("cd \"$CLAUDE_PROJECT_DIR\"", "deny"),
        # Git Bash のドライブ表記 /f/... -> deny
        ("cd /f/Repositories/Skyflash/mmd-toolbox", "deny"),
        ("cd /f/Repositories/Skyflash/mmd-toolbox/", "deny"),
        # 先頭の cd は複数行でも捕捉する -> deny
        ("cd f:/Repositories/Skyflash/mmd-toolbox\npytest", "deny"),
        # subdirectory / parent / other / uncertain -> pass
        ("cd shakevmd", "pass"),
        ("cd shakevmd && pytest", "pass"),
        ("cd mmd_toolbox/vmd", "pass"),
        ("cd ..", "pass"),
        ("cd ../mmd-toolbox-other", "pass"),
        ("cd /", "pass"),
        ("cd /c/other", "pass"),
        ("cd -", "pass"),
        ("cd", "pass"),
        ("cd ~", "pass"),
        ("cd ~/mmd-toolbox", "pass"),
        ("cd \"$OTHER_DIR\"", "pass"),
        ("cd $(pwd)", "pass"),
        # 後続行に置かれた冗長 cd は取りこぼす(引用/here-doc データとの区別にパーサが要るため、
        # 誤 deny より無害な誤 pass を優先する) -> pass
        ("pytest\ncd .", "pass"),
        ("pytest\ncd shakevmd", "pass"),
        # 行継続・複数行引用・here-doc 本文中の cd は実行コマンドでない -> 誤 deny しない -> pass
        ("echo first \\\ncd .", "pass"),
        ("echo \"a\ncd .\"", "pass"),
        ("python - <<'PY'\ncd .\nPY", "pass"),
        ("python - <<'PY'\necho x; cd .\nPY", "pass"),  # here-doc 本文の ; cd . も誤 deny しない
        # 引用不整合(構文エラー)は不確定 -> pass
        ("echo \"unterminated cd f:/Repositories/Skyflash/mmd-toolbox", "pass"),
        # bash の # コメント内は実行されない -> pass
        ("echo ok # comment; cd .", "pass"),
        ("echo ok  # cd f:/Repositories/Skyflash/mmd-toolbox", "pass"),
        ("echo foo#bar; cd .", "pass"),  # 単語途中 # は厳密にはコメントでないが取りこぼし(無害)
        # not a cd at all / literal mention -> pass
        ("pytest", "pass"),
        ("echo 'cd f:/Repositories/Skyflash/mmd-toolbox'", "pass"),
        ("git status", "pass"),
        ("", "pass"),
        # PowerShell verbs do not run in Git Bash -> not treated as cd -> pass
        ("sl .", "pass"),
        ("Set-Location f:/Repositories/Skyflash/mmd-toolbox", "pass"),
    ]
    failures = []
    for command, expected in cases:
        actual, _ = classify(command, root)
        if actual != expected:
            failures.append((command, expected, actual))
    if failures:
        for command, expected, actual in failures:
            print(f"FAIL expected={expected} actual={actual}: {command!r}")
        sys.exit(1)
    print(f"ALL PASS ({len(cases)} cases)")
    sys.exit(0)


if __name__ == "__main__":
    main()
