#!/usr/bin/env python3
"""PreToolUse hook for the commit skill's git writes.

Only these forms pass silently:

    git add -- <explicit-file>...
    git commit -m <message>

Every recognized add/commit form other than these is denied, so the agent retries with the
regular form instead of asking the user for permission. A form this hook does not recognize as
add/commit (e.g. a launcher not in WRAPPERS such as watch/strace, or git inside a `bash -c`
string) falls through here, but it does not begin with `git add`/`git commit`, so it misses the
settings allow globs and prompts via allow-miss rather than auto-running. Unrelated commands pass.

Additionally every `git reset` form is denied outright (it rewrites the index/HEAD and has no
safe variant to allow); the agent must ask the user to add an explicit allow rule if one is ever
truly needed, rather than running it.

Usage: configured as a Bash PreToolUse hook. Run with --selftest.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


CONTROL_CHARS = ";&|<>\n"
GLOB_CHARS = "*?[]{}"
TARGET_WORD = re.compile(r"(?<![\w-])(add|commit)(?![\w-])")
# Process wrappers that run the FOLLOWING command. Claude Code strips a documented set of these
# before matching a command against the allow list, so e.g. `time git commit --amend` would
# otherwise become `git commit --amend`, match `Bash(git commit *)`, and auto-run. This set MUST
# cover Claude Code's strip-list (as of 2.1.x: env, time, timeout, nice, nohup, stdbuf) so the
# inner add/commit is denied here instead (deny beats allow). The remaining entries
# (command/builtin/exec/ionice/setsid/xargs/sudo/doas/taskset/chrt) are NOT stripped — a git
# add/commit behind them does not match the allow glob and merely prompts — but denying them is
# safe-side and future-proofs against the strip-list growing, so they are kept as defense.
WRAPPERS = {
    "command", "builtin", "env", "exec", "time", "timeout", "nice", "ionice",
    "nohup", "setsid", "stdbuf", "xargs", "sudo", "doas", "taskset", "chrt",
}
GIT_VALUE_OPTIONS = {
    "-C", "-c", "--git-dir", "--work-tree", "--namespace",
    "--super-prefix", "--exec-path", "--config-env",
}


def _has_shell_syntax(text):
    """Detect control operators or live expansion outside single quotes."""
    single = double = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\" and not single and i + 1 < len(text):
            i += 2
            continue
        if char == "'" and not double:
            single = not single
        elif char == '"' and not single:
            double = not double
        elif not single and char in "$`":
            return True
        elif not single and not double and char in CONTROL_CHARS:
            return True
        i += 1
    return single or double


def _tokens(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _is_git(token):
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name in {"git", "git.exe"}


def _is_plain_git(token):
    return token == "git"


def _mentions_target(tokens):
    """Conservatively recognize a non-plain add/commit invocation."""
    for index, token in enumerate(tokens):
        if not _is_git(token):
            continue
        pos = index + 1
        while pos < len(tokens) and tokens[pos].startswith("-"):
            option = tokens[pos].split("=", 1)[0]
            pos += 1
            if option in GIT_VALUE_OPTIONS and "=" not in tokens[pos - 1]:
                pos += 1
        if pos >= len(tokens):
            continue
        subcommand = tokens[pos]
        # A substituted subcommand (`git $(printf commit)`) could be add/commit -> deny. Only the
        # subcommand position is checked, so `git log "$(date)"` (a non-add/commit read whose ARG
        # holds the substitution) is left to the normal flow, not denied here.
        if "$" in subcommand or "`" in subcommand:
            return True
        if subcommand not in {"add", "commit"}:
            continue
        prefix = tokens[:index]
        if (
            index == 0
            or not _is_plain_git(token)
            or any(item in WRAPPERS or item in {"cd", "&&", "||", ";", "|"} for item in prefix)
            or any(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", item) for item in prefix)
        ):
            return True
    return False


def _mentions_reset(tokens):
    """True when a git invocation actually runs the reset subcommand (any form).

    git reset is denied outright: it rewrites the index/HEAD and has no safe variant to allow.
    Mirrors _mentions_target's prefix analysis so a non-executing mention (`echo git reset`) is
    not denied -- only git at argv[0], behind a wrapper/control-op/VAR= prefix, or a non-plain
    git path counts as a real invocation.
    """
    for index, token in enumerate(tokens):
        if not _is_git(token):
            continue
        pos = index + 1
        while pos < len(tokens) and tokens[pos].startswith("-"):
            option = tokens[pos].split("=", 1)[0]
            pos += 1
            if option in GIT_VALUE_OPTIONS and "=" not in tokens[pos - 1]:
                pos += 1
        if pos >= len(tokens):
            continue
        if tokens[pos] != "reset":
            continue
        prefix = tokens[:index]
        if (
            index == 0
            or not _is_plain_git(token)
            or any(item in WRAPPERS or item in {"cd", "&&", "||", ";", "|"} for item in prefix)
            or any(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", item) for item in prefix)
        ):
            return True
    return False


def _tracked_file(root, path):
    """True only when path exactly names one tracked file, including a deletion."""
    relative = path.relative_to(root).as_posix()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--error-unmatch", "--", relative],
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    # Compare BYTES: git emits paths as UTF-8, and -z avoids quoting/escaping. Decoding via
    # text=True would use the locale codec (e.g. cp932 here) and corrupt or raise on a non-ASCII
    # name (PMX仕様.txt). Split the raw bytes on NUL and compare to the UTF-8-encoded path.
    entries = [entry for entry in result.stdout.split(b"\0") if entry]
    return result.returncode == 0 and entries == [relative.encode("utf-8")]


def _safe_add(args, root):
    if len(args) < 2 or args[0] != "--":
        return False
    root = Path(root).resolve()
    for raw_path in args[1:]:
        if (
            not raw_path
            or Path(raw_path).is_absolute()
            or raw_path.startswith(("~", ":"))
            or any(char in raw_path for char in GLOB_CHARS)
        ):
            return False
        path = (root / raw_path).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError:
            return False
        if path.is_symlink() or path.is_file():
            continue
        if path.is_dir() or not _tracked_file(root, path):
            return False
    return True


def _safe_commit(args):
    return len(args) == 2 and args[0] == "-m" and bool(args[1].strip())


def classify(command, root=None):
    """Return ("deny", reason) or ("pass", None)."""
    if not isinstance(command, str) or not command.strip():
        return "pass", None
    try:
        tokens = _tokens(command)
    except ValueError:
        if re.search(r"\bgit\b", command) and re.search(r"\breset\b", command):
            return "deny", "git reset is denied; ask the user to add an allow rule if truly needed"
        if re.search(r"\bgit\b", command) and TARGET_WORD.search(command):
            return "deny", "Unparseable git add/commit; retry with the regular form"
        return "pass", None

    if _mentions_reset(tokens):
        return "deny", "git reset is denied; ask the user to add an allow rule if truly needed"

    plain_target = (
        len(tokens) >= 2
        and _is_plain_git(tokens[0])
        and tokens[1] in {"add", "commit"}
    )
    if not plain_target:
        if _mentions_target(tokens):
            return "deny", "Use plain git add/commit from the repository cwd"
        return "pass", None
    if _has_shell_syntax(command):
        return "deny", "Do not combine or expand git add/commit commands"

    args = tokens[2:]
    root = root or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    safe = _safe_add(args, root) if tokens[1] == "add" else _safe_commit(args)
    if safe:
        return "pass", None
    return "deny", (
        "Retry with git add -- <explicit-file>... or "
        "git commit -m <message>"
    )


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
    cases = [
        ("git add -- .claude/hooks/guard-git-write.py", "pass"),
        ("git add .claude/hooks/guard-git-write.py", "deny"),
        ("git add -- .claude/hooks", "deny"),
        ("git add -- ../outside.py", "deny"),
        ("git add -- C:/outside.py", "deny"),
        ("git add -- missing-file.py", "deny"),
        ("git commit -m '件名\n\n本文 $5 `literal` > text'", "pass"),
        ("git commit -m 'it'\\''s fixed'", "pass"),
        ("git add", "deny"),
        ("git add .", "deny"),
        ("git add -A", "deny"),
        ("git add 'src/*'", "deny"),
        ("git add -- :/", "deny"),
        ("git commit", "deny"),
        ("git commit -m ''", "deny"),
        ("git commit -m 'x' --amend", "deny"),
        ("git commit -m 'x' -- file.py", "deny"),
        ("git add a.py && git commit -m 'x'", "deny"),
        ("git commit -m \"$MESSAGE\"", "deny"),
        ("git commit -m \"$(date)\"", "deny"),
        ("git add src/{a,b}.py", "deny"),
        ("git add a.py > staged.txt", "deny"),
        ("cd repo && git add a.py", "deny"),
        ("git -C repo commit -m 'x'", "deny"),
        ("VAR=x git commit -m 'x'", "deny"),
        ("command git add a.py", "deny"),
        ("time git commit --amend", "deny"),
        ("stdbuf git add -A", "deny"),
        ("xargs git commit -m 'x'", "deny"),
        ("git.exe commit -m 'x'", "deny"),
        ("/usr/bin/git commit -m 'x'", "deny"),
        ("git reset", "deny"),
        ("git reset --soft HEAD~1", "deny"),
        ("git reset --hard origin/main", "deny"),
        ("git reset -- file.py", "deny"),
        ("git -C repo reset --hard", "deny"),
        ("git.exe reset", "deny"),
        ("/usr/bin/git reset --hard", "deny"),
        ("time git reset --hard", "deny"),
        ("VAR=x git reset", "deny"),
        ("cd repo && git reset --hard", "deny"),
        ("echo git reset", "pass"),
        ("echo 'git reset --hard'", "pass"),
        ("git restore --staged a.py", "pass"),
        ("git status --short", "pass"),
        ("git diff --staged -- a.py", "pass"),
        ("git log --oneline -10", "pass"),
        ("git log commit", "pass"),
        ("echo git commit", "pass"),
        ("echo 'git commit -m x'", "pass"),
        ("", "pass"),
    ]
    failures = []
    for command, expected in cases:
        actual, _ = classify(command, Path(__file__).resolve().parents[2])
        if actual != expected:
            failures.append((command, expected, actual))
    if failures:
        for command, expected, actual in failures:
            print(f"FAIL expected={expected} actual={actual}: {command!r}")
        raise SystemExit(1)
    # Regression for the tracked-deletion byte comparison: git emits paths as UTF-8, so a
    # tracked non-ASCII name must compare equal (decoding via the locale codec, e.g. cp932,
    # used to corrupt or raise). The repo's sole tracked non-ASCII path is PMX仕様.txt.
    repo = Path(__file__).resolve().parents[2]
    if not _tracked_file(repo, repo / "docs/specs/pmx/PMX仕様.txt"):
        print("FAIL _tracked_file rejected a tracked non-ASCII path")
        raise SystemExit(1)
    print(f"ALL PASS ({len(cases)} cases + non-ASCII tracked-path check)")


if __name__ == "__main__":
    main()
