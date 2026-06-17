#!/usr/bin/env python3
"""PreToolUse hook: gate destructive/scope-widening `git add` / `git commit` args.

Purpose: in autonomous runs we want plain `git add <files>` and `git commit -m ...`
to pass without a prompt (they are reversible), while arguments that rewrite history,
bypass verification, or widen the staged/committed scope are routed back to a human
prompt. The broad `Bash(git add *)` / `Bash(git commit *)` allow rules in settings.json
would otherwise auto-approve those dangerous forms too.

Design (mirror of auto-approve-readonly.py's safety stance, inverted):
  * This hook ONLY ever emits "ask" (or stays silent / pass-through). It never emits
    "allow" and never "deny". Worst case on a bug it re-introduces a prompt — it can
    never wrongly approve a write and never hard-block a command outright. Returning
    "ask" beats the broad "allow" rule because PreToolUse composes deny > ask > allow.
  * It is a SEPARATE hook so the existing auto-approve-readonly.py keeps its invariant
    ("only allow read-only, never deny"). The two run on the same Bash matcher; ask wins.
  * Only `git add` / `git commit` are inspected. Every other command passes through
    untouched (so this hook never adds prompts to anything that works today).
  * Fail-safe to "ask": any shell expansion ($ / backtick / $()/ <()/ >()), env-var
    prefix, parse failure, or unrecognized shape around a git add/commit segment yields
    "ask" rather than silently letting a possibly-hidden dangerous flag through.

Behavior:
  * git add/commit with a gated flag/pathspec, or ambiguous -> "ask".
  * Anything else -> prints nothing, exits 0 (pass-through): normal flow decides
    (the broad allow rule auto-approves plain add/commit; unrelated commands unaffected).

Usage: configured as a PreToolUse hook for the Bash tool. Run with --selftest.
"""
import sys
import json
import re
import shlex

OPS = {";", "|", "||", "&&", "&", "|&", "\n"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Global `git` options that consume the following token as their value. Used to skip
# past them when locating the subcommand (e.g. `git -C path commit` -> subcommand commit).
GIT_GLOBAL_VALUE_OPTS = {"-C", "--git-dir", "--work-tree", "--namespace",
                         "--super-prefix", "--exec-path"}

# commit flags that must not auto-pass. Long forms compared after stripping a trailing
# "=value"; short flags matched per-char inside a bundle like "-an".
COMMIT_GATED_LONG = {
    "amend", "no-verify", "all", "allow-empty", "allow-empty-message",
    "reuse-message", "reedit-message", "fixup", "squash", "include", "only", "reset-author",
    "no-gpg-sign",  # bypasses configured commit signing (verification bypass)
    # pathspec read from an external file: committed set is opaque in the command line.
    "pathspec-from-file", "pathspec-file-nul",
}
COMMIT_GATED_SHORT = set("anCcio")  # -a all, -n no-verify, -C/-c reuse/reedit, -i include, -o only
# Non-gated commit flags that consume the FOLLOWING token as their value. Skip that token
# so a message/path value (e.g. `-m .`) is never mistaken for a broad pathspec.
COMMIT_VALUE_SHORT = set("mFt")  # -m message, -F file, -t template
COMMIT_VALUE_LONG = {"message", "file", "template", "author", "date", "cleanup", "trailer"}

# add flags that must not auto-pass.
ADD_GATED_LONG = {
    "all", "update", "no-ignore-removal", "intent-to-add",
    "patch", "interactive", "edit", "force", "chmod", "sparse", "renormalize",
    "ignore-errors",  # continues past failures -> silent partial staging
    # pathspec read from an external file: staged set is opaque in the command line.
    "pathspec-from-file", "pathspec-file-nul",
}
ADD_GATED_SHORT = set("AuNpief")  # -A -u -N -p -i -e -f

# Broad pathspecs for `git add` that stage far more than an explicit file.
BROAD_PATHSPECS = {".", "*", ":/", ":", "::", ":/:"}


_SQUOTE = re.compile(r"'[^']*'")          # single-quoted span: fully literal in shell
_DQUOTE = re.compile(r'"(?:[^"\\]|\\.)*"')  # double-quoted span: literal redirects, live $

def _has_shell_expansion(s):
    # $ / backtick / process-substitution that the shell would expand. Single-quoted text
    # is literal, so strip it first (a literal $ in 'msg' is not expansion). Double-quoted
    # $ IS expanded by the shell, so it is intentionally kept.
    t = _SQUOTE.sub(" ", s)
    if "$" in t or "`" in t or "<(" in t or ">(" in t:
        return True
    # Brace expansion ({a,b}, {1..5}) expands to multiple words, so the actual staged set
    # is not the literal token. It does NOT expand inside single OR double quotes, nor when
    # a brace is backslash-escaped, so strip double-quoted spans and escaped chars first and
    # check only the unquoted, unescaped remainder.
    t = _DQUOTE.sub(" ", t)
    t = re.sub(r"\\.", " ", t)  # \{ \} etc. are literal -> not brace expansion
    return bool(re.search(r"\{[^{}]*(?:,|\.\.)[^{}]*\}", t))

def _has_redirect(s):
    # A redirect can create/overwrite a file. Redirect operators inside quotes (single OR
    # double) are literal, so strip both kinds of quoted spans first; then ignore the
    # harmless /dev/null and fd-dup forms before looking for any remaining < or >.
    t = _DQUOTE.sub(" ", _SQUOTE.sub(" ", s))
    for pat in (r"\d*>>?\s*/dev/null\b", r"&>>?\s*/dev/null\b", r"2>&1", r"1>&2"):
        t = re.sub(pat, " ", t)
    return bool(re.search(r"[<>]", t))


def _split_on_unquoted_newlines(cmd):
    """Split cmd at newlines that are OUTSIDE single/double quotes, honoring backslash
    escapes. A newline inside a quoted argument (e.g. a multi-line `git commit -m`
    message) stays within its line, so shlex can parse the quoted value as one token;
    real newline-separated commands still split into separate lines (the compound-command
    security boundary is unchanged). Single quotes are literal (no escapes inside);
    outside and inside double quotes a backslash escapes the next char."""
    lines = []
    buf = []
    sq = dq = False  # inside single- / double-quoted span
    i, n = 0, len(cmd)
    while i < n:
        c = cmd[i]
        # Backslash escape: outside quotes, or inside double quotes (not inside single).
        if c == "\\" and not sq and i + 1 < n:
            buf.append(c)
            buf.append(cmd[i + 1])
            i += 2
            continue
        if c == "'" and not dq:
            sq = not sq
            buf.append(c)
        elif c == '"' and not sq:
            dq = not dq
            buf.append(c)
        elif c == "\n" and not sq and not dq:
            lines.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    lines.append("".join(buf))
    return lines


def _split_segments(cmd):
    """Split a command line into subcommand token-lists. Raises on parse failure."""
    segments = []
    for line in _split_on_unquoted_newlines(cmd):
        if not line.strip():
            continue
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.commenters = ""  # '#' is a literal word, not a comment (match sibling hook)
        seg = []
        for t in list(lex):
            if t in OPS:
                if seg:
                    segments.append(seg)
                    seg = []
            else:
                seg.append(t)
        if seg:
            segments.append(seg)
    return segments


def _gated_short(token, gated_set):
    """token like '-an'; return True if any char (after the dash) is gated."""
    return token.startswith("-") and not token.startswith("--") and \
        any(ch in gated_set for ch in token[1:])


def _long_name(token):
    """'--no-verify=x' -> 'no-verify'; '--amend' -> 'amend'."""
    return token[2:].split("=", 1)[0]


def _is_gated_long(name, gated_set):
    """git accepts unambiguous prefix abbreviations (--amen == --amend), so gate when the
    typed name is a prefix of (or equals) any gated option name. Ambiguous short prefixes
    over-match to ask, which is the safe direction."""
    return bool(name) and any(g.startswith(name) for g in gated_set)


def _commit_gated(args):
    after_ddash = False
    i = 0
    while i < len(args):
        t = args[i]
        if t == "--" and not after_ddash:
            after_ddash = True
            i += 1
            continue
        if not after_ddash and t.startswith("--"):
            name = _long_name(t)
            if _is_gated_long(name, COMMIT_GATED_LONG):
                return True
            if "=" not in t and name in COMMIT_VALUE_LONG:
                i += 2  # skip the space-form value
                continue
            i += 1
            continue
        if not after_ddash and t.startswith("-") and len(t) > 1:
            if _gated_short(t, COMMIT_GATED_SHORT):
                return True
            if t[-1] in COMMIT_VALUE_SHORT:
                i += 2  # bundle ends in a value-taking flag (-m/-F/-t): skip its value
                continue
            i += 1
            continue
        # positional / pathspec (a bare `git commit <pathspec>` commits working-tree
        # content of those paths, same scope-widening risk as `git add`).
        if t in BROAD_PATHSPECS or t.startswith(":") or any(c in t for c in "*?[]"):
            return True
        i += 1
    return False


def _add_gated(args):
    after_ddash = False
    for t in args:
        if t == "--" and not after_ddash:
            after_ddash = True
            continue
        if not after_ddash:
            if t.startswith("--"):
                if _is_gated_long(_long_name(t), ADD_GATED_LONG):
                    return True
                continue
            if t.startswith("-") and len(t) > 1:
                if _gated_short(t, ADD_GATED_SHORT):
                    return True
                continue
        # positional / pathspec (either before or after '--')
        if t in BROAD_PATHSPECS or t.startswith(":") or any(c in t for c in "*?[]"):
            return True
    return False


def _subcommand_and_args(seg):
    """For a segment, return (subcommand, args, danger) if it's a git invocation, else
    (None, None, False). subcommand is '?' when it's git but the subcommand can't be
    resolved (-> caller asks). danger is True when a `-c`/`--config-env` global config
    injection is present (can set commit.gpgsign=false / core.hookspath=... to bypass
    verification), so the caller asks for add/commit."""
    i = 0
    # skip leading env-var assignments (FOO=bar git ...) — handled as ambiguous by caller
    had_assign = False
    while i < len(seg) and ASSIGN.match(seg[i]):
        had_assign = True
        i += 1
    if i >= len(seg):
        return (None, None, False)
    verb = seg[i]
    base = verb.rsplit("/", 1)[-1]  # tolerate /usr/bin/git
    if base != "git":
        return (None, None, False)
    i += 1
    # skip git global options; flag config-injection ones as dangerous
    danger = False
    while i < len(seg):
        t = seg[i]
        if t == "-c" or t == "--config-env":
            danger = True
            i += 2  # option + its value (`-c key=val`, `--config-env name=VAR`)
            continue
        if t.startswith("--config-env="):
            danger = True
            i += 1
            continue
        if t in GIT_GLOBAL_VALUE_OPTS:
            i += 2  # option + its value
            continue
        if t.startswith("-"):
            i += 1  # valueless global option (e.g. -p, --no-pager, --paginate)
            continue
        break
    if i >= len(seg):
        return ("?", None, danger)  # git with no resolvable subcommand
    sub = seg[i]
    args = seg[i + 1:]
    return ("?" if had_assign else sub), args, danger


def classify(cmd):
    """Return ("ask", reason) to force a prompt, or ("pass", None) for normal flow.

    A command is only allowed to pass (deferring to the broad git add/commit allow rule)
    when EVERY segment is a non-gated git add/commit. If a git add/commit is compounded
    with anything else (another git subcommand, rm, a pipe target, ...), we ask: the broad
    allow rule could otherwise auto-approve the whole compound, and we will not depend on
    the host's compound-decomposition for a security boundary."""
    if not isinstance(cmd, str) or not cmd.strip():
        return ("pass", None)
    # Cheap pre-filter: if it can't contain a git add/commit, never interfere.
    if "git" not in cmd or ("add" not in cmd and "commit" not in cmd):
        return ("pass", None)
    try:
        segments = _split_segments(cmd)
    except Exception:
        # Parse failed but it mentions git + add/commit -> can't verify safety.
        return ("ask", "git add/commit が解析できないため確認します")
    saw_git_write = False
    all_segments_safe_write = True  # True only while every segment is a safe git add/commit
    for seg in segments:
        sub, args, danger = _subcommand_and_args(seg)
        if sub == "?":
            # git, but subcommand hidden behind env prefix / unresolved -> ambiguous
            if "add" in seg or "commit" in seg:
                return ("ask", "git add/commit の引数が確定できないため確認します")
            all_segments_safe_write = False
            continue
        if sub is None or sub not in ("add", "commit"):
            all_segments_safe_write = False  # non-git, or a different git subcommand
            continue
        # It IS git add / git commit.
        saw_git_write = True
        if danger:
            return ("ask", "git の -c 設定注入(署名やフックの迂回の恐れ)があるため確認します")
        if sub == "commit" and _commit_gated(args):
            return ("ask", "git commit に履歴改変/検証迂回/一括コミット系の引数が含まれるため確認します")
        if sub == "add" and _add_gated(args):
            return ("ask", "git add に一括ステージ/広域 pathspec 系の引数が含まれるため確認します")
    # No git add/commit invocation present -> never interfere (the redirect/expansion
    # checks below must not fire on unrelated commands like `echo git commit < in.txt`).
    if not saw_git_write:
        return ("pass", None)
    # A real git add/commit invocation exists. Apply whole-command safety so the broad
    # allow rule can't auto-approve more than a plain add/commit.
    if _has_shell_expansion(cmd):
        return ("ask", "git add/commit にシェル展開が含まれ引数を検証できないため確認します")
    if _has_redirect(cmd):
        return ("ask", "git add/commit にリダイレクトが含まれるため確認します")
    if not all_segments_safe_write:
        return ("ask", "git add/commit が他のコマンドと複合しており全体を検証できないため確認します")
    return ("pass", None)


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
    decision, reason = classify(cmd)
    if decision == "ask":
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": reason,
        }}))
    # else: print nothing -> normal permission flow.
    sys.exit(0)


def _selftest():
    cases = [
        # plain, reversible -> pass (broad allow auto-approves)
        ("git commit -m 'msg'", "pass"),
        ("git add path/to/file.py", "pass"),
        ("git add -- a.py b.py", "pass"),
        ("git commit -m 'x' -- a.py", "pass"),
        ("git add .claude/skills/autonomous-dev/SKILL.md", "pass"),
        ("git commit -s -m 'signed'", "pass"),
        # history rewrite / verify bypass / bulk -> ask
        ("git commit --amend -m 'x'", "ask"),
        ("git commit -m 'x' --amend", "ask"),
        ("git commit --amend=reword", "ask"),
        ("git commit --no-verify -m 'x'", "ask"),
        ("git commit -n -m 'x'", "ask"),
        ("git commit -a -m 'x'", "ask"),
        ("git commit -an -m 'x'", "ask"),
        ("git commit --all -m 'x'", "ask"),
        ("git commit --fixup HEAD", "ask"),
        ("git commit --squash HEAD", "ask"),
        ("git commit --allow-empty -m 'x'", "ask"),
        ("git commit -C HEAD~1", "ask"),
        # bulk / broad add -> ask
        ("git add -A", "ask"),
        ("git add --all", "ask"),
        ("git add .", "ask"),
        ("git add -u", "ask"),
        ("git add -p", "ask"),
        ("git add -N x", "ask"),
        ("git add 'src/*'", "ask"),
        ("git add -- .", "ask"),
        ("git add :/", "ask"),
        ("git add --pathspec-from-file=list.txt", "ask"),
        ("git add --pathspec-from-file list.txt", "ask"),
        ("git commit --pathspec-from-file=list.txt -m 'x'", "ask"),
        # broad pathspec on commit (working-tree scope grab) -> ask
        ("git commit -m 'x' -- .", "ask"),
        ("git commit -m 'x' .", "ask"),
        ("git commit -m 'x' -- :/", "ask"),
        ("git commit -m 'x' -- 'src/*'", "ask"),
        ("git commit --include . -m 'x'", "ask"),
        ("git commit -i . -m 'x'", "ask"),
        # commit message/value that happens to look like a pathspec -> still pass
        ("git commit -m .", "pass"),
        ("git commit -sm 'msg'", "pass"),
        ("git commit -m 'x' -- a.py", "pass"),
        # redirects around git add/commit -> ask; harmless /dev/null form stays pass
        ("git add f.py > out.txt", "ask"),
        ("git commit -m 'x' > out.txt", "ask"),
        ("git add f.py 2>/dev/null", "pass"),
        # metadata / scope-widening flags -> ask
        ("git add --chmod=+x script.sh", "ask"),
        ("git add --sparse path/to/file", "ask"),
        ("git commit --reset-author -m 'x'", "ask"),
        ("git commit --no-gpg-sign -m 'x'", "ask"),
        ("git add --ignore-errors f.py", "ask"),
        # redirect/expansion must NOT fire on non-git-add/commit commands
        ("echo git commit < in.txt", "pass"),
        ("echo 'git add .' > note.txt", "pass"),
        # literal $ / > inside single quotes is not expansion/redirect -> pass
        ("git commit -m 'cost is $5'", "pass"),
        ("git commit -m 'use a > b'", "pass"),
        # live expansion in double quotes / unquoted -> ask
        ("git commit -m \"cost $TOTAL\"", "ask"),
        # angle bracket inside double quotes is a literal, not a redirect -> pass
        ("git commit -m \"a > b\"", "pass"),
        # renormalize restages tracked files broadly -> ask
        ("git add --renormalize path/to/file", "ask"),
        ("git add --renormalize .", "ask"),
        # brace expansion enumerates multiple paths via the shell -> ask; quoted is literal
        ("git add src/{a,b}.py", "ask"),
        ("git commit -m 'fix {a,b}'", "pass"),
        ("git commit -m \"fix {a,b}\"", "pass"),
        ("git add src/\\{a,b\\}.py", "pass"),  # escaped braces are literal, no expansion
        # compound: all-safe git writes -> pass; anything else mixed in -> ask
        ("git add -A && git commit -m 'x'", "ask"),
        ("git add f.py && git commit -m 'x'", "pass"),
        ("git add a.py && git add b.py && git commit -m 'x'", "pass"),
        ("git add f.py && rm -rf build", "ask"),
        ("git commit -m 'x'; git push", "ask"),
        ("git add f.py | tee out", "ask"),
        ("git add f.py || true", "ask"),
        ("git commit -m 'x' && git status", "ask"),
        ("cd repo && git commit -m 'x'", "ask"),
        # multi-line commit message: newlines are INSIDE the double-quoted value, so the
        # whole command is one line/segment -> plain git commit -> pass (no stray prompt)
        ('git commit -m "subject\n\n- bullet1\n- bullet2\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>"', "pass"),
        ('git commit -m "subject line\n\nbody paragraph"', "pass"),
        # UNQUOTED newline still separates commands -> compound boundary preserved (security)
        ("git add f.py\ngit commit -m 'x'", "pass"),   # both are safe git writes
        ("git add f.py\nrm x", "ask"),                  # a non-write command after a newline
        ("git commit -m 'x'\ngit push", "ask"),         # newline-separated push must still ask
        # newlines inside single quotes are part of the value, not command boundaries -> pass
        ("git commit -m 'subject\n\n- bullet1\n- bullet2'", "pass"),
        # a backslash-escaped newline (line continuation) keeps it one command -> pass
        ("git add a.py\\\nb.py", "pass"),
        # shell expansion around git add/commit -> ask
        ("git commit -m \"$MSG\"", "ask"),
        ("git add $FILES", "ask"),
        ("git commit -m `whoami`", "ask"),
        # env prefix hides behavior -> ask
        ("GIT_AUTHOR_NAME=x git commit -m 'y'", "ask"),
        # global option then subcommand
        ("git -C /repo commit --amend", "ask"),
        ("git -C /repo commit -m 'y'", "pass"),  # -C (run in dir) is benign
        # -c config injection can disable signing/hooks -> ask (even benign-looking config)
        ("git -c user.name=x commit -m 'y'", "ask"),
        ("git -c commit.gpgSign=false commit -m 'y'", "ask"),
        ("git -c core.hooksPath=/dev/null commit -m 'y'", "ask"),
        ("git --config-env=COMMIT_GPGSIGN=v commit -m 'y'", "ask"),
        # unambiguous long-option abbreviations of gated flags -> ask
        ("git commit --amen -m 'x'", "ask"),
        ("git commit --no-veri -m 'x'", "ask"),
        ("git add --renorm path/to/file", "ask"),
        # --only / -o commit (working-tree scope change, mirror of --include) -> ask
        ("git commit --only file.txt -m 'x'", "ask"),
        ("git commit -o file.txt -m 'x'", "ask"),
        # unrelated commands -> pass (never interfere)
        ("git status", "pass"),
        ("git log --oneline | head", "pass"),
        ("git diff --staged", "pass"),
        ("echo git commit --amend", "pass"),
        ("pytest -q", "pass"),
        ("", "pass"),
    ]
    ok = True
    for cmd, want in cases:
        got, _ = classify(cmd)
        flag = "OK  " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"{flag} want={want:4} got={got:4} :: {cmd!r}")
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
