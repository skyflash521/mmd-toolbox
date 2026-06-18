#!/usr/bin/env python3
"""PreToolUse hook: gate destructive/scope-widening `git add` / `git commit` args.

Purpose: in autonomous runs we want plain `git add <files>` and `git commit -m ...`
to pass without a prompt (they are reversible), while arguments that rewrite history,
bypass verification, or widen the staged/committed scope are routed back to a human
prompt. The broad `Bash(git add *)` / `Bash(git commit *)` allow rules in settings.json
would otherwise auto-approve those dangerous forms too.

Staying auto-approved (callers, read this before constructing a git command): THREE things
can stop a plain auto-approve, with different causes and severities:
  (1) Hard DENY (this hook): command substitution (`$(...)`, backtick) or a heredoc (`<<`)
      inside a git add/commit segment — the prompt-inducing `$(cat <<'EOF'...)` form. It is
      blocked outright, not merely prompted. (`$(`/backtick expand even inside double quotes;
      `<<` is the heredoc operator outside quotes. A backslash-escaped `\$(` is literal and
      not denied; a non-git segment's substitution is not denied either.)
  (2) Allow-glob miss (NOT this hook): the settings.json globs match only commands that
      BEGIN with `git add` / `git commit`. A form that shifts those leading tokens passes
      this hook yet matches no allow rule, so it still prompts — e.g.
      `git -C <path> commit ...` (begins `git -C`, not `git commit`).
  (3) This hook emits "ask": an env-var prefix (`VAR=x git ...`), a compound that mixes in
      a non-(git add/commit) segment (`cd <path> && git ...`, `git add f && rm x`), a bare
      `$` / process-substitution `<()`/`>()` / brace `{a,b}` expansion, or a redirect around
      an add/commit. Note a compound of ONLY safe git writes still passes
      (`git add f && git commit -m "..."`).
The Bash tool's cwd is already the repo root, so run plain `git add` / `git commit` from
the cwd (no `cd`, no `-C`/`--git-dir`/`--work-tree`, no env prefix) and keep the message in
a single `-m "..."` double-quoted string — NEVER `$(...)`/backtick/heredoc (those are denied)
and avoid a bare `$` (it prompts as expansion). Braces, newlines, and the `<noreply@...>`
trailer inside the quotes are fine. (`-F <file>` is NOT gated and matches the allow rule, so
it does not prompt; it is discouraged only to keep one message style.)

Design (mirror of auto-approve-readonly.py's safety stance, inverted):
  * This hook emits "ask" (force a prompt) or, for ONE narrow case, "deny" (hard block);
    otherwise it stays silent / pass-through. It never emits "allow". The single "deny"
    case is command substitution / heredoc on a git add/commit (`$(...)`, backtick, `<<`):
    these have no legitimate use there (the message is a literal `-m "..."`, not assembled
    by a subshell or fed from a heredoc) and are the prompt-inducing `$(cat <<'EOF'...)`
    pattern, so they are blocked outright rather than merely prompted. Everything else that
    is risky is "ask", never "allow"; worst case on a bug it re-introduces a prompt. "deny"
    beats "ask" beats "allow" (PreToolUse composes deny > ask > allow).
  * It is a SEPARATE hook so the existing auto-approve-readonly.py keeps its invariant
    ("only allow read-only, never deny"). The two run on the same Bash matcher; ask wins
    over that hook's allow, and this hook's deny wins over everything.
  * Only `git add` / `git commit` are inspected. Every other command passes through
    untouched (so this hook never adds prompts to anything that works today). Plain human
    commit forms (`git commit -m "msg"`, multiple `-m`, an editor commit, `-F <file>`) have
    no `$(`/backtick/`<<` and are never denied.
  * Fail-safe to "ask": any other shell expansion ($ / brace / <()/ >()), env-var prefix,
    redirect, parse failure, or unrecognized shape around a git add/commit segment yields
    "ask" rather than silently letting a possibly-hidden dangerous flag through.

Behavior:
  * git add/commit with command substitution / heredoc (`$(...)`/backtick/`<<`) -> "deny".
  * git add/commit with a gated flag/pathspec, other expansion/redirect, or ambiguous -> "ask".
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


_DQUOTE = re.compile(r'"(?:[^"\\]|\\.)*"')  # double-quoted span (run after single quotes blanked)


def _blank_single_quoted(s):
    """Return s with the CONTENTS of single-quoted spans replaced by spaces, tracking
    double-quote state so an apostrophe INSIDE a double-quoted string (`"it's $x"`) is literal
    text, NOT a single-quote delimiter. A naive `'[^']*'` regex cannot tell the two apart and
    would wrongly swallow `$x` between two such apostrophes, hiding live expansion/substitution
    (a deny/ask bypass). Backslash escapes are honored outside single quotes. _DQUOTE (run
    AFTER this) can then safely strip double-quoted spans, since any `"` that sat inside single
    quotes is already blanked."""
    out = []
    sq = dq = False
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and not sq and i + 1 < n:
            out.append(c)
            out.append(s[i + 1])
            i += 2
            continue
        if c == "'" and not dq:
            sq = not sq
            out.append(" ")
        elif c == '"' and not sq:
            dq = not dq
            out.append(c)
        else:
            out.append(" " if sq else c)
        i += 1
    return "".join(out)


def _has_shell_expansion(s):
    # $ / backtick / process-substitution that the shell would expand. Single-quoted text
    # is literal, so blank it first (a literal $ in 'msg' is not expansion). Double-quoted
    # $ IS expanded by the shell, so it is intentionally kept.
    t = _blank_single_quoted(s)
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
    t = _DQUOTE.sub(" ", _blank_single_quoted(s))
    for pat in (r"\d*>>?\s*/dev/null\b", r"&>>?\s*/dev/null\b", r"2>&1", r"1>&2"):
        t = re.sub(pat, " ", t)
    return bool(re.search(r"[<>]", t))


def _has_cmdsubst_or_heredoc(s):
    # Command substitution (`$(...)` / backtick) and heredocs (`<<`) have NO legitimate use
    # in a git add/commit: the message must be a literal `-m "..."` string, not assembled
    # by a subshell or fed from a heredoc. They are the prompt-inducing `$(cat <<'EOF'...)`
    # pattern. Unlike other expansions (handled by "ask"), these are hard-DENIED so the
    # only path is the correct literal form. Single-quoted spans are literal (strip them);
    # backslash-escaped chars (`\$`, `` \` ``) are literal too (neutralize them) so an
    # escaped `\$(` is not mistaken for live substitution. Command substitution expands even
    # inside double quotes, so `$(`/backtick are checked with double quotes kept; the heredoc
    # `<<` operator lives outside quotes, so both quote kinds are stripped for it.
    t = _blank_single_quoted(s)
    t = re.sub(r"\\\n", "", t)   # join line continuations FIRST: `$\<nl>(` is live `$(`
    t = re.sub(r"\\.", " ", t)   # then neutralize other backslash-escapes (`\$`, `` \` ``)
    if "$(" in t or "`" in t:
        return True
    t = _DQUOTE.sub(" ", t)
    return "<<" in t


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


def _raw_segments(cmd):
    """Split cmd into RAW segment strings on unquoted `;`/`|`/`&`/newline (quote- and
    escape-aware). Unlike _split_segments (which returns shlex token lists), this keeps the
    original text so callers can scan a single segment for substitution/heredoc syntax that
    tokenizing would normalize away. Consecutive operator chars (`&&`, `||`, `|&`) collapse
    into one boundary."""
    segs = []
    buf = []
    sq = dq = False
    i, n = 0, len(cmd)
    while i < n:
        c = cmd[i]
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
        elif not sq and not dq and c in ";|&\n":
            if buf:
                segs.append("".join(buf))
                buf = []
            while i + 1 < n and cmd[i + 1] in ";|&":
                i += 1
        else:
            buf.append(c)
        i += 1
    if buf:
        segs.append("".join(buf))
    return segs


def _tokenize_segment(rs):
    """shlex-tokenize a single raw segment (no operator splitting). Used only to identify
    whether the segment is a git add/commit; raises on parse failure."""
    lex = shlex.shlex(rs, posix=True, punctuation_chars=True)
    lex.commenters = ""
    return [t for t in lex if t not in OPS]


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


def _locate_subcommand(seg):
    """Locate the git subcommand in a token list. Returns (sub_index, had_assign, danger),
    or None if the segment is not a git invocation. sub_index is the index of the subcommand
    token, or None when git has no subcommand. had_assign is True if an env-var prefix
    (`FOO=bar git ...`) precedes git. danger is True for a `-c`/`--config-env` global config
    injection (can set commit.gpgsign=false / core.hooksPath=... to bypass verification).
    Shared by _subcommand_and_args (ask/pass tier) and _deny_target_subcommand (deny tier)
    so the skip logic lives in one place."""
    i = 0
    had_assign = False
    while i < len(seg) and ASSIGN.match(seg[i]):
        had_assign = True
        i += 1
    if i >= len(seg) or seg[i].rsplit("/", 1)[-1] != "git":  # tolerate /usr/bin/git
        return None
    i += 1
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
    return (i if i < len(seg) else None, had_assign, danger)


def _subcommand_and_args(seg):
    """For a segment, return (subcommand, args, danger) if it's a git invocation, else
    (None, None, False). subcommand is '?' when it's git but the subcommand can't be
    resolved (env prefix, or none present) -> caller asks. danger flags `-c`/`--config-env`
    global config injection, so the caller asks for add/commit."""
    loc = _locate_subcommand(seg)
    if loc is None:
        return (None, None, False)
    idx, had_assign, danger = loc
    if idx is None:
        return ("?", None, danger)  # git with no resolvable subcommand
    sub = seg[idx]
    return ("?" if had_assign else sub), seg[idx + 1:], danger


def _deny_target_subcommand(seg):
    """True if the segment is a git command whose REAL subcommand is add/commit, or cannot
    be statically confirmed to be a different plain subcommand (the subcommand position is
    itself a command substitution). Lets the deny tier hard-block cmdsubst/heredoc inside a
    git add/commit even behind an env prefix, while NOT denying a confirmed non-write
    subcommand (log/status/...) — so env-prefixed `git status` with `commit` only as an
    ARGUMENT is not mistaken for a commit."""
    loc = _locate_subcommand(seg)
    if loc is None:
        return False
    idx, _had_assign, _danger = loc
    if idx is None:
        return False  # `git` with no subcommand
    sub = seg[idx]
    # add/commit is the target. Anything that is NOT a clean lowercase subcommand word
    # cannot be confirmed to be a non-write subcommand — e.g. a command-substituted
    # subcommand leaves a `$`/`(`/`$(...)` fragment after tokenizing — so treat it as a
    # target too. A real non-write subcommand (log/status/rev-parse/...) is a clean word and
    # is NOT denied (its substitution would be in args, not the subcommand position).
    return sub in ("add", "commit") or not re.fullmatch(r"[a-z][a-z0-9-]*", sub)


def classify(cmd):
    """Return ("deny", reason) to hard-block, ("ask", reason) to force a prompt, or
    ("pass", None) for normal flow.

    A command is only allowed to pass (deferring to the broad git add/commit allow rule)
    when EVERY segment is a non-gated git add/commit. If a git add/commit is compounded
    with anything else (another git subcommand, rm, a pipe target, ...), we ask: the broad
    allow rule could otherwise auto-approve the whole compound, and we will not depend on
    the host's compound-decomposition for a security boundary."""
    if not isinstance(cmd, str) or not cmd.strip():
        return ("pass", None)
    # Cheap pre-filter: skip only when there is neither a git add/commit NOR any
    # substitution/heredoc to inspect, so an obfuscated subcommand (`git $(...)` whose text
    # never spells "commit") still reaches the deny tier below.
    if "git" not in cmd or (
        "add" not in cmd and "commit" not in cmd
        and "$(" not in cmd and "`" not in cmd and "<<" not in cmd
    ):
        return ("pass", None)
    # DENY tier (evaluated FIRST — before parsing and before any "ask"): command substitution
    # / heredoc inside a git add/commit segment is the prompt-inducing `$(cat <<'EOF'...)`
    # form and has no legitimate use; hard-block it so it cannot be approved away even when
    # combined with a gated flag or env prefix. Run BEFORE _split_segments so a shlex parse
    # failure (e.g. a stray quote inside a heredoc body) cannot drop it to "ask". Checked per
    # RAW segment so substitution in a NON-git segment (`echo "$(date)" && git commit ...`)
    # is NOT denied (it falls through to the "ask" tier).
    deny_reason = ("git add/commit でのコマンド置換/ヒアドキュメント($(...)・バッククォート・<<)"
                   "は禁止。メッセージは -m にダブルクォート文字列で渡すこと(複数行は単一 -m の"
                   "実改行、または段落ごとに -m を分ける)")
    for rs in _raw_segments(cmd):
        if not rs.strip():
            continue
        try:
            toks = _tokenize_segment(rs)
        except Exception:
            # Tokenizing failed (e.g. an unbalanced quote in a half-written message). Fall
            # back to a textual check so a git add/commit with cmdsubst/heredoc is still
            # denied here, not dropped to the parse-fail "ask" below. A segment that merely
            # starts with (optional env-prefix then) `git` and carries the dangerous syntax
            # is blocked conservatively — such input is malformed and won't run cleanly anyway.
            if re.match(r"\s*(?:[A-Za-z_]\w*=\S*\s+)*git\b", rs) and _has_cmdsubst_or_heredoc(rs):
                return ("deny", deny_reason)
            continue
        if _deny_target_subcommand(toks) and _has_cmdsubst_or_heredoc(rs):
            return ("deny", deny_reason)
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
    # A real git add/commit invocation exists (and already passed the DENY tier above).
    # Apply whole-command safety so the broad allow rule can't auto-approve more than a
    # plain add/commit.
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
    if decision in ("ask", "deny"):
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
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
        # shell expansion around git add/commit -> ask (bare $ / unquoted var, no $()/backtick)
        ("git commit -m \"$MSG\"", "ask"),
        ("git add $FILES", "ask"),
        # command substitution / heredoc on git add/commit -> DENY (hard block; the
        # prompt-inducing `$(cat <<'EOF'...)` anti-pattern). backtick/$() expand even inside
        # double quotes; `<<` is the heredoc operator outside quotes.
        ("git commit -m `whoami`", "deny"),
        ("git commit -m \"$(date)\"", "deny"),
        ('git commit -m "$(cat <<\'EOF\'\nmsg\nEOF\n)"', "deny"),
        ("git commit -F - <<EOF\nmsg\nEOF", "deny"),
        ("git add `echo f.py`", "deny"),
        # DENY runs BEFORE ask: a gated flag or env prefix must not let cmdsubst slip to ask
        ("git commit --amend -m \"$(date)\"", "deny"),
        ("GIT_AUTHOR_NAME=x git commit -m \"$(date)\"", "deny"),
        # per-segment: cmdsubst in a NON-git segment -> not denied (compound -> ask);
        # cmdsubst in the git segment -> deny even alongside another segment
        ("echo \"$(date)\" && git commit -m 'msg'", "ask"),
        ("git commit -m \"$(date)\" && git status", "deny"),
        # backslash-escaped substitution is literal, not live -> not denied (bare $ -> ask)
        ("git commit -m \"\\$(date)\"", "ask"),
        # deny runs BEFORE parsing: a stray quote in a heredoc body breaks shlex, but the
        # heredoc git segment is still denied (not dropped to the parse-fail "ask")
        ("git commit -F - <<EOF\n'\nEOF", "deny"),
        # subcommand produced by substitution (quoted OR unquoted, single OR multi) -> the
        # subcommand isn't a clean word, can't confirm non-write -> deny
        ("git \"$(printf commit)\" -m msg", "deny"),
        ("git $(printf commit) -m msg", "deny"),
        ("git $(printf com;printf mit) -m msg", "deny"),
        # tokenize failure (unbalanced quote) must NOT drop a git+cmdsubst to "ask" -> deny
        ("git commit -m \"$(date)", "deny"),
        # a clean non-write subcommand with substitution in ARGS is not denied (sub is a word)
        ("git log \"$(date)\"", "pass"),
        ("git p4 \"$(date)\"", "pass"),                  # digit in a real subcommand is clean
        ("git l \"$(date)\"", "pass"),                    # single-char alias is a clean word
        # apostrophes INSIDE a double-quoted message are literal, not single-quote delimiters;
        # the substitution/expansion between them must still be seen (no quote-state bypass)
        ("git commit -m \"it's $(date) isn't\"", "deny"),
        ("git commit -m \"it's $TOTAL isn't\"", "ask"),
        ("git commit -m 'it is $(date) fine'", "pass"),    # truly single-quoted -> literal
        # line continuation forms live substitution/heredoc after joining -> deny
        ("git commit -m \"$\\\n(date)\"", "deny"),
        ("git commit -F - <\\\n<EOF", "deny"),
        # env-prefixed non-write subcommand with `commit` only as an ARG -> not denied as a
        # commit (real subcommand is `status`); the env prefix still routes it to "ask"
        ("X=1 git status commit \"$(date)\"", "ask"),
        # plain human commit forms must NOT be denied (no $()/backtick/<<)
        ("git commit -m 'subject' -m 'body'", "pass"),   # multiple -m (paragraphs)
        ("git commit", "pass"),                          # editor commit
        ("git commit -m \"see func() call\"", "pass"),   # parens without $ are literal
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
