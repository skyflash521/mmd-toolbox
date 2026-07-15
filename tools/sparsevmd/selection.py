"""ボーン選択ルールの解決。

入力VMDのボーン名集合に対して include / exclude セレクタを適用し、処理対象の
ボーン名と警告を返す。セレクタは名前完全一致(name)、glob、組み込みグループ(group)。

選択の流れ:
- include 指定が1つも無ければ全ボーンを include 扱いとする。その後 exclude を引く。
- ハード エラー(SelectionError、CLI で終了コード2に対応づけ):
  空文字 NAME、include と exclude の同名衝突、--bone 明示名が入力に存在しない
  (ボーンセクションが空の場合を含む)、ボーンキーが存在するのに最終選択が0件。
- ソフト事象(警告して継続): exclude 名のみの不在、glob/group の不一致。

SelectionError はそこまでに蓄積した警告を .warnings に保持する。警告文字列には
対象セレクタ値を含め、原因を区別できるようにする。
"""

from dataclasses import dataclass, field
from fnmatch import fnmatchcase

# 組み込みグループ。各グループは glob の集合として定義する。
_CORE = (
    "センター", "グルーブ", "全ての親", "上半身*", "下半身", "首", "頭",
    "center", "groove", "root", "upper body*", "lower body", "neck", "head",
)
_ARMS = (
    "*肩*", "*腕*", "*ひじ*", "*肘*", "*手首*",
    "*shoulder*", "*arm*", "*elbow*", "*wrist*",
)
_LEGS = (
    "*足*", "*脚*", "*ひざ*", "*膝*", "*つま先*",
    "*leg*", "*knee*", "*ankle*", "*toe*",
)
_FINGERS = (
    "*指*", "*finger*", "*thumb*", "*index*", "*middle*", "*ring*",
    "*pinky*", "*little*",
)
_IK = ("*IK*", "*ＩＫ*", "*ik*")

# mocap は core + arms + legs + fingers の glob 集合。ik 固有の glob は含めない。
GROUPS = {
    "core": _CORE,
    "arms": _ARMS,
    "legs": _LEGS,
    "fingers": _FINGERS,
    "ik": _IK,
    "mocap": _CORE + _ARMS + _LEGS + _FINGERS,
}

_VALID_KINDS = ("name", "glob", "group")


@dataclass(frozen=True)
class Selector:
    """ボーン選択子。kind は name / glob / group。"""

    kind: str
    value: str


class SelectionError(ValueError):
    """ボーン選択のハード エラー(CLI で終了コード2)。

    .warnings にエラーまでに蓄積したソフト警告を保持する。
    """

    def __init__(self, message, warnings=None):
        super().__init__(message)
        self.warnings = list(warnings) if warnings else []


@dataclass
class SelectionResult:
    """選択結果。selected は入力ボーン名順を保つ。"""

    selected: list
    warnings: list = field(default_factory=list)


def _matches(name, selector):
    """単一ボーン名がセレクタに一致するか。"""
    if selector.kind == "name":
        return name == selector.value
    if selector.kind == "glob":
        return fnmatchcase(name, selector.value)
    if selector.kind == "group":
        globs = GROUPS.get(selector.value)
        if globs is None:
            raise SelectionError(f"未知のグループ: {selector.value!r}")
        return any(fnmatchcase(name, g) for g in globs)
    raise SelectionError(f"未知のセレクタ種別: {selector.kind!r}")


def _matched_names(bone_names, selector):
    """セレクタに一致するボーン名の集合(入力順は呼び出し側で復元)。"""
    return {n for n in bone_names if _matches(n, selector)}


def resolve_selection(bone_names, includes, excludes, undecodable=None):
    """include/exclude セレクタを適用し SelectionResult を返す。

    undecodable はデコード不能なボーン名(置換文字入りの表示名)の集合。これらは
    name 種別の `--bone` / `--exclude-bone` では一致不可とする。glob/group
    やデフォルト全件選択では通常どおり対象になる。
    """
    includes = list(includes or [])
    excludes = list(excludes or [])
    undecodable = set(undecodable or ())
    warnings = []

    # 空文字 NAME はエラー。
    for sel in includes + excludes:
        if sel.kind == "name" and sel.value == "":
            raise SelectionError("空文字のボーン名は指定できない", warnings)

    # include と exclude に同じ NAME → エラー。
    inc_names = {s.value for s in includes if s.kind == "name"}
    exc_names = {s.value for s in excludes if s.kind == "name"}
    dup = inc_names & exc_names
    if dup:
        raise SelectionError(
            f"--bone と --exclude-bone に同じ名前: {sorted(dup)}", warnings
        )

    universe = list(bone_names)
    universe_set = set(universe)
    # name 種別の照合に使う集合。デコード不能名は一致不可とする。
    name_universe = universe_set - undecodable

    # include 集合を構築。
    if not includes:
        included = set(universe)
    else:
        included = set()
        missing_names = []
        for sel in includes:
            if sel.kind == "name":
                if sel.value in name_universe:
                    included.add(sel.value)
                else:
                    # --bone 明示名が不在(デコード不能名も含む)→ ハード エラー
                    # (空 universe を含む)。
                    missing_names.append(sel.value)
            else:
                hit = _matched_names(universe, sel)
                if hit:
                    included |= hit
                else:
                    warnings.append(
                        f"{sel.kind} に一致するボーンがありません: {sel.value}"
                    )
        if missing_names:
            raise SelectionError(
                f"--bone 名が入力に存在しません: {missing_names}", warnings
            )

    # exclude 集合を構築(不在・不一致は警告して継続)。
    excluded = set()
    for sel in excludes:
        if sel.kind == "name":
            if sel.value in name_universe:
                excluded.add(sel.value)
            else:
                warnings.append(
                    f"--exclude-bone 名が入力に存在しません: {sel.value}"
                )
        else:
            hit = _matched_names(universe, sel)
            if hit:
                excluded |= hit
            else:
                warnings.append(
                    f"除外 {sel.kind} に一致するボーンがありません: {sel.value}"
                )

    selected = [n for n in universe if n in included and n not in excluded]

    # ボーンキーが存在するのに最終0件 → エラー。空 universe の扱いは CLI 側の責務。
    if universe and not selected:
        raise SelectionError("ボーン選択の結果が0件です", warnings)

    return SelectionResult(selected=selected, warnings=warnings)


def parse_bone_file(text):
    """bone-file テキストを (includes, excludes) のセレクタ列に解析する。

    空行と # で始まる行は無視。各行は name:/glob:/group:/exclude:name:/
    exclude:glob:/exclude:group: のいずれか。接頭辞なしは name: 扱い。
    同じ選択子の重複は1つに正規化する。
    """
    includes = []
    excludes = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("exclude:"):
            sel = _parse_selector(line[len("exclude:"):])
            _add_unique(excludes, sel)
        else:
            _add_unique(includes, _parse_selector(line))
    return includes, excludes


def _parse_selector(body):
    """1行(exclude: 接頭辞除去後)を Selector に解析する。接頭辞なしは name:。"""
    for kind in _VALID_KINDS:
        prefix = kind + ":"
        if body.startswith(prefix):
            return Selector(kind, body[len(prefix):].strip())
    return Selector("name", body.strip())


def _add_unique(seq, sel):
    if sel not in seq:
        seq.append(sel)
