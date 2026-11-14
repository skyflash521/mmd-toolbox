from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Literal

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
    kind: Literal["name", "glob", "group"]
    value: str


class SelectionError(ValueError):
    def __init__(self, message, warnings=None):
        super().__init__(message)
        self.warnings = list(warnings) if warnings else []


@dataclass
class SelectionResult:
    """selected は resolve_selection に渡した bone_names と同じ並び。"""

    selected: list
    warnings: list = field(default_factory=list)


def _matches(name, selector):
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
    return {n for n in bone_names if _matches(n, selector)}


def resolve_selection(bone_names, includes, excludes, undecodable=None):
    """undecodable はデコードできなかったボーン名(置換文字入りの表示名)の集合。"""
    includes = list(includes or [])
    excludes = list(excludes or [])
    undecodable = set(undecodable or ())
    warnings = []

    for sel in includes + excludes:
        if sel.kind == "name" and sel.value == "":
            raise SelectionError("空文字のボーン名は指定できない", warnings)

    inc_names = {s.value for s in includes if s.kind == "name"}
    exc_names = {s.value for s in excludes if s.kind == "name"}
    dup = inc_names & exc_names
    if dup:
        raise SelectionError(
            f"--bone と --exclude-bone に同じ名前: {sorted(dup)}", warnings
        )

    universe = list(bone_names)
    universe_set = set(universe)
    name_universe = universe_set - undecodable

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

    if universe and not selected:
        raise SelectionError("ボーン選択の結果が0件です", warnings)

    return SelectionResult(selected=selected, warnings=warnings)


def parse_bone_file(text):
    """戻り値は (includes, excludes)。"""
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
    for kind in _VALID_KINDS:
        prefix = kind + ":"
        if body.startswith(prefix):
            return Selector(kind, body[len(prefix):].strip())
    return Selector("name", body.strip())


def _add_unique(seq, sel):
    if sel not in seq:
        seq.append(sel)
