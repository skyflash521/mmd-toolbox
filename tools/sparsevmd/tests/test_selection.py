import pytest

from sparsevmd import selection
from sparsevmd.selection import SelectionError, Selector, resolve_selection

BONES = [
    "センター",
    "上半身",
    "上半身2",
    "頭",
    "head",
    "Head",
    "左腕",
    "右腕",
    "左手首",
    "左親指１",
    "左足",
    "左足ＩＫ",
]


def names(result):
    return list(result.selected)


def warned_about(result_or_exc, needle):
    ws = result_or_exc.warnings
    return any(needle in w for w in ws)


def test_no_include_selects_all():
    r = resolve_selection(BONES, includes=[], excludes=[])
    assert set(names(r)) == set(BONES)


def test_name_include_exact():
    r = resolve_selection(BONES, includes=[Selector("name", "センター")], excludes=[])
    assert names(r) == ["センター"]


def test_name_include_partial_does_not_match():
    r = resolve_selection(BONES, includes=[Selector("name", "上半身")], excludes=[])
    assert names(r) == ["上半身"]


def test_glob_case_sensitive():
    r = resolve_selection(BONES, includes=[Selector("glob", "head")], excludes=[])
    assert names(r) == ["head"]


def test_glob_star():
    r = resolve_selection(BONES, includes=[Selector("glob", "上半身*")], excludes=[])
    assert set(names(r)) == {"上半身", "上半身2"}


def test_glob_question():
    r = resolve_selection(BONES, includes=[Selector("glob", "?腕")], excludes=[])
    assert set(names(r)) == {"左腕", "右腕"}


def test_glob_charset():
    r = resolve_selection(BONES, includes=[Selector("glob", "[左右]腕")], excludes=[])
    assert set(names(r)) == {"左腕", "右腕"}


def test_group_include_arms_exact():
    r = resolve_selection(BONES, includes=[Selector("group", "arms")], excludes=[])
    assert set(names(r)) == {"左腕", "右腕", "左手首"}


def test_group_include_fingers_and_legs():
    rf = resolve_selection(BONES, includes=[Selector("group", "fingers")], excludes=[])
    assert set(names(rf)) == {"左親指１"}
    rl = resolve_selection(BONES, includes=[Selector("group", "legs")], excludes=[])
    assert set(names(rl)) == {"左足", "左足ＩＫ"}


def test_exclude_after_include():
    r = resolve_selection(
        BONES,
        includes=[Selector("glob", "[左右]腕")],
        excludes=[Selector("name", "右腕")],
    )
    assert names(r) == ["左腕"]


def test_no_include_then_exclude_glob():
    r = resolve_selection(BONES, includes=[], excludes=[Selector("glob", "*ＩＫ")])
    assert "左足ＩＫ" not in names(r)
    assert "センター" in names(r)


def test_groups_exact_contents():
    g = selection.GROUPS
    assert set(g["core"]) == {
        "センター", "グルーブ", "全ての親", "上半身*", "下半身", "首", "頭",
        "center", "groove", "root", "upper body*", "lower body", "neck", "head",
    }
    assert set(g["arms"]) == {
        "*肩*", "*腕*", "*ひじ*", "*肘*", "*手首*",
        "*shoulder*", "*arm*", "*elbow*", "*wrist*",
    }
    assert set(g["legs"]) == {
        "*足*", "*脚*", "*ひざ*", "*膝*", "*つま先*",
        "*leg*", "*knee*", "*ankle*", "*toe*",
    }
    assert set(g["fingers"]) == {
        "*指*", "*finger*", "*thumb*", "*index*", "*middle*", "*ring*",
        "*pinky*", "*little*",
    }
    assert set(g["ik"]) == {"*IK*", "*ＩＫ*", "*ik*"}


def test_mocap_is_union_without_ik_globs():
    mocap = set(selection.GROUPS["mocap"])
    expected = (
        set(selection.GROUPS["core"])
        | set(selection.GROUPS["arms"])
        | set(selection.GROUPS["legs"])
        | set(selection.GROUPS["fingers"])
    )
    assert mocap == expected
    assert not (set(selection.GROUPS["ik"]) & mocap)


def test_mocap_resolution_includes_ik_named_bone_via_legs():
    r = resolve_selection(BONES, includes=[Selector("group", "mocap")], excludes=[])
    assert "左足ＩＫ" in names(r)


def test_name_include_not_found_raises():
    with pytest.raises(SelectionError):
        resolve_selection(BONES, includes=[Selector("name", "存在しない")], excludes=[])


def test_include_exclude_same_name_raises():
    with pytest.raises(SelectionError):
        resolve_selection(
            BONES,
            includes=[Selector("name", "センター")],
            excludes=[Selector("name", "センター")],
        )


@pytest.mark.parametrize(
    "includes,excludes",
    [
        pytest.param([Selector("name", "")], [], id="include"),
        pytest.param([], [Selector("name", "")], id="exclude"),
    ],
)
def test_empty_name_raises(includes, excludes):
    with pytest.raises(SelectionError):
        resolve_selection(BONES, includes=includes, excludes=excludes)


def test_exclude_removing_every_bone_raises():
    with pytest.raises(SelectionError):
        resolve_selection(BONES, includes=[], excludes=[Selector("glob", "*")])


def test_sole_unmatched_glob_warns_then_errors():
    with pytest.raises(SelectionError) as exc:
        resolve_selection(BONES, includes=[Selector("glob", "存在しない*")], excludes=[])
    assert warned_about(exc.value, "存在しない*")


def test_exclude_name_not_found_warns_and_continues():
    r = resolve_selection(BONES, includes=[], excludes=[Selector("name", "幻ボーン")])
    assert set(names(r)) == set(BONES)
    assert warned_about(r, "幻ボーン")


def test_unmatched_include_glob_warns_but_other_matches():
    r = resolve_selection(
        BONES,
        includes=[Selector("glob", "幻*"), Selector("name", "頭")],
        excludes=[],
    )
    assert names(r) == ["頭"]
    assert warned_about(r, "幻*")


def test_unmatched_include_group_warns_but_other_matches():
    universe = ["頭", "センター"]
    r = resolve_selection(
        universe,
        includes=[Selector("group", "arms"), Selector("name", "頭")],
        excludes=[],
    )
    assert names(r) == ["頭"]
    assert warned_about(r, "arms")


def test_unmatched_exclude_glob_warns():
    r = resolve_selection(BONES, includes=[], excludes=[Selector("glob", "幻*")])
    assert set(names(r)) == set(BONES)
    assert warned_about(r, "幻*")


def test_empty_universe_without_include_returns_empty_without_error():
    r = resolve_selection([], includes=[], excludes=[])
    assert names(r) == []


def test_empty_universe_explicit_include_name_raises():
    with pytest.raises(SelectionError):
        resolve_selection([], includes=[Selector("name", "センター")], excludes=[])


def test_parse_bone_file_skips_comments_and_reads_every_prefix():
    text = (
        "# コメント\n"
        "\n"
        "センター\n"
        "name:頭\n"
        "glob:*腕\n"
        "group:legs\n"
        "exclude:name:右腕\n"
        "exclude:glob:*ＩＫ\n"
        "exclude:group:ik\n"
    )
    includes, excludes = selection.parse_bone_file(text)
    assert Selector("name", "センター") in includes
    assert Selector("name", "頭") in includes
    assert Selector("glob", "*腕") in includes
    assert Selector("group", "legs") in includes
    assert Selector("name", "右腕") in excludes
    assert Selector("glob", "*ＩＫ") in excludes
    assert Selector("group", "ik") in excludes


@pytest.mark.parametrize(
    "text,expect_inc,expect_exc",
    [
        pytest.param(
            "センター\nname:センター\n", [Selector("name", "センター")], [], id="unprefixed_and_name_prefix"
        ),
        pytest.param(
            "exclude:glob:*腕\nexclude:glob:*腕\n",
            [],
            [Selector("glob", "*腕")],
            id="exclude_glob",
        ),
        pytest.param("group:arms\ngroup:arms\n", [Selector("group", "arms")], [], id="group"),
    ],
)
def test_parse_bone_file_dedup(text, expect_inc, expect_exc):
    includes, excludes = selection.parse_bone_file(text)
    assert includes == expect_inc
    assert excludes == expect_exc


UNDECODABLE_NAME = "\ufffd"


def test_name_selector_cannot_target_undecodable():
    with pytest.raises(SelectionError):
        resolve_selection(
            BONES + [UNDECODABLE_NAME],
            includes=[Selector("name", UNDECODABLE_NAME)],
            excludes=[],
            undecodable={UNDECODABLE_NAME},
        )


def test_exclude_name_undecodable_warns_not_matched():
    r = resolve_selection(
        BONES + [UNDECODABLE_NAME],
        includes=[],
        excludes=[Selector("name", UNDECODABLE_NAME)],
        undecodable={UNDECODABLE_NAME},
    )
    assert UNDECODABLE_NAME in r.selected
    assert warned_about(r, "存在しません")


def test_default_all_and_glob_still_include_undecodable():
    r_all = resolve_selection(BONES + [UNDECODABLE_NAME], includes=[], excludes=[], undecodable={UNDECODABLE_NAME})
    assert UNDECODABLE_NAME in r_all.selected
    r_glob = resolve_selection(
        BONES + [UNDECODABLE_NAME], includes=[Selector("glob", "*")], excludes=[], undecodable={UNDECODABLE_NAME}
    )
    assert UNDECODABLE_NAME in r_glob.selected


def test_selected_keeps_input_order():
    r = resolve_selection(BONES, includes=[Selector("glob", "左*"), Selector("name", "センター")], excludes=[])
    assert names(r) == [n for n in BONES if n == "センター" or n.startswith("左")]
