"""ボーン選択ルールのテスト(sparsevmd.md §2.2)。

selection.resolve_selection は、入力VMDのボーン名集合に対して include/exclude
セレクタを適用し、処理対象ボーン名と警告を返す。ハード エラー(--bone 不在、
include と exclude の名前衝突、空文字、最終0件など)は SelectionError を送出する。
SelectionError はそれまでに蓄積した警告を .warnings に保持する。
ソフト事象(exclude のみの不在名、glob/group の不一致)は警告して継続する。

警告はセレクタ値を含む文字列で、原因(どの名前/パターンか)を区別できる。

bone-file のパス検証(不在・非通常ファイル)は CLI(argparse)層の責務であり、
parse_bone_file はテキストを受ける。パス検証は CLI テスト(Step4)で扱う。
"""

import pytest

from sparsevmd import selection
from sparsevmd.selection import Selector, SelectionError, resolve_selection


# テスト用のボーン名集合(キーが存在するボーン)。case 比較用に head/Head を併置。
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


# --- include の基本 ---------------------------------------------------------


def test_no_include_selects_all():
    # include 指定が無ければ全ボーンを include 扱い(§2.2)。
    r = resolve_selection(BONES, includes=[], excludes=[])
    assert set(names(r)) == set(BONES)


def test_name_include_exact():
    r = resolve_selection(BONES, includes=[Selector("name", "センター")], excludes=[])
    assert names(r) == ["センター"]


def test_name_include_partial_does_not_match():
    # 部分一致しない(§2.2)。"上半身" は "上半身2" を含まない。
    r = resolve_selection(BONES, includes=[Selector("name", "上半身")], excludes=[])
    assert names(r) == ["上半身"]


def test_glob_case_sensitive():
    # GLOB は fnmatchcase 相当で大文字小文字を区別(§2.2)。"head" は "Head" に一致しない。
    r = resolve_selection(BONES, includes=[Selector("glob", "head")], excludes=[])
    assert names(r) == ["head"]


def test_glob_star():
    r = resolve_selection(BONES, includes=[Selector("glob", "上半身*")], excludes=[])
    assert set(names(r)) == {"上半身", "上半身2"}


def test_glob_question():
    # ? は1文字。"?腕" は2文字目が腕の2文字名に一致。
    r = resolve_selection(BONES, includes=[Selector("glob", "?腕")], excludes=[])
    assert set(names(r)) == {"左腕", "右腕"}


def test_glob_charset():
    # [左右] は文字集合。
    r = resolve_selection(BONES, includes=[Selector("glob", "[左右]腕")], excludes=[])
    assert set(names(r)) == {"左腕", "右腕"}


def test_group_include_arms_exact():
    # arms は *肩*/*腕*/*手首* 等。BONES では 左腕/右腕/左手首 のちょうど3件。
    r = resolve_selection(BONES, includes=[Selector("group", "arms")], excludes=[])
    assert set(names(r)) == {"左腕", "右腕", "左手首"}


def test_group_include_fingers_and_legs():
    rf = resolve_selection(BONES, includes=[Selector("group", "fingers")], excludes=[])
    assert set(names(rf)) == {"左親指１"}
    rl = resolve_selection(BONES, includes=[Selector("group", "legs")], excludes=[])
    # legs の *足* は 左足 と 左足ＩＫ の両方に一致する。
    assert set(names(rl)) == {"左足", "左足ＩＫ"}


# --- exclude ----------------------------------------------------------------


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


# --- groups の定義(§2.2 の表に厳密一致) ------------------------------------


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
    # ik 固有のグロブは含まない(§2.2)。
    assert not (set(selection.GROUPS["ik"]) & mocap)


def test_mocap_resolution_includes_ik_named_bone_via_legs():
    # 「ik は含めない」は ik グロブを足さない意味であり、IK 名のボーンを能動的に
    # 除外する意味ではない。左足ＩＫ は legs の *足* に一致して mocap に入る。
    r = resolve_selection(BONES, includes=[Selector("group", "mocap")], excludes=[])
    assert "左足ＩＫ" in names(r)


# --- ハード エラー(SelectionError) ------------------------------------------


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
        ([Selector("name", "")], []),
        ([], [Selector("name", "")]),
    ],
)
def test_empty_name_raises(includes, excludes):
    # include 側・exclude 側いずれの空文字 NAME もエラー(§2.2)。
    with pytest.raises(SelectionError):
        resolve_selection(BONES, includes=includes, excludes=excludes)


def test_final_zero_with_bones_present_raises():
    # 全 include 後に全 exclude で0件(ボーンキーは存在) → エラー(§2.2)。
    with pytest.raises(SelectionError):
        resolve_selection(BONES, includes=[], excludes=[Selector("glob", "*")])


def test_sole_unmatched_glob_warns_then_errors():
    # 唯一の include が不一致 glob → 警告を出した上で最終0件 → SelectionError(§2.2)。
    # 警告はエラーに載せて観測できる。
    with pytest.raises(SelectionError) as exc:
        resolve_selection(BONES, includes=[Selector("glob", "存在しない*")], excludes=[])
    assert warned_about(exc.value, "存在しない*")


# --- ソフト警告(継続)。原因をセレクタ値で区別できること --------------------


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


def test_unmatched_include_group_warns():
    # グループがどのボーンにも一致しない場合は警告継続(他に一致があれば継続)。
    # arms はこの小universe(頭/センター)のどれにも一致しない。
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


def test_empty_universe_no_include_is_empty_no_error():
    # ボーンセクションが空(キー無し)で include 指定も無ければ0件を返しエラーにしない
    # (空セクションの扱いは CLI 側 §3.1)。
    r = resolve_selection([], includes=[], excludes=[])
    assert names(r) == []


def test_empty_universe_explicit_name_raises():
    # 空 universe でも --bone NAME 明示名が不在ならエラー(§2.2、ボーンセクション空を含む)。
    with pytest.raises(SelectionError):
        resolve_selection([], includes=[Selector("name", "センター")], excludes=[])


# --- bone-file 解析(テキストのみ。パス検証は CLI 層) -----------------------


def test_parse_bone_file_basic():
    text = (
        "# コメント\n"
        "\n"
        "センター\n"  # 接頭辞なし → name:
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
        # include 側: bare と name: の重複。
        ("センター\nname:センター\n", [Selector("name", "センター")], []),
        # exclude 側の重複。
        (
            "exclude:glob:*腕\nexclude:glob:*腕\n",
            [],
            [Selector("glob", "*腕")],
        ),
        # group 重複。
        ("group:arms\ngroup:arms\n", [Selector("group", "arms")], []),
    ],
)
def test_parse_bone_file_dedup(text, expect_inc, expect_exc):
    # 同じ選択子の重複は1つに正規化(§2.2)。
    includes, excludes = selection.parse_bone_file(text)
    assert includes == expect_inc
    assert excludes == expect_exc
