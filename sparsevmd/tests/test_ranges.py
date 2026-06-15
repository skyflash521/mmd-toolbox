"""範囲指定の解析・展開・積集合のテスト(sparsevmd.md §2.2 の RANGE)。

- parse_range: START:END / START: / :END を (start, end)(省略側は None)に解析。
  両端は0以上の10進整数、両端含む。両方指定で START>END はエラー。
- expand_and_normalize: 省略端を対象トラック全体の最小/最大に1回だけ展開し、
  昇順に正規化、1フレームでも重複(端の接触含む)したらエラー。展開後 START>END もエラー。
- intersect: グローバル範囲と当該トラックの先頭/末尾フレームの積集合(空なら空リスト)。
"""

import pytest

from sparsevmd import ranges
from sparsevmd.ranges import RangeError, expand_and_normalize, intersect, parse_range


# --- parse_range ------------------------------------------------------------


def test_parse_both_ends():
    assert parse_range("0:240") == (0, 240)


def test_parse_start_only():
    assert parse_range("10:") == (10, None)


def test_parse_end_only():
    assert parse_range(":420") == (None, 420)


def test_parse_single_frame_inclusive():
    assert parse_range("5:5") == (5, 5)


def test_parse_start_greater_than_end_raises():
    with pytest.raises(ValueError):
        parse_range("20:10")


@pytest.mark.parametrize("bad", ["-1:5", "1:-3"])
def test_parse_negative_raises(bad):
    with pytest.raises(ValueError):
        parse_range(bad)


@pytest.mark.parametrize("bad", ["abc", "1.5:2", "1:2:3", "5", ""])
def test_parse_malformed_raises(bad):
    with pytest.raises(ValueError):
        parse_range(bad)


def test_parse_both_omitted_raises():
    # §2.2 は START:END / START: / :END を挙げ「一方は省略可」とする。
    # 両端省略(:)は仕様外でエラー。
    with pytest.raises(ValueError):
        parse_range(":")


# --- expand_and_normalize ---------------------------------------------------


def test_expand_omitted_end():
    assert expand_and_normalize([(None, 240)], 0, 300) == [(0, 240)]


def test_expand_omitted_start():
    assert expand_and_normalize([(10, None)], 0, 300) == [(10, 300)]


def test_normalize_sorts_ascending():
    assert expand_and_normalize([(300, 420), (0, 240)], 0, 500) == [
        (0, 240),
        (300, 420),
    ]


def test_touching_ranges_overlap_error():
    # 10:20 と 20:30 は 20 が重複 → エラー(§2.2)。
    with pytest.raises(RangeError):
        expand_and_normalize([(10, 20), (20, 30)], 0, 100)


def test_overlapping_ranges_error():
    with pytest.raises(RangeError):
        expand_and_normalize([(0, 100), (50, 150)], 0, 200)


def test_out_of_order_overlap_error():
    # 順不同で重複(50:150 と 0:100)→ ソート前後に関わらずエラーを検出すること。
    with pytest.raises(RangeError):
        expand_and_normalize([(50, 150), (0, 100)], 0, 200)


def test_expansion_induced_touch_error():
    # 省略端の展開後に接触するケース。:20 → (0,20)、20: → (20,100) が 20 で重複。
    with pytest.raises(RangeError):
        expand_and_normalize([(None, 20), (20, None)], 0, 100)


def test_adjacent_non_touching_ok():
    # 0:20 と 21:30 は接触しない(20 と 21)→ OK。
    assert expand_and_normalize([(0, 20), (21, 30)], 0, 100) == [(0, 20), (21, 30)]


def test_start_greater_than_end_after_expansion_error():
    # 999: は END=末尾60 に展開され 999>60 → エラー(§2.2)。
    with pytest.raises(RangeError):
        expand_and_normalize([(999, None)], 0, 60)


# --- intersect --------------------------------------------------------------


def test_intersect_full_track_keeps_ranges():
    g = [(0, 240), (300, 420)]
    assert intersect(g, 0, 500) == [(0, 240), (300, 420)]


def test_intersect_clips_to_track_extent():
    assert intersect([(0, 240)], 100, 150) == [(100, 150)]


def test_intersect_partial_clip():
    assert intersect([(0, 240)], 100, 400) == [(100, 240)]


def test_intersect_empty_when_outside():
    # トラックがグローバル範囲外 → 積集合は空。
    # (空=削減対象なし。トラックを元のまま保持する処理は呼び出し側の責務で、
    #  後段の reducer/CLI テストで検証する。§2.2/§3.1)
    assert intersect([(0, 240)], 600, 700) == []


def test_intersect_drops_empty_subranges():
    # 一部の範囲だけ積集合が残る。
    g = [(0, 50), (300, 420)]
    assert intersect(g, 0, 100) == [(0, 50)]
