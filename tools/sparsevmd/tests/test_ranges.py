import pytest

from sparsevmd.ranges import RangeError, expand_and_normalize, intersect, parse_range


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
    with pytest.raises(ValueError):
        parse_range(":")


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
    with pytest.raises(RangeError):
        expand_and_normalize([(10, 20), (20, 30)], 0, 100)


def test_overlapping_ranges_error():
    with pytest.raises(RangeError):
        expand_and_normalize([(0, 100), (50, 150)], 0, 200)


def test_out_of_order_overlap_error():
    with pytest.raises(RangeError):
        expand_and_normalize([(50, 150), (0, 100)], 0, 200)


def test_expansion_induced_touch_error():
    with pytest.raises(RangeError):
        expand_and_normalize([(None, 20), (20, None)], 0, 100)


def test_adjacent_non_touching_ok():
    assert expand_and_normalize([(0, 20), (21, 30)], 0, 100) == [(0, 20), (21, 30)]


def test_start_greater_than_end_after_expansion_error():
    with pytest.raises(RangeError):
        expand_and_normalize([(999, None)], 0, 60)


def test_intersect_full_track_keeps_ranges():
    g = [(0, 240), (300, 420)]
    assert intersect(g, 0, 500) == [(0, 240), (300, 420)]


def test_intersect_clips_to_track_extent():
    assert intersect([(0, 240)], 100, 150) == [(100, 150)]


def test_intersect_partial_clip():
    assert intersect([(0, 240)], 100, 400) == [(100, 240)]


def test_intersect_empty_when_outside():
    assert intersect([(0, 240)], 600, 700) == []


def test_intersect_drops_empty_subranges():
    g = [(0, 50), (300, 420)]
    assert intersect(g, 0, 100) == [(0, 50)]
