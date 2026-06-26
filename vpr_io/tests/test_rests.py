"""休符導出のテスト(vpr_io.md §2.1)。

休符は同一トラック内の発音区間 [start_tick, start_tick+duration_tick) の和集合の補集合として
[0, end_tick) の範囲で導出する。重なり・隣接で偽の休符を作らない。
"""

import pytest

# rest_intervals が未実装の間は import が ImportError になる(これだけを xfail で許す)。
pytestmark = pytest.mark.xfail(
    reason="impl pending: vpr_io rest_intervals",
    raises=ImportError,
)


def _note(start, duration):
    from vpr_io import Note

    return Note(start_tick=start, duration_tick=duration, pitch=60, lyric="x", velocity=64)


def test_rest_intervals_no_notes_is_all_silence():
    from vpr_io import rest_intervals

    assert rest_intervals([], 1920) == [(0, 1920)]


def test_rest_intervals_single_note_yields_leading_and_trailing():
    from vpr_io import rest_intervals

    # 発音 [480, 720)。前後が休符。
    assert rest_intervals([_note(480, 240)], 1920) == [(0, 480), (720, 1920)]


def test_rest_intervals_note_from_zero_yields_only_trailing():
    from vpr_io import rest_intervals

    assert rest_intervals([_note(0, 480)], 960) == [(480, 960)]


def test_rest_intervals_note_filling_range_yields_no_rest():
    from vpr_io import rest_intervals

    assert rest_intervals([_note(0, 1920)], 1920) == []


def test_rest_intervals_adjacent_notes_make_no_false_rest():
    from vpr_io import rest_intervals

    # [480,720) と [720,960) は隣接。間に休符を作らない。
    notes = [_note(480, 240), _note(720, 240)]
    assert rest_intervals(notes, 1920) == [(0, 480), (960, 1920)]


def test_rest_intervals_overlapping_notes_make_no_false_rest():
    from vpr_io import rest_intervals

    # [480,720) と [600,840) は重なる。和集合 [480,840) とし偽休符を作らない。
    notes = [_note(480, 240), _note(600, 240)]
    assert rest_intervals(notes, 1920) == [(0, 480), (840, 1920)]


def test_rest_intervals_unsorted_input():
    from vpr_io import rest_intervals

    # 入力順が時刻順でなくても和集合・補集合は同じ。
    notes = [_note(960, 240), _note(480, 240)]
    assert rest_intervals(notes, 1920) == [(0, 480), (720, 960), (1200, 1920)]
