"""口形イベント確定のテスト(vpr2vmd.md §3、重なり音符の非重複化)。

`resolve_overlaps` は collect_notes が整列した音符列(start_tick 昇順・duration_tick 降順・
パート出現順・索引昇順)を受け、単音前提の採用音符列へ非重複化する。2段階:
(1) 同一 start_tick は整列順の先頭(最長)だけ残す、(2) 各音符の終端を次音符の開始へ切り詰め、
長さ 0 以下は除外する。
"""

import pytest

events = pytest.importorskip("vpr2vmd.events", reason="impl pending: P-2 重なり解決")

from vpr_io import Note  # noqa: E402


def _note(start, dur, *, lyric="x"):
    return Note(start_tick=start, duration_tick=dur, pitch=60, lyric=lyric, velocity=64, phonemes=[])


def _spans(notes):
    return [(n.start_tick, n.start_tick + n.duration_tick) for n in notes]


def test_empty():
    assert events.resolve_overlaps([]) == []


def test_no_overlap_unchanged():
    notes = [_note(0, 100), _note(200, 100)]
    assert _spans(events.resolve_overlaps(notes)) == [(0, 100), (200, 300)]


def test_same_start_keeps_longest_only():
    # collect_notes の整列で同一 start は duration 降順。先頭(最長)だけ残す。
    notes = [_note(0, 120, lyric="long"), _note(0, 80, lyric="short")]
    adopted = events.resolve_overlaps(notes)
    assert [n.lyric for n in adopted] == ["long"]
    assert _spans(adopted) == [(0, 120)]


def test_same_start_same_duration_keeps_first_in_order():
    # 同一 start・同一 duration(別パートの同時同長など)は、collect_notes の整列順
    # (パート出現順・索引昇順)の先頭を残す。同長時に後続を選ぶ実装を落とすため。
    notes = [_note(0, 100, lyric="first"), _note(0, 100, lyric="second")]
    adopted = events.resolve_overlaps(notes)
    assert [n.lyric for n in adopted] == ["first"]
    assert _spans(adopted) == [(0, 100)]


def test_truncate_to_next_start():
    # 後続開始へ切り詰め: [0,480) は次音符 start=240 まで詰めて [0,240)。
    notes = [_note(0, 480), _note(240, 240)]
    assert _spans(events.resolve_overlaps(notes)) == [(0, 240), (240, 480)]


def test_contained_notes_truncated_in_chain():
    notes = [_note(0, 1000), _note(100, 200), _note(150, 900)]
    assert _spans(events.resolve_overlaps(notes)) == [(0, 100), (100, 150), (150, 1050)]


def test_zero_duration_note_dropped():
    # 長さ 0 以下の音符は採用しない(発音区間を持たない)。
    assert events.resolve_overlaps([_note(0, 0)]) == []


def test_zero_duration_note_dropped_others_kept():
    notes = [_note(0, 0), _note(100, 100)]
    assert _spans(events.resolve_overlaps(notes)) == [(100, 200)]
