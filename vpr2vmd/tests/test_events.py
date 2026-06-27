"""口形イベント確定のテスト(vpr2vmd.md §3、重なり音符の非重複化)。

`resolve_overlaps` は collect_notes が整列した音符列(start_tick 昇順・duration_tick 降順・
パート出現順・索引昇順)を受け、単音前提の採用音符列へ非重複化する。2段階:
(1) 同一 start_tick は整列順の先頭(最長)だけ残す、(2) 各音符の終端を次音符の開始へ切り詰め、
長さ 0 以下は除外する。
"""

from lipsync import MouthShape
from vpr_io import Note, TempoEvent

from vpr2vmd import events


def _note(start, dur, *, lyric="x", phonemes=None):
    return Note(
        start_tick=start, duration_tick=dur, pitch=60, lyric=lyric,
        velocity=64, phonemes=phonemes if phonemes is not None else [],
    )


def _spans(notes):
    return [(n.start_tick, n.start_tick + n.duration_tick) for n in notes]


# 120bpm・resolution 480 では 1 tick = 1/32 フレーム(tick 240→7.5, 480→15)。
_TEMPOS = [TempoEvent(0, 120.0)]
_RES = 480


def _build_se(adopted, *, use_n_morph=True):
    return [
        (e.shape, e.start, e.end)
        for e in events.build_mouth_events(adopted, _TEMPOS, _RES, use_n_morph=use_n_morph)
    ]


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


# --- 組み立て: 採用音符列→フレーム変換→休符 SILENCE・直前口形継続・全時間軸被覆の MouthEvent 列 ---

def test_build_empty_is_empty():
    assert events.build_mouth_events([], _TEMPOS, _RES, use_n_morph=True) == []


def test_build_single_vowel_note():
    # [a] [0,480) → frame [0,15)。先頭休符なし・末尾休符なし。
    assert _build_se([_note(0, 480, phonemes=["a"])]) == [(MouthShape.A, 0.0, 15.0)]


def test_build_leading_rest_is_silence():
    # 先頭〜最初の音符は SILENCE。[a] [240,480) → frame [7.5,15)、先頭 [0,7.5) は SILENCE。
    assert _build_se([_note(240, 240, phonemes=["a"])]) == [
        (MouthShape.SILENCE, 0.0, 7.5),
        (MouthShape.A, 7.5, 15.0),
    ]


def test_build_gap_between_notes_is_silence():
    # 採用音符列の発音区間の補集合が休符。[a][0,240) と [i][480,720) の間 [7.5,15) は SILENCE。
    adopted = [_note(0, 240, phonemes=["a"]), _note(480, 240, phonemes=["i"])]
    assert _build_se(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.SILENCE, 7.5, 15.0),
        (MouthShape.I, 15.0, 22.5),
    ]


def test_build_adjacent_notes_have_no_silence():
    # 隙間なく隣接する音符の間には SILENCE を挟まない。
    adopted = [_note(0, 240, phonemes=["a"]), _note(240, 240, phonemes=["i"])]
    assert _build_se(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.I, 7.5, 15.0),
    ]


def test_build_continuation_holds_previous_vowel():
    # 継続「-」(母音なし音符)は直前の母音口形を保つ。[a] の後の [-] は あ を継続。
    adopted = [_note(0, 240, phonemes=["a"]), _note(240, 240, phonemes=["-"])]
    assert _build_se(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.A, 7.5, 15.0),
    ]


def test_build_continuation_after_moraic_nasal_holds_n_when_on():
    # ん ON では撥音の直後の継続は「ん」を保つ(直前の口形を保つ)。
    adopted = [_note(0, 240, phonemes=["N\\"]), _note(240, 240, phonemes=["-"])]
    assert _build_se(adopted, use_n_morph=True) == [
        (MouthShape.N, 0.0, 7.5),
        (MouthShape.N, 7.5, 15.0),
    ]


def test_build_continuation_after_moraic_nasal_stays_closed_when_off():
    # ん OFF では撥音は閉口(SILENCE)。直後の継続も閉口を保ち口を開けない(「ん——」OFF)。
    adopted = [_note(0, 240, phonemes=["N\\"]), _note(240, 240, phonemes=["-"])]
    assert _build_se(adopted, use_n_morph=False) == [
        (MouthShape.SILENCE, 0.0, 7.5),
        (MouthShape.SILENCE, 7.5, 15.0),
    ]


def test_build_continuation_after_rest_holds_closed_not_pre_rest_vowel():
    # 休符(閉口)の直後の母音なし音符は、休符前の母音を再開せず閉口を継続する。
    # [a][0,240) frame[0,7.5)、休符[7.5,22.5)、[-][720,960) frame[22.5,30)。
    adopted = [_note(0, 240, phonemes=["a"]), _note(720, 240, phonemes=["-"])]
    assert _build_se(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.SILENCE, 7.5, 22.5),
        (MouthShape.SILENCE, 22.5, 30.0),
    ]


def test_build_first_note_voweless_with_no_previous_is_silence():
    # 直前口形が無い(曲頭の母音なし音符)は無音(閉口)。既定母音「あ」へ倒さない。
    assert _build_se([_note(0, 480, phonemes=["t"])]) == [(MouthShape.SILENCE, 0.0, 15.0)]


def test_build_geminate_is_silence():
    # 促音(Q)は無音(閉口)。
    assert _build_se([_note(0, 480, phonemes=["w", "Q"])]) == [(MouthShape.SILENCE, 0.0, 15.0)]


def test_build_is_contiguous_and_covers_full_axis():
    # 連続(終端=次の始端)・先頭 0・末尾=採用音符列の最後の終端、で全時間軸を被覆する。
    adopted = [_note(240, 240, phonemes=["m", "a"]), _note(720, 240, phonemes=["i"])]
    result = events.build_mouth_events(adopted, _TEMPOS, _RES, use_n_morph=True)
    assert result[0].start == 0.0
    for a, b in zip(result, result[1:]):
        assert a.end == b.start
    assert result[-1].end == 30.0  # 960/32
