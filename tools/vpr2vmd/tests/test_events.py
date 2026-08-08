"""口形イベント確定のテスト(重なり音符の非重複化)。

`resolve_overlaps` は collect_notes が整列した音符列(start_tick 昇順・duration_tick 降順・
パート出現順・索引昇順)を受け、単音前提の採用音符列へ非重複化する。2段階:
(1) 同一 start_tick は整列順の先頭(最長)だけ残す、(2) 各音符の終端を次音符の開始へ切り詰め、
長さ 0 以下は除外する。
"""

from lipsync import MouthShape
from vpr import Note, TempoEvent
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


def _build_se(adopted, *, use_n_morph=True, legato_max_frames=None):
    kw = {} if legato_max_frames is None else {"legato_max_frames": legato_max_frames}
    result, _diag = events.build_mouth_events(
        adopted, _TEMPOS, _RES, use_n_morph=use_n_morph, **kw
    )
    return [(e.shape, e.start, e.end) for e in result]


def test_empty():
    assert events.resolve_overlaps([])[0] == []


def test_no_overlap_unchanged():
    notes = [_note(0, 100), _note(200, 100)]
    assert _spans(events.resolve_overlaps(notes)[0]) == [(0, 100), (200, 300)]


def test_same_start_keeps_longest_only():
    # collect_notes の整列で同一 start は duration 降順。先頭(最長)だけ残す。
    notes = [_note(0, 120, lyric="long"), _note(0, 80, lyric="short")]
    adopted, _diag = events.resolve_overlaps(notes)
    assert [n.lyric for n in adopted] == ["long"]
    assert _spans(adopted) == [(0, 120)]


def test_same_start_same_duration_keeps_first_in_order():
    # 同一 start・同一 duration(別パートの同時同長など)は、collect_notes の整列順
    # (パート出現順・索引昇順)の先頭を残す。同長時に後続を選ぶ実装を落とすため。
    notes = [_note(0, 100, lyric="first"), _note(0, 100, lyric="second")]
    adopted, _diag = events.resolve_overlaps(notes)
    assert [n.lyric for n in adopted] == ["first"]
    assert _spans(adopted) == [(0, 100)]


def test_truncate_to_next_start():
    # 後続開始へ切り詰め: [0,480) は次音符 start=240 まで詰めて [0,240)。
    notes = [_note(0, 480), _note(240, 240)]
    assert _spans(events.resolve_overlaps(notes)[0]) == [(0, 240), (240, 480)]


def test_contained_notes_truncated_in_chain():
    notes = [_note(0, 1000), _note(100, 200), _note(150, 900)]
    assert _spans(events.resolve_overlaps(notes)[0]) == [(0, 100), (100, 150), (150, 1050)]


def test_zero_duration_note_dropped():
    # 長さ 0 以下の音符は採用しない(発音区間を持たない)。
    assert events.resolve_overlaps([_note(0, 0)])[0] == []


def test_zero_duration_note_dropped_others_kept():
    notes = [_note(0, 0), _note(100, 100)]
    assert _spans(events.resolve_overlaps(notes)[0]) == [(100, 200)]


# --- 組み立て: 採用音符列→フレーム変換→休符 SILENCE・直前口形継続・全時間軸被覆の MouthEvent 列 ---

def test_build_empty_is_empty():
    result, _diag = events.build_mouth_events([], _TEMPOS, _RES, use_n_morph=True)
    assert result == []


def test_build_single_vowel_note():
    # [a] [0,480) → frame [0,15)。先頭休符なし・末尾休符なし。
    assert _build_se([_note(0, 480, phonemes=["a"])]) == [(MouthShape.A, 0.0, 15.0)]


def test_build_leading_rest_is_silence():
    # 先頭〜最初の音符は SILENCE。[a] [240,480) → frame [7.5,15)、先頭 [0,7.5) は SILENCE。
    assert _build_se([_note(240, 240, phonemes=["a"])]) == [
        (MouthShape.SILENCE, 0.0, 7.5),
        (MouthShape.A, 7.5, 15.0),
    ]


def test_build_short_vowel_gap_is_legato():
    # 前後とも母音的で短い間隙はレガート間隙(LEGATO_GAP)。[a][0,240) と [i][480,720) の間
    # [7.5,15)(8分音符相当=7.5f ≤ legato_max 8.0)は LEGATO_GAP(完全閉口でなく谷で繋ぐ)。
    adopted = [_note(0, 240, phonemes=["a"]), _note(480, 240, phonemes=["i"])]
    assert _build_se(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.LEGATO_GAP, 7.5, 15.0),
        (MouthShape.I, 15.0, 22.5),
    ]


def test_build_short_gap_into_continuation_is_legato():
    # 次音符が継続「-」(note_mouth_events=None)でも、直前の確定口形(母音的)へ解決してから分類する。
    # [a][0,240) frame[0,7.5)、[-][480,720) frame[15,22.5) は あ を継続。間 [7.5,15)=7.5f は前後とも
    # 母音的(左 A・右は A へ解決)で短いため LEGATO_GAP。継続イベントは あ を保つ(間隙が谷で繋ぐため)。
    adopted = [_note(0, 240, phonemes=["a"]), _note(480, 240, phonemes=["-"])]
    assert _build_se(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.LEGATO_GAP, 7.5, 15.0),
        (MouthShape.A, 15.0, 22.5),
    ]


def test_build_long_vowel_gap_is_silence():
    # 前後とも母音的でも、間隙が legato_max を超えれば休符(SILENCE、完全閉口)。
    # [a][0,240) frame[0,7.5)、[i][720,960) frame[22.5,30) の間 [7.5,22.5)=15f > 8.0 → SILENCE。
    adopted = [_note(0, 240, phonemes=["a"]), _note(720, 240, phonemes=["i"])]
    assert _build_se(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.SILENCE, 7.5, 22.5),
        (MouthShape.I, 22.5, 30.0),
    ]


def test_build_gap_before_bilabial_onset_is_silence():
    # 次音符の実効口形(語頭)が母音的でない(両唇閉鎖)間隙は、短くても SILENCE(閉口を優先)。
    # [a][0,240) frame[0,7.5)、[m,i][480,720) は語頭閉鎖。間 [7.5,15)=7.5f でも次が BILABIAL → SILENCE。
    adopted = [_note(0, 240, phonemes=["a"]), _note(480, 240, phonemes=["m", "i"])]
    result = _build_se(adopted)
    assert result[0] == (MouthShape.A, 0.0, 7.5)
    assert result[1] == (MouthShape.SILENCE, 7.5, 15.0)


def test_build_gap_after_closure_is_silence():
    # 前音符の実効口形が母音的でない(促音=閉口)間隙は、短くても SILENCE。
    # [w,Q][0,240)→SILENCE frame[0,7.5)、[i][480,720) frame[15,22.5) の間 [7.5,15)=7.5f、
    # 左が閉口 → SILENCE。
    adopted = [_note(0, 240, phonemes=["w", "Q"]), _note(480, 240, phonemes=["i"])]
    result = _build_se(adopted)
    assert result[0] == (MouthShape.SILENCE, 0.0, 7.5)
    assert result[1] == (MouthShape.SILENCE, 7.5, 15.0)


def test_build_legato_max_frames_override():
    # legato_max_frames を下げると同じ母音-母音短間隙でも SILENCE になる(視覚チューニング点)。
    # 間隙 7.5f に対し legato_max=4.0 → 7.5 > 4.0 → SILENCE。
    adopted = [_note(0, 240, phonemes=["a"]), _note(480, 240, phonemes=["i"])]
    assert _build_se(adopted, legato_max_frames=4.0) == [
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


def test_build_moraic_nasal_is_silence_when_use_n_morph_omitted():
    # build_mouth_events 自身の use_n_morph 既定値(off)を直接検証する(_build_se ヘルパーは
    # 既定 use_n_morph=True を明示するため、この既定値の回帰は検出できない)。
    adopted = [_note(0, 240, phonemes=["N\\"])]
    result, _diag = events.build_mouth_events(adopted, _TEMPOS, _RES)
    assert [(e.shape, e.start, e.end) for e in result] == [(MouthShape.SILENCE, 0.0, 7.5)]


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
    result, _diag = events.build_mouth_events(adopted, _TEMPOS, _RES, use_n_morph=True)
    assert result[0].start == 0.0
    for a, b in zip(result, result[1:], strict=False):
        assert a.end == b.start
    assert result[-1].end == 30.0  # 960/32


# --- 開き量の付与: 母音的口形イベントへ音符別開き量を刻印、無音は 0 ---

def _build_so(adopted, *, use_n_morph=True, open_by_note=None):
    result, _diag = events.build_mouth_events(
        adopted, _TEMPOS, _RES, use_n_morph=use_n_morph, open_by_note=open_by_note
    )
    return [(e.shape, e.open_amount) for e in result]


def test_build_open_defaults_zero_without_open_by_note():
    # open_by_note を渡さなければ全イベントの開き量は 0(既定・後方互換)。
    assert _build_so([_note(0, 240, phonemes=["a"])]) == [(MouthShape.A, 0.0)]


def test_build_stamps_note_open_on_vowel():
    # 母音イベントへ該当音符の開き量を刻印する。
    assert _build_so([_note(0, 240, phonemes=["a"])], open_by_note=[0.6]) == [(MouthShape.A, 0.6)]


def test_build_open_shared_across_moras_of_one_note():
    # 1音符の複数母音(モーラ)は同じ音符の開き量を共有する。
    assert _build_so([_note(0, 480, phonemes=["a", "i"])], open_by_note=[0.5]) == [
        (MouthShape.A, 0.5),
        (MouthShape.I, 0.5),
    ]


def test_build_open_on_moraic_nasal_when_on():
    # 撥音「ん」は母音的口形なので開き量を刻印する(ん ON)。
    assert _build_so([_note(0, 240, phonemes=["N\\"])], use_n_morph=True, open_by_note=[0.3]) == [
        (MouthShape.N, 0.3),
    ]


def test_build_open_zero_on_moraic_nasal_when_off():
    # ん OFF では撥音は無音(SILENCE)なので母音的口形でなく開き量 0(音符の開き量を無視)。
    assert _build_so([_note(0, 240, phonemes=["N\\"])], use_n_morph=False, open_by_note=[0.3]) == [
        (MouthShape.SILENCE, 0.0),
    ]


def test_build_open_zero_on_geminate_silence():
    # 促音(無音)は母音的口形でないので音符の開き量を無視し 0。
    assert _build_so([_note(0, 480, phonemes=["w", "Q"])], open_by_note=[0.9]) == [
        (MouthShape.SILENCE, 0.0),
    ]


def test_build_open_zero_on_leading_bilabial():
    # 語頭両唇閉鎖(BILABIAL)も母音的口形でないので開き量 0。後続母音は音符の開き量。
    # [m,a] [0,480) frame[0,15) → 語頭閉鎖 d_b=min(3.0,15*0.5)=3、BILABIAL[0,3)・A[3,15)。
    assert _build_so([_note(0, 480, phonemes=["m", "a"])], open_by_note=[0.6]) == [
        (MouthShape.BILABIAL, 0.0),
        (MouthShape.A, 0.6),
    ]


def test_build_open_zero_on_rest_silence():
    # 先頭休符は音符に紐づかないので開き量 0。続く母音は自音符の開き量。
    assert _build_so([_note(240, 240, phonemes=["a"])], open_by_note=[0.7]) == [
        (MouthShape.SILENCE, 0.0),
        (MouthShape.A, 0.7),
    ]


def test_build_open_on_continuation_uses_continuation_note_open():
    # 継続(母音なし→直前口形を継続)の保持イベントは、継続音符自身の開き量を持つ
    # (直前音符の開き量ではない)。
    adopted = [_note(0, 240, phonemes=["a"]), _note(240, 240, phonemes=["-"])]
    assert _build_so(adopted, open_by_note=[0.4, 0.8]) == [
        (MouthShape.A, 0.4),
        (MouthShape.A, 0.8),
    ]


def test_build_open_on_continuation_holding_moraic_nasal():
    # 継続が直前の撥音「ん」(N)を保持する場合、N は母音的口形なので継続音符自身の開き量を刻印する。
    adopted = [_note(0, 240, phonemes=["N\\"]), _note(240, 240, phonemes=["-"])]
    assert _build_so(adopted, use_n_morph=True, open_by_note=[0.3, 0.8]) == [
        (MouthShape.N, 0.3),
        (MouthShape.N, 0.8),
    ]


def test_build_open_zero_on_continuation_holding_closed():
    # 継続が直前の閉口(SILENCE)を保持する場合は母音的口形でないので、継続音符の開き量を
    # 無視して 0 にする(休符後の継続。[a][0,240)→休符[7.5,22.5)→[-][720,960) は閉口継続)。
    adopted = [_note(0, 240, phonemes=["a"]), _note(720, 240, phonemes=["-"])]
    assert _build_so(adopted, open_by_note=[0.4, 0.8]) == [
        (MouthShape.A, 0.4),
        (MouthShape.SILENCE, 0.0),
        (MouthShape.SILENCE, 0.0),
    ]
