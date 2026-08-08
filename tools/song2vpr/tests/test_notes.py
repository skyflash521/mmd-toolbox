"""song2vpr の音符分割のテスト。

時刻ごとの音高と音素セグメントから音符の区間と音高を切る規則を検証する。表示歌詞・音素列・強弱の
付与は後段が持つので、ここでは区間と音高だけを見る。
"""

import numpy as np
import pytest

from song2vpr import notes, pitch
from vocal_analysis import Segment

FRAME = 0.01


def _track(spans):
    """(秒, MIDI or None) の並びから、前段が返す形のピッチ列を作る。None は無声。"""
    midi = []
    for seconds, value in spans:
        midi += [np.nan if value is None else float(value)] * int(round(seconds / FRAME))
    midi = np.array(midi)
    return pitch.PitchTrack(
        times_sec=np.arange(len(midi)) * FRAME,
        f0_hz=440.0 * 2.0 ** ((midi - 69.0) / 12.0),
        voiced=~np.isnan(midi), midi=midi)


def _vowel(start, end, phoneme="a"):
    return Segment(type="vowel", start_sec=start, end_sec=end, phoneme=phoneme, confidence=0.9)


def _consonant(start, end, phoneme="k"):
    return Segment(type="consonant", start_sec=start, end_sec=end, phoneme=phoneme, confidence=0.9)


def _gap(start, end):
    return Segment(type="gap", start_sec=start, end_sec=end, phoneme=None, confidence=None)


def _one_vowel(track):
    """全体を1つの母音セグメントで覆う(母音の切り替わりを分割理由にしない場合に使う)。"""
    return [_vowel(0.0, float(track.times_sec[-1]) + FRAME)]


# --- 半音丸めと無声の扱い ----------------------------------------------------


def test_steady_pitch_becomes_one_note():
    track = _track([(0.5, 69)])
    result = notes.split(track, _one_vowel(track)).notes
    assert len(result) == 1
    assert result[0].midi == 69


def test_pitch_is_rounded_to_the_nearest_semitone():
    """音高は最も近い半音へ丸める。"""
    track = _track([(0.5, 69.4)])
    assert notes.split(track, _one_vowel(track)).notes[0].midi == 69


def test_unvoiced_span_produces_no_note():
    """無声の区間には音符を作らない(休符は書き出し側が補集合として導く)。"""
    track = _track([(0.3, 69), (0.3, None), (0.3, 71)])
    result = notes.split(track, _one_vowel(track)).notes
    assert [note.midi for note in result] == [69, 71]
    assert result[0].end_sec <= result[1].start_sec


def test_all_unvoiced_produces_no_note():
    track = _track([(0.5, None)])
    assert notes.split(track, _one_vowel(track)).notes == []


# --- 音節の頭の子音 ----------------------------------------------------------


def test_unvoiced_leading_consonant_joins_the_following_vowel():
    """後続の母音と同じ音節をなす子音は、無声でもその母音の音符に含める。

    含めないと、音符の音素列から音節の頭の子音が落ちる。
    """
    track = _track([(0.1, None), (0.4, 69)])  # 子音の区間は無声
    segments = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.5, "a")]
    result = notes.split(track, segments).notes
    assert len(result) == 1
    assert result[0].start_sec == pytest.approx(0.0, abs=FRAME)


def test_leading_consonant_does_not_eat_into_the_previous_note():
    """子音を取り込んでも、直前の音符の区間は削らない(音符どうしを重ねない)。"""
    track = _track([(0.3, 67), (0.1, 69), (0.3, 69)])
    segments = [_vowel(0.0, 0.3, "a"), _consonant(0.3, 0.4, "k"), _vowel(0.4, 0.7, "i")]
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [67, 69]
    assert result[0].end_sec == pytest.approx(0.3, abs=FRAME)
    assert result[1].start_sec == pytest.approx(0.3, abs=FRAME)


def test_moraic_nasal_becomes_its_own_note():
    """撥音は母音を伴わなくても1つの音節をなすので、独立した音符にする。"""
    track = _track([(0.3, 69), (0.2, 69)])  # 音高は変わらない
    segments = [_vowel(0.0, 0.3, "a"), _consonant(0.3, 0.5, "ɴ")]
    result = notes.split(track, segments).notes
    assert len(result) == 2
    assert result[1].start_sec == pytest.approx(0.3, abs=FRAME)


def test_syllable_head_goes_to_its_vowel_even_before_a_moraic_nasal():
    """「かん」のような並びで、頭の子音は母音の音符へ、撥音は独立した音符になる。"""
    track = _track([(0.1, None), (0.3, 69), (0.2, 69)])
    segments = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.4, "a"), _consonant(0.4, 0.6, "ɴ")]
    result = notes.split(track, segments).notes
    assert len(result) == 2
    assert result[0].start_sec == pytest.approx(0.0, abs=FRAME)
    assert result[1].start_sec == pytest.approx(0.4, abs=FRAME)


# --- 同一音高の連結 ----------------------------------------------------------


def test_same_pitch_without_a_break_is_one_note():
    """丸めた音高が同じで、間に休符も母音の切り替わりも無ければ1音符。"""
    track = _track([(0.3, 68.6), (0.3, 69.4)])  # どちらも 69 へ丸まる
    result = notes.split(track, _one_vowel(track)).notes
    assert len(result) == 1
    assert result[0].start_sec == pytest.approx(0.0, abs=FRAME)
    assert result[0].end_sec == pytest.approx(0.6, abs=FRAME)


def test_pitch_change_splits_the_note():
    track = _track([(0.3, 69), (0.3, 71)])
    assert [note.midi for note in notes.split(track, _one_vowel(track)).notes] == [69, 71]


def test_vowel_change_splits_the_note_at_the_same_pitch():
    """同じ音高が続いても、別の母音になれば別の音符にする。"""
    track = _track([(0.3, 69), (0.3, 69)])
    segments = [_vowel(0.0, 0.3, "a"), _vowel(0.3, 0.6, "i")]
    result = notes.split(track, segments).notes
    assert len(result) == 2
    assert [note.midi for note in result] == [69, 69]


def test_gap_alone_does_not_split_the_note():
    """gap は音素を割り当てなかった区間で、発声の継続中にも出る。切れ目とみなさない。"""
    track = _track([(0.3, 69), (0.3, 69)])
    segments = [_vowel(0.0, 0.3, "a"), _gap(0.3, 0.6)]
    assert len(notes.split(track, segments).notes) == 1


# --- 短い音符の扱い ----------------------------------------------------------


def test_short_note_merges_into_the_neighbour_of_the_same_pitch():
    """最小長に満たない音符は、同じ音高の隣接音符へ併合する。"""
    # 母音の切り替わりで 69 が2つに分かれ、そのうち後ろが最小長に満たない配置。
    track = _track([(0.3, 69), (0.03, 69), (0.3, 71)])
    segments = [_vowel(0.0, 0.3, "a"), _vowel(0.3, 0.33, "i"), _vowel(0.33, 0.63, "a")]
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [69, 71]
    assert result[0].end_sec == pytest.approx(0.33, abs=FRAME)


def test_short_note_is_absorbed_by_the_nearest_pitch():
    """同じ音高の隣接が無ければ、音高が最も近い隣接音符へ吸収する。"""
    track = _track([(0.3, 60), (0.03, 71), (0.3, 72)])
    result = notes.split(track, _one_vowel(track)).notes
    assert [note.midi for note in result] == [60, 72]
    # 吸収先は音高を保ち、短音符の区間を足して延びる。
    assert result[1].start_sec == pytest.approx(0.3, abs=FRAME)


def test_absorption_prefers_the_earlier_side_on_a_tie():
    """音高の距離が同じなら直前側へ吸収する。"""
    track = _track([(0.3, 67), (0.03, 69), (0.3, 71)])
    result = notes.split(track, _one_vowel(track)).notes
    assert [note.midi for note in result] == [67, 71]
    assert result[0].end_sec == pytest.approx(0.33, abs=FRAME)


def test_isolated_short_note_survives():
    """休符で隔てられ吸収先が無い短音符は、そのまま単独の音符として残す。"""
    track = _track([(0.2, None), (0.03, 69), (0.2, None)])
    result = notes.split(track, _one_vowel(track)).notes
    assert len(result) == 1
    assert result[0].midi == 69


def test_short_note_is_not_absorbed_across_a_rest():
    """休符を挟む音符は連続していないので吸収先にしない。"""
    track = _track([(0.3, 60), (0.2, None), (0.03, 71)])
    result = notes.split(track, _one_vowel(track)).notes
    assert [note.midi for note in result] == [60, 71]


# --- 音符の並び --------------------------------------------------------------


def test_notes_are_ordered_and_do_not_overlap():
    track = _track([(0.3, 69), (0.3, 71), (0.1, None), (0.3, 67)])
    result = notes.split(track, _one_vowel(track)).notes
    assert len(result) == 3
    for earlier, later in zip(result, result[1:], strict=False):
        assert earlier.end_sec <= later.start_sec
    assert all(note.end_sec > note.start_sec for note in result)


# --- 決定論 ------------------------------------------------------------------


def test_same_input_gives_the_same_notes():
    track = _track([(0.3, 69), (0.03, 71), (0.3, 67), (0.2, None), (0.3, 69)])
    segments = _one_vowel(track)
    first = notes.split(track, segments).notes
    second = notes.split(track, segments).notes
    assert [(n.start_sec, n.end_sec, n.midi) for n in first] == \
           [(n.start_sec, n.end_sec, n.midi) for n in second]


# --- 診断 --------------------------------------------------------------------


def test_short_notes_left_alone_are_counted():
    """まとめる先が無く短いまま残った音符は、件数を診断に出す(最小長は内部の値のため)。"""
    track = _track([(0.2, None), (0.03, 69), (0.2, None), (0.03, 71), (0.2, None)])
    result = notes.split(track, _one_vowel(track))
    assert len(result.notes) == 2
    assert result.diagnostics.short_notes == 2


def test_absorbed_short_notes_are_not_counted():
    """隣へ吸収された短音符は残っていないので数えない。

    母音の切り替わりで同じ音高が2つに分かれ、後ろが最小長に満たない配置(吸収が実際に起きる)。
    """
    track = _track([(0.3, 69), (0.03, 69)])
    segments = [_vowel(0.0, 0.3, "a"), _vowel(0.3, 0.33, "i")]
    result = notes.split(track, segments)
    assert len(result.notes) == 1
    assert result.diagnostics.short_notes == 0
