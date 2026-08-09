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


def _flickering_voicing(first, middle, last):
    """頭子音の中で有声の判定が途切れ、短い音符が同じ音節の音符に前後を挟まれる配置を作る。

    無声で隔てられた3つの有声区間は、頭子音への開始の延長で隣り合う。有声の子音では F0 が
    断続的にしか得られないことがあるため、この並びは実際の入力でも起こる。
    """
    track = _track([(0.10, first), (0.02, None), (0.03, middle), (0.02, None), (0.43, last)])
    return track, [_consonant(0.0, 0.20), _vowel(0.20, 0.60)]


# --- 半音丸めと無声の扱い ----------------------------------------------------


def test_steady_pitch_becomes_one_note():
    track = _track([(0.5, 69)])
    result = notes.split(track, _one_vowel(track)).notes
    assert len(result) == 1
    assert result[0].midi == 69


@pytest.mark.parametrize(("value", "expected"), [(69.4, 69), (69.6, 70)])
def test_pitch_is_rounded_to_the_nearest_semitone(value, expected):
    """音高は最も近い半音へ丸める。"""
    track = _track([(0.5, value)])
    assert notes.split(track, _one_vowel(track)).notes[0].midi == expected


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


def test_a_momentary_pitch_change_does_not_split_the_note():
    """ビブラートやしゃくりで瞬間的に半音を超えて動いても、それだけでは別の音符にしない。"""
    track = _track([(0.2, 69), (0.03, 71), (0.2, 69)])
    result = notes.split(track, _one_vowel(track)).notes
    assert len(result) == 1
    assert result[0].midi == 69


def test_a_pitch_change_that_lasts_splits_the_note():
    """最小長ぶん続く音高の変化は、瞬間的な揺れではないので別の音符にする。"""
    track = _track([(0.2, 69), (0.08, 71), (0.2, 69)])
    assert [note.midi for note in notes.split(track, _one_vowel(track)).notes] == [69, 71, 69]


def test_the_note_pitch_is_the_median_of_its_frames():
    """音符の音高は、その音符の中のフレームの MIDI ノート番号の中央値にする。

    どの値もそれだけでは最小長ぶん続かないので1音符になる。先頭のフレームの値なら 60、平均を
    丸めると 64、最も多い値なら 61 になる配置で、中央値の 67 になることを見る。
    """
    track = _track([(0.03, 60), (0.07, 61), (0.06, 67), (0.05, 68)])
    result = notes.split(track, _one_vowel(track)).notes
    assert len(result) == 1
    assert result[0].midi == 67


def test_the_median_is_taken_before_rounding_to_a_semitone():
    """中央値は半音へ丸める前の値で求め、その中央値を丸める。

    先に各フレームを半音へ丸めてから中央値を求めると 69 と 70 の中間になり 70 へ倒れる配置で、
    丸める前の中央値 69.1 から 69 になることを見る。
    """
    track = _track([(0.07, 68.6), (0.07, 69.6)])
    result = notes.split(track, _one_vowel(track)).notes
    assert len(result) == 1
    assert result[0].midi == 69


def test_the_median_pitch_is_not_truncated():
    """中央値は切り捨てず、最も近い半音へ丸める。"""
    # 中央値は 69.6 なので、切り捨てなら 69、丸めれば 70 になる。
    track = _track([(0.07, 69.2), (0.07, 70.0)])
    assert notes.split(track, _one_vowel(track)).notes[0].midi == 70


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


def test_short_note_is_not_absorbed_across_a_syllable():
    """まとめる先は同じ音節に属する隣接音符だけ。別の音節しか隣に無ければそのまま残す。

    音節をまたいでまとめると音節=モーラが消え、1音符が1つの発声に対応しなくなる。
    """
    # 中央の短い音符だけが別の音節に属し、前後の隣接はどちらも別音節になる配置。
    track = _track([(0.3, 69), (0.03, 69), (0.3, 71)])
    segments = [_vowel(0.0, 0.3, "a"), _vowel(0.3, 0.33, "i"), _vowel(0.33, 0.63, "a")]
    result = notes.split(track, segments)
    assert [note.syllable for note in result.notes] == [0, 1, 2]
    assert result.diagnostics.short_notes == 1


def test_short_note_is_absorbed_by_the_nearest_pitch():
    """両隣が同じ音節にあるときは、音高が最も近い隣接音符へ吸収する。"""
    track, segments = _flickering_voicing(60, 71, 72)
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [60, 72]
    # 吸収先は音高を保ち、短音符の区間を足して延びる。
    assert result[1].start_sec == pytest.approx(0.10, abs=FRAME)


def test_absorption_prefers_the_earlier_side_on_a_tie():
    """音高の距離が同じなら直前側へ吸収する。"""
    track, segments = _flickering_voicing(67, 69, 71)
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [67, 71]
    assert result[0].end_sec == pytest.approx(0.15, abs=FRAME)


def test_short_note_is_absorbed_by_the_preceding_note_in_the_same_syllable():
    """最小長に満たない音符は、同じ音節の直前の音符へ吸収する。

    音高の変化が続いて音符になった後、音節の切り替わりで切り詰められると最小長を割ることがある。
    """
    # 音高は 0.30 秒で変わって続くので音符になるが、0.36 秒の音節の境界で切り詰められる。
    track = _track([(0.30, 69), (0.30, 71)])
    segments = [_vowel(0.0, 0.36, "a"), _vowel(0.36, 0.60, "i")]
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [69, 71]
    # 吸収先は音高を保ち、短音符の区間を足して延びる。
    assert result[0].end_sec == pytest.approx(0.36, abs=FRAME)


def test_short_note_is_absorbed_by_the_following_note_in_the_same_syllable():
    """音節の最初の音符が最小長に満たなければ、同じ音節の直後の音符へ吸収する。

    吸収先の音高を保つので、最初の音節の音高は 69 ではなく 71 になる。音高の変化が続くかどうかは
    音節の境界で区切らずに見るため、0.07 秒からの 71 は音節の先まで続いて音符になる。
    """
    track = _track([(0.07, 69), (0.30, 71)])
    segments = [_vowel(0.0, 0.13, "a"), _vowel(0.13, 0.37, "i")]
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [71, 71]
    assert result[0].start_sec == pytest.approx(0.0, abs=FRAME)


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


# --- 音節の帰属 --------------------------------------------------------------


def test_note_carries_its_syllable():
    """音符は自分が属する音節を持つ(付与の段が時間の重なりで決め直さないため)。

    最初の音節は音高の変わり目で2つの音符に分かれるので、音符の通し番号を入れる実装では
    期待値に一致しない。
    """
    track = _track([(0.1, None), (0.3, 69), (0.2, 71), (0.3, 62)])
    segments = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.6, "a"), _vowel(0.6, 0.9, "i")]
    assert [note.syllable for note in notes.split(track, segments).notes] == [0, 0, 1]


def test_a_gap_before_the_nucleus_does_not_pull_the_head_consonant_back():
    """頭の子音と後続の核の間に gap があっても、子音は後続の音節に属する。

    音符の帰属とセグメントの帰属が食い違うと、付与の段が別の音節の音素を載せる。
    """
    track = _track([(0.2, 69), (0.2, 69), (0.2, 71)])
    head = _consonant(0.2, 0.3, "k")
    first = _vowel(0.0, 0.2, "a")
    second = _vowel(0.4, 0.6, "i")
    result = notes.split(track, [first, head, _gap(0.3, 0.4), second])
    assert [note.syllable for note in result.notes] == [0, 1, 1]
    assert result.syllable_segments == [[first], [head, second]]


def test_a_nucleus_shorter_than_the_frame_still_takes_its_head_consonant():
    """時刻の刻みより短くフレームを1つも覆わない核でも、その頭の子音は核と同じ音節に属する。

    フレームの被覆から後続の核を数えると、この核は見つからず子音が直前の音節へ倒れる。
    """
    track = _track([(0.5, 69)])
    head = _consonant(0.2, 0.3, "k")
    first = _vowel(0.0, 0.2, "a")
    second = _vowel(0.402, 0.408, "i")
    result = notes.split(track, [first, head, _gap(0.3, 0.402), second, _gap(0.408, 0.5)])
    assert [note.syllable for note in result.notes] == [0, 1]
    assert result.syllable_segments == [[first], [head, second]]


def test_a_gap_after_a_frame_less_nucleus_belongs_to_that_nucleus():
    """フレームを1つも覆わない核でも、その後の gap はその核の音節を引き継ぐ。

    頭の子音が無いと番号を先に置いてくれる音符が無いので、引き継ぎだけでは前の音節に留まる。
    """
    track = _track([(0.5, 69)])
    first = _vowel(0.0, 0.2, "a")
    second = _vowel(0.402, 0.408, "i")
    result = notes.split(track, [first, _gap(0.2, 0.402), second, _gap(0.408, 0.5)])
    assert [note.syllable for note in result.notes] == [0, 1]
    assert result.syllable_segments == [[first], [second]]


def test_split_result_lists_the_segments_of_each_syllable():
    """分割結果は音節ごとのセグメント帰属を持つ。頭の子音は後続の音節に属する。"""
    track = _track([(0.1, None), (0.3, 69), (0.3, 71)])
    head = _consonant(0.0, 0.1, "k")
    first = _vowel(0.1, 0.4, "a")
    second = _vowel(0.4, 0.7, "i")
    result = notes.split(track, [head, first, second])
    assert result.syllable_segments == [[head, first], [second]]


def test_gap_segments_are_not_assigned_to_a_syllable():
    """gap は音素を割り当てなかった区間なので、音節のセグメント列に載せない。"""
    track = _track([(0.3, 69), (0.3, 69)])
    vowel = _vowel(0.0, 0.3, "a")
    result = notes.split(track, [vowel, _gap(0.3, 0.6)])
    assert result.syllable_segments == [[vowel]]


# --- 音節と結びつかない音符の抑制 --------------------------------------------


def test_voiced_span_belonging_to_no_syllable_is_not_output():
    """前後に核が無く帰属を決められない有声区間は音符にしない。"""
    track = _track([(0.3, 69), (0.2, None), (0.3, 71)])
    result = notes.split(track, [_gap(0.0, 0.5), _vowel(0.5, 0.8, "a")])
    assert [note.midi for note in result.notes] == [71]
    assert result.diagnostics.suppressed_notes == 1


def test_component_starting_far_from_the_nucleus_is_dropped_whole():
    """核へ届く音符を含まない連続成分は、先頭が核の終端から離れていれば全件落とす。

    分離しきれなかった伴奏が、gap の引き継ぎで核から遠く離れたまま音符化するのを抑える。
    """
    track = _track([(0.3, 69), (2.2, None), (0.3, 60), (0.3, 62)])
    result = notes.split(track, [_vowel(0.0, 0.3, "a"), _gap(0.3, 3.1)])
    assert [note.midi for note in result.notes] == [69]
    assert result.diagnostics.suppressed_notes == 2


def test_component_near_the_nucleus_end_survives_whole():
    """核の終端の近くから始まる成分は全件残す。判定は成分の先頭だけで行い、途中では行わない。

    核を長くとってあるので、核の開始から測る実装ならこの成分は落ちる(基準点が終端であることの
    確認)。成分の2つ目の音符は核の終端から境目より遠くで始まるので、音符ごとに判定する実装でも
    落ちる(判定の単位が成分であることの確認)。
    """
    track = _track([(0.3, 69), (2.7, None), (1.6, 60), (0.6, 62)])
    result = notes.split(track, [_vowel(0.0, 2.5, "a"), _gap(2.5, 5.2)])
    assert [note.midi for note in result.notes] == [69, 60, 62]
    assert result.diagnostics.suppressed_notes == 0


def test_component_reaching_the_nucleus_survives_however_far_it_runs():
    """核と重なる音符を含む成分は全件残す。

    3つ目の音符は核の終端から境目より遠くで始まるので、音符ごとに判定する実装なら落ちる。
    """
    track = _track([(0.3, 69), (2.3, 71), (0.6, 72)])
    result = notes.split(track, [_vowel(0.0, 0.3, "a"), _gap(0.3, 3.2)])
    assert [note.midi for note in result.notes] == [69, 71, 72]
    assert result.diagnostics.suppressed_notes == 0


def test_a_component_does_not_span_two_syllables():
    """成分は同じ音節に属する音符の並び。切れ目が無くても音節が変われば別の成分にする。

    次の音節の核へ届く音符と接しているだけで、前の音節の遠い音符まで残ってしまわないこと。
    """
    track = _track([(0.3, 69), (2.2, None), (0.3, 60), (0.3, 62)])
    segments = [_vowel(0.0, 0.3, "a"), _gap(0.3, 2.8), _vowel(2.8, 3.1, "i")]
    result = notes.split(track, segments)
    assert [note.midi for note in result.notes] == [69, 62]
    assert result.diagnostics.suppressed_notes == 1


def test_suppressed_notes_are_not_counted_as_short():
    """抑制で出力しない音符は、短いまま残った音符として数えない。"""
    track = _track([(0.3, 69), (2.5, None), (0.03, 60), (0.2, None)])
    result = notes.split(track, [_vowel(0.0, 0.3, "a"), _gap(0.3, 3.03)])
    assert [note.midi for note in result.notes] == [69]
    assert result.diagnostics.suppressed_notes == 1
    assert result.diagnostics.short_notes == 0


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

    音節の最初の音符が最小長に満たず、同じ音節の直後の音符へ吸収される配置。
    """
    track = _track([(0.07, 69), (0.30, 71)])
    segments = [_vowel(0.0, 0.13, "a"), _vowel(0.13, 0.37, "i")]
    result = notes.split(track, segments)
    assert len(result.notes) == 2
    assert result.diagnostics.short_notes == 0
