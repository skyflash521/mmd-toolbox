"""S-1 認識ゲートの参照ラベル写像テスト。

歌唱の実音素記号を a/i/u/e/o(母音)・c(子音)・sil(無音/息)へ写像する規則と、モノフォンラベル
形式(秒単位・HTK 100ns単位)のパーサを、実際の音素記号・時刻値を使って決定論的に検証する。
"""

import pytest


@pytest.mark.parametrize(
    "symbol,expected",
    [("a", "a"), ("i", "i"), ("u", "u"), ("e", "e"), ("o", "o")],
)
def test_reference_symbol_to_category_vowels(symbol, expected):
    from vocal_analysis.gate import reference_symbol_to_category

    assert reference_symbol_to_category(symbol) == expected


@pytest.mark.parametrize("symbol", ["pau", "br"])
def test_reference_symbol_to_category_silence(symbol):
    from vocal_analysis.gate import reference_symbol_to_category

    assert reference_symbol_to_category(symbol) == "sil"


@pytest.mark.parametrize(
    "symbol",
    [
        "b", "by", "ch", "cl", "d", "f", "g", "gy", "h", "hy", "j", "k", "ky",
        "m", "my", "n", "N", "ny", "p", "py", "r", "ry", "s", "sh", "t", "ts",
        "v", "w", "y", "z",
    ],
)
def test_reference_symbol_to_category_consonants(symbol):
    from vocal_analysis.gate import reference_symbol_to_category

    # cl(促音の閉鎖)・N(撥音)・上記以外の全子音は一律 c。
    assert reference_symbol_to_category(symbol) == "c"


def test_reference_symbol_to_category_xx_is_excluded():
    from vocal_analysis.gate import reference_symbol_to_category

    # xx(未定義区間)は母音でも子音でもsilでもなく、採点から除外する対象として None を返す。
    assert reference_symbol_to_category("xx") is None


@pytest.mark.parametrize("symbol", ["sy", "ty", "zy", "q", "I", "U", "O", ""])
def test_reference_symbol_to_category_unknown_symbol_raises(symbol):
    from vocal_analysis.gate import UnknownReferenceSymbolError, reference_symbol_to_category

    # 網羅した記号集合に無い想定外記号はエラーで停止する(黙って捨てない)。
    with pytest.raises(UnknownReferenceSymbolError):
        reference_symbol_to_category(symbol)


def test_parse_seconds_monophone_label_reads_seconds_and_maps_category(tmp_path):
    from vocal_analysis.gate import parse_seconds_monophone_label

    # 秒単位のモノフォンラベルは「開始秒 終了秒 音素」形式。
    label_path = tmp_path / "001.lab"
    label_path.write_text(
        "0.0000000 18.6263777 pau\n"
        "18.6263777 19.0916217 br\n"
        "19.0916217 19.1636238 k\n"
        "19.1636238 19.3223705 i\n",
        encoding="utf-8",
    )

    segments = parse_seconds_monophone_label(label_path)

    assert [s.category for s in segments] == ["sil", "sil", "c", "i"]
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(18.6263777)
    assert segments[-1].start_sec == pytest.approx(19.1636238)
    assert segments[-1].end_sec == pytest.approx(19.3223705)


def test_parse_seconds_monophone_label_skips_blank_lines(tmp_path):
    from vocal_analysis.gate import parse_seconds_monophone_label

    label_path = tmp_path / "001.lab"
    label_path.write_text("0.0 1.0 pau\n\n1.0 2.0 a\n", encoding="utf-8")

    segments = parse_seconds_monophone_label(label_path)

    assert len(segments) == 2


def test_parse_htk100ns_monophone_label_converts_units_to_seconds(tmp_path):
    from vocal_analysis.gate import parse_htk100ns_monophone_label

    # HTK 100ns単位のモノフォンラベルは「開始 終了 音素」形式(時刻は整数)。
    # 35841272 * 1e-7 = 3.5841272 秒。
    label_path = tmp_path / "001.lab"
    label_path.write_text(
        "0 35841272 pau\n35841272 36371876 m\n36371876 38893424 a\n",
        encoding="utf-8",
    )

    segments = parse_htk100ns_monophone_label(label_path)

    assert [s.category for s in segments] == ["sil", "c", "a"]
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(3.5841272)
    assert segments[1].start_sec == pytest.approx(3.5841272)
    assert segments[1].end_sec == pytest.approx(3.6371876)


def test_parse_seconds_monophone_label_xx_category_is_none(tmp_path):
    from vocal_analysis.gate import parse_seconds_monophone_label

    label_path = tmp_path / "001.lab"
    label_path.write_text("0.0 1.0 xx\n", encoding="utf-8")

    segments = parse_seconds_monophone_label(label_path)

    assert segments[0].category is None


def test_parse_seconds_monophone_label_unknown_symbol_raises(tmp_path):
    from vocal_analysis.gate import UnknownReferenceSymbolError, parse_seconds_monophone_label

    label_path = tmp_path / "001.lab"
    label_path.write_text("0.0 1.0 zy\n", encoding="utf-8")

    with pytest.raises(UnknownReferenceSymbolError):
        parse_seconds_monophone_label(label_path)


def test_parse_seconds_monophone_label_delegates_to_reference_symbol_to_category(tmp_path, monkeypatch):
    from vocal_analysis import gate as gate_module

    # パーサが記号写像を独自実装(重複)せず reference_symbol_to_category を呼び出すことを、
    # 差し替えた戻り値がそのままセグメントへ伝播するかで直接検証する(単に同じ例外が出ることの
    # 確認だけでは、パーサ側に独自の写像表を重複実装していても通ってしまうため不十分)。
    calls = []

    def fake_reference_symbol_to_category(symbol):
        calls.append(symbol)
        return "SENTINEL"

    monkeypatch.setattr(gate_module, "reference_symbol_to_category", fake_reference_symbol_to_category)

    label_path = tmp_path / "001.lab"
    label_path.write_text("0.0 1.0 k\n", encoding="utf-8")

    segments = gate_module.parse_seconds_monophone_label(label_path)

    assert calls == ["k"]
    assert segments[0].category == "SENTINEL"


def test_parse_htk100ns_monophone_label_malformed_line_raises_clear_error(tmp_path):
    from vocal_analysis.gate import MonophoneLabelFormatError, parse_htk100ns_monophone_label

    label_path = tmp_path / "001.lab"
    label_path.write_text("0 35841272\n", encoding="utf-8")  # 音素記号が欠落

    with pytest.raises(MonophoneLabelFormatError, match="001.lab"):
        parse_htk100ns_monophone_label(label_path)


def test_parse_htk100ns_monophone_label_delegates_to_reference_symbol_to_category(tmp_path, monkeypatch):
    from vocal_analysis import gate as gate_module

    # HTK 100ns単位のパーサも秒単位のパーサと同じく reference_symbol_to_category に委譲することを、
    # 差し替えた戻り値がそのままセグメントへ伝播するかで検証する(このパーサ自身が独自の写像表を
    # 重複実装していないことの確認)。
    calls = []

    def fake_reference_symbol_to_category(symbol):
        calls.append(symbol)
        return "SENTINEL"

    monkeypatch.setattr(gate_module, "reference_symbol_to_category", fake_reference_symbol_to_category)

    label_path = tmp_path / "001.lab"
    label_path.write_text("0 10000000 k\n", encoding="utf-8")

    segments = gate_module.parse_htk100ns_monophone_label(label_path)

    assert calls == ["k"]
    assert segments[0].category == "SENTINEL"


def _seg(category, start, end):
    from vocal_analysis.gate import CategorySegment

    return CategorySegment(category=category, start_sec=start, end_sec=end)


def test_clip_segments_to_audio_duration_leaves_in_range_segment_unchanged():
    from vocal_analysis.gate import clip_segments_to_audio_duration

    segments = [_seg("a", 0.0, 1.0)]

    result = clip_segments_to_audio_duration(segments, audio_duration_sec=2.0)

    assert result == segments


def test_clip_segments_to_audio_duration_trims_segment_extending_past_end():
    from vocal_analysis.gate import clip_segments_to_audio_duration

    # ラベルが音声より数秒長いケース: 音声長を超える区間は切り捨てる。
    segments = [_seg("a", 0.0, 3.0)]

    result = clip_segments_to_audio_duration(segments, audio_duration_sec=2.0)

    assert result == [_seg("a", 0.0, 2.0)]


def test_clip_segments_to_audio_duration_drops_segment_entirely_beyond_range():
    from vocal_analysis.gate import clip_segments_to_audio_duration

    segments = [_seg("a", 0.0, 1.0), _seg("i", 2.5, 3.0)]

    result = clip_segments_to_audio_duration(segments, audio_duration_sec=2.0)

    assert result == [_seg("a", 0.0, 1.0)]


def test_clip_segments_to_audio_duration_clips_negative_start_to_zero():
    from vocal_analysis.gate import clip_segments_to_audio_duration

    segments = [_seg("a", -0.5, 1.0)]

    result = clip_segments_to_audio_duration(segments, audio_duration_sec=2.0)

    assert result == [_seg("a", 0.0, 1.0)]


def test_clip_segments_to_audio_duration_drops_segment_entirely_before_zero():
    from vocal_analysis.gate import clip_segments_to_audio_duration

    segments = [_seg("a", -2.0, -1.0), _seg("i", 0.0, 1.0)]

    result = clip_segments_to_audio_duration(segments, audio_duration_sec=2.0)

    assert result == [_seg("i", 0.0, 1.0)]


def test_clip_segments_to_audio_duration_leaves_uncovered_tail_unfilled():
    from vocal_analysis.gate import clip_segments_to_audio_duration

    # ラベルが音声より早く終わる(末尾欠落)場合、末尾を埋める合成区間は追加しない
    # (被覆しない区間は採点から除外=結果に含めないだけでよい)。
    segments = [_seg("a", 0.0, 1.0)]

    result = clip_segments_to_audio_duration(segments, audio_duration_sec=5.0)

    assert result == [_seg("a", 0.0, 1.0)]


def test_remove_invalid_time_segments_leaves_valid_segments_unchanged():
    from vocal_analysis.gate import remove_invalid_time_segments

    segments = [_seg("a", 0.0, 1.0), _seg("i", 1.0, 2.0)]

    result = remove_invalid_time_segments(segments)

    assert result == segments


def test_remove_invalid_time_segments_drops_zero_length_segment():
    from vocal_analysis.gate import remove_invalid_time_segments

    segments = [_seg("a", 0.0, 1.0), _seg("i", 1.0, 1.0), _seg("u", 1.0, 2.0)]

    result = remove_invalid_time_segments(segments)

    assert result == [_seg("a", 0.0, 1.0), _seg("u", 1.0, 2.0)]


def test_remove_invalid_time_segments_drops_time_reversed_segment():
    from vocal_analysis.gate import remove_invalid_time_segments

    segments = [_seg("a", 0.0, 1.0), _seg("i", 2.0, 1.5), _seg("u", 2.0, 3.0)]

    result = remove_invalid_time_segments(segments)

    assert result == [_seg("a", 0.0, 1.0), _seg("u", 2.0, 3.0)]


def test_remove_invalid_time_segments_excludes_overlapping_range_from_both_sides():
    from vocal_analysis.gate import remove_invalid_time_segments

    # 2つの参照区間が [0.8, 1.0) で重複する。重複時間範囲はどちらの区間にも属させず、
    # 採点から除外する(隙間として残す)。
    segments = [_seg("a", 0.0, 1.0), _seg("i", 0.8, 2.0)]

    result = remove_invalid_time_segments(segments)

    assert result == [_seg("a", 0.0, 0.8), _seg("i", 1.0, 2.0)]


def test_remove_invalid_time_segments_excludes_fully_contained_overlap():
    from vocal_analysis.gate import remove_invalid_time_segments

    # 短い区間 [2.0, 3.0) が長い区間 [0.0, 10.0) に完全に包含される場合、包含された区間全体が
    # 重複時間範囲として除外され、長い区間はその前後2つに分かれて残る(非重複部分を失わない)。
    segments = [_seg("a", 0.0, 10.0), _seg("b", 2.0, 3.0)]

    result = remove_invalid_time_segments(segments)

    assert result == [_seg("a", 0.0, 2.0), _seg("a", 3.0, 10.0)]


def test_remove_invalid_time_segments_excludes_overlap_with_non_adjacent_segment():
    from vocal_analysis.gate import remove_invalid_time_segments

    # 長い区間 [0.0, 10.0) が、互いには重ならない2つの短い区間 [3.0, 4.0)・[8.0, 9.0) の
    # それぞれと重複する。隣接ペアの重複解決だけでは検出できない、隔たった区間との重複も
    # 正しく除外できることを確認する(長い区間は3つに分かれ、短い区間はどちらも消える)。
    segments = [_seg("a", 0.0, 10.0), _seg("b", 3.0, 4.0), _seg("c", 8.0, 9.0)]

    result = remove_invalid_time_segments(segments)

    assert result == [_seg("a", 0.0, 3.0), _seg("a", 4.0, 8.0), _seg("a", 9.0, 10.0)]


def test_find_midi_mismatch_ranges_no_mismatch_when_vowel_covered_by_note():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    segments = [_seg("a", 0.0, 1.0)]
    midi_notes = [(0.0, 1.0)]

    assert find_midi_mismatch_ranges(segments, midi_notes) == []


def test_find_midi_mismatch_ranges_detects_vowel_with_no_note_for_300ms_or_more():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    # 母音区間[0.0, 0.5)と時間的に重なるMIDIノートが1つも無く、300ms以上続く。
    segments = [_seg("a", 0.0, 0.5)]
    midi_notes = []

    assert find_midi_mismatch_ranges(segments, midi_notes) == [(0.0, 0.5)]


def test_find_midi_mismatch_ranges_ignores_vowel_gap_just_under_threshold():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    # ノートに重ならない区間が300ms未満(0.299秒)なので候補異常にしない(閾値の境界)。
    segments = [_seg("a", 0.0, 0.299)]
    midi_notes = []

    assert find_midi_mismatch_ranges(segments, midi_notes) == []


def test_find_midi_mismatch_ranges_detects_vowel_gap_of_exactly_threshold():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    # ちょうど300msは「300ms以上」に含む(境界を含む)。
    segments = [_seg("a", 0.0, 0.3)]
    midi_notes = []

    assert find_midi_mismatch_ranges(segments, midi_notes) == [(0.0, 0.3)]


def test_find_midi_mismatch_ranges_detects_sil_covered_by_note_for_300ms_or_more():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    segments = [_seg("sil", 0.0, 0.5)]
    midi_notes = [(0.0, 0.5)]

    assert find_midi_mismatch_ranges(segments, midi_notes) == [(0.0, 0.5)]


def test_find_midi_mismatch_ranges_sil_brief_note_overlap_is_not_a_mismatch():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    # sil区間[0.0, 0.5)のうち、ノートに覆われるのは[0.0, 0.1)の100msだけ(300ms未満)。
    segments = [_seg("sil", 0.0, 0.5)]
    midi_notes = [(0.0, 0.1)]

    assert find_midi_mismatch_ranges(segments, midi_notes) == []


def test_find_midi_mismatch_ranges_sil_continuously_covered_by_adjacent_notes():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    # 隙間なく連続する2つのノート[0.0,0.2)・[0.2,0.35)が合わせて350ms連続でsilを覆う。
    segments = [_seg("sil", 0.0, 0.5)]
    midi_notes = [(0.0, 0.2), (0.2, 0.35)]

    assert find_midi_mismatch_ranges(segments, midi_notes) == [(0.0, 0.35)]


def test_find_midi_mismatch_ranges_sil_covered_with_gap_is_not_continuous():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    # ノート[0.0,0.2)と[0.25,0.5)の間に隙間があり、どちらの被覆も単独では300ms未満のため
    # 連続被覆にならない(食い違いにしない)。
    segments = [_seg("sil", 0.0, 0.5)]
    midi_notes = [(0.0, 0.2), (0.25, 0.5)]

    assert find_midi_mismatch_ranges(segments, midi_notes) == []


def test_find_midi_mismatch_ranges_sil_without_note_is_not_a_mismatch():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    # sil区間がMIDIノートに覆われていない(=通常どおり無音)のは異常ではない。
    segments = [_seg("sil", 0.0, 0.5)]
    midi_notes = []

    assert find_midi_mismatch_ranges(segments, midi_notes) == []


def test_find_midi_mismatch_ranges_ignores_consonant_segments():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    # 子音区間はこの粗整合の対象外(母音とsilだけを見る)。
    segments = [_seg("c", 0.0, 1.0)]
    midi_notes = []

    assert find_midi_mismatch_ranges(segments, midi_notes) == []


def test_exclude_ranges_from_segments_splits_segment_around_excluded_middle():
    from vocal_analysis.gate import exclude_ranges_from_segments

    segments = [_seg("a", 0.0, 10.0)]
    ranges_to_exclude = [(3.0, 4.0)]

    result = exclude_ranges_from_segments(segments, ranges_to_exclude)

    assert result == [_seg("a", 0.0, 3.0), _seg("a", 4.0, 10.0)]


def test_exclude_ranges_from_segments_drops_fully_excluded_segment():
    from vocal_analysis.gate import exclude_ranges_from_segments

    segments = [_seg("a", 0.0, 1.0), _seg("i", 1.0, 2.0)]
    ranges_to_exclude = [(0.0, 1.0)]

    result = exclude_ranges_from_segments(segments, ranges_to_exclude)

    assert result == [_seg("i", 1.0, 2.0)]


def test_exclude_ranges_from_segments_leaves_unaffected_segment_unchanged():
    from vocal_analysis.gate import exclude_ranges_from_segments

    segments = [_seg("a", 0.0, 1.0)]
    ranges_to_exclude = [(5.0, 6.0)]

    result = exclude_ranges_from_segments(segments, ranges_to_exclude)

    assert result == segments


@pytest.mark.xfail(reason="impl pending: vocal_analysis.gate tick to seconds conversion", strict=True)
def test_ticks_to_seconds_at_tick_zero_is_zero():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import ticks_to_seconds

    tempos = [TempoEvent(tick=0, bpm=120.0)]

    assert ticks_to_seconds(0, tempos, resolution=480) == pytest.approx(0.0)


@pytest.mark.xfail(reason="impl pending: vocal_analysis.gate tick to seconds conversion", strict=True)
def test_ticks_to_seconds_single_tempo_one_quarter_note():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import ticks_to_seconds

    # BPM120では四分音符(1拍)は60/120=0.5秒。resolution(tick/四分音符)が480なら480tickで0.5秒。
    tempos = [TempoEvent(tick=0, bpm=120.0)]

    assert ticks_to_seconds(480, tempos, resolution=480) == pytest.approx(0.5)


@pytest.mark.xfail(reason="impl pending: vocal_analysis.gate tick to seconds conversion", strict=True)
def test_ticks_to_seconds_across_tempo_change():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import ticks_to_seconds

    # tick 0〜480: BPM120(0.5秒)。tick 480〜960: BPM60(四分音符1拍=60/60=1.0秒)。
    # 合計 tick=960 で 0.5+1.0=1.5秒。
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=480, bpm=60.0)]

    assert ticks_to_seconds(960, tempos, resolution=480) == pytest.approx(1.5)


@pytest.mark.xfail(reason="impl pending: vocal_analysis.gate tick to seconds conversion", strict=True)
def test_ticks_to_seconds_mid_segment_after_tempo_change():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import ticks_to_seconds

    # tick=720は tempo変化後(tick 480, BPM60)の区間の途中(240tick=0.5拍=0.5秒分)。
    # 0.5(最初の区間) + 0.5(2番目の区間の途中) = 1.0秒。
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=480, bpm=60.0)]

    assert ticks_to_seconds(720, tempos, resolution=480) == pytest.approx(1.0)


@pytest.mark.xfail(reason="impl pending: vocal_analysis.gate tick to seconds conversion", strict=True)
def test_ticks_to_seconds_accepts_unsorted_tempo_list():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import ticks_to_seconds

    # テンポイベントが tick 降順で渡されても、内部で並べ替えて正しく計算する。
    tempos = [TempoEvent(tick=480, bpm=60.0), TempoEvent(tick=0, bpm=120.0)]

    assert ticks_to_seconds(960, tempos, resolution=480) == pytest.approx(1.5)
