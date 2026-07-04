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


def test_ticks_to_seconds_at_tick_zero_is_zero():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import ticks_to_seconds

    tempos = [TempoEvent(tick=0, bpm=120.0)]

    assert ticks_to_seconds(0, tempos, resolution=480) == pytest.approx(0.0)


def test_ticks_to_seconds_single_tempo_one_quarter_note():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import ticks_to_seconds

    # BPM120では四分音符(1拍)は60/120=0.5秒。resolution(tick/四分音符)が480なら480tickで0.5秒。
    tempos = [TempoEvent(tick=0, bpm=120.0)]

    assert ticks_to_seconds(480, tempos, resolution=480) == pytest.approx(0.5)


def test_ticks_to_seconds_across_tempo_change():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import ticks_to_seconds

    # tick 0〜480: BPM120(0.5秒)。tick 480〜960: BPM60(四分音符1拍=60/60=1.0秒)。
    # 合計 tick=960 で 0.5+1.0=1.5秒。
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=480, bpm=60.0)]

    assert ticks_to_seconds(960, tempos, resolution=480) == pytest.approx(1.5)


def test_ticks_to_seconds_mid_segment_after_tempo_change():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import ticks_to_seconds

    # tick=720は tempo変化後(tick 480, BPM60)の区間の途中(240tick=0.5拍=0.5秒分)。
    # 0.5(最初の区間) + 0.5(2番目の区間の途中) = 1.0秒。
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=480, bpm=60.0)]

    assert ticks_to_seconds(720, tempos, resolution=480) == pytest.approx(1.0)


def test_ticks_to_seconds_accepts_unsorted_tempo_list():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import ticks_to_seconds

    # テンポイベントが tick 降順で渡されても、内部で並べ替えて正しく計算する。
    tempos = [TempoEvent(tick=480, bpm=60.0), TempoEvent(tick=0, bpm=120.0)]

    assert ticks_to_seconds(960, tempos, resolution=480) == pytest.approx(1.5)


def _note(start_tick, duration_tick, phonemes):
    from vpr.types import Note

    return Note(start_tick=start_tick, duration_tick=duration_tick, pitch=60, lyric="", velocity=100, phonemes=phonemes)


def _part(notes):
    from vpr.types import Part

    return Part(name="", start_tick=0, notes=notes)


def test_generate_vpr_reference_segments_single_vowel_note():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import generate_vpr_reference_segments

    part = _part([_note(0, 480, ["a"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == [_seg("a", 0.0, 0.5)]


def test_generate_vpr_reference_segments_uses_last_phoneme_as_representative():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import generate_vpr_reference_segments

    # 子音+母音(例: "s","a")の音符は、末尾の音素(母音)を代表として1区間にする
    # (音符内の音素別タイミングは持たないため)。
    part = _part([_note(0, 480, ["s", "a"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == [_seg("a", 0.0, 0.5)]


def test_generate_vpr_reference_segments_consonant_only_phoneme_maps_to_c():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import generate_vpr_reference_segments

    part = _part([_note(0, 480, ["k"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == [_seg("c", 0.0, 0.5)]


def test_generate_vpr_reference_segments_continuation_inherits_and_merges_with_previous():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import generate_vpr_reference_segments

    # 継続記号「-」の音符は直前の音符の音素を継承し、同一カテゴリで連続するため1区間に結合する。
    part = _part([_note(0, 480, ["a"]), _note(480, 480, ["-"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == [_seg("a", 0.0, 1.0)]


def test_generate_vpr_reference_segments_includes_internal_rest_as_sil():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import generate_vpr_reference_segments

    # 音符間の隙間[480tick, 960tick)(0.5〜1.0秒)は休符としてsil区間になる。
    part = _part([_note(0, 480, ["a"]), _note(960, 480, ["i"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == [_seg("a", 0.0, 0.5), _seg("sil", 0.5, 1.0), _seg("i", 1.0, 1.5)]


def test_generate_vpr_reference_segments_leading_continuation_with_no_previous_is_skipped():
    from vpr.types import TempoEvent

    from vocal_analysis.gate import generate_vpr_reference_segments

    # 最初の音符が継続記号のみで継承元が無い場合、解決できないためその音符は区間を生成しない。
    part = _part([_note(0, 480, ["-"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == []


def test_compute_vowel_accuracy_all_correct():
    from vocal_analysis.gate import compute_vowel_accuracy

    reference = [_seg("a", 0.0, 0.1)]
    predicted = [_seg("a", 0.0, 0.1)]

    assert compute_vowel_accuracy(predicted, reference, duration_sec=0.1) == pytest.approx(1.0)


def test_compute_vowel_accuracy_wrong_vowel_is_zero():
    from vocal_analysis.gate import compute_vowel_accuracy

    reference = [_seg("a", 0.0, 0.1)]
    predicted = [_seg("i", 0.0, 0.1)]

    assert compute_vowel_accuracy(predicted, reference, duration_sec=0.1) == pytest.approx(0.0)


def test_compute_vowel_accuracy_undetected_counts_as_mismatch():
    from vocal_analysis.gate import compute_vowel_accuracy

    reference = [_seg("a", 0.0, 0.1)]
    predicted = []  # 予測が何も無い(未検出)

    assert compute_vowel_accuracy(predicted, reference, duration_sec=0.1) == pytest.approx(0.0)


def test_compute_vowel_accuracy_ignores_non_vowel_reference_frames():
    from vocal_analysis.gate import compute_vowel_accuracy

    # 基準がc(子音)のフレームは分母に含めない。母音フレームだけを見て全一致なら1.0。
    reference = [_seg("c", 0.0, 0.05), _seg("a", 0.05, 0.1)]
    predicted = [_seg("i", 0.0, 0.05), _seg("a", 0.05, 0.1)]  # 子音区間の予測は何でもよい

    assert compute_vowel_accuracy(predicted, reference, duration_sec=0.1) == pytest.approx(1.0)


def test_compute_vowel_accuracy_half_correct():
    from vocal_analysis.gate import compute_vowel_accuracy

    # 母音フレーム10個中、前半5個が正解・後半5個が不一致で正解率0.5。
    reference = [_seg("a", 0.0, 0.1)]
    predicted = [_seg("a", 0.0, 0.05), _seg("i", 0.05, 0.1)]

    assert compute_vowel_accuracy(predicted, reference, duration_sec=0.1) == pytest.approx(0.5)


def test_compute_over_opening_rate_detects_vowel_bleed():
    from vocal_analysis.gate import compute_over_opening_rate

    reference = [_seg("sil", 0.0, 0.1)]
    predicted = [_seg("a", 0.0, 0.1)]  # 無音のはずが母音と誤検出(過開口)

    assert compute_over_opening_rate(predicted, reference, duration_sec=0.1) == pytest.approx(1.0)


def test_compute_over_opening_rate_correct_silence_is_zero():
    from vocal_analysis.gate import compute_over_opening_rate

    reference = [_seg("sil", 0.0, 0.1)]
    predicted = [_seg("sil", 0.0, 0.1)]

    assert compute_over_opening_rate(predicted, reference, duration_sec=0.1) == pytest.approx(0.0)


def test_compute_over_opening_rate_includes_consonant_reference_frames():
    from vocal_analysis.gate import compute_over_opening_rate

    # 基準がc(子音)のフレームも過開口率の分母に含む(母音フレームだけ除外)。
    reference = [_seg("c", 0.0, 0.1)]
    predicted = [_seg("a", 0.0, 0.1)]

    assert compute_over_opening_rate(predicted, reference, duration_sec=0.1) == pytest.approx(1.0)


def test_compute_over_opening_rate_ignores_vowel_reference_frames():
    from vocal_analysis.gate import compute_over_opening_rate

    # 基準が母音のフレーム(前半)は過開口率の分母に含めない。前半は予測を非母音にしておくことで、
    # もし母音フレームを誤って分母に含める実装なら分母が10フレームに増え、分子(母音誤検出は
    # 後半の5フレームだけ)は変わらないため結果が0.5に落ちて判別できる(母音フレームを正しく
    # 除外していれば分母は後半の5フレームだけなので1.0のまま)。
    reference = [_seg("a", 0.0, 0.05), _seg("sil", 0.05, 0.1)]
    predicted = [_seg("sil", 0.0, 0.05), _seg("a", 0.05, 0.1)]

    assert compute_over_opening_rate(predicted, reference, duration_sec=0.1) == pytest.approx(1.0)


def test_match_segments_matches_overlapping_same_vowel_pair():
    from vocal_analysis.gate import match_segments

    reference = [_seg("a", 0.0, 1.0)]
    predicted = [_seg("a", 0.1, 1.1)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == [(reference[0], predicted[0])]
    assert undetected == []
    assert excess == []


def test_match_segments_different_vowel_category_is_not_a_valid_pair():
    from vocal_analysis.gate import match_segments

    # 時間は完全に重なるが母音種別が異なるため対応しない。
    reference = [_seg("a", 0.0, 1.0)]
    predicted = [_seg("i", 0.0, 1.0)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == []
    assert undetected == [reference[0]]
    assert excess == [predicted[0]]


def test_match_segments_no_time_overlap_is_not_a_valid_pair():
    from vocal_analysis.gate import match_segments

    # 母音種別は一致するが時間重なりが無いため対応しない。
    reference = [_seg("a", 0.0, 1.0)]
    predicted = [_seg("a", 2.0, 3.0)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == []
    assert undetected == [reference[0]]
    assert excess == [predicted[0]]


def test_match_segments_ignores_non_vowel_segments_entirely():
    from vocal_analysis.gate import match_segments

    # c/silの区間は母音種別を持たないため対応付けの対象外(未検出にも余剰にも数えない)。
    reference = [_seg("c", 0.0, 1.0), _seg("a", 1.0, 2.0)]
    predicted = [_seg("sil", 0.0, 1.0), _seg("a", 1.0, 2.0)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == [(reference[1], predicted[1])]
    assert undetected == []
    assert excess == []


def test_match_segments_finds_globally_optimal_total_overlap():
    from vocal_analysis.gate import match_segments

    # 基準区間どうしは重複しない(ref0=[0,3)、ref1=[3,7.05))。ref0はpred0とだけ重なる(重複3.0)。
    # ref1はpred0(重複4.0)ともpred1(重複0.05)とも重なる。局所的にはref0がpred0を早い者勝ちで
    # 取りたいところだが、全体の総重複時間を最大化する割当は「ref1をpred0に対応させ、ref0は
    # 未検出・pred1は余剰にする」(総重複4.0)であり、「ref0をpred0に対応させる」
    # (総重複3.0+0.05=3.05)より大きい。
    reference = [_seg("a", 0.0, 3.0), _seg("a", 3.0, 7.05)]
    predicted = [_seg("a", 0.0, 7.0), _seg("a", 7.0, 7.1)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == [(reference[1], predicted[0])]
    assert undetected == [reference[0]]
    assert excess == [predicted[1]]


def test_match_segments_tie_break_prefers_lexicographically_smallest_assignment():
    from vocal_analysis.gate import match_segments

    # ref0(幅広い1区間)はpred0・pred1のどちらとも重複時間1.0で等しく対応できるため、総重複時間は
    # どちらを選んでも同点(1.0)。仕様の辞書式最小規則により、基準番号・予測番号の組が辞書式に
    # 小さい方((0基準番号, 0予測番号)、すなわちref0をpred0に対応させる)を選び、pred1が余剰になる。
    reference = [_seg("a", 0.0, 3.0)]
    predicted = [_seg("a", 0.0, 1.0), _seg("a", 1.0, 2.0)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == [(reference[0], predicted[0])]
    assert undetected == []
    assert excess == [predicted[1]]


def test_compute_boundary_deviation_single_pair():
    from vocal_analysis.gate import compute_boundary_deviation

    # 予測開始時刻が基準より50ms(0.05秒)遅い。1件だけなら中央値・95パーセンタイルとも同じ値。
    matched_pairs = [(_seg("a", 0.0, 1.0), _seg("a", 0.05, 1.05))]

    result = compute_boundary_deviation(matched_pairs)

    assert result == pytest.approx((50.0, 50.0))


def test_compute_boundary_deviation_median_and_p95_over_multiple_pairs():
    from vocal_analysis.gate import compute_boundary_deviation

    # 開始時刻差(ミリ秒、符号は問わないため正負を混在): 10, 20, 30, 40, 50。
    # 中央値(50パーセンタイル、線形補間)は30。95パーセンタイルは40と50の間を0.8で補間した48。
    # 各組の基準区間はmatch_segmentsが実際に出力しうる形(基準区間どうしが重複しない)に揃えている。
    matched_pairs = [
        (_seg("a", 0.0, 1.0), _seg("a", 0.01, 1.01)),
        (_seg("a", 2.0, 3.0), _seg("a", 1.98, 2.98)),
        (_seg("a", 4.0, 5.0), _seg("a", 4.03, 5.03)),
        (_seg("a", 6.0, 7.0), _seg("a", 6.04, 7.04)),
        (_seg("a", 8.0, 9.0), _seg("a", 8.05, 9.05)),
    ]

    result = compute_boundary_deviation(matched_pairs)

    assert result == pytest.approx((30.0, 48.0))


def test_compute_boundary_deviation_no_matched_pairs_is_undefined():
    from vocal_analysis.gate import compute_boundary_deviation

    # 対応区間が0件の曲は境界ずれが未定義(マクロ平均からの除外・ゲート不合格判定は集計処理側の責務)。
    assert compute_boundary_deviation([]) is None
