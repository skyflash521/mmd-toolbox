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

    assert reference_symbol_to_category(symbol) == "c"


def test_reference_symbol_to_category_xx_is_excluded():
    from vocal_analysis.gate import reference_symbol_to_category

    assert reference_symbol_to_category("xx") is None


@pytest.mark.parametrize("symbol", ["sy", "ty", "zy", "q", "I", "U", "O", ""])
def test_reference_symbol_to_category_unknown_symbol_raises(symbol):
    from vocal_analysis.gate import UnknownReferenceSymbolError, reference_symbol_to_category

    with pytest.raises(UnknownReferenceSymbolError):
        reference_symbol_to_category(symbol)


def test_parse_seconds_monophone_label_reads_seconds_and_maps_category(tmp_path):
    from vocal_analysis.gate import parse_seconds_monophone_label

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


def test_parse_htk100ns_monophone_label_line_missing_symbol_raises_error_naming_file(tmp_path):
    from vocal_analysis.gate import MonophoneLabelFormatError, parse_htk100ns_monophone_label

    label_path = tmp_path / "001.lab"
    label_path.write_text("0 35841272\n", encoding="utf-8")

    with pytest.raises(MonophoneLabelFormatError, match="001.lab"):
        parse_htk100ns_monophone_label(label_path)


def test_parse_htk100ns_monophone_label_delegates_to_reference_symbol_to_category(tmp_path, monkeypatch):
    from vocal_analysis import gate as gate_module

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

    segments = [_seg("a", 0.0, 1.0), _seg("i", 0.8, 2.0)]

    result = remove_invalid_time_segments(segments)

    assert result == [_seg("a", 0.0, 0.8), _seg("i", 1.0, 2.0)]


def test_remove_invalid_time_segments_drops_contained_segment_and_splits_container():
    from vocal_analysis.gate import remove_invalid_time_segments

    segments = [_seg("a", 0.0, 10.0), _seg("b", 2.0, 3.0)]

    result = remove_invalid_time_segments(segments)

    assert result == [_seg("a", 0.0, 2.0), _seg("a", 3.0, 10.0)]


def test_remove_invalid_time_segments_excludes_overlap_with_non_adjacent_segment():
    from vocal_analysis.gate import remove_invalid_time_segments

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

    segments = [_seg("a", 0.0, 0.5)]
    midi_notes = []

    assert find_midi_mismatch_ranges(segments, midi_notes) == [(0.0, 0.5)]


def test_find_midi_mismatch_ranges_ignores_vowel_gap_just_under_threshold():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    segments = [_seg("a", 0.0, 0.299)]
    midi_notes = []

    assert find_midi_mismatch_ranges(segments, midi_notes) == []


def test_find_midi_mismatch_ranges_detects_vowel_gap_of_exactly_threshold():
    from vocal_analysis.gate import find_midi_mismatch_ranges

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

    segments = [_seg("sil", 0.0, 0.5)]
    midi_notes = [(0.0, 0.1)]

    assert find_midi_mismatch_ranges(segments, midi_notes) == []


def test_find_midi_mismatch_ranges_sil_continuously_covered_by_adjacent_notes():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    segments = [_seg("sil", 0.0, 0.5)]
    midi_notes = [(0.0, 0.2), (0.2, 0.35)]

    assert find_midi_mismatch_ranges(segments, midi_notes) == [(0.0, 0.35)]


def test_find_midi_mismatch_ranges_sil_covered_with_gap_is_not_continuous():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    segments = [_seg("sil", 0.0, 0.5)]
    midi_notes = [(0.0, 0.2), (0.25, 0.5)]

    assert find_midi_mismatch_ranges(segments, midi_notes) == []


def test_find_midi_mismatch_ranges_sil_without_note_is_not_a_mismatch():
    from vocal_analysis.gate import find_midi_mismatch_ranges

    segments = [_seg("sil", 0.0, 0.5)]
    midi_notes = []

    assert find_midi_mismatch_ranges(segments, midi_notes) == []


def test_find_midi_mismatch_ranges_ignores_consonant_segments():
    from vocal_analysis.gate import find_midi_mismatch_ranges

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
    from vocal_analysis.gate import ticks_to_seconds
    from vpr.types import TempoEvent

    tempos = [TempoEvent(tick=0, bpm=120.0)]

    assert ticks_to_seconds(0, tempos, resolution=480) == pytest.approx(0.0)


def test_ticks_to_seconds_single_tempo_one_quarter_note():
    from vocal_analysis.gate import ticks_to_seconds
    from vpr.types import TempoEvent

    tempos = [TempoEvent(tick=0, bpm=120.0)]

    assert ticks_to_seconds(480, tempos, resolution=480) == pytest.approx(0.5)


def test_ticks_to_seconds_across_tempo_change():
    from vocal_analysis.gate import ticks_to_seconds
    from vpr.types import TempoEvent

    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=480, bpm=60.0)]

    assert ticks_to_seconds(960, tempos, resolution=480) == pytest.approx(0.5 + 1.0)


def test_ticks_to_seconds_mid_segment_after_tempo_change():
    from vocal_analysis.gate import ticks_to_seconds
    from vpr.types import TempoEvent

    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=480, bpm=60.0)]

    assert ticks_to_seconds(720, tempos, resolution=480) == pytest.approx(0.5 + 0.5)


def test_ticks_to_seconds_accepts_unsorted_tempo_list():
    from vocal_analysis.gate import ticks_to_seconds
    from vpr.types import TempoEvent

    tempos = [TempoEvent(tick=480, bpm=60.0), TempoEvent(tick=0, bpm=120.0)]

    assert ticks_to_seconds(960, tempos, resolution=480) == pytest.approx(1.5)


def _note(start_tick, duration_tick, phonemes):
    from vpr.types import Note

    return Note(start_tick=start_tick, duration_tick=duration_tick, pitch=60, lyric="", velocity=100, phonemes=phonemes)


def _part(notes):
    from vpr.types import Part

    return Part(name="", start_tick=0, notes=notes)


def test_generate_vpr_reference_segments_single_vowel_note():
    from vocal_analysis.gate import generate_vpr_reference_segments
    from vpr.types import TempoEvent

    part = _part([_note(0, 480, ["a"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == [_seg("a", 0.0, 0.5)]


def test_generate_vpr_reference_segments_uses_last_phoneme_as_representative():
    from vocal_analysis.gate import generate_vpr_reference_segments
    from vpr.types import TempoEvent

    part = _part([_note(0, 480, ["s", "a"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == [_seg("a", 0.0, 0.5)]


def test_generate_vpr_reference_segments_consonant_only_phoneme_maps_to_c():
    from vocal_analysis.gate import generate_vpr_reference_segments
    from vpr.types import TempoEvent

    part = _part([_note(0, 480, ["k"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == [_seg("c", 0.0, 0.5)]


def test_generate_vpr_reference_segments_continuation_inherits_and_merges_with_previous():
    from vocal_analysis.gate import generate_vpr_reference_segments
    from vpr.types import TempoEvent

    part = _part([_note(0, 480, ["a"]), _note(480, 480, ["-"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == [_seg("a", 0.0, 1.0)]


def test_generate_vpr_reference_segments_includes_internal_rest_as_sil():
    from vocal_analysis.gate import generate_vpr_reference_segments
    from vpr.types import TempoEvent

    part = _part([_note(0, 480, ["a"]), _note(960, 480, ["i"])])

    result = generate_vpr_reference_segments(part, tempos=[TempoEvent(tick=0, bpm=120.0)], resolution=480)

    assert result == [_seg("a", 0.0, 0.5), _seg("sil", 0.5, 1.0), _seg("i", 1.0, 1.5)]


def test_generate_vpr_reference_segments_leading_continuation_with_no_previous_is_skipped():
    from vocal_analysis.gate import generate_vpr_reference_segments
    from vpr.types import TempoEvent

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
    predicted = []

    assert compute_vowel_accuracy(predicted, reference, duration_sec=0.1) == pytest.approx(0.0)


def test_compute_vowel_accuracy_ignores_non_vowel_reference_frames():
    from vocal_analysis.gate import compute_vowel_accuracy

    reference = [_seg("c", 0.0, 0.05), _seg("a", 0.05, 0.1)]
    predicted = [_seg("i", 0.0, 0.05), _seg("a", 0.05, 0.1)]

    assert compute_vowel_accuracy(predicted, reference, duration_sec=0.1) == pytest.approx(1.0)


def test_compute_vowel_accuracy_half_correct():
    from vocal_analysis.gate import compute_vowel_accuracy

    reference = [_seg("a", 0.0, 0.1)]
    predicted = [_seg("a", 0.0, 0.05), _seg("i", 0.05, 0.1)]

    assert compute_vowel_accuracy(predicted, reference, duration_sec=0.1) == pytest.approx(0.5)


def test_compute_over_opening_rate_detects_vowel_bleed():
    from vocal_analysis.gate import compute_over_opening_rate

    reference = [_seg("sil", 0.0, 0.1)]
    predicted = [_seg("a", 0.0, 0.1)]

    assert compute_over_opening_rate(predicted, reference, duration_sec=0.1) == pytest.approx(1.0)


def test_compute_over_opening_rate_correct_silence_is_zero():
    from vocal_analysis.gate import compute_over_opening_rate

    reference = [_seg("sil", 0.0, 0.1)]
    predicted = [_seg("sil", 0.0, 0.1)]

    assert compute_over_opening_rate(predicted, reference, duration_sec=0.1) == pytest.approx(0.0)


def test_compute_over_opening_rate_includes_consonant_reference_frames():
    from vocal_analysis.gate import compute_over_opening_rate

    reference = [_seg("c", 0.0, 0.1)]
    predicted = [_seg("a", 0.0, 0.1)]

    assert compute_over_opening_rate(predicted, reference, duration_sec=0.1) == pytest.approx(1.0)


def test_compute_over_opening_rate_ignores_vowel_reference_frames():
    from vocal_analysis.gate import compute_over_opening_rate

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

    reference = [_seg("a", 0.0, 1.0)]
    predicted = [_seg("i", 0.0, 1.0)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == []
    assert undetected == [reference[0]]
    assert excess == [predicted[0]]


def test_match_segments_no_time_overlap_is_not_a_valid_pair():
    from vocal_analysis.gate import match_segments

    reference = [_seg("a", 0.0, 1.0)]
    predicted = [_seg("a", 2.0, 3.0)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == []
    assert undetected == [reference[0]]
    assert excess == [predicted[0]]


def test_match_segments_ignores_non_vowel_segments_entirely():
    from vocal_analysis.gate import match_segments

    reference = [_seg("c", 0.0, 1.0), _seg("a", 1.0, 2.0)]
    predicted = [_seg("sil", 0.0, 1.0), _seg("a", 1.0, 2.0)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == [(reference[1], predicted[1])]
    assert undetected == []
    assert excess == []


def test_match_segments_prefers_greater_total_overlap_over_first_come_pairing():
    from vocal_analysis.gate import match_segments

    reference = [_seg("a", 0.0, 3.0), _seg("a", 3.0, 7.05)]
    predicted = [_seg("a", 0.0, 7.0), _seg("a", 7.0, 7.1)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == [(reference[1], predicted[0])]
    assert undetected == [reference[0]]
    assert excess == [predicted[1]]


def test_match_segments_tie_break_prefers_lexicographically_smallest_assignment():
    from vocal_analysis.gate import match_segments

    reference = [_seg("a", 0.0, 3.0)]
    predicted = [_seg("a", 0.0, 1.0), _seg("a", 1.0, 2.0)]

    matched, undetected, excess = match_segments(predicted, reference)

    assert matched == [(reference[0], predicted[0])]
    assert undetected == []
    assert excess == [predicted[1]]


def test_compute_boundary_deviation_single_pair():
    from vocal_analysis.gate import compute_boundary_deviation

    matched_pairs = [(_seg("a", 0.0, 1.0), _seg("a", 0.05, 1.05))]

    result = compute_boundary_deviation(matched_pairs)

    assert result == pytest.approx((50.0, 50.0))


def test_compute_boundary_deviation_median_and_p95_over_multiple_pairs_ignore_sign():
    from vocal_analysis.gate import compute_boundary_deviation

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

    assert compute_boundary_deviation([]) is None


def test_compute_song_metrics_basic_composition():
    from vocal_analysis.gate import compute_song_metrics

    reference = [_seg("a", 0.0, 0.5), _seg("sil", 0.5, 1.0)]
    predicted = [_seg("a", 0.0, 0.5), _seg("sil", 0.5, 1.0)]

    result = compute_song_metrics(predicted, reference, duration_sec=1.0)

    assert result.vowel_accuracy == pytest.approx(1.0)
    assert result.over_opening_rate == pytest.approx(0.0)
    assert result.boundary_deviation == pytest.approx((0.0, 0.0))
    assert result.undetected_count == 0
    assert result.excess_count == 0
    assert result.reference_vowel_count == 1


def test_compute_song_metrics_reference_vowel_count_excludes_non_vowel_segments():
    from vocal_analysis.gate import compute_song_metrics

    reference = [_seg("c", 0.0, 0.5), _seg("a", 0.5, 1.0)]
    predicted = [_seg("c", 0.0, 0.5), _seg("i", 0.5, 1.0)]

    result = compute_song_metrics(predicted, reference, duration_sec=1.0)

    assert result.reference_vowel_count == 1
    assert result.undetected_count == 1
    assert result.excess_count == 1


def test_compute_song_metrics_boundary_deviation_is_none_without_matches():
    from vocal_analysis.gate import compute_song_metrics

    reference = [_seg("a", 0.0, 0.5), _seg("sil", 0.5, 1.0)]
    predicted = [_seg("i", 0.0, 0.5), _seg("sil", 0.5, 1.0)]

    result = compute_song_metrics(predicted, reference, duration_sec=1.0)

    assert result.boundary_deviation is None
