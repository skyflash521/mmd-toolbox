import numpy as np
import pytest

from song2vpr import notes, pitch
from vocal_analysis import Segment

FRAME = 0.01


def _track(seconds_and_midi_or_unvoiced):
    midi = []
    for seconds, value in seconds_and_midi_or_unvoiced:
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


def _one_vowel_covering(track):
    return [_vowel(0.0, float(track.times_sec[-1]) + FRAME)]


def _voicing_flickering_in_the_head_consonant(first, middle, last):
    track = _track([(0.10, first), (0.02, None), (0.03, middle), (0.02, None), (0.43, last)])
    return track, [_consonant(0.0, 0.20), _vowel(0.20, 0.60)]


def test_steady_pitch_becomes_one_note():
    track = _track([(0.5, 69)])
    result = notes.split(track, _one_vowel_covering(track)).notes
    assert len(result) == 1
    assert result[0].midi == 69


@pytest.mark.parametrize(("value", "expected"), [(69.4, 69), (69.6, 70)])
def test_pitch_is_rounded_to_the_nearest_semitone(value, expected):
    track = _track([(0.5, value)])
    assert notes.split(track, _one_vowel_covering(track)).notes[0].midi == expected


def test_unvoiced_span_produces_no_note():
    track = _track([(0.3, 69), (0.3, None), (0.3, 71)])
    result = notes.split(track, _one_vowel_covering(track)).notes
    assert [note.midi for note in result] == [69, 71]
    assert result[0].end_sec <= result[1].start_sec


def test_all_unvoiced_produces_no_note():
    track = _track([(0.5, None)])
    assert notes.split(track, _one_vowel_covering(track)).notes == []


def test_unvoiced_leading_consonant_joins_the_following_vowel():
    track = _track([(0.1, None), (0.4, 69)])
    segments = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.5, "a")]
    result = notes.split(track, segments).notes
    assert len(result) == 1
    assert result[0].start_sec == pytest.approx(0.0, abs=FRAME)


def test_leading_consonant_does_not_eat_into_the_previous_note():
    track = _track([(0.3, 67), (0.1, 69), (0.3, 69)])
    segments = [_vowel(0.0, 0.3, "a"), _consonant(0.3, 0.4, "k"), _vowel(0.4, 0.7, "i")]
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [67, 69]
    assert result[0].end_sec == pytest.approx(0.3, abs=FRAME)
    assert result[1].start_sec == pytest.approx(0.3, abs=FRAME)


def test_moraic_nasal_becomes_its_own_note_at_the_same_pitch():
    track = _track([(0.3, 69), (0.2, 69)])
    segments = [_vowel(0.0, 0.3, "a"), _consonant(0.3, 0.5, "ɴ")]
    result = notes.split(track, segments).notes
    assert len(result) == 2
    assert result[1].start_sec == pytest.approx(0.3, abs=FRAME)


def test_geminate_closure_joins_the_following_syllable():
    track = _track([(0.3, 69), (0.15, None), (0.25, 69)])
    closure, onset = _consonant(0.3, 0.4, "k"), _consonant(0.4, 0.45, "k")
    segments = [_vowel(0.0, 0.3, "a"), closure, onset, _vowel(0.45, 0.7, "a")]
    split = notes.split(track, segments)
    assert len(split.notes) == 2
    assert split.notes[1].start_sec == pytest.approx(0.3, abs=FRAME)
    assert closure in split.syllable_segments[split.notes[1].syllable]
    assert closure not in split.syllable_segments[split.notes[0].syllable]


def test_syllable_head_goes_to_its_vowel_even_before_a_moraic_nasal():
    track = _track([(0.1, None), (0.3, 69), (0.2, 69)])
    segments = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.4, "a"), _consonant(0.4, 0.6, "ɴ")]
    result = notes.split(track, segments).notes
    assert len(result) == 2
    assert result[0].start_sec == pytest.approx(0.0, abs=FRAME)
    assert result[1].start_sec == pytest.approx(0.4, abs=FRAME)


def test_frames_rounding_to_the_same_semitone_without_a_break_are_one_note():
    track = _track([(0.3, 68.6), (0.3, 69.4)])
    result = notes.split(track, _one_vowel_covering(track)).notes
    assert len(result) == 1
    assert result[0].start_sec == pytest.approx(0.0, abs=FRAME)
    assert result[0].end_sec == pytest.approx(0.6, abs=FRAME)


def test_pitch_change_splits_the_note():
    track = _track([(0.4, 69), (0.4, 71)])
    assert [note.midi for note in notes.split(track, _one_vowel_covering(track)).notes] == [69, 71]


def test_a_momentary_pitch_change_does_not_split_the_note():
    track = _track([(0.2, 69), (0.03, 71), (0.2, 69)])
    result = notes.split(track, _one_vowel_covering(track)).notes
    assert len(result) == 1
    assert result[0].midi == 69


def test_a_pitch_change_lasting_longer_than_the_hold_length_splits_the_note():
    track = _track([(0.4, 69), (0.36, 71), (0.4, 69)])
    assert [note.midi for note in notes.split(track, _one_vowel_covering(track)).notes] == [69, 71, 69]


def test_a_pitch_change_at_exactly_the_hold_length_splits_the_note():
    track = _track([(0.4, 69), (0.34, 71), (0.4, 69)])
    assert [note.midi for note in notes.split(track, _one_vowel_covering(track)).notes] == [69, 71, 69]


def test_a_pitch_change_one_frame_shorter_than_the_hold_length_does_not_split():
    track = _track([(0.4, 69), (0.33, 71), (0.4, 69)])
    result = notes.split(track, _one_vowel_covering(track)).notes
    assert len(result) == 1
    assert result[0].midi == 69


def test_the_note_pitch_is_the_most_frequent_semitone_not_the_first_frame_nor_the_median():
    track = _track([(0.03, 60), (0.07, 61), (0.06, 67), (0.05, 68)])
    result = notes.split(track, _one_vowel_covering(track)).notes
    assert len(result) == 1
    assert result[0].midi == 61


def test_frames_are_rounded_to_semitones_before_counting_the_mode():
    track = _track([(0.03, 59.6), (0.03, 60.4), (0.05, 70.0), (0.05, 71.0)])
    assert notes.split(track, _one_vowel_covering(track)).notes[0].midi == 60


def test_the_mode_tie_falls_to_the_lower_semitone():
    track = _track([(0.07, 69.0), (0.07, 71.0)])
    assert notes.split(track, _one_vowel_covering(track)).notes[0].midi == 69


def test_vowel_change_splits_the_note_at_the_same_pitch():
    track = _track([(0.3, 69), (0.3, 69)])
    segments = [_vowel(0.0, 0.3, "a"), _vowel(0.3, 0.6, "i")]
    result = notes.split(track, segments).notes
    assert len(result) == 2
    assert [note.midi for note in result] == [69, 69]


def test_gap_alone_does_not_split_the_note():
    track = _track([(0.3, 69), (0.3, 69)])
    segments = [_vowel(0.0, 0.3, "a"), _gap(0.3, 0.6)]
    assert len(notes.split(track, segments).notes) == 1


def test_short_note_whose_neighbours_are_other_syllables_is_not_absorbed():
    track = _track([(0.3, 69), (0.03, 69), (0.3, 71)])
    segments = [_vowel(0.0, 0.3, "a"), _vowel(0.3, 0.33, "i"), _vowel(0.33, 0.63, "a")]
    result = notes.split(track, segments)
    assert [note.syllable for note in result.notes] == [0, 1, 2]


def test_short_note_is_absorbed_by_the_nearest_pitch_neighbour_which_keeps_its_pitch_and_extends():
    track, segments = _voicing_flickering_in_the_head_consonant(60, 71, 72)
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [60, 72]
    assert result[1].start_sec == pytest.approx(0.10, abs=FRAME)


def test_absorption_prefers_the_earlier_side_on_a_pitch_distance_tie():
    track, segments = _voicing_flickering_in_the_head_consonant(67, 69, 71)
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [67, 71]
    assert result[0].end_sec == pytest.approx(0.15, abs=FRAME)


def test_note_cut_short_by_a_syllable_boundary_is_absorbed_by_the_preceding_note_in_the_same_syllable():
    track = _track([(0.36, 69), (0.40, 71)])
    segments = [_vowel(0.0, 0.42, "a"), _vowel(0.42, 0.76, "i")]
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [69, 71]
    assert result[0].end_sec == pytest.approx(0.42, abs=FRAME)


def test_short_first_note_of_a_syllable_is_absorbed_by_the_following_note_keeping_its_pitch():
    track = _track([(0.07, 69), (0.40, 71)])
    segments = [_vowel(0.0, 0.13, "a"), _vowel(0.13, 0.47, "i")]
    result = notes.split(track, segments).notes
    assert [note.midi for note in result] == [71, 71]
    assert result[0].start_sec == pytest.approx(0.0, abs=FRAME)


def test_isolated_short_note_survives():
    track = _track([(0.2, None), (0.03, 69), (0.2, None)])
    result = notes.split(track, _one_vowel_covering(track)).notes
    assert len(result) == 1
    assert result[0].midi == 69


def test_short_note_is_not_absorbed_across_a_rest():
    track = _track([(0.3, 60), (0.2, None), (0.03, 71)])
    result = notes.split(track, _one_vowel_covering(track)).notes
    assert [note.midi for note in result] == [60, 71]


def test_note_carries_its_syllable_index_not_its_note_index():
    track = _track([(0.1, None), (0.4, 69), (0.4, 71), (0.4, 62)])
    segments = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.9, "a"), _vowel(0.9, 1.3, "i")]
    assert [note.syllable for note in notes.split(track, segments).notes] == [0, 0, 1]


def test_a_gap_before_the_nucleus_does_not_pull_the_head_consonant_back():
    track = _track([(0.2, 69), (0.2, 69), (0.4, 71)])
    head = _consonant(0.2, 0.3, "k")
    first = _vowel(0.0, 0.2, "a")
    second = _vowel(0.4, 0.8, "i")
    result = notes.split(track, [first, head, _gap(0.3, 0.4), second])
    assert [note.syllable for note in result.notes] == [0, 1, 1]
    assert result.syllable_segments == [[first], [head, second]]


def test_a_nucleus_shorter_than_the_frame_still_takes_its_head_consonant():
    track = _track([(0.5, 69)])
    head = _consonant(0.2, 0.3, "k")
    first = _vowel(0.0, 0.2, "a")
    second = _vowel(0.402, 0.408, "i")
    result = notes.split(track, [first, head, _gap(0.3, 0.402), second, _gap(0.408, 0.5)])
    assert [note.syllable for note in result.notes] == [0, 1]
    assert result.syllable_segments == [[first], [head, second]]


def test_a_gap_after_a_frame_less_nucleus_without_head_consonant_belongs_to_that_nucleus():
    track = _track([(0.5, 69)])
    first = _vowel(0.0, 0.2, "a")
    second = _vowel(0.402, 0.408, "i")
    result = notes.split(track, [first, _gap(0.2, 0.402), second, _gap(0.408, 0.5)])
    assert [note.syllable for note in result.notes] == [0, 1]
    assert result.syllable_segments == [[first], [second]]


def test_split_result_lists_the_segments_of_each_syllable_with_the_head_consonant_in_its_syllable():
    track = _track([(0.1, None), (0.3, 69), (0.3, 71)])
    head = _consonant(0.0, 0.1, "k")
    first = _vowel(0.1, 0.4, "a")
    second = _vowel(0.4, 0.7, "i")
    result = notes.split(track, [head, first, second])
    assert result.syllable_segments == [[head, first], [second]]


def test_gap_segments_are_not_assigned_to_a_syllable():
    track = _track([(0.3, 69), (0.3, 69)])
    vowel = _vowel(0.0, 0.3, "a")
    result = notes.split(track, [vowel, _gap(0.3, 0.6)])
    assert result.syllable_segments == [[vowel]]


def test_voiced_span_with_no_nucleus_on_either_side_is_not_output_and_is_counted():
    track = _track([(0.3, 69), (0.2, None), (0.3, 71)])
    result = notes.split(track, [_gap(0.0, 0.5), _vowel(0.5, 0.8, "a")])
    assert [note.midi for note in result.notes] == [71]
    assert result.diagnostics.suppressed_notes == 1


def test_component_not_reaching_the_nucleus_and_starting_far_from_its_end_is_dropped_whole():
    track = _track([(0.3, 69), (2.2, None), (0.4, 60), (0.4, 62)])
    result = notes.split(track, [_vowel(0.0, 0.3, "a"), _gap(0.3, 3.3)])
    assert [note.midi for note in result.notes] == [69]
    assert result.diagnostics.suppressed_notes == 2


def test_component_starting_near_the_long_nucleus_end_survives_whole_though_its_later_note_starts_far():
    track = _track([(0.3, 69), (2.7, None), (1.6, 60), (0.6, 62)])
    result = notes.split(track, [_vowel(0.0, 2.5, "a"), _gap(2.5, 5.2)])
    assert [note.midi for note in result.notes] == [69, 60, 62]
    assert result.diagnostics.suppressed_notes == 0


def test_component_reaching_the_nucleus_survives_however_far_it_runs():
    track = _track([(0.3, 69), (2.3, 71), (0.6, 72)])
    result = notes.split(track, [_vowel(0.0, 0.3, "a"), _gap(0.3, 3.2)])
    assert [note.midi for note in result.notes] == [69, 71, 72]
    assert result.diagnostics.suppressed_notes == 0


def test_a_component_does_not_span_two_syllables():
    track = _track([(0.3, 69), (2.2, None), (0.3, 60), (0.3, 62)])
    segments = [_vowel(0.0, 0.3, "a"), _gap(0.3, 2.8), _vowel(2.8, 3.1, "i")]
    result = notes.split(track, segments)
    assert [note.midi for note in result.notes] == [69, 62]
    assert result.diagnostics.suppressed_notes == 1


def test_a_suppressed_short_note_is_counted_as_suppressed():
    track = _track([(0.3, 69), (2.5, None), (0.03, 60), (0.2, None)])
    result = notes.split(track, [_vowel(0.0, 0.3, "a"), _gap(0.3, 3.03)])
    assert [note.midi for note in result.notes] == [69]
    assert result.diagnostics.suppressed_notes == 1


def test_notes_are_ordered_and_do_not_overlap():
    track = _track([(0.4, 69), (0.4, 71), (0.1, None), (0.4, 67)])
    result = notes.split(track, _one_vowel_covering(track)).notes
    assert len(result) == 3
    for earlier, later in zip(result, result[1:], strict=False):
        assert earlier.end_sec <= later.start_sec
    assert all(note.end_sec > note.start_sec for note in result)


def test_same_input_gives_the_same_notes():
    track = _track([(0.3, 69), (0.03, 71), (0.3, 67), (0.2, None), (0.3, 69)])
    segments = _one_vowel_covering(track)
    first = notes.split(track, segments).notes
    second = notes.split(track, segments).notes
    assert [(n.start_sec, n.end_sec, n.midi) for n in first] == \
           [(n.start_sec, n.end_sec, n.midi) for n in second]


def test_short_notes_without_an_absorb_target_are_left_in_place():
    track = _track([(0.2, None), (0.03, 69), (0.2, None), (0.03, 71), (0.2, None)])
    result = notes.split(track, _one_vowel_covering(track))
    assert len(result.notes) == 2
