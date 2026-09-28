import pytest

from song2vpr import project
from song2vpr.lyrics import SungNote
from song2vpr.tempo import TempoEstimate
from vpr import read, write_file


def _tempo(bpm=120.0, numerator=4, denominator=4):
    source = "default" if (numerator, denominator) == (4, 4) else "option"
    return TempoEstimate(bpm=bpm, numerator=numerator, denominator=denominator,
                         tempo_source="estimated", time_signature_source=source)


def _sung(start_sec, end_sec, midi=60, lyric="あ", phonemes=None, velocity=64,
          is_protected=False):
    return SungNote(start_sec=start_sec, end_sec=end_sec, midi=midi, lyric=lyric,
                    phonemes=list(phonemes) if phonemes is not None else ["a"], velocity=velocity,
                    is_protected=is_protected)


def test_collapsed_notes_take_lyric_from_the_syllable_head_and_pitch_from_the_longest():
    result = project.build([_sung(0.0, 0.01, midi=64, lyric="さ", phonemes=["s", "a"],
                                  velocity=30, is_protected=True),
                            _sung(0.01, 0.0105, midi=62, lyric="-", phonemes=["-"], velocity=20),
                            _sung(0.0105, 0.0106, midi=60, lyric="か", phonemes=["k", "a"],
                                  velocity=10, is_protected=True)],
                           _tempo(), name="song")
    assert result.diagnostics.quantized_merged_notes == 1
    note = result.project.tracks[0].parts[0].notes[1]
    assert (note.lyric, note.phonemes, note.is_protected) == ("か", ["k", "a"], True)
    assert (note.pitch, note.velocity) == (62, 20)


def test_a_collapsed_syllable_head_without_phonemes_still_wins():
    result = project.build([_sung(0.0, 0.0001, lyric="あ", phonemes=[]),
                            _sung(0.0001, 0.0006, lyric="-", phonemes=["-"])],
                           _tempo(), name="song")
    note = result.project.tracks[0].parts[0].notes[0]
    assert (note.lyric, note.phonemes, note.is_protected) == ("あ", [], False)


def test_the_earlier_syllable_head_wins_regardless_of_length_when_two_collapse():
    result = project.build([_sung(0.0, 0.0001, lyric="か", phonemes=["k", "a"], is_protected=True),
                            _sung(0.0001, 0.0006, lyric="き", phonemes=[])],
                           _tempo(), name="song")
    note = result.project.tracks[0].parts[0].notes[0]
    assert (note.lyric, note.phonemes, note.is_protected) == ("か", ["k", "a"], True)


@pytest.mark.parametrize("is_protected", [True, False])
def test_the_phoneme_protection_reaches_the_written_note(is_protected):
    note = SungNote(start_sec=0.0, end_sec=0.5, midi=60, lyric="か", phonemes=["k", "a"],
                    velocity=64, is_protected=is_protected)
    result = project.build([note], _tempo(), name="song")
    assert result.project.tracks[0].parts[0].notes[0].is_protected is is_protected


def test_project_has_one_singing_track_with_one_part():
    result = project.build([_sung(0.0, 0.5)], _tempo(), name="song")
    assert len(result.project.tracks) == 1
    assert len(result.project.tracks[0].parts) == 1


def test_title_track_and_part_names_use_the_output_base_name():
    result = project.build([_sung(0.0, 0.5)], _tempo(), name="my_song")
    track = result.project.tracks[0]
    assert (result.project.title, track.name, track.parts[0].name) == \
           ("my_song", "my_song", "my_song")


def test_part_starts_at_zero_and_ends_at_the_last_note():
    tempo = _tempo()
    result = project.build([_sung(0.0, 0.5), _sung(0.5, 1.0)], tempo, name="song")
    part = result.project.tracks[0].parts[0]
    assert part.start_tick == 0
    assert part.duration_tick == tempo.to_tick(1.0)


def test_part_of_a_project_without_notes_is_one_bar_long():
    result = project.build([], _tempo(bpm=120.0, numerator=3, denominator=8), name="song")
    part = result.project.tracks[0].parts[0]
    assert part.notes == []
    assert part.duration_tick == 3 * 480 * 4 // 8


def test_tempo_and_time_signature_are_single_events_at_the_head():
    result = project.build([_sung(0.0, 0.5)], _tempo(bpm=136.0, numerator=3, denominator=8),
                           name="song")
    assert [(t.tick, t.bpm) for t in result.project.tempos] == [(0, 136.0)]
    assert [(s.tick, s.numerator, s.denominator) for s in result.project.time_signatures] == \
           [(0, 3, 8)]


def test_part_has_no_controller_curves():
    result = project.build([_sung(0.0, 0.5)], _tempo(), name="song")
    assert result.project.tracks[0].parts[0].controllers == []


def test_part_specifies_a_voice_bank():
    voice = project.build([_sung(0.0, 0.5)], _tempo(), name="song").project.tracks[0].parts[0].voice
    assert voice is not None
    assert voice.comp_id and voice.name


def test_note_fields_are_carried_over():
    tempo = _tempo()
    note = _sung(0.5, 1.0, midi=62, lyric="い", phonemes=["i"], velocity=100)
    written = project.build([note], tempo, name="song").project.tracks[0].parts[0].notes[0]
    assert (written.start_tick, written.pitch, written.lyric, written.velocity) == \
           (tempo.to_tick(0.5), 62, "い", 100)
    assert written.phonemes == ["i"]
    assert written.duration_tick == tempo.to_tick(1.0) - tempo.to_tick(0.5)


def test_notes_are_not_shifted_onto_the_bar_line():
    tempo = _tempo()
    result = project.build([_sung(0.5, 1.0)], tempo, name="song")
    part = result.project.tracks[0].parts[0]
    assert part.start_tick == 0
    assert part.notes[0].start_tick == tempo.to_tick(0.5)
    assert [t.tick for t in result.project.tempos] == [0]
    assert [s.tick for s in result.project.time_signatures] == [0]


def test_adjacent_notes_share_the_boundary_tick():
    notes = project.build([_sung(0.0, 0.31), _sung(0.31, 0.62)], _tempo(),
                          name="song").project.tracks[0].parts[0].notes
    assert notes[0].start_tick + notes[0].duration_tick == notes[1].start_tick


def test_note_that_rounds_to_zero_length_is_stretched_to_one_tick():
    result = project.build([_sung(0.0, 0.05, lyric="あ"),
                            _sung(0.05, 0.0501, lyric="-", phonemes=["-"])],
                           _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert notes[1].duration_tick == 1
    assert result.diagnostics.quantized_stretched_notes == 1


def test_notes_collapsing_to_one_tick_merge_up_to_the_last_end_with_pitch_from_the_longest():
    result = project.build([_sung(0.0, 0.0001, midi=60, lyric="あ", phonemes=["a"], velocity=10),
                            _sung(0.0001, 0.0005, midi=62, lyric="い", phonemes=["i"], velocity=20),
                            _sung(0.0005, 0.0006, midi=64, lyric="う", phonemes=["M"], velocity=30),
                            _sung(0.0006, 0.02, lyric="え")],
                           _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert len(notes) == 2
    assert (notes[0].pitch, notes[0].velocity) == (62, 20)
    assert (notes[0].lyric, notes[0].phonemes) == ("あ", ["a"])
    assert notes[0].start_tick + notes[0].duration_tick == _tempo().to_tick(0.0006)
    assert result.diagnostics.quantized_merged_notes == 2


def test_notes_of_the_same_length_are_merged_into_the_earlier_one():
    notes = project.build([_sung(0.0, 0.0005, midi=60), _sung(0.0005, 0.001, midi=62)],
                          _tempo(), name="song").project.tracks[0].parts[0].notes
    assert [n.pitch for n in notes] == [60]


@pytest.mark.parametrize(("midi", "expected"), [(-3, 0), (200, 127)])
def test_pitch_outside_the_storable_range_is_clamped(midi, expected):
    result = project.build([_sung(0.0, 0.5, midi=midi)], _tempo(), name="song")
    assert result.project.tracks[0].parts[0].notes[0].pitch == expected
    assert result.diagnostics.pitch_clamped_notes == 1


def test_notes_do_not_overlap_after_merging_and_stretching():
    result = project.build([_sung(0.0, 0.0005), _sung(0.0005, 0.0011),
                            _sung(0.01, 0.0105), _sung(0.0115, 0.5)], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert (result.diagnostics.quantized_merged_notes,
            result.diagnostics.quantized_stretched_notes) == (1, 1)
    ends = [n.start_tick + n.duration_tick for n in notes]
    assert all(end <= nxt.start_tick for end, nxt in zip(ends, notes[1:], strict=False))
    assert all(n.duration_tick > 0 for n in notes)


def test_built_project_can_be_written_and_read_back(tmp_path):
    path = tmp_path / "song.vpr"
    result = project.build([_sung(0.0, 0.5, lyric="あ", phonemes=["a"])], _tempo(), name="song")
    write_file(result.project, path)

    restored, _warnings = read(path.read_bytes())
    note = restored.tracks[0].parts[0].notes[0]
    assert (restored.title, note.lyric, note.phonemes) == ("song", "あ", ["a"])


def test_same_input_gives_the_same_project():
    notes = [_sung(0.0, 0.5), _sung(0.5, 1.1)]
    assert project.build(notes, _tempo(), name="song") == project.build(notes, _tempo(),
                                                                        name="song")


def _sec_at_tick(ticks):
    return ticks / 960.0


def test_short_lyric_note_in_a_run_reaches_the_minimum_by_proportional_redistribution():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(20), lyric="あ"),
                            _sung(_sec_at_tick(20), _sec_at_tick(60), lyric="-", phonemes=["-"]),
                            _sung(_sec_at_tick(60), _sec_at_tick(150), lyric="い")], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert [(n.start_tick, n.duration_tick) for n in notes] == [(0, 27), (27, 39), (66, 84)]
    assert result.diagnostics.short_notes == 0


def test_isolated_short_lyric_note_extends_to_the_minimum():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(20), lyric="あ"),
                            _sung(_sec_at_tick(500), _sec_at_tick(600), lyric="い")], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert (notes[0].start_tick, notes[0].duration_tick) == (0, 27)
    assert result.diagnostics.short_notes == 0


def test_isolated_short_lyric_note_stops_at_the_next_note_and_is_counted_as_short():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(20), lyric="あ"),
                            _sung(_sec_at_tick(24), _sec_at_tick(200), lyric="い")], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert (notes[0].start_tick, notes[0].duration_tick) == (0, 24)
    assert (notes[1].start_tick, notes[1].duration_tick) == (24, 176)
    assert result.diagnostics.short_notes == 1


def test_run_too_short_to_afford_the_minimum_is_left_unchanged_and_counted_as_short():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(20), lyric="あ"),
                            _sung(_sec_at_tick(20), _sec_at_tick(40), lyric="い")], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert [(n.start_tick, n.duration_tick) for n in notes] == [(0, 20), (20, 20)]
    assert result.diagnostics.short_notes == 2


def test_short_continuation_note_does_not_trigger_redistribution():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(100), lyric="あ"),
                            _sung(_sec_at_tick(100), _sec_at_tick(110), lyric="-", phonemes=["-"])],
                           _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert [(n.start_tick, n.duration_tick) for n in notes] == [(0, 100), (100, 10)]
    assert result.diagnostics.short_notes == 0


def test_redistribution_does_not_touch_notes_outside_the_run():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(50), lyric="ま"),
                            _sung(_sec_at_tick(100), _sec_at_tick(120), lyric="あ"),
                            _sung(_sec_at_tick(120), _sec_at_tick(220), lyric="い"),
                            _sung(_sec_at_tick(300), _sec_at_tick(400), lyric="も")], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert (notes[0].start_tick, notes[0].duration_tick) == (0, 50)
    assert (notes[3].start_tick, notes[3].duration_tick) == (300, 100)
    assert notes[1].start_tick == 100
    assert notes[1].duration_tick >= 27
    assert notes[2].start_tick + notes[2].duration_tick == 220


def test_equal_fractions_give_the_leftover_tick_to_the_earlier_note():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(20), lyric="あ"),
                            _sung(_sec_at_tick(20), _sec_at_tick(67), lyric="い"),
                            _sung(_sec_at_tick(67), _sec_at_tick(114), lyric="う")], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert [(n.start_tick, n.duration_tick) for n in notes] == [(0, 27), (27, 44), (71, 43)]


def test_last_isolated_short_note_extends_and_updates_part_length():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(20), lyric="あ")], _tempo(), name="song")
    note = result.project.tracks[0].parts[0].notes[0]
    assert (note.start_tick, note.duration_tick) == (0, 27)
    assert result.project.tracks[0].parts[0].duration_tick == 27


def test_unaffordable_run_counts_continuation_notes_left_below_the_minimum_too():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(20), lyric="あ"),
                            _sung(_sec_at_tick(20), _sec_at_tick(40), lyric="-", phonemes=["-"])],
                           _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert [(n.start_tick, n.duration_tick) for n in notes] == [(0, 20), (20, 20)]
    assert result.diagnostics.short_notes == 2


def test_redistributed_notes_do_not_overlap_and_keep_positive_length():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(10), lyric="あ"),
                            _sung(_sec_at_tick(10), _sec_at_tick(20), lyric="-", phonemes=["-"]),
                            _sung(_sec_at_tick(20), _sec_at_tick(120), lyric="い")], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    ends = [n.start_tick + n.duration_tick for n in notes]
    assert all(end == nxt.start_tick for end, nxt in zip(ends, notes[1:], strict=False))
    assert all(n.duration_tick >= 27 for n in notes)
    assert ends[-1] == 120


def test_equal_fractions_are_compared_exactly_not_after_float_rounding():
    result = project.build([_sung(_sec_at_tick(0), _sec_at_tick(25), lyric="あ"),
                            _sung(_sec_at_tick(25), _sec_at_tick(53), lyric="い"),
                            _sung(_sec_at_tick(53), _sec_at_tick(84), lyric="う"),
                            _sung(_sec_at_tick(84), _sec_at_tick(112), lyric="え")], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert [(n.start_tick, n.duration_tick) for n in notes] == \
           [(0, 27), (27, 28), (55, 30), (85, 27)]
