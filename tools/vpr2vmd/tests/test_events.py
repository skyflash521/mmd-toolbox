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


_TEMPOS = [TempoEvent(0, 120.0)]
_RES = 480


def _built_shapes_and_spans(adopted, *, use_n_morph=True, legato_max_frames=None):
    kw = {} if legato_max_frames is None else {"legato_max_frames": legato_max_frames}
    result, _diag = events.build_mouth_events(
        adopted, _TEMPOS, _RES, use_n_morph=use_n_morph, **kw
    )
    return [(e.shape, e.start, e.end) for e in result]


def test_resolve_overlaps_of_no_notes_is_empty():
    assert events.resolve_overlaps([])[0] == []


def test_resolve_overlaps_keeps_non_overlapping_notes():
    notes = [_note(0, 100), _note(200, 100)]
    assert _spans(events.resolve_overlaps(notes)[0]) == [(0, 100), (200, 300)]


def test_same_start_keeps_longest_only():
    notes = [_note(0, 120, lyric="long"), _note(0, 80, lyric="short")]
    adopted, _diag = events.resolve_overlaps(notes)
    assert [n.lyric for n in adopted] == ["long"]
    assert _spans(adopted) == [(0, 120)]


def test_same_start_same_duration_keeps_first_in_order():
    notes = [_note(0, 100, lyric="first"), _note(0, 100, lyric="second")]
    adopted, _diag = events.resolve_overlaps(notes)
    assert [n.lyric for n in adopted] == ["first"]
    assert _spans(adopted) == [(0, 100)]


def test_truncate_to_next_start():
    notes = [_note(0, 480), _note(240, 240)]
    assert _spans(events.resolve_overlaps(notes)[0]) == [(0, 240), (240, 480)]


def test_contained_notes_truncated_in_chain():
    notes = [_note(0, 1000), _note(100, 200), _note(150, 900)]
    assert _spans(events.resolve_overlaps(notes)[0]) == [(0, 100), (100, 150), (150, 1050)]


def test_zero_duration_note_dropped():
    assert events.resolve_overlaps([_note(0, 0)])[0] == []


def test_zero_duration_note_dropped_others_kept():
    notes = [_note(0, 0), _note(100, 100)]
    assert _spans(events.resolve_overlaps(notes)[0]) == [(100, 200)]


def test_build_empty_is_empty():
    result, _diag = events.build_mouth_events([], _TEMPOS, _RES, use_n_morph=True)
    assert result == []


def test_build_single_vowel_note():
    assert _built_shapes_and_spans([_note(0, 480, phonemes=["a"])]) == [(MouthShape.A, 0.0, 15.0)]


def test_build_leading_rest_is_silence():
    assert _built_shapes_and_spans([_note(240, 240, phonemes=["a"])]) == [
        (MouthShape.SILENCE, 0.0, 7.5),
        (MouthShape.A, 7.5, 15.0),
    ]


def test_build_short_vowel_gap_is_legato():
    adopted = [_note(0, 240, phonemes=["a"]), _note(480, 240, phonemes=["i"])]
    assert _built_shapes_and_spans(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.LEGATO_GAP, 7.5, 15.0),
        (MouthShape.I, 15.0, 22.5),
    ]


def test_build_short_gap_into_continuation_is_legato():
    adopted = [_note(0, 240, phonemes=["a"]), _note(480, 240, phonemes=["-"])]
    assert _built_shapes_and_spans(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.LEGATO_GAP, 7.5, 15.0),
        (MouthShape.A, 15.0, 22.5),
    ]


def test_build_long_vowel_gap_is_silence():
    adopted = [_note(0, 240, phonemes=["a"]), _note(720, 240, phonemes=["i"])]
    assert _built_shapes_and_spans(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.SILENCE, 7.5, 22.5),
        (MouthShape.I, 22.5, 30.0),
    ]


def test_build_gap_before_bilabial_onset_is_silence():
    adopted = [_note(0, 240, phonemes=["a"]), _note(480, 240, phonemes=["m", "i"])]
    result = _built_shapes_and_spans(adopted)
    assert result[0] == (MouthShape.A, 0.0, 7.5)
    assert result[1] == (MouthShape.SILENCE, 7.5, 15.0)


def test_build_gap_after_closure_is_silence():
    adopted = [_note(0, 240, phonemes=["w", "Q"]), _note(480, 240, phonemes=["i"])]
    result = _built_shapes_and_spans(adopted)
    assert result[0] == (MouthShape.SILENCE, 0.0, 7.5)
    assert result[1] == (MouthShape.SILENCE, 7.5, 15.0)


def test_build_gap_longer_than_legato_max_frames_is_silence():
    adopted = [_note(0, 240, phonemes=["a"]), _note(480, 240, phonemes=["i"])]
    assert _built_shapes_and_spans(adopted, legato_max_frames=4.0) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.SILENCE, 7.5, 15.0),
        (MouthShape.I, 15.0, 22.5),
    ]


def test_build_gap_equal_to_legato_max_frames_is_legato():
    adopted = [_note(0, 240, phonemes=["a"]), _note(480, 240, phonemes=["i"])]
    assert _built_shapes_and_spans(adopted, legato_max_frames=7.5)[1] == (
        MouthShape.LEGATO_GAP, 7.5, 15.0,
    )


def test_build_adjacent_notes_have_no_silence():
    adopted = [_note(0, 240, phonemes=["a"]), _note(240, 240, phonemes=["i"])]
    assert _built_shapes_and_spans(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.I, 7.5, 15.0),
    ]


def test_build_continuation_holds_previous_vowel():
    adopted = [_note(0, 240, phonemes=["a"]), _note(240, 240, phonemes=["-"])]
    assert _built_shapes_and_spans(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.A, 7.5, 15.0),
    ]


def test_build_continuation_after_moraic_nasal_holds_n_when_on():
    adopted = [_note(0, 240, phonemes=["N\\"]), _note(240, 240, phonemes=["-"])]
    assert _built_shapes_and_spans(adopted, use_n_morph=True) == [
        (MouthShape.N, 0.0, 7.5),
        (MouthShape.N, 7.5, 15.0),
    ]


def test_build_continuation_after_moraic_nasal_stays_closed_when_off():
    adopted = [_note(0, 240, phonemes=["N\\"]), _note(240, 240, phonemes=["-"])]
    assert _built_shapes_and_spans(adopted, use_n_morph=False) == [
        (MouthShape.SILENCE, 0.0, 7.5),
        (MouthShape.SILENCE, 7.5, 15.0),
    ]


def test_build_moraic_nasal_is_silence_when_use_n_morph_omitted():
    adopted = [_note(0, 240, phonemes=["N\\"])]
    result, _diag = events.build_mouth_events(adopted, _TEMPOS, _RES)
    assert [(e.shape, e.start, e.end) for e in result] == [(MouthShape.SILENCE, 0.0, 7.5)]


def test_build_continuation_after_rest_holds_closed_not_pre_rest_vowel():
    adopted = [_note(0, 240, phonemes=["a"]), _note(720, 240, phonemes=["-"])]
    assert _built_shapes_and_spans(adopted) == [
        (MouthShape.A, 0.0, 7.5),
        (MouthShape.SILENCE, 7.5, 22.5),
        (MouthShape.SILENCE, 22.5, 30.0),
    ]


def test_build_first_note_voweless_with_no_previous_is_silence():
    assert _built_shapes_and_spans([_note(0, 480, phonemes=["t"])]) == [(MouthShape.SILENCE, 0.0, 15.0)]


def test_build_geminate_is_silence():
    assert _built_shapes_and_spans([_note(0, 480, phonemes=["w", "Q"])]) == [(MouthShape.SILENCE, 0.0, 15.0)]


def test_build_is_contiguous_and_covers_full_axis():
    adopted = [_note(240, 240, phonemes=["m", "a"]), _note(720, 240, phonemes=["i"])]
    result, _diag = events.build_mouth_events(adopted, _TEMPOS, _RES, use_n_morph=True)
    assert result[0].start == 0.0
    for a, b in zip(result, result[1:], strict=False):
        assert a.end == b.start
    assert result[-1].end == 30.0


def _built_shapes_and_open_amounts(adopted, *, use_n_morph=True, open_by_note=None):
    result, _diag = events.build_mouth_events(
        adopted, _TEMPOS, _RES, use_n_morph=use_n_morph, open_by_note=open_by_note
    )
    return [(e.shape, e.open_amount) for e in result]


def test_build_open_defaults_zero_without_open_by_note():
    assert _built_shapes_and_open_amounts([_note(0, 240, phonemes=["a"])]) == [(MouthShape.A, 0.0)]


def test_build_stamps_note_open_on_vowel():
    assert _built_shapes_and_open_amounts([_note(0, 240, phonemes=["a"])], open_by_note=[0.6]) == [(MouthShape.A, 0.6)]


def test_build_open_shared_across_moras_of_one_note():
    assert _built_shapes_and_open_amounts([_note(0, 480, phonemes=["a", "i"])], open_by_note=[0.5]) == [
        (MouthShape.A, 0.5),
        (MouthShape.I, 0.5),
    ]


def test_build_open_on_moraic_nasal_when_on():
    assert _built_shapes_and_open_amounts([_note(0, 240, phonemes=["N\\"])], use_n_morph=True, open_by_note=[0.3]) == [
        (MouthShape.N, 0.3),
    ]


def test_build_open_zero_on_moraic_nasal_when_off():
    assert _built_shapes_and_open_amounts([_note(0, 240, phonemes=["N\\"])], use_n_morph=False, open_by_note=[0.3]) == [
        (MouthShape.SILENCE, 0.0),
    ]


def test_build_open_zero_on_geminate_silence():
    assert _built_shapes_and_open_amounts([_note(0, 480, phonemes=["w", "Q"])], open_by_note=[0.9]) == [
        (MouthShape.SILENCE, 0.0),
    ]


def test_build_open_zero_on_leading_bilabial():
    assert _built_shapes_and_open_amounts([_note(0, 480, phonemes=["m", "a"])], open_by_note=[0.6]) == [
        (MouthShape.BILABIAL, 0.0),
        (MouthShape.A, 0.6),
    ]


def test_build_open_zero_on_rest_silence():
    assert _built_shapes_and_open_amounts([_note(240, 240, phonemes=["a"])], open_by_note=[0.7]) == [
        (MouthShape.SILENCE, 0.0),
        (MouthShape.A, 0.7),
    ]


def test_build_open_on_continuation_uses_continuation_note_open():
    adopted = [_note(0, 240, phonemes=["a"]), _note(240, 240, phonemes=["-"])]
    assert _built_shapes_and_open_amounts(adopted, open_by_note=[0.4, 0.8]) == [
        (MouthShape.A, 0.4),
        (MouthShape.A, 0.8),
    ]


def test_build_open_on_continuation_holding_moraic_nasal():
    adopted = [_note(0, 240, phonemes=["N\\"]), _note(240, 240, phonemes=["-"])]
    assert _built_shapes_and_open_amounts(adopted, use_n_morph=True, open_by_note=[0.3, 0.8]) == [
        (MouthShape.N, 0.3),
        (MouthShape.N, 0.8),
    ]


def test_build_open_zero_on_continuation_holding_closed():
    adopted = [_note(0, 240, phonemes=["a"]), _note(720, 240, phonemes=["-"])]
    assert _built_shapes_and_open_amounts(adopted, open_by_note=[0.4, 0.8]) == [
        (MouthShape.A, 0.4),
        (MouthShape.SILENCE, 0.0),
        (MouthShape.SILENCE, 0.0),
    ]
