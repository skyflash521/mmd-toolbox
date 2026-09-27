from lipsync import ApertureClass, ConsonantClass, MouthShape
from vpr2vmd import mapping


def _shapes_spans(eventlist):
    return [(e.shape, e.start, e.end) for e in eventlist]


def _classes(eventlist):
    return [(e.shape, e.consonant_class) for e in eventlist]


def _apertures(eventlist):
    return [(e.shape, e.aperture_class) for e in eventlist]


def test_single_vowel_fills_note():
    assert _shapes_spans(mapping.note_mouth_events(["a"], 0.0, 30.0)) == [
        (MouthShape.A, 0.0, 30.0),
    ]


def test_leading_bilabial_takes_nominal_share():
    assert _shapes_spans(mapping.note_mouth_events(["m", "a"], 0.0, 100.0)) == [
        (MouthShape.BILABIAL, 0.0, 3.0),
        (MouthShape.A, 3.0, 100.0),
    ]


def test_leading_bilabial_share_capped_on_short_note():
    assert _shapes_spans(mapping.note_mouth_events(["b", "e"], 0.0, 5.0)) == [
        (MouthShape.BILABIAL, 0.0, 2.5),
        (MouthShape.E, 2.5, 5.0),
    ]


def test_multiple_vowels_equally_divided():
    assert _shapes_spans(mapping.note_mouth_events(["a", "i"], 0.0, 100.0)) == [
        (MouthShape.A, 0.0, 50.0),
        (MouthShape.I, 50.0, 100.0),
    ]


def test_leading_bilabial_then_multiple_vowels_divided_in_remainder():
    assert _shapes_spans(mapping.note_mouth_events(["m", "a", "i"], 0.0, 100.0)) == [
        (MouthShape.BILABIAL, 0.0, 3.0),
        (MouthShape.A, 3.0, 51.5),
        (MouthShape.I, 51.5, 100.0),
    ]


def test_last_vowel_ends_exactly_at_note_end():
    events = mapping.note_mouth_events(["a", "i", "M", "e", "o", "a", "i"], 0.1, 1.0)
    assert len(events) == 7
    assert events[-1].end == 1.0


def test_respects_nonzero_start_frame():
    assert _shapes_spans(mapping.note_mouth_events(["m", "a", "i"], 10.0, 110.0)) == [
        (MouthShape.BILABIAL, 10.0, 13.0),
        (MouthShape.A, 13.0, 61.5),
        (MouthShape.I, 61.5, 110.0),
    ]


def test_non_leading_bilabial_makes_no_event():
    assert _shapes_spans(mapping.note_mouth_events(["k", "m", "a"], 0.0, 30.0)) == [
        (MouthShape.A, 0.0, 30.0),
    ]
    assert _shapes_spans(mapping.note_mouth_events(["a", "m"], 0.0, 30.0)) == [
        (MouthShape.A, 0.0, 30.0),
    ]


def test_no_vowel_returns_none_for_hold():
    assert mapping.note_mouth_events(["t"], 0.0, 30.0) is None
    assert mapping.note_mouth_events(["-"], 0.0, 30.0) is None
    assert mapping.note_mouth_events(["m"], 0.0, 100.0) is None
    assert mapping.note_mouth_events([], 0.0, 30.0) is None


def test_moraic_nasal_is_silence_by_default():
    assert _shapes_spans(mapping.note_mouth_events(["N\\"], 0.0, 30.0)) == [
        (MouthShape.SILENCE, 0.0, 30.0),
    ]


def test_moraic_nasal_fills_note_with_n_when_on():
    assert _shapes_spans(mapping.note_mouth_events(["N\\"], 0.0, 30.0, use_n_morph=True)) == [
        (MouthShape.N, 0.0, 30.0),
    ]


def test_moraic_nasal_is_silence_when_off():
    assert _shapes_spans(mapping.note_mouth_events(["N\\"], 0.0, 30.0, use_n_morph=False)) == [
        (MouthShape.SILENCE, 0.0, 30.0),
    ]


def test_moraic_nasal_mixed_with_consonant_is_not_standalone():
    assert mapping.note_mouth_events(["t", "N\\"], 0.0, 30.0) is None


def test_geminate_stop_fills_note_with_silence():
    assert _shapes_spans(mapping.note_mouth_events(["Q"], 0.0, 30.0)) == [
        (MouthShape.SILENCE, 0.0, 30.0),
    ]
    assert _shapes_spans(mapping.note_mouth_events(["w", "Q"], 0.0, 30.0)) == [
        (MouthShape.SILENCE, 0.0, 30.0),
    ]


def test_other_consonant_before_vowel_has_no_own_event():
    assert _shapes_spans(mapping.note_mouth_events(["k", "o"], 0.0, 30.0)) == [
        (MouthShape.O, 0.0, 30.0),
    ]


def test_bilabial_fricative_is_not_bilabial_event():
    assert _shapes_spans(mapping.note_mouth_events(["p\\", "M"], 0.0, 30.0)) == [
        (MouthShape.U, 0.0, 30.0),
    ]


def test_onset_consonant_class_attached_to_leading_vowel():
    assert _classes(mapping.note_mouth_events(["S", "a"], 0.0, 30.0)) == [
        (MouthShape.A, ConsonantClass.SPREAD),
    ]
    assert _classes(mapping.note_mouth_events(["p\\", "a"], 0.0, 30.0)) == [
        (MouthShape.A, ConsonantClass.ROUNDED),
    ]
    assert _classes(mapping.note_mouth_events(["k", "o"], 0.0, 30.0)) == [
        (MouthShape.O, ConsonantClass.NEUTRAL),
    ]
    assert _classes(mapping.note_mouth_events(["a"], 0.0, 30.0)) == [
        (MouthShape.A, ConsonantClass.NONE),
    ]


def test_bilabial_onset_excluded_palatalized_keeps_spread():
    assert _classes(mapping.note_mouth_events(["b", "a"], 0.0, 30.0)) == [
        (MouthShape.BILABIAL, ConsonantClass.NONE),
        (MouthShape.A, ConsonantClass.NONE),
    ]
    assert _classes(mapping.note_mouth_events(["m", "j", "a"], 0.0, 30.0)) == [
        (MouthShape.BILABIAL, ConsonantClass.NONE),
        (MouthShape.A, ConsonantClass.SPREAD),
    ]


def test_onset_class_only_on_first_mora_vowel():
    assert _classes(mapping.note_mouth_events(["k", "a", "i"], 0.0, 30.0)) == [
        (MouthShape.A, ConsonantClass.NEUTRAL),
        (MouthShape.I, ConsonantClass.NONE),
    ]


def test_onset_class_priority_when_multiple_consonants():
    assert mapping.note_mouth_events(["k", "j", "a"], 0.0, 30.0)[0].consonant_class is (
        ConsonantClass.SPREAD
    )
    assert mapping.note_mouth_events(["k", "w", "a"], 0.0, 30.0)[0].consonant_class is (
        ConsonantClass.ROUNDED
    )
    assert mapping.note_mouth_events(["w", "j", "a"], 0.0, 30.0)[0].consonant_class is (
        ConsonantClass.ROUNDED
    )


def test_unknown_onset_consonant_is_neutral():
    assert _classes(mapping.note_mouth_events(["zzz", "a"], 0.0, 30.0)) == [
        (MouthShape.A, ConsonantClass.NEUTRAL),
    ]


def test_open_amount_placeholder_zero_for_all_event_kinds():
    for phonemes in (["a"], ["m", "a"], ["N\\"], ["Q"]):
        events = mapping.note_mouth_events(phonemes, 0.0, 30.0)
        assert events
        assert all(e.open_amount == 0.0 for e in events)


def test_onset_aperture_class_attached_to_leading_vowel():
    assert _apertures(mapping.note_mouth_events(["t", "a"], 0.0, 30.0)) == [
        (MouthShape.A, ApertureClass.FIRM_CLOSURE),
    ]
    assert _apertures(mapping.note_mouth_events(["s", "a"], 0.0, 30.0)) == [
        (MouthShape.A, ApertureClass.NARROW_CHANNEL),
    ]
    assert _apertures(mapping.note_mouth_events(["k", "o"], 0.0, 30.0)) == [
        (MouthShape.O, ApertureClass.SLIGHT_CLOSURE),
    ]
    assert _apertures(mapping.note_mouth_events(["a"], 0.0, 30.0)) == [
        (MouthShape.A, ApertureClass.NONE),
    ]


def test_onset_aperture_only_on_first_mora_vowel():
    assert _apertures(mapping.note_mouth_events(["k", "a", "i"], 0.0, 30.0)) == [
        (MouthShape.A, ApertureClass.SLIGHT_CLOSURE),
        (MouthShape.I, ApertureClass.NONE),
    ]


def test_onset_aperture_priority_when_multiple_consonants():
    assert mapping.note_mouth_events(["k", "s", "a"], 0.0, 30.0)[0].aperture_class is (
        ApertureClass.NARROW_CHANNEL
    )
    assert mapping.note_mouth_events(["s", "t", "a"], 0.0, 30.0)[0].aperture_class is (
        ApertureClass.FIRM_CLOSURE
    )


def test_onset_aperture_and_consonant_class_are_independent():
    event = mapping.note_mouth_events(["w", "t", "a"], 0.0, 30.0)[0]
    assert event.consonant_class is ConsonantClass.ROUNDED
    assert event.aperture_class is ApertureClass.FIRM_CLOSURE


def test_onset_aperture_class_depends_on_bilabial_position():
    assert mapping.note_mouth_events(["k", "m", "a"], 0.0, 30.0)[0].aperture_class is (
        ApertureClass.NONE
    )
    assert mapping.note_mouth_events(["m", "k", "a"], 0.0, 30.0)[1].aperture_class is (
        ApertureClass.SLIGHT_CLOSURE
    )
