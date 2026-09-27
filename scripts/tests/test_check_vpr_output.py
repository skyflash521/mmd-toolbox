import check_vpr_output as check
import pytest

from vpr import Note, Part, TempoEvent, TimeSignature, Track, VprProject

RESOLUTION = 480


def _note(start: int, dur: int, lyric: str = "あ") -> Note:
    return Note(start_tick=start, duration_tick=dur, pitch=60, lyric=lyric, velocity=64)


def _project(notes, *, tempos=None, signatures=None) -> VprProject:
    return VprProject(
        resolution=RESOLUTION,
        tempos=tempos or [TempoEvent(tick=0, bpm=120.0)],
        time_signatures=signatures
        or [TimeSignature(tick=0, numerator=4, denominator=4)],
        tracks=[Track(name="song", parts=[Part(name="part", start_tick=0, notes=notes)])],
    )


def _reference(lyric_durs, cont_durs=(), *, gap=960) -> VprProject:
    notes = []
    pos = 0
    for dur in lyric_durs:
        notes.append(_note(pos, dur))
        pos += dur + gap
    for dur in cont_durs:
        notes.append(_note(pos, dur, lyric="-"))
        pos += dur + gap
    return _project(notes)


def test_seconds_at_follows_piecewise_tempo():
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=960, bpm=60.0)]
    assert check.seconds_at(480, tempos, RESOLUTION) == pytest.approx(0.5)
    assert check.seconds_at(960, tempos, RESOLUTION) == pytest.approx(1.0)
    assert check.seconds_at(1440, tempos, RESOLUTION) == pytest.approx(2.0)


def test_seconds_at_single_tempo():
    tempos = [TempoEvent(tick=0, bpm=120.0)]
    assert check.seconds_at(0, tempos, RESOLUTION) == pytest.approx(0.0)
    assert check.seconds_at(2400, tempos, RESOLUTION) == pytest.approx(2.5)


def test_measure_number_is_one_based():
    signatures = [TimeSignature(tick=0, numerator=4, denominator=4)]
    assert check.measure_number(0, signatures, RESOLUTION) == 1
    assert check.measure_number(1919, signatures, RESOLUTION) == 1
    assert check.measure_number(1920, signatures, RESOLUTION) == 2


def test_measure_number_follows_time_signature_change():
    signatures = [
        TimeSignature(tick=0, numerator=4, denominator=4),
        TimeSignature(tick=3840, numerator=3, denominator=4),
    ]
    assert check.measure_number(3839, signatures, RESOLUTION) == 2
    assert check.measure_number(3840, signatures, RESOLUTION) == 3
    assert check.measure_number(3840 + 1440, signatures, RESOLUTION) == 4


def test_thresholds_take_min_duration_and_max_ratio_across_references():
    ref1 = _reference([96, 192])
    ref2 = _reference([192, 384], cont_durs=[192])
    thresholds = check.compute_thresholds([ref1, ref2])
    assert thresholds.min_lyric_note_sec == pytest.approx(0.1)
    assert thresholds.max_continuation_ratio == pytest.approx(1.5 * (1 / 3))


def test_thresholds_mora_uses_min_across_references():
    ref1 = _reference([240])
    ref2 = _reference([96])
    thresholds = check.compute_thresholds([ref1, ref2])
    assert thresholds.min_mora_sec == pytest.approx(0.1)


def test_thresholds_come_from_given_reference():
    thresholds = check.compute_thresholds([_reference([240, 480])])
    assert thresholds.min_lyric_note_sec == pytest.approx(0.25)
    assert thresholds.min_mora_sec == pytest.approx(0.25)
    assert thresholds.max_continuation_ratio == pytest.approx(0.0)


def test_thresholds_mora_comes_from_runs_not_note_minimum():
    run = [_note(0, 96), _note(96, 96, lyric="-"), _note(192, 96)]
    isolated = [_note(5000, 240)]
    ref = _project(run + isolated)
    thresholds = check.compute_thresholds([ref])
    assert thresholds.min_mora_sec == pytest.approx(0.15)
    assert thresholds.min_lyric_note_sec == pytest.approx(0.1)


def test_thresholds_convert_ticks_after_tempo_change_piecewise():
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=960, bpm=60.0)]
    ref = _project([_note(960, 96)], tempos=tempos)
    thresholds = check.compute_thresholds([ref])
    assert thresholds.min_lyric_note_sec == pytest.approx(0.2)


def test_overlapping_notes_are_reported_with_measure():
    project = _project([_note(0, 960), _note(480, 480)])
    thresholds = check.compute_thresholds([_reference([96])])
    violations = [v for v in check.check_project(project, thresholds) if v.kind == "overlap"]
    assert len(violations) == 1
    assert violations[0].measure == 1


def test_adjacent_notes_do_not_overlap():
    project = _project([_note(0, 480), _note(480, 480)])
    thresholds = check.compute_thresholds([_reference([96])])
    assert [v for v in check.check_project(project, thresholds) if v.kind == "overlap"] == []


def test_zero_duration_note_is_reported():
    project = _project([_note(1920, 0)])
    thresholds = check.compute_thresholds([_reference([96])])
    violations = [
        v for v in check.check_project(project, thresholds) if v.kind == "zero_duration"
    ]
    assert len(violations) == 1
    assert violations[0].measure == 2


def test_zero_duration_note_inside_another_is_not_an_overlap():
    thresholds = check.compute_thresholds([_reference([96])])
    project = _project([_note(0, 960), _note(480, 0)])
    violations = check.check_project(project, thresholds)
    assert [v for v in violations if v.kind == "overlap"] == []
    assert len([v for v in violations if v.kind == "zero_duration"]) == 1


def test_short_lyric_note_is_reported_against_reference_minimum():
    thresholds = check.compute_thresholds([_reference([96, 192])])
    project = _project([_note(0, 48), _note(1920, 96)])
    violations = [
        v for v in check.check_project(project, thresholds) if v.kind == "short_lyric_note"
    ]
    assert len(violations) == 1
    assert violations[0].measure == 1


def test_thresholds_convert_note_crossing_tempo_change_piecewise():
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=960, bpm=60.0)]
    ref = _project([_note(480, 960)], tempos=tempos)
    thresholds = check.compute_thresholds([ref])
    assert thresholds.min_lyric_note_sec == pytest.approx(1.5)
    assert thresholds.min_mora_sec == pytest.approx(1.5)


def test_check_mora_run_crossing_tempo_change_uses_piecewise_seconds():
    thresholds = check.compute_thresholds([_reference([96])])
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=960, bpm=60.0)]
    run = [_note(864, 96), _note(960, 48)]
    project = _project(run, tempos=tempos)
    assert [
        v for v in check.check_project(project, thresholds) if v.kind == "short_mora_run"
    ] == []


def test_gap_splits_runs_in_checked_project():
    thresholds = check.compute_thresholds([_reference([96])])
    notes = [_note(0, 48), _note(2000, 480)]
    violations = [
        v for v in check.check_project(_project(notes), thresholds) if v.kind == "short_mora_run"
    ]
    assert len(violations) == 1
    assert violations[0].measure == 1


def test_check_converts_ticks_after_tempo_change_piecewise():
    thresholds = check.compute_thresholds([_reference([96])])
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=960, bpm=60.0)]
    project = _project([_note(960, 48)], tempos=tempos)
    assert [
        v for v in check.check_project(project, thresholds) if v.kind == "short_lyric_note"
    ] == []


def test_lyric_note_equal_to_reference_minimum_passes():
    thresholds = check.compute_thresholds([_reference([96])])
    project = _project([_note(0, 96)])
    assert [
        v for v in check.check_project(project, thresholds) if v.kind == "short_lyric_note"
    ] == []


def test_short_continuation_note_is_not_a_lyric_violation():
    thresholds = check.compute_thresholds([_reference([96])])
    project = _project([_note(0, 480), _note(480, 48, lyric="-")])
    assert [
        v for v in check.check_project(project, thresholds) if v.kind == "short_lyric_note"
    ] == []


def test_short_mora_run_is_reported():
    thresholds = check.compute_thresholds([_reference([96])])
    run = [_note(0, 48), _note(48, 48, lyric="-"), _note(96, 48)]
    violations = [
        v for v in check.check_project(_project(run), thresholds) if v.kind == "short_mora_run"
    ]
    assert len(violations) == 1
    assert violations[0].measure == 1


def test_mora_run_with_enough_length_passes():
    thresholds = check.compute_thresholds([_reference([96])])
    run = [_note(0, 192), _note(192, 96, lyric="-"), _note(288, 96)]
    assert [
        v for v in check.check_project(_project(run), thresholds) if v.kind == "short_mora_run"
    ] == []


def test_mora_run_equal_to_reference_minimum_passes():
    thresholds = check.compute_thresholds([_reference([96])])
    run = [_note(0, 96), _note(96, 96)]
    assert [
        v for v in check.check_project(_project(run), thresholds) if v.kind == "short_mora_run"
    ] == []


def test_violation_measure_follows_time_signature_denominator():
    signatures = [
        TimeSignature(tick=0, numerator=4, denominator=4),
        TimeSignature(tick=1920, numerator=3, denominator=8),
    ]
    thresholds = check.compute_thresholds([_reference([96])])
    project = _project([_note(3360, 0)], signatures=signatures)
    violations = [
        v for v in check.check_project(project, thresholds) if v.kind == "zero_duration"
    ]
    assert len(violations) == 1
    assert violations[0].measure == 4


def test_continuation_ratio_over_limit_is_reported():
    ref = _reference([96, 96], cont_durs=[96])
    thresholds = check.compute_thresholds([ref])
    notes = []
    pos = 0
    for lyric in ["あ", "-", "-", "い", "-"]:
        notes.append(_note(pos, 96, lyric=lyric))
        pos += 96 + 960
    violations = [
        v
        for v in check.check_project(_project(notes), thresholds)
        if v.kind == "continuation_ratio"
    ]
    assert len(violations) == 1
    assert violations[0].measure is None


def test_continuation_ratio_within_limit_passes():
    ref = _reference([96, 96], cont_durs=[96])
    thresholds = check.compute_thresholds([ref])
    notes = [_note(0, 96), _note(2000, 96, lyric="-"), _note(4000, 96)]
    assert [
        v
        for v in check.check_project(_project(notes), thresholds)
        if v.kind == "continuation_ratio"
    ] == []


def test_continuation_ratio_equal_to_limit_passes():
    ref = _reference([96, 96], cont_durs=[96])
    thresholds = check.compute_thresholds([ref])
    notes = [_note(0, 96), _note(2000, 96, lyric="-")]
    assert [
        v
        for v in check.check_project(_project(notes), thresholds)
        if v.kind == "continuation_ratio"
    ] == []


def test_clean_project_has_no_violations():
    thresholds = check.compute_thresholds([_reference([96, 192], cont_durs=[192])])
    notes = [_note(0, 192), _note(192, 192, lyric="-"), _note(384, 192), _note(2000, 240)]
    assert check.check_project(_project(notes), thresholds) == []
