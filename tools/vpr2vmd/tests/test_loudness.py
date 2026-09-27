import pytest

from vpr import ControllerCurve, ControllerEvent, Note, Part
from vpr2vmd import loudness


def _part(controllers):
    return Part(name="p", start_tick=0, controllers=controllers)


def _curve(name, points):
    return ControllerCurve(name=name, events=[ControllerEvent(t, v) for t, v in points])


def _note(start, dur):
    return Note(
        start_tick=start, duration_tick=dur, pitch=60, lyric="x", velocity=64, phonemes=["a"]
    )


def test_choose_dynamics_ignoring_non_loudness_controllers():
    parts = [_part([_curve("s5Expression", [(0, 30)]), _curve("dynamics", [(0, 64)])])]
    assert loudness.choose_loudness_controller(parts) == "dynamics"


def test_choose_none_when_only_s5expression():
    parts = [_part([_curve("s5Expression", [(0, 30)])])]
    assert loudness.choose_loudness_controller(parts) is None


def test_choose_none_when_no_loudness_controller():
    parts = [_part([_curve("brightness", [(0, 64)]), _curve("character", [(0, 5)])])]
    assert loudness.choose_loudness_controller(parts) is None


def test_choose_ignores_empty_loudness_curve():
    parts = [_part([_curve("dynamics", [])])]
    assert loudness.choose_loudness_controller(parts) is None


def test_merged_events_keep_input_order_for_same_tick():
    parts = [_part([_curve("dynamics", [(100, 80), (100, 50)])])]
    assert loudness._merged_events(parts, "dynamics") == [(100, 80), (100, 50)]


def test_step_average_constant():
    curve = loudness._StepCurve([(0, 64), (1000, 64)], value_before_first=64)
    assert curve.average(0, 480) == pytest.approx(64.0)


def test_step_average_is_time_weighted_across_a_step():
    curve = loudness._StepCurve([(0, 0), (240, 120)], value_before_first=0)
    assert curve.average(0, 480) == pytest.approx(60.0)


def test_step_average_before_first_point_uses_value_before_first():
    curve = loudness._StepCurve([(1000, 100)], value_before_first=50)
    assert curve.average(0, 480) == pytest.approx(50.0)


def test_step_average_spanning_first_point_mixes_value_before_first_and_curve():
    curve = loudness._StepCurve([(480, 100)], value_before_first=20)
    assert curve.average(0, 960) == pytest.approx(60.0)


def test_step_average_zero_length_interval_is_start_value():
    curve = loudness._StepCurve([(0, 30), (480, 90)], value_before_first=30)
    assert curve.average(480, 480) == pytest.approx(90.0)


def test_step_curve_value_at_takes_last_point_on_duplicate_tick():
    curve = loudness._StepCurve([(100, 80), (100, 50)], value_before_first=80)
    assert curve.value_at(100) == 50
    assert curve.value_at(99) == 80


def test_open_amounts_loud_higher_than_quiet():
    parts = [_part([_curve("dynamics", [(0, 120), (480, 10)])])]
    notes = [_note(0, 480), _note(480, 480)]
    out = loudness.open_amounts_from_loudness(parts, notes, lo=0.3, hi=0.75, open_max=0.9, gamma=0.6)
    assert out is not None
    assert out[0] > out[1]
    assert out[0] == pytest.approx(min(0.3 + 0.45 * (120 / 127) ** 0.6, 0.75, 0.9))
    assert out[1] == pytest.approx(min(0.3 + 0.45 * (10 / 127) ** 0.6, 0.75, 0.9))


def test_open_amounts_none_when_no_loudness_controller():
    parts = [_part([_curve("brightness", [(0, 64)])])]
    out = loudness.open_amounts_from_loudness(
        parts, [_note(0, 480)], lo=0.3, hi=0.75, open_max=0.9, gamma=0.6
    )
    assert out is None


def test_open_amounts_dynamics_max_value_maps_to_high():
    parts = [_part([_curve("dynamics", [(0, 127)])])]
    out = loudness.open_amounts_from_loudness(
        parts, [_note(0, 480)], lo=0.3, hi=0.75, open_max=0.9, gamma=0.6
    )
    assert out[0] == pytest.approx(0.75)


def test_open_amounts_clamped_by_open_max():
    parts = [_part([_curve("dynamics", [(0, 127)])])]
    out = loudness.open_amounts_from_loudness(
        parts, [_note(0, 480)], lo=0.3, hi=0.9, open_max=0.5, gamma=0.6
    )
    assert out[0] == pytest.approx(0.5)


def test_open_amounts_note_before_curve_start_uses_first_point_value():
    parts = [_part([_curve("dynamics", [(480, 127)])])]
    out = loudness.open_amounts_from_loudness(
        parts, [_note(0, 480)], lo=0.3, hi=0.75, open_max=0.9, gamma=0.6
    )
    assert out[0] == pytest.approx(0.75)


def test_open_amounts_merge_dynamics_from_all_parts():
    parts = [
        _part([_curve("dynamics", [(0, 120)])]),
        _part([_curve("dynamics", [(480, 10)])]),
    ]
    notes = [_note(0, 480), _note(480, 480)]
    out = loudness.open_amounts_from_loudness(parts, notes, lo=0.3, hi=0.75, open_max=0.9, gamma=0.6)
    assert out[1] == pytest.approx(0.3 + 0.45 * (10 / 127) ** 0.6)
    assert out[0] > out[1]
