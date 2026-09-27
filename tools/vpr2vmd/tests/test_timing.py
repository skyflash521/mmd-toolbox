import pytest

from vpr import Note, TempoEvent
from vpr2vmd import timing


def _note(start, dur):
    return Note(
        start_tick=start, duration_tick=dur, pitch=60, lyric="x",
        velocity=64, phonemes=["a"],
    )


def test_tick_zero_is_frame_zero():
    assert timing.tick_to_frame(0, [TempoEvent(0, 120.0)], 480) == 0.0


def test_single_tempo_one_beat():
    assert timing.tick_to_frame(480, [TempoEvent(0, 120.0)], 480) == pytest.approx(15.0)


def test_single_tempo_fractional_frame_not_quantized():
    assert timing.tick_to_frame(240, [TempoEvent(0, 120.0)], 480) == pytest.approx(7.5)


def test_tempo_change_integrates_piecewise():
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    assert timing.tick_to_frame(960, tempos, 480) == pytest.approx(22.5)


def test_region_boundary_cumulative_is_preceding_integral():
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    assert timing.tick_to_frame(480, tempos, 480) == pytest.approx(15.0)


def test_just_after_boundary_uses_new_region_rate():
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    assert timing.tick_to_frame(481, tempos, 480) == pytest.approx(15.015625)


def test_first_tempo_not_at_zero_applies_retroactively():
    tempos = [TempoEvent(240, 120.0)]
    assert timing.tick_to_frame(480, tempos, 480) == pytest.approx(15.0)


def test_seconds_helper_matches_frame_over_30():
    tempos = [TempoEvent(0, 120.0)]
    assert timing.tick_to_seconds(480, tempos, 480) == pytest.approx(0.5)
    assert timing.tick_to_frame(480, tempos, 480) == pytest.approx(
        timing.tick_to_seconds(480, tempos, 480) * 30.0
    )


def test_seconds_integrates_tempo_change_directly():
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    assert timing.tick_to_seconds(960, tempos, 480) == pytest.approx(0.75)


def test_seconds_three_regions_accumulate():
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 60.0), TempoEvent(960, 240.0)]
    assert timing.tick_to_seconds(1440, tempos, 480) == pytest.approx(1.75)


def test_seconds_first_tempo_not_at_zero_applies_retroactively():
    assert timing.tick_to_seconds(480, [TempoEvent(240, 120.0)], 480) == pytest.approx(0.5)


def test_three_regions_accumulate():
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 60.0), TempoEvent(960, 240.0)]
    assert timing.tick_to_frame(1440, tempos, 480) == pytest.approx(52.5)


def test_note_effective_bpm_single_tempo_equals_bpm():
    assert timing.note_effective_bpm(_note(0, 480), [TempoEvent(0, 150.0)], 480) == pytest.approx(150.0)
    assert timing.note_effective_bpm(_note(960, 120), [TempoEvent(0, 150.0)], 480) == pytest.approx(150.0)


def test_note_effective_bpm_spans_tempo_change_is_time_weighted():
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    assert timing.note_effective_bpm(_note(0, 960), tempos, 480) == pytest.approx(160.0)


def test_note_effective_bpm_zero_duration_is_none():
    assert timing.note_effective_bpm(_note(100, 0), [TempoEvent(0, 120.0)], 480) is None


def test_representative_bpm_uniform_returns_that_bpm():
    tempos = [TempoEvent(0, 190.0)]
    notes = [_note(0, 240), _note(240, 240), _note(480, 240)]
    assert timing.representative_bpm(notes, tempos, 480) == pytest.approx(190.0)


def test_representative_bpm_long_fast_note_dominates():
    tempos = [TempoEvent(0, 100.0), TempoEvent(480, 200.0)]
    slow = _note(0, 480)
    fast = _note(480, 480 * 4)
    assert timing.representative_bpm([slow, fast], tempos, 480) == pytest.approx(200.0)


def test_representative_bpm_slow_note_dominates_when_longer_in_seconds():
    tempos = [TempoEvent(0, 100.0), TempoEvent(480, 200.0)]
    assert timing.representative_bpm([_note(0, 480), _note(480, 480)], tempos, 480) == pytest.approx(100.0)


def test_representative_bpm_equal_weights_pick_lower_bpm():
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    notes = [_note(0, 480), _note(480, 960)]
    assert timing.representative_bpm(notes, tempos, 480) == pytest.approx(120.0)


def test_representative_bpm_empty_returns_default():
    assert timing.representative_bpm([], [TempoEvent(0, 120.0)], 480) == pytest.approx(120.0)


def test_representative_bpm_all_zero_duration_returns_default():
    notes = [_note(0, 0), _note(100, 0)]
    assert timing.representative_bpm(notes, [TempoEvent(0, 120.0)], 480, default_bpm=120.0) == pytest.approx(120.0)
