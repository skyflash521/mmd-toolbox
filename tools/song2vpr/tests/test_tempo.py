import numpy as np
import pytest

from song2vpr import tempo
from vocal_analysis import AudioPcm

SR = 22050


def _click_track(bpm, seconds=8.0, offset_sec=0.0, accent_every=None, sample_rate=SR):
    n = int(seconds * sample_rate)
    samples = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    rng = np.random.default_rng(0)
    samples += rng.normal(0.0, 0.001, n).astype(np.float32)
    index = 0
    while True:
        at = offset_sec + index * period
        start = int(at * sample_rate)
        if start >= n:
            break
        length = int(0.02 * sample_rate)
        strong = accent_every is not None and index % accent_every == 0
        envelope = np.exp(-np.arange(length) / (0.004 * sample_rate))
        click = envelope * (1.0 if strong else 0.4)
        samples[start:start + length] += click[:max(0, min(length, n - start))].astype(np.float32)
        index += 1
    return AudioPcm(samples=samples[:, None], sample_rate=sample_rate)


def _silence(seconds=8.0, sample_rate=SR):
    n = int(seconds * sample_rate)
    return AudioPcm(samples=np.zeros((n, 1), dtype=np.float32), sample_rate=sample_rate)


def _subdivided_click_track(bpm, subdivisions, sub_amp, seconds=16.0, sample_rate=SR):
    n = int(seconds * sample_rate)
    samples = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    samples += rng.normal(0.0, 0.001, n).astype(np.float32)
    period = 60.0 / bpm
    length = int(0.02 * sample_rate)
    envelope = np.exp(-np.arange(length) / (0.004 * sample_rate))

    def add_click(at, amp):
        start = int(at * sample_rate)
        if start >= n:
            return False
        segment = (envelope * amp)[:max(0, min(length, n - start))]
        samples[start:start + len(segment)] += segment.astype(np.float32)
        return True

    index = 0
    while add_click(index * period, 1.0):
        index += 1
    index = 0
    while index * period < seconds:
        for k in range(1, subdivisions):
            add_click(index * period + k * period / subdivisions, sub_amp)
        index += 1
    return AudioPcm(samples=samples[:, None], sample_rate=sample_rate)


@pytest.mark.parametrize("bpm", [90.0, 120.0, 150.0])
def test_bpm_is_estimated_from_the_beats(bpm):
    result = tempo.estimate(_click_track(bpm))
    assert result.bpm == pytest.approx(bpm, rel=0.03)


def test_estimated_bpm_is_not_an_octave_off():
    result = tempo.estimate(_click_track(120.0))
    assert 60.0 < result.bpm < 240.0


def test_a_fast_beat_far_above_the_weight_centre_is_not_taken_at_half_speed():
    result = tempo.estimate(_click_track(190.0, seconds=16.0))
    assert result.bpm == pytest.approx(190.0, rel=0.03)


def test_a_slow_beat_far_below_the_weight_centre_is_not_taken_at_half_speed():
    result = tempo.estimate(_click_track(90.0, seconds=16.0))
    assert result.bpm == pytest.approx(90.0, rel=0.03)


def test_given_tempo_is_used_as_is():
    result = tempo.estimate(_click_track(90.0), tempo_bpm=140.0)
    assert result.bpm == 140.0
    assert not result.tempo_defaulted


def test_a_fast_plain_beat_folded_to_half_is_promoted_back_to_a_storable_bpm():
    result = tempo.estimate(_click_track(200.0, seconds=16.0))
    assert result.bpm == pytest.approx(200.0, rel=0.03)
    assert result.bpm * 100 == round(result.bpm * 100)
    assert result.tempo_source == "estimated"


def test_eighth_subdivisions_do_not_promote_the_correct_tempo():
    result = tempo.estimate(_subdivided_click_track(120.0, 2, 0.7))
    assert result.bpm == pytest.approx(120.0, rel=0.03)


def test_sixteenth_subdivisions_do_not_promote_the_correct_tempo():
    result = tempo.estimate(_subdivided_click_track(100.0, 4, 0.4))
    assert result.bpm == pytest.approx(100.0, rel=0.03)


def test_promotion_above_the_target_cap_is_skipped_keeping_the_half_tempo():
    result = tempo.estimate(_click_track(270.0, seconds=16.0))
    assert result.bpm == pytest.approx(135.0, rel=0.03)


def test_given_tempo_is_never_promoted():
    result = tempo.estimate(_click_track(200.0, seconds=16.0), tempo_bpm=100.0)
    assert result.bpm == 100.0


def test_numerator_is_four_unless_given_even_with_accents_every_three_beats():
    result = tempo.estimate(_click_track(120.0, seconds=16.0, accent_every=3))
    assert result.numerator == 4


def test_denominator_is_a_quarter_note_when_not_given():
    result = tempo.estimate(_click_track(120.0))
    assert result.denominator == 4


def test_given_time_signature_is_used_as_is():
    result = tempo.estimate(_click_track(120.0), time_signature=(6, 8))
    assert (result.numerator, result.denominator) == (6, 8)


def test_silence_falls_back_to_the_default_tempo_and_reports_it():
    result = tempo.estimate(_silence())
    assert result.bpm == 120.0
    assert (result.numerator, result.denominator) == (4, 4)
    assert result.tempo_defaulted


def test_default_keeps_the_given_time_signature():
    result = tempo.estimate(_silence(), time_signature=(3, 4))
    assert result.bpm == 120.0
    assert (result.numerator, result.denominator) == (3, 4)
    assert result.tempo_defaulted


def test_given_tempo_is_not_defaulted_on_silence():
    result = tempo.estimate(_silence(), tempo_bpm=100.0)
    assert result.bpm == 100.0
    assert not result.tempo_defaulted


def test_the_time_signature_is_the_default_even_without_a_usable_period():
    result = tempo.estimate(_silence(), tempo_bpm=100.0)
    assert not result.tempo_defaulted
    assert (result.numerator, result.denominator) == (4, 4)


def test_tick_conversion_starts_at_the_input_origin():
    result = tempo.estimate(_click_track(120.0))
    assert result.to_tick(0.0) == 0


def test_a_quarter_note_at_120_bpm_is_one_resolution_of_ticks():
    result = tempo.estimate(_click_track(120.0), tempo_bpm=120.0)
    assert result.to_tick(0.5) == result.resolution


def test_tick_conversion_is_monotonic():
    result = tempo.estimate(_click_track(120.0), tempo_bpm=120.0)
    ticks = [result.to_tick(t) for t in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)]
    assert ticks == sorted(ticks)
    assert len(set(ticks)) == len(ticks)


def test_adopted_tempo_is_rounded_to_hundredths_of_a_bpm():
    result = tempo.estimate(_click_track(120.0), tempo_bpm=120.005)
    assert result.bpm * 100 == round(result.bpm * 100)
    assert result.bpm != 120.005


def test_resolution_matches_the_format_layer():
    from vpr.constants import RESOLUTION

    assert tempo.RESOLUTION == RESOLUTION


def test_same_input_gives_the_same_estimate():
    pcm = _click_track(132.0)
    first = tempo.estimate(pcm)
    second = tempo.estimate(pcm)
    assert (first.bpm, first.numerator, first.denominator) == \
           (second.bpm, second.numerator, second.denominator)


def test_given_values_are_reported_as_options():
    result = tempo.estimate(_click_track(120.0), tempo_bpm=96.0, time_signature=(3, 4))
    assert (result.tempo_source, result.time_signature_source) == ("option", "option")


def test_the_unspecified_time_signature_is_reported_as_the_default():
    result = tempo.estimate(_click_track(120.0, seconds=16.0, accent_every=3))
    assert (result.tempo_source, result.time_signature_source) == ("estimated", "default")


def test_values_not_taken_from_the_audio_are_reported_as_defaults():
    result = tempo.estimate(_silence())
    assert (result.tempo_source, result.time_signature_source) == ("default", "default")


def test_tempo_source_default_agrees_with_tempo_defaulted():
    for result in (tempo.estimate(_silence()),
                   tempo.estimate(_silence(), tempo_bpm=100.0),
                   tempo.estimate(_click_track(120.0))):
        assert result.tempo_defaulted == (result.tempo_source == "default")


def test_estimate_does_not_expose_a_beat_phase():
    result = tempo.estimate(_click_track(120.0))
    assert not hasattr(result, "beat_offset_sec")
