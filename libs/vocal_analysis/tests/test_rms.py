import math

import numpy as np
import pytest


def test_frame_rms_center_times_and_count():
    from vocal_analysis.rms import _frame_rms

    sample_rate = 1000
    frame_samples = 25
    samples = np.full((100, 1), 0.5, dtype=np.float32)

    times_sec, raw_rms = _frame_rms(samples, sample_rate)

    assert len(times_sec) == 8
    assert len(raw_rms) == 8
    expected_starts = np.arange(0, 71, 10)
    expected_centers_sec = (expected_starts + frame_samples / 2) / sample_rate
    assert np.allclose(times_sec, expected_centers_sec)


def test_frame_rms_rounds_frame_and_hop_samples():
    from vocal_analysis.rms import _frame_rms

    sample_rate_where_round_differs_from_ceil_for_frame_and_floor_for_hop = 1370
    sample_rate = sample_rate_where_round_differs_from_ceil_for_frame_and_floor_for_hop
    length = 200
    samples = np.full((length, 1), 0.5, dtype=np.float32)

    times_sec, raw_rms = _frame_rms(samples, sample_rate)

    frame_samples = round(0.025 * sample_rate)
    hop_samples = round(0.010 * sample_rate)
    expected_starts = np.arange(0, length - frame_samples + 1, hop_samples)
    expected_centers_sec = (expected_starts + frame_samples / 2) / sample_rate

    assert len(times_sec) == len(expected_starts)
    assert len(raw_rms) == len(expected_starts)
    assert np.allclose(times_sec, expected_centers_sec)


def test_frame_rms_constant_amplitude_matches_amplitude():
    from vocal_analysis.rms import _frame_rms

    sample_rate = 1000
    samples = np.full((100, 1), 0.5, dtype=np.float32)

    _, raw_rms = _frame_rms(samples, sample_rate)

    assert np.allclose(raw_rms, 0.5, atol=1e-6)


def test_frame_rms_pools_across_channels():
    from vocal_analysis.rms import _frame_rms

    sample_rate = 1000
    left = np.full(100, 0.6, dtype=np.float32)
    right = np.full(100, 0.8, dtype=np.float32)
    samples = np.stack([left, right], axis=1)

    _, raw_rms = _frame_rms(samples, sample_rate)

    pooled_rms = math.sqrt((0.6**2 + 0.8**2) / 2)
    assert np.allclose(raw_rms, pooled_rms, atol=1e-6)


def test_frame_rms_signal_shorter_than_frame_returns_empty():
    from vocal_analysis.rms import _frame_rms

    sample_rate = 1000
    samples = np.full((10, 1), 0.5, dtype=np.float32)

    times_sec, raw_rms = _frame_rms(samples, sample_rate)

    assert len(times_sec) == 0
    assert len(raw_rms) == 0


def test_normalize_rms_reference_formula():
    from vocal_analysis.rms import _normalize_rms

    raw_rms = np.linspace(0.0, 1.0, 11)

    values = _normalize_rms(raw_rms)

    p10 = np.percentile(raw_rms, 10, method="linear")
    p90 = np.percentile(raw_rms, 90, method="linear")
    expected = np.clip((raw_rms - p10) / (p90 - p10), 0.0, 1.0)
    assert np.allclose(values, expected)


def test_normalize_rms_p90_equals_p10_returns_zero():
    from vocal_analysis.rms import _normalize_rms

    raw_rms = np.full(20, 0.3)

    values = _normalize_rms(raw_rms)

    assert np.array_equal(values, np.zeros(20))


def test_normalize_rms_is_gain_invariant():
    from vocal_analysis.rms import _normalize_rms

    raw_rms = np.array([0.0, 0.05, 0.2, 0.5, 0.8, 1.0])
    gained = raw_rms * 3.0

    assert np.allclose(_normalize_rms(raw_rms), _normalize_rms(gained), atol=1e-9)


def test_dynamic_range_db_reference_formula():
    from vocal_analysis.rms import _dynamic_range_db

    raw_rms = np.linspace(0.0, 1.0, 11)

    result = _dynamic_range_db(raw_rms)

    p5 = np.percentile(raw_rms, 5, method="linear")
    p95 = np.percentile(raw_rms, 95, method="linear")
    expected = 20 * math.log10(p95 / p5)
    assert result == pytest.approx(expected)


def test_dynamic_range_db_p95_zero_returns_zero():
    from vocal_analysis.rms import _dynamic_range_db

    raw_rms = np.zeros(20)

    assert _dynamic_range_db(raw_rms) == 0.0


def test_dynamic_range_db_p5_zero_uses_floor_relative_to_p95():
    from vocal_analysis.rms import _dynamic_range_db

    raw_rms = np.concatenate([np.zeros(10), np.ones(10)])

    assert _dynamic_range_db(raw_rms) == pytest.approx(20 * math.log10(1.0 / 1e-6))


def test_dynamic_range_db_is_gain_invariant_including_p5_zero_floor():
    from vocal_analysis.rms import _dynamic_range_db

    raw_rms = np.concatenate([np.zeros(10), np.ones(10)])
    gained = raw_rms * 4.0

    assert _dynamic_range_db(raw_rms) == pytest.approx(_dynamic_range_db(gained))


def test_compute_rms_returns_envelope_with_expected_shape():
    from vocal_analysis.rms import compute_rms
    from vocal_analysis.types import AudioPcm, RmsEnvelope

    sample_rate = 1000
    rng = np.random.default_rng(0)
    samples = (rng.random((500, 1), dtype=np.float32) - 0.5) * 2
    pcm = AudioPcm(samples=samples, sample_rate=sample_rate)

    envelope = compute_rms(pcm)

    assert isinstance(envelope, RmsEnvelope)
    assert len(envelope.times_sec) == len(envelope.values)
    assert len(envelope.times_sec) > 0
    assert np.all(envelope.values >= 0.0) and np.all(envelope.values <= 1.0)


def test_compute_rms_is_gain_invariant():
    from vocal_analysis.rms import compute_rms
    from vocal_analysis.types import AudioPcm

    sample_rate = 1000
    rng = np.random.default_rng(1)
    samples = (rng.random((500, 1), dtype=np.float32) - 0.5) * 0.4
    pcm = AudioPcm(samples=samples, sample_rate=sample_rate)
    gained_pcm = AudioPcm(samples=(samples * 2.5).astype(np.float32), sample_rate=sample_rate)

    envelope = compute_rms(pcm)
    gained_envelope = compute_rms(gained_pcm)

    assert np.allclose(envelope.times_sec, gained_envelope.times_sec)
    assert np.allclose(envelope.values, gained_envelope.values, atol=1e-5)
    assert envelope.dynamic_range_db == pytest.approx(gained_envelope.dynamic_range_db, abs=1e-3)


def test_compute_rms_signal_shorter_than_frame_returns_empty_envelope():
    from vocal_analysis.rms import compute_rms
    from vocal_analysis.types import AudioPcm

    pcm = AudioPcm(samples=np.full((10, 1), 0.5, dtype=np.float32), sample_rate=1000)

    envelope = compute_rms(pcm)

    assert len(envelope.times_sec) == 0
    assert len(envelope.values) == 0
    assert envelope.dynamic_range_db == 0.0
