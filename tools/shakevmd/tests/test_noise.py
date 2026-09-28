import numpy as np
import pytest

from shakevmd import noise


class TestEffectiveOctaves:
    @pytest.mark.parametrize(
        "freq,octaves,expected",
        [
            pytest.param(1.2, 3, 3, id="all_within_band"),
            pytest.param(1.2, 5, 3, id="octaves_above_band_dropped"),
            pytest.param(6.0, 4, 1, id="only_base_within_band"),
            pytest.param(8.0, 2, 1, id="octave_exactly_at_limit_kept"),
            pytest.param(10.0, 1, 0, id="base_above_band"),
            pytest.param(4.0, 2, 2, id="top_octave_exactly_at_limit"),
        ],
    )
    def test_counts_leading_octaves_within_8hz(self, freq, octaves, expected):
        assert noise.effective_octaves(freq, octaves) == expected

    def test_negative_freq_is_judged_by_magnitude(self):
        assert noise.effective_octaves(-1.2, 3) == 3
        assert noise.effective_octaves(-10.0, 1) == 0

    def test_custom_bandlimit(self):
        assert noise.effective_octaves(1.0, 4, bandlimit_hz=4.0) == 3


class TestDeriveSeed:
    def test_deterministic(self):
        assert noise.derive_seed(1, "X") == noise.derive_seed(1, "X")
        assert noise.derive_seed(1, "X", 2) == noise.derive_seed(1, "X", 2)

    def test_distinct_by_channel(self):
        assert noise.derive_seed(1, "X") != noise.derive_seed(1, "Y")

    def test_distinct_by_segment(self):
        assert noise.derive_seed(1, "X", 0) != noise.derive_seed(1, "X", 1)

    def test_distinct_by_base(self):
        assert noise.derive_seed(1, "X") != noise.derive_seed(2, "X")

    def test_returns_int(self):
        assert isinstance(noise.derive_seed(1, "X"), int)

    def test_distinct_by_part_type(self):
        assert noise.derive_seed(1, 2) != noise.derive_seed(1, "2")

    def test_part_boundaries_do_not_collide(self):
        assert noise.derive_seed(1, "ab", "c") != noise.derive_seed(1, "a", "bc")


T_30FPS = np.arange(300) / 30.0


class TestBandLimitedNoiseBasics:
    def test_shape_matches_t(self):
        vals, _ = noise.band_limited_noise(7, T_30FPS, 1.2)
        assert vals.shape == T_30FPS.shape

    def test_reproducible_same_seed(self):
        a, _ = noise.band_limited_noise(7, T_30FPS, 1.2)
        b, _ = noise.band_limited_noise(7, T_30FPS, 1.2)
        assert np.array_equal(a, b)

    def test_different_seed_differs(self):
        a, _ = noise.band_limited_noise(7, T_30FPS, 1.2)
        b, _ = noise.band_limited_noise(8, T_30FPS, 1.2)
        assert not np.allclose(a, b)

    def test_no_clamp_warning_when_within_band(self):
        _, warns = noise.band_limited_noise(7, T_30FPS, 1.2, octaves=3)
        assert warns == []

    def test_clamp_warning_when_exceeding_band(self):
        _, warns = noise.band_limited_noise(7, T_30FPS, 6.0, octaves=4)
        assert len(warns) >= 1


class TestPhaseIndependence:
    def test_adjacent_segment_seeds_give_uncorrelated_noise(self):
        s_x = noise.derive_seed(1, "X", 0)
        s_x2 = noise.derive_seed(1, "X", 1)
        a, _ = noise.band_limited_noise(s_x, T_30FPS, 1.2)
        b, _ = noise.band_limited_noise(s_x2, T_30FPS, 1.2)
        corr = np.corrcoef(a, b)[0, 1]
        assert abs(corr) < 0.5

    def test_no_synchronized_zeros_at_lattice_frames(self):
        t = np.arange(300) / 30.0
        series = np.array(
            [noise.band_limited_noise(noise.derive_seed(1, ch), t, 1.2)[0]
             for ch in ("X", "Y", "Z", "rot")]
        )
        aligned = np.arange(25, 300, 25)
        max_over_channels = np.max(np.abs(series[:, aligned]), axis=0)
        assert np.all(max_over_channels > 1e-6)


class TestSpectralBandLimit:
    def test_energy_above_12hz_is_under_10_percent(self):
        fs = 200.0
        t = np.arange(int(fs * 20)) / fs
        vals, _ = noise.band_limited_noise(7, t, 1.2, octaves=3)
        vals = vals - vals.mean()
        spec = np.abs(np.fft.rfft(vals)) ** 2
        freqs = np.fft.rfftfreq(len(vals), 1.0 / fs)
        total = spec.sum()
        above = spec[freqs > 12.0].sum()
        assert above / total < 0.1


class TestContinuityAndAmplitude:
    def test_single_octave_amplitude_bounded(self):
        t = np.linspace(0, 50, 5000)
        vals, _ = noise.band_limited_noise(7, t, 1.0, octaves=1, persistence=1.0)
        assert np.max(np.abs(vals)) <= 1.2

    def test_no_jumps_c0(self):
        h = 1e-3
        t = np.arange(0, 5, h)
        vals, _ = noise.band_limited_noise(7, t, 1.2, octaves=1, persistence=1.0)
        assert np.max(np.abs(np.diff(vals))) < 0.1

    def test_derivative_continuous_c1(self):
        h = 1e-3
        t = np.arange(0, 5, h)
        vals, _ = noise.band_limited_noise(7, t, 1.2, octaves=1, persistence=1.0)
        deriv = (vals[2:] - vals[:-2]) / (2 * h)
        assert np.max(np.abs(np.diff(deriv))) < 0.5


class TestOctaveComponents:
    def test_count_and_amplitude_bound(self):
        t = np.arange(0, 2, 1 / 30.0)
        comps, warns = noise.octave_components(123, t, 1.2, octaves=3)
        assert len(comps) == noise.effective_octaves(1.2, 3)
        assert not warns
        for c in comps:
            c = np.asarray(c)
            assert c.shape == t.shape
            assert np.max(np.abs(c)) <= 1.0 + 1e-9

    def test_each_component_equals_its_octave(self):
        t = np.arange(0, 3, 1 / 30.0)
        freq = 1.2
        comps, _ = noise.octave_components(7, t, freq, octaves=3)
        for i in range(len(comps)):
            upto_hi, _ = noise.band_limited_noise(7, t, freq, octaves=i + 1, persistence=1.0)
            upto_lo, _ = noise.band_limited_noise(7, t, freq, octaves=i, persistence=1.0)
            assert np.allclose(np.asarray(comps[i]), upto_hi - upto_lo, atol=1e-9)

    def test_persistence_weighted_sum_matches_band_limited(self):
        t = np.arange(0, 3, 1 / 30.0)
        comps, _ = noise.octave_components(7, t, 1.2, octaves=3)
        ref, _ = noise.band_limited_noise(
            7, t, 1.2, octaves=3, persistence=noise.DEFAULT_PERSISTENCE)
        summed = sum((noise.DEFAULT_PERSISTENCE ** i) * np.asarray(comps[i])
                     for i in range(len(comps)))
        assert np.allclose(summed, ref, atol=1e-9)

    def test_bandlimit_clamp_reduces_components_with_warning(self):
        t = np.arange(0, 2, 1 / 30.0)
        comps, warns = noise.octave_components(1, t, 5.0, octaves=3)
        assert len(comps) == noise.effective_octaves(5.0, 3) < 3
        assert warns

    def test_all_octaves_clamped_returns_empty(self):
        t = np.arange(0, 2, 1 / 30.0)
        comps, warns = noise.octave_components(1, t, 10.0, octaves=3)
        assert noise.effective_octaves(10.0, 3) == 0
        assert len(comps) == 0
        assert warns
