"""noise のテスト(shakevmd.md §6.1)。

検証可能な性質に落とす:
- オクターブの帯域制限(実効周波数 ≤ 8Hz)と警告
- 決定論・シード再現性
- チャンネル/セグメントごとの独立シード派生(位相非連続)
- スペクトルの帯域制限(FFT)
- C1連続性(無段差 + 微分の連続)・振幅有界
"""

import math

import numpy as np
import pytest

from shakevmd import noise


# ---------------------------------------------------------------------------
# effective_octaves: 帯域制限のオクターブ数(§6.1)
# ---------------------------------------------------------------------------


class TestEffectiveOctaves:
    @pytest.mark.parametrize(
        "freq,octaves,expected",
        [
            (1.2, 3, 3),    # 1.2, 2.4, 4.8 すべて ≤8
            (1.2, 5, 3),    # 9.6, 19.2 が超過 → 3
            (6.0, 4, 1),    # 6 のみ ≤8(12,24,48 超過)
            (8.0, 2, 1),    # 8 は境界(≤8)で可、16 は超過
            (10.0, 1, 0),   # freq 自体が 8 超過 → 0
            (4.0, 2, 2),    # 4, 8 ともに ≤8
        ],
    )
    def test_counts(self, freq, octaves, expected):
        assert noise.effective_octaves(freq, octaves) == expected

    def test_custom_bandlimit(self):
        assert noise.effective_octaves(1.0, 4, bandlimit_hz=4.0) == 3  # 1,2,4 ≤4; 8 超過


# ---------------------------------------------------------------------------
# derive_seed: 決定論+独立性(§5.3, §6.1)
# ---------------------------------------------------------------------------


class TestDeriveSeed:
    def test_deterministic(self):
        assert noise.derive_seed(1, "X") == noise.derive_seed(1, "X")
        assert noise.derive_seed(1, "X", 2) == noise.derive_seed(1, "X", 2)

    def test_distinct_by_channel(self):
        assert noise.derive_seed(1, "X") != noise.derive_seed(1, "Y")

    def test_distinct_by_segment(self):
        # 同一チャンネルでもセグメント番号が違えば別シード(位相を連続させない)
        assert noise.derive_seed(1, "X", 0) != noise.derive_seed(1, "X", 1)

    def test_distinct_by_base(self):
        assert noise.derive_seed(1, "X") != noise.derive_seed(2, "X")

    def test_returns_int(self):
        assert isinstance(noise.derive_seed(1, "X"), int)


# ---------------------------------------------------------------------------
# band_limited_noise: 決定論・形状・クランプ警告
# ---------------------------------------------------------------------------

T_30FPS = np.arange(300) / 30.0  # 10秒・30fps


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
        # freq=6, octaves=4 → 1オクターブのみ採用、3つクランプ → 警告
        _, warns = noise.band_limited_noise(7, T_30FPS, 6.0, octaves=4)
        assert len(warns) >= 1


# ---------------------------------------------------------------------------
# 独立シードの位相独立(相関が低い)
# ---------------------------------------------------------------------------


class TestPhaseIndependence:
    def test_derived_seeds_low_correlation(self):
        s_x = noise.derive_seed(1, "X", 0)
        s_x2 = noise.derive_seed(1, "X", 1)  # 次セグメント
        a, _ = noise.band_limited_noise(s_x, T_30FPS, 1.2)
        b, _ = noise.band_limited_noise(s_x2, T_30FPS, 1.2)
        corr = np.corrcoef(a, b)[0, 1]
        assert abs(corr) < 0.5  # 位相が連続していない


# ---------------------------------------------------------------------------
# スペクトルの帯域制限(FFT)
# ---------------------------------------------------------------------------


class TestSpectralBandLimit:
    def test_negligible_energy_near_nyquist(self):
        # 連続ノイズを高レート(200Hz)で長く生成し、12Hz超のエネルギーが小さいこと。
        # 最高オクターブ基本周波数 4.8Hz の滑らかなノイズはナイキスト近傍に
        # ほとんどエネルギーを持たない(§6.1 の帯域制限の sanity check)。
        fs = 200.0
        t = np.arange(int(fs * 20)) / fs
        vals, _ = noise.band_limited_noise(7, t, 1.2, octaves=3)
        vals = vals - vals.mean()
        spec = np.abs(np.fft.rfft(vals)) ** 2
        freqs = np.fft.rfftfreq(len(vals), 1.0 / fs)
        total = spec.sum()
        above = spec[freqs > 12.0].sum()
        assert above / total < 0.1


# ---------------------------------------------------------------------------
# C1連続性・振幅有界(単一オクターブで基底ノイズの性質を見る)
# ---------------------------------------------------------------------------


class TestContinuityAndAmplitude:
    def test_single_octave_amplitude_bounded(self):
        t = np.linspace(0, 50, 5000)
        vals, _ = noise.band_limited_noise(7, t, 1.0, octaves=1, persistence=1.0)
        assert np.max(np.abs(vals)) <= 1.2  # 概ね単位振幅

    def test_no_jumps_c0(self):
        # 微小刻みで連続(段差がない)
        h = 1e-3
        t = np.arange(0, 5, h)
        vals, _ = noise.band_limited_noise(7, t, 1.2, octaves=1, persistence=1.0)
        assert np.max(np.abs(np.diff(vals))) < 0.1

    def test_derivative_continuous_c1(self):
        # 中心差分による微分が連続(その差分が小さい)= C1 の指標
        h = 1e-3
        t = np.arange(0, 5, h)
        vals, _ = noise.band_limited_noise(7, t, 1.2, octaves=1, persistence=1.0)
        deriv = (vals[2:] - vals[:-2]) / (2 * h)
        assert np.max(np.abs(np.diff(deriv))) < 0.5
