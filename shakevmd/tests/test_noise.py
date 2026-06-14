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

    def test_no_synchronized_zeros_at_lattice_frames(self):
        # グラディエントノイズは整数格子点で厳密ゼロ。位相オフセットがないと
        # 既定 freq=1.2・t=frame/30 で25フレームごとに全チャンネルが同時ゼロになる。
        # 各チャンネルが別位相を持ち、整列フレームで同期ゼロにならないこと。
        t = np.arange(300) / 30.0
        series = np.array(
            [noise.band_limited_noise(noise.derive_seed(1, ch), t, 1.2)[0]
             for ch in ("X", "Y", "Z", "rot")]
        )
        aligned = np.arange(25, 300, 25)  # f_i*t が整数になるフレーム
        max_over_channels = np.max(np.abs(series[:, aligned]), axis=0)
        assert np.all(max_over_channels > 1e-6)


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


class TestOctaveComponents:
    """プロファイルクロスフェード用の per-octave 成分(§6.2)。"""

    def test_count_and_amplitude_bound(self):
        t = np.arange(0, 2, 1 / 30.0)
        comps, warns = noise.octave_components(123, t, 1.2, octaves=3)
        assert len(comps) == noise.effective_octaves(1.2, 3)
        assert not warns                               # 1.2×4=4.8≤8 でクランプなし → 警告なし
        for c in comps:
            c = np.asarray(c)
            assert c.shape == t.shape
            assert np.max(np.abs(c)) <= 1.0 + 1e-9     # 単一オクターブ perlin は [-1,1]

    def test_each_component_equals_its_octave(self):
        # 各成分 i が「実際の i 番目のオクターブ」であることを公開APIだけで検証する
        # (偽の再分配=和は合うが個々が別物、を排除)。band_limited_noise(persistence=1) は
        # 先頭 i+1 本のオクターブの単純和なので、その階差が i 番目のオクターブそのもの。
        t = np.arange(0, 3, 1 / 30.0)
        freq = 1.2
        comps, _ = noise.octave_components(7, t, freq, octaves=3)
        for i in range(len(comps)):
            upto_hi, _ = noise.band_limited_noise(7, t, freq, octaves=i + 1, persistence=1.0)
            upto_lo, _ = noise.band_limited_noise(7, t, freq, octaves=i, persistence=1.0)
            assert np.allclose(np.asarray(comps[i]), upto_hi - upto_lo, atol=1e-9)

    def test_persistence_weighted_sum_matches_band_limited(self):
        # Σ persistence^i × components[i] == band_limited_noise(同パラメータ)。
        # 成分の per-octave シード派生・位相が band_limited_noise と一致することを担保する。
        t = np.arange(0, 3, 1 / 30.0)
        comps, _ = noise.octave_components(7, t, 1.2, octaves=3)
        ref, _ = noise.band_limited_noise(
            7, t, 1.2, octaves=3, persistence=noise.DEFAULT_PERSISTENCE)
        summed = sum((noise.DEFAULT_PERSISTENCE ** i) * np.asarray(comps[i])
                     for i in range(len(comps)))
        assert np.allclose(summed, ref, atol=1e-9)

    def test_bandlimit_clamp_reduces_components_with_warning(self):
        # 高 freq で帯域制限クランプ → 成分数が減り警告が出る(§6.1)。
        t = np.arange(0, 2, 1 / 30.0)
        comps, warns = noise.octave_components(1, t, 5.0, octaves=3)
        assert len(comps) == noise.effective_octaves(5.0, 3) < 3
        assert warns

    def test_all_octaves_clamped_returns_empty(self):
        # freq>8Hz は基本オクターブから帯域外 → 成分0本 + 警告(§6.1: 8Hz超は生成しない)。
        # 帯域外を1本でも強制生成する実装を排除する。
        t = np.arange(0, 2, 1 / 30.0)
        comps, warns = noise.octave_components(1, t, 10.0, octaves=3)
        assert noise.effective_octaves(10.0, 3) == 0
        assert len(comps) == 0
        assert warns
