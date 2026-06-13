"""motion の確定的コアのテスト(shakevmd.md §5.1 フェード, §6.2 速度適応)。

このサブステップは fade_envelope / frame_speeds / adaptive_amplitude を対象とする。
settle・インパルス・プロファイルブレンドは後続サブステップ。
"""

import numpy as np
import pytest

from shakevmd import motion


def _ref_smoothstep(t):
    # 仕様から独立に書いた参照(3t^2-2t^3)
    t = np.asarray(t, dtype=float)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# fade_envelope(§5.1)
# ---------------------------------------------------------------------------


class TestFadeEnvelope:
    def test_length_and_zero_ends(self):
        env = motion.fade_envelope(100, fade_sec=0.5, fps=30.0)
        assert env.shape == (100,)
        assert env[0] == pytest.approx(0.0, abs=1e-9)
        assert env[-1] == pytest.approx(0.0, abs=1e-9)

    def test_plateau_is_one(self):
        # 中央(フェード区間の外)は 1
        env = motion.fade_envelope(100, fade_sec=0.3, fps=30.0)
        assert env[50] == pytest.approx(1.0, abs=1e-9)
        assert np.max(env) == pytest.approx(1.0, abs=1e-9)

    def test_monotonic_in_out(self):
        env = motion.fade_envelope(100, fade_sec=0.5, fps=30.0)
        half = len(env) // 2
        assert np.all(np.diff(env[:half]) >= -1e-12)   # 立ち上がり: 非減少
        assert np.all(np.diff(env[half:]) <= 1e-12)    # 立ち下がり: 非増加

    def test_bounded_0_1(self):
        env = motion.fade_envelope(80, fade_sec=0.6, fps=30.0)
        assert np.all(env >= -1e-12) and np.all(env <= 1.0 + 1e-12)

    def test_ramp_follows_smoothstep(self):
        # 立ち上がり f フレームは smoothstep(linspace(0,1,f)) 形状、立ち下がりはその反転
        n, f = 120, 15  # fade 0.5s × 30fps = 15、2×15=30 <= 120 なので短縮なし
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        ref = _ref_smoothstep(np.linspace(0.0, 1.0, f))
        assert np.allclose(env[:f], ref, atol=1e-9)
        assert np.allclose(env[n - f:], ref[::-1], atol=1e-9)

    def test_velocity_continuous_at_both_ends(self):
        # smoothstep の端微分0 により、両端の初差分(速度)が小さい(0接続で速度連続)
        env = motion.fade_envelope(120, fade_sec=0.5, fps=30.0)
        assert abs(env[1] - env[0]) < 0.02
        assert abs(env[-1] - env[-2]) < 0.02

    def test_short_segment_shortens_fade_to_half(self):
        # 2×fade(=30フレーム)未満のセグメントはフェードを n//2 に短縮。
        # n=20 → f=10。立ち上がりは smoothstep(linspace(0,1,10))、中央(index 9,10)で1。
        n = 20
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        ref = _ref_smoothstep(np.linspace(0.0, 1.0, 10))
        assert np.allclose(env[:10], ref, atol=1e-9)         # 立ち上がり
        assert np.allclose(env[10:], ref[::-1], atol=1e-9)   # 立ち下がり(反転)
        assert env[9] == pytest.approx(1.0, abs=1e-9)
        assert env[10] == pytest.approx(1.0, abs=1e-9)
        assert env[0] == pytest.approx(0.0, abs=1e-9)
        assert env[-1] == pytest.approx(0.0, abs=1e-9)

    def test_odd_short_segment_structure(self):
        # 奇数短尺 n=21 → f=10。立ち上がり env[:10]=ramp(末尾1)、index10 は untouched の1、
        # 立ち下がり env[11:]=ramp 反転(先頭1)。よって index 9,10,11 が連続して1になる。
        n = 21
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        ref = _ref_smoothstep(np.linspace(0.0, 1.0, 10))
        assert np.allclose(env[:10], ref, atol=1e-9)
        assert env[10] == pytest.approx(1.0, abs=1e-9)        # untouched plateau
        assert np.allclose(env[11:], ref[::-1], atol=1e-9)
        assert np.allclose(env[9:12], 1.0, atol=1e-9)         # 連続して1の領域
        assert env[0] == pytest.approx(0.0, abs=1e-9)
        assert env[-1] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, []),
            (1, [0.0]),
            (2, [0.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
        ],
    )
    def test_tiny_segments(self, n, expected):
        # 極短セグメントの挙動を定義: 端は常に0、間があれば1
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        assert env.shape == (n,)
        assert np.allclose(env, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# frame_speeds(§6.2)
# ---------------------------------------------------------------------------


class TestFrameSpeeds:
    def test_static_is_zero(self):
        speeds = motion.frame_speeds(np.full(50, 3.0))
        assert np.allclose(speeds, 0.0)

    def test_normalized_to_unit_max(self):
        # 一定速度で増加 → 差分一定 → 正規化後はほぼ一定の最大1
        vals = np.arange(50, dtype=float) * 2.0
        speeds = motion.frame_speeds(vals)
        assert np.max(speeds) == pytest.approx(1.0, abs=1e-9)
        assert np.all(speeds >= -1e-12) and np.all(speeds <= 1.0 + 1e-12)

    def test_relative_magnitude(self):
        # 後半で速度2倍 → 正規化後、後半が前半の約2倍、最大が1
        vals = np.concatenate([np.arange(0, 10, 1.0), np.arange(10, 30, 2.0)])
        speeds = motion.frame_speeds(vals)
        assert np.max(speeds) == pytest.approx(1.0, abs=1e-9)
        slow = np.median(speeds[2:9])
        fast = np.median(speeds[12:19])
        assert fast == pytest.approx(2.0 * slow, rel=0.2)

    def test_length_matches(self):
        speeds = motion.frame_speeds(np.zeros(37))
        assert speeds.shape == (37,)

    def test_frame_alignment_explicit(self):
        # speed[i] = |x[i]-x[i-1]|、speed[0]=0。値が変化したフレームに速度が現れる。
        vals = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        speeds = motion.frame_speeds(vals)
        # 差分は index 3 のみ(10)。最大で正規化 → index3=1、他0
        assert np.allclose(speeds, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def test_scalar_speed_is_absolute(self):
        # 速度は差の「大きさ」。減少列でも絶対値で扱う。
        vals = np.array([3.0, 2.0, 1.0])
        speeds = motion.frame_speeds(vals)  # 生 [0,1,1] → 正規化 [0,1,1]
        assert np.allclose(speeds, [0.0, 1.0, 1.0])

    def test_vector_norm_is_euclidean(self):
        # ベクトル差の大きさはユークリッドノルム([3,4,0] の差 → 5)
        vals = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
        speeds = motion.frame_speeds(vals)
        # 生速度 [0,5,0] → 正規化 [0,1,0]
        assert np.allclose(speeds, [0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# adaptive_amplitude(§6.2)
# ---------------------------------------------------------------------------


class TestAdaptiveAmplitude:
    def test_zero_speed_is_base(self):
        assert motion.adaptive_amplitude(0.8, 0.0) == pytest.approx(0.8)

    def test_full_speed_scales_by_motion_scale(self):
        # 速度1・motion_scale=0.5 → base×1.5
        assert motion.adaptive_amplitude(0.8, 1.0, motion_scale=0.5) == pytest.approx(1.2)

    def test_motion_scale_zero_disables(self):
        assert motion.adaptive_amplitude(0.8, 1.0, motion_scale=0.0) == pytest.approx(0.8)

    def test_array_input(self):
        out = motion.adaptive_amplitude(1.0, np.array([0.0, 0.5, 1.0]), motion_scale=0.4)
        assert np.allclose(out, [1.0, 1.2, 1.4])
