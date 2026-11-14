import numpy as np
import pytest

from shakevmd import motion


def _ref_smoothstep(t):
    t = np.asarray(t, dtype=float)
    return t * t * (3.0 - 2.0 * t)


class TestFadeEnvelope:
    def test_length_and_zero_ends(self):
        env = motion.fade_envelope(100, fade_sec=0.5, fps=30.0)
        assert env.shape == (100,)
        assert env[0] == pytest.approx(0.0, abs=1e-9)
        assert env[-1] == pytest.approx(0.0, abs=1e-9)

    def test_plateau_is_one(self):
        env = motion.fade_envelope(100, fade_sec=0.3, fps=30.0)
        assert env[50] == pytest.approx(1.0, abs=1e-9)
        assert np.max(env) == pytest.approx(1.0, abs=1e-9)

    def test_monotonic_in_out(self):
        env = motion.fade_envelope(100, fade_sec=0.5, fps=30.0)
        half = len(env) // 2
        assert np.all(np.diff(env[:half]) >= -1e-12)
        assert np.all(np.diff(env[half:]) <= 1e-12)

    def test_bounded_0_1(self):
        env = motion.fade_envelope(80, fade_sec=0.6, fps=30.0)
        assert np.all(env >= -1e-12) and np.all(env <= 1.0 + 1e-12)

    def test_ramp_follows_smoothstep(self):
        n, f = 120, 15
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        ref = _ref_smoothstep(np.linspace(0.0, 1.0, f))
        assert np.allclose(env[:f], ref, atol=1e-9)
        assert np.allclose(env[n - f:], ref[::-1], atol=1e-9)

    def test_velocity_continuous_at_both_ends(self):
        env = motion.fade_envelope(120, fade_sec=0.5, fps=30.0)
        assert abs(env[1] - env[0]) < 0.02
        assert abs(env[-1] - env[-2]) < 0.02

    def test_short_segment_shortens_fade_to_half(self):
        n = 20
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        ref = _ref_smoothstep(np.linspace(0.0, 1.0, 10))
        assert np.allclose(env[:10], ref, atol=1e-9)
        assert np.allclose(env[10:], ref[::-1], atol=1e-9)
        assert env[9] == pytest.approx(1.0, abs=1e-9)
        assert env[10] == pytest.approx(1.0, abs=1e-9)
        assert env[0] == pytest.approx(0.0, abs=1e-9)
        assert env[-1] == pytest.approx(0.0, abs=1e-9)

    def test_odd_short_segment_keeps_middle_frame_at_one(self):
        n = 21
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        ref = _ref_smoothstep(np.linspace(0.0, 1.0, 10))
        assert np.allclose(env[:10], ref, atol=1e-9)
        assert env[10] == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(env[11:], ref[::-1], atol=1e-9)
        assert np.allclose(env[9:12], 1.0, atol=1e-9)
        assert env[0] == pytest.approx(0.0, abs=1e-9)
        assert env[-1] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize(
        "n,expected",
        [
            pytest.param(0, [], id="empty"),
            pytest.param(1, [0.0], id="one_frame"),
            pytest.param(2, [0.0, 0.0], id="two_frames"),
            pytest.param(3, [0.0, 1.0, 0.0], id="three_frames"),
        ],
    )
    def test_tiny_segments_have_zero_ends(self, n, expected):
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        assert env.shape == (n,)
        assert np.allclose(env, expected, atol=1e-9)


class TestFrameSpeeds:
    def test_static_is_zero(self):
        speeds = motion.frame_speeds(np.full(50, 3.0), 1.0)
        assert np.allclose(speeds, 0.0)

    def test_speed_below_ref_is_proportional(self):
        speeds = motion.frame_speeds(np.arange(0, 12, 2.0), ref=8.0)
        assert np.allclose(speeds[1:], 0.25)

    def test_clamped_at_ref(self):
        speeds = motion.frame_speeds(np.array([0.0, 20.0, 60.0]), ref=5.0)
        assert np.allclose(speeds, [0.0, 1.0, 1.0])

    def test_speed_appears_at_frame_where_value_changed(self):
        vals = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        speeds = motion.frame_speeds(vals, ref=10.0)
        assert np.allclose(speeds, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def test_scalar_speed_is_absolute(self):
        vals = np.array([3.0, 2.0, 1.0])
        speeds = motion.frame_speeds(vals, ref=1.0)
        assert np.allclose(speeds, [0.0, 1.0, 1.0])

    def test_vector_norm_is_euclidean(self):
        vals = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
        speeds = motion.frame_speeds(vals, ref=5.0)
        assert np.allclose(speeds, [0.0, 1.0, 0.0])

    def test_nonpositive_ref_is_zero(self):
        for ref in (0.0, -3.0, -1e-9):
            speeds = motion.frame_speeds(np.arange(0, 5, 1.0), ref=ref)
            assert np.allclose(speeds, 0.0), ref


class TestDetectStops:
    def test_crossing_below_is_stop(self):
        speeds = np.array([0.5, 0.5, 0.0, 0.0])
        assert motion.detect_stops(speeds, threshold=0.1) == [2]

    def test_no_stop_when_always_moving(self):
        assert motion.detect_stops(np.full(10, 0.5), threshold=0.1) == []

    def test_no_stop_when_always_stopped(self):
        assert motion.detect_stops(np.zeros(10), threshold=0.1) == []

    def test_multiple_stops(self):
        speeds = np.array([0.5, 0.0, 0.0, 0.5, 0.5, 0.0])
        assert motion.detect_stops(speeds, threshold=0.1) == [1, 5]

    def test_only_downward_crossing(self):
        speeds = np.array([0.0, 0.0, 0.5, 0.5])
        assert motion.detect_stops(speeds, threshold=0.1) == []

    def test_threshold_boundary(self):
        assert motion.detect_stops(np.array([0.1, 0.0]), threshold=0.1) == [1]
        assert motion.detect_stops(np.array([0.5, 0.1]), threshold=0.1) == []


class TestSettleOscillation:
    def test_zero_at_start(self):
        assert motion.settle_oscillation(0.0, amp=0.3) == pytest.approx(0.0, abs=1e-9)


class TestProfileCrossfade:
    def test_array_input_returns_weights_per_frame(self):
        s = np.array([0.0, 0.5, 1.0])
        w = np.asarray(motion.profile_weights(s))
        assert w.shape == (3, len(motion.STILL_PROFILE))
