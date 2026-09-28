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

    def test_absolute_not_peak_normalized(self):
        speeds = motion.frame_speeds(np.arange(0, 5, 1.0), ref=10.0)
        assert np.allclose(speeds[1:], 0.1)
        assert np.max(speeds) == pytest.approx(0.1)

    def test_speed_below_ref_is_proportional(self):
        speeds = motion.frame_speeds(np.arange(0, 12, 2.0), ref=8.0)
        assert np.allclose(speeds[1:], 0.25)

    def test_clamped_at_ref(self):
        speeds = motion.frame_speeds(np.array([0.0, 20.0, 60.0]), ref=5.0)
        assert np.allclose(speeds, [0.0, 1.0, 1.0])

    def test_relative_magnitude_preserved_below_ref(self):
        vals = np.concatenate([np.arange(0, 10, 1.0), np.arange(10, 30, 2.0)])
        speeds = motion.frame_speeds(vals, ref=8.0)
        slow = np.median(speeds[2:9])
        fast = np.median(speeds[12:19])
        assert fast == pytest.approx(2.0 * slow, rel=0.2)

    def test_length_matches(self):
        speeds = motion.frame_speeds(np.zeros(37), 1.0)
        assert speeds.shape == (37,)

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


class TestAdaptiveAmplitude:
    def test_zero_speed_is_base(self):
        assert motion.adaptive_amplitude(0.8, 0.0, motion_damp=1.0) == pytest.approx(0.8)

    def test_full_speed_damps_by_motion_damp(self):
        assert motion.adaptive_amplitude(0.8, 1.0, motion_damp=0.5) == pytest.approx(0.4)

    def test_motion_damp_zero_disables(self):
        assert motion.adaptive_amplitude(0.8, 1.0, motion_damp=0.0) == pytest.approx(0.8)

    def test_clamped_to_zero(self):
        assert motion.adaptive_amplitude(0.8, 1.0, motion_damp=2.0) == pytest.approx(0.0)

    def test_array_input(self):
        out = motion.adaptive_amplitude(1.0, np.array([0.0, 0.5, 1.0]), motion_damp=0.4)
        assert np.allclose(out, [1.0, 0.8, 0.6])


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

    def test_matches_formula(self):
        amp, st, freq = 0.3, 1.0, 3.0
        t = np.linspace(0.0, 2.0, 257)
        vals = np.asarray(motion.settle_oscillation(t, amp=amp, settle_time=st, freq=freq))
        expected = amp * np.exp(-t / (st / 4.0)) * np.sin(2 * np.pi * freq * t)
        assert np.allclose(vals, expected, atol=1e-9)

    def test_envelope_upper_bound(self):
        amp, st, freq = 0.3, 1.0, 3.0
        t = np.linspace(0.0, 2.0, 400)
        vals = np.asarray(motion.settle_oscillation(t, amp=amp, settle_time=st, freq=freq))
        env = amp * np.exp(-t / (st / 4.0))
        assert np.all(np.abs(vals) <= env + 1e-9)

    def test_converged_after_settle_time(self):
        amp, st, freq = 0.3, 1.0, 3.0
        t = np.linspace(st, 2 * st, 200)
        vals = np.asarray(motion.settle_oscillation(t, amp=amp, settle_time=st, freq=freq))
        assert np.all(np.abs(vals) < 0.02 * amp)

    def test_oscillates(self):
        amp, st, freq = 0.3, 1.0, 3.0
        t = np.linspace(0.0, 1.0, 600)
        vals = np.asarray(motion.settle_oscillation(t, amp=amp, settle_time=st, freq=freq))
        nonzero = vals[np.abs(vals) > 1e-6]
        sign_changes = int(np.sum(np.diff(np.sign(nonzero)) != 0))
        assert sign_changes >= 2

    def test_deterministic(self):
        a = motion.settle_oscillation(0.3, amp=0.3, freq=3.0)
        b = motion.settle_oscillation(0.3, amp=0.3, freq=3.0)
        assert a == b


class TestImpulseEnvelope:
    def test_zero_before_frame(self):
        env = motion.impulse_envelope(60, frame=30, strength=2.0, decay_sec=0.5, fps=30.0)
        assert env.shape == (60,)
        assert np.allclose(env[:30], 0.0)

    def test_peak_at_frame(self):
        env = motion.impulse_envelope(60, frame=30, strength=2.0, decay_sec=0.5, fps=30.0)
        assert env[30] == pytest.approx(2.0, abs=1e-9)

    def test_exponential_decay(self):
        env = motion.impulse_envelope(120, frame=10, strength=2.0, decay_sec=1.0, fps=30.0)
        assert env[10 + 30] == pytest.approx(2.0 / np.e, rel=1e-6)

    def test_monotonic_decay_after_frame(self):
        env = motion.impulse_envelope(120, frame=10, strength=2.0, decay_sec=1.0, fps=30.0)
        assert np.all(np.diff(env[10:]) <= 1e-12)

    def test_matches_exponential_formula(self):
        F, S, D, fps = 10, 2.0, 0.8, 30.0
        env = motion.impulse_envelope(120, frame=F, strength=S, decay_sec=D, fps=fps)
        idx = np.arange(F, 120)
        dt = (idx - F) / fps
        assert np.allclose(env[F:], S * np.exp(-dt / D), atol=1e-9)


class TestProfileCrossfade:
    def test_endpoints_and_midpoint(self):
        assert list(motion.profile_weights(0.0)) == pytest.approx(list(motion.STILL_PROFILE))
        assert list(motion.profile_weights(1.0)) == pytest.approx(list(motion.MOVING_PROFILE))
        mid = motion.profile_weights(0.5)
        expect = [(motion.STILL_PROFILE[i] + motion.MOVING_PROFILE[i]) / 2
                  for i in range(len(motion.STILL_PROFILE))]
        assert list(mid) == pytest.approx(expect)
        q = motion.profile_weights(0.25)
        expect_q = [0.75 * motion.STILL_PROFILE[i] + 0.25 * motion.MOVING_PROFILE[i]
                    for i in range(len(motion.STILL_PROFILE))]
        assert list(q) == pytest.approx(expect_q)

    def test_array_input_per_frame(self):
        s = np.array([0.0, 0.5, 1.0])
        w = np.asarray(motion.profile_weights(s))
        assert w.shape == (3, len(motion.STILL_PROFILE))
        assert list(w[0]) == pytest.approx(list(motion.STILL_PROFILE))
        assert list(w[-1]) == pytest.approx(list(motion.MOVING_PROFILE))
        mid = [(motion.STILL_PROFILE[i] + motion.MOVING_PROFILE[i]) / 2
               for i in range(len(motion.STILL_PROFILE))]
        assert list(w[1]) == pytest.approx(mid)

    def test_moving_weights_top_octave_more_and_still_keeps_it_nonzero(self):
        assert tuple(motion.STILL_PROFILE) != tuple(motion.MOVING_PROFILE)
        assert motion.MOVING_PROFILE[-1] > motion.STILL_PROFILE[-1]
        assert motion.STILL_PROFILE[-1] > 0.0


class TestBreathingDrift:
    def test_is_0_3hz_sine_with_given_amplitude(self):
        amp = 0.5
        assert motion.BREATHING_HZ == pytest.approx(0.3)
        period = 1.0 / 0.3
        t = np.linspace(0.0, 2.0 * period, 2001)
        x = np.asarray(motion.breathing_drift(t, amp))
        assert np.max(np.abs(x)) <= amp + 1e-9
        q = float(np.asarray(motion.breathing_drift(np.array([period / 4.0]), amp))[0])
        assert q == pytest.approx(amp, abs=1e-3)
        z = float(np.asarray(motion.breathing_drift(np.array([period]), amp))[0])
        assert abs(z) < 1e-3
        assert float(np.asarray(motion.breathing_drift(np.array([0.0]), amp))[0]) == pytest.approx(0.0, abs=1e-9)
        neg = float(np.asarray(motion.breathing_drift(np.array([3.0 * period / 4.0]), amp))[0])
        assert neg == pytest.approx(-amp, abs=1e-3)
