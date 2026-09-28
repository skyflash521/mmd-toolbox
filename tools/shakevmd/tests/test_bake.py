import inspect

import numpy as np
import pytest

from shakevmd import bake, motion
from vmd import camera, interp
from vmd.types import CameraKey


def cam(distance=-30.0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), fov=30, perspective=0):
    return CameraKey(0, distance, center, rotation, bytes(24), fov, perspective)


def rebuilt(key, res):
    return CameraKey(0, key.distance, res["position"], res["rotation"],
                     bytes(24), key.fov, key.perspective)


class TestApplyGazeShake:
    def test_zero_noise_is_identity(self):
        key = cam(distance=-30.0, center=(1.0, 2.0, 3.0), rotation=(0.1, 0.2, 0.3))
        res = bake.apply_gaze_shake(key, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert res["position"] == pytest.approx(key.position, abs=1e-6)
        assert res["rotation"] == pytest.approx(key.rotation, abs=1e-6)

    def test_rotation_is_componentwise_euler_addition(self):
        key = cam(distance=-30.0, center=(1.0, 2.0, 3.0), rotation=(0.1, -0.2, 0.3))
        noise = (0.05, 0.07, -0.04)
        res = bake.apply_gaze_shake(key, noise, (0.0, 0.0, 0.0))
        expected = tuple(key.rotation[i] + noise[i] for i in range(3))
        assert res["rotation"] == pytest.approx(expected, abs=1e-9)

    def test_rotation_sum_past_pi_is_not_wrapped(self):
        key = cam(distance=-30.0, center=(0.0, 0.0, 0.0), rotation=(0.0, 3.10, 0.0))
        noise = (0.0, 0.1, 0.0)
        res = bake.apply_gaze_shake(key, noise, (0.0, 0.0, 0.0))
        assert res["rotation"][1] == pytest.approx(3.20, abs=1e-9)

    def test_rotation_only_keeps_camera_world_position(self):
        key = cam(distance=-30.0, center=(1.0, 2.0, 3.0), rotation=(0.1, 0.2, 0.0))
        res = bake.apply_gaze_shake(key, (0.05, -0.03, 0.02), (0.0, 0.0, 0.0))
        before = camera.to_world(key).position
        after = camera.to_world(rebuilt(key, res)).position
        assert after == pytest.approx(before, abs=1e-5)

    def test_position_noise_shifts_world_position_regardless_of_orientation(self):
        key = cam(distance=-30.0, center=(1.0, 2.0, 3.0), rotation=(0.3, -0.4, 0.2))
        pn = np.array([5.0, -2.0, 1.0])
        res = bake.apply_gaze_shake(key, (0.0, 0.0, 0.0), tuple(pn))
        p0 = np.array(camera.to_world(key).position)
        p1 = np.array(camera.to_world(rebuilt(key, res)).position)
        assert p1 == pytest.approx(p0 + pn, abs=1e-5)
        assert res["rotation"] == pytest.approx(key.rotation, abs=1e-9)

    def test_distance_zero_matches_naive_addition(self):
        key = cam(distance=0.0, center=(1.0, 2.0, 3.0), rotation=(0.2, 0.1, -0.1))
        noise = (0.1, -0.05, 0.03)
        res = bake.apply_gaze_shake(key, noise, (0.0, 0.0, 0.0))
        assert res["position"] == pytest.approx((1.0, 2.0, 3.0), abs=1e-6)
        expected = tuple(key.rotation[i] + noise[i] for i in range(3))
        assert res["rotation"] == pytest.approx(expected, abs=1e-9)

    def test_distance_zero_position_noise_moves_center_directly(self):
        key = cam(distance=0.0, center=(1.0, 2.0, 3.0), rotation=(0.2, 0.1, 0.0))
        res = bake.apply_gaze_shake(key, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        assert res["position"] == pytest.approx((3.0, 2.0, 3.0), abs=1e-5)

    def test_rotation_and_position_combined(self):
        key = cam(distance=-40.0, center=(2.0, 1.0, -3.0), rotation=(0.2, 0.3, 0.1))
        noise, pn = (0.04, -0.06, 0.02), np.array([1.0, 2.0, -1.0])
        res = bake.apply_gaze_shake(key, noise, tuple(pn))
        assert res["rotation"] == pytest.approx(
            tuple(key.rotation[i] + noise[i] for i in range(3)), abs=1e-9
        )
        p0 = np.array(camera.to_world(key).position)
        p1 = np.array(camera.to_world(rebuilt(key, res)).position)
        assert p1 == pytest.approx(p0 + pn, abs=1e-5)

    def test_deterministic(self):
        key = cam(rotation=(0.1, 0.2, 0.3))
        a = bake.apply_gaze_shake(key, (0.05, 0.05, 0.05), (1.0, 0.0, 0.0))
        b = bake.apply_gaze_shake(key, (0.05, 0.05, 0.05), (1.0, 0.0, 0.0))
        assert a["position"] == pytest.approx(b["position"], abs=1e-12)
        assert a["rotation"] == pytest.approx(b["rotation"], abs=1e-12)


NONLINEAR_INTERP = bytes([54, 12, 91, 110]) * 6


def kf(frame, distance=-30.0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
       fov=30, perspective=0, interp_block=NONLINEAR_INTERP):
    return CameraKey(frame, distance, center, rotation, interp_block, fov, perspective)


MOVING_CONSTANT_FOV = [
    kf(0, distance=-30.0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), fov=30, perspective=0),
    kf(30, distance=-25.0, center=(10.0, 5.0, 2.0), rotation=(0.2, 0.1, 0.0), fov=30, perspective=1),
    kf(60, distance=-20.0, center=(20.0, 0.0, -3.0), rotation=(-0.1, 0.3, 0.05), fov=30, perspective=1),
]

CUT_AT_31 = [
    kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)),
    kf(30, center=(10.0, 5.0, 2.0), rotation=(0.2, 0.1, 0.0)),
    kf(31, center=(40.0, 5.0, 2.0), rotation=(0.2, 0.1, 0.0)),
    kf(60, center=(50.0, 0.0, -3.0), rotation=(-0.1, 0.3, 0.05)),
]

_LIN = bake.LINEAR_CAMERA_INTERP


PAN_STOP_AT_30 = [
    kf(0, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(30, rotation=(0.0, 0.5, 0.0), interp_block=_LIN),
    kf(60, rotation=(0.0, 0.5, 0.0), interp_block=_LIN),
]

PAN_UNTIL_CUT_AT_30_THEN_HOLD = [
    kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(29, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.5, 0.0), interp_block=_LIN),
    kf(30, center=(40.0, 0.0, 0.0), rotation=(0.0, 0.5, 0.0), interp_block=_LIN),
    kf(60, center=(40.0, 0.0, 0.0), rotation=(0.0, 0.5, 0.0), interp_block=_LIN),
]

POSITION_PAN_STOP_AT_30 = [
    kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(30, center=(20.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(60, center=(20.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
]

PAN_STOP_AT_50 = [
    kf(0, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(50, rotation=(0.0, 0.5, 0.0), interp_block=_LIN),
    kf(60, rotation=(0.0, 0.5, 0.0), interp_block=_LIN),
]

STATIC = [
    kf(0, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(60, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
]

STATIC_KEYED_EVERY_20 = [
    kf(0, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(20, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(40, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(60, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
]


def by_frame(result):
    return {k.frame: k for k in result.camera_keys}


def key_tuple(k):
    return (k.frame, k.distance, tuple(k.position), tuple(k.rotation),
            bytes(k.interpolation), k.fov, k.perspective)


def rot_dev(res_map, frame):
    s = interp.sample_camera(MOVING_CONSTANT_FOV, frame)
    return float(np.linalg.norm(np.array(res_map[frame].rotation) - np.array(s["rotation"])))


class TestBake:
    def test_full_range_bakes_every_frame(self):
        res = bake.bake(MOVING_CONSTANT_FOV, seed=1)
        frames = sorted(k.frame for k in res.camera_keys)
        assert frames == list(range(0, 61))

    def test_baked_keys_use_linear_interp_even_for_nonlinear_input(self):
        res = bake.bake(MOVING_CONSTANT_FOV, seed=1)
        for k in res.camera_keys:
            assert k.interpolation == bake.LINEAR_CAMERA_INTERP

    def test_empty_camera_keys_raises(self):
        with pytest.raises(ValueError):
            bake.bake([], seed=1)

    def test_unsorted_input_uses_normalized_view(self):
        shuffled = [MOVING_CONSTANT_FOV[2], MOVING_CONSTANT_FOV[0], MOVING_CONSTANT_FOV[1]]
        a = [key_tuple(k) for k in bake.bake(MOVING_CONSTANT_FOV, seed=1, amp_rot=8.0, amp_pos=1.0).camera_keys]
        b = [key_tuple(k) for k in bake.bake(shuffled, seed=1, amp_rot=8.0, amp_pos=1.0).camera_keys]
        assert a == b

    def test_duplicate_frame_last_wins(self):
        dup_late = kf(30, distance=-22.0, center=(99.0, 99.0, 99.0),
                      rotation=(0.5, -0.5, 0.2), fov=20, perspective=0)
        keys = [MOVING_CONSTANT_FOV[0], MOVING_CONSTANT_FOV[1], dup_late, MOVING_CONSTANT_FOV[2]]
        res = by_frame(bake.bake(keys, seed=1, amp_rot=0.0, amp_pos=0.0))
        assert res[30].position == pytest.approx((99.0, 99.0, 99.0), abs=1e-6)
        assert res[30].rotation == pytest.approx((0.5, -0.5, 0.2), abs=1e-6)
        assert res[30].distance == pytest.approx(-22.0, abs=1e-6)
        assert res[30].fov == 20
        assert res[30].perspective == 0

    def test_duplicate_frame_warns_and_clean_input_does_not(self):
        dup = kf(30, center=(99.0, 99.0, 99.0))
        res_dup = bake.bake([MOVING_CONSTANT_FOV[0], MOVING_CONSTANT_FOV[1], dup, MOVING_CONSTANT_FOV[2]], seed=1)
        res_clean = bake.bake(MOVING_CONSTANT_FOV, seed=1)
        assert res_dup.warnings
        assert not res_clean.warnings

    def test_reproducible_full_record(self):
        a = bake.bake(MOVING_CONSTANT_FOV, seed=7, amp_rot=8.0, amp_pos=1.0)
        b = bake.bake(MOVING_CONSTANT_FOV, seed=7, amp_rot=8.0, amp_pos=1.0)
        assert [key_tuple(k) for k in a.camera_keys] == [key_tuple(k) for k in b.camera_keys]
        assert a.warnings == b.warnings

    def test_different_seed_differs(self):
        r1 = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=1, fade_sec=0.1))
        r2 = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=2, fade_sec=0.1))
        assert r1[30].rotation != pytest.approx(r2[30].rotation, abs=1e-9)

    def test_zero_amplitude_matches_sampled_all_channels(self):
        res = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0))
        for f in range(0, 61):
            s = interp.sample_camera(MOVING_CONSTANT_FOV, f)
            assert res[f].position == pytest.approx(s["position"], abs=1e-6)
            assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-6)
            assert res[f].distance == pytest.approx(s["distance"], abs=1e-6)
            assert res[f].fov == bake.round_half_up(s["fov"])
            assert res[f].interpolation == bake.LINEAR_CAMERA_INTERP

    def test_fade_zero_at_range_ends(self):
        res = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=1, amp_rot=30.0, amp_pos=5.0, fade_sec=0.7))
        for f in (0, 60):
            s = interp.sample_camera(MOVING_CONSTANT_FOV, f)
            assert res[f].position == pytest.approx(s["position"], abs=1e-4)
            assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-4)

    def test_fade_keeps_shake_and_its_velocity_small_at_range_ends(self):
        res = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.7))
        devs = [rot_dev(res, f) for f in range(0, 61)]
        vel = [abs(devs[f + 1] - devs[f]) for f in range(0, 60)]
        mid_max = max(devs[20:41])
        mid_vel_max = max(vel[20:40])
        assert mid_max > 1e-3
        assert max(devs[0:3]) < 0.2 * mid_max
        assert max(devs[58:61]) < 0.2 * mid_max
        assert mid_vel_max > 1e-4
        assert max(vel[0:2]) < 0.25 * mid_vel_max
        assert max(vel[58:60]) < 0.25 * mid_vel_max

    def test_midrange_actually_shakes(self):
        res = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.3))
        s = interp.sample_camera(MOVING_CONSTANT_FOV, 30)
        assert res[30].rotation != pytest.approx(s["rotation"], abs=1e-3)

    def test_short_range_shortens_fade_with_warning(self):
        res = bake.bake(MOVING_CONSTANT_FOV, ranges=[(0, 30)], seed=1, amp_rot=20.0, amp_pos=3.0, fade_sec=0.7)
        rm = by_frame(res)
        assert res.warnings
        for f in (0, 30):
            s = interp.sample_camera(MOVING_CONSTANT_FOV, f)
            assert rm[f].rotation == pytest.approx(s["rotation"], abs=1e-4)
            assert rm[f].position == pytest.approx(s["position"], abs=1e-4)

    def test_amp_rot_is_degrees_not_radians(self):
        res = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.3))
        peak = max(rot_dev(res, f) for f in range(10, 51))
        assert 5e-3 < peak < 1.0

    def test_zero_amp_pos_keeps_camera_world_path(self):
        res = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.3))
        for f in range(1, 60):
            s = interp.sample_camera(MOVING_CONSTANT_FOV, f)
            orig = kf(f, distance=s["distance"], center=s["position"], rotation=s["rotation"])
            p_orig = np.array(camera.to_world(orig).position)
            p_baked = np.array(camera.to_world(res[f]).position)
            assert p_baked == pytest.approx(p_orig, abs=1e-4)

    def test_position_noise_shifts_camera_world(self):
        res = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=1, amp_rot=0.0, amp_pos=3.0, fade_sec=0.3))

        def world_dev(f):
            s = interp.sample_camera(MOVING_CONSTANT_FOV, f)
            orig = kf(f, distance=s["distance"], center=s["position"], rotation=s["rotation"])
            return float(np.linalg.norm(
                np.array(camera.to_world(res[f]).position)
                - np.array(camera.to_world(orig).position)))

        assert max(world_dev(f) for f in range(10, 51)) > 1e-2
        assert world_dev(0) < 1e-4 and world_dev(60) < 1e-4

    def test_motion_adaptation_responds_to_rotation_only(self):
        keys = [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), distance=-30.0),
                kf(60, center=(0.0, 0.0, 0.0), rotation=(0.0, 1.0, 0.0), distance=-30.0)]
        a = by_frame(bake.bake(keys, seed=1, amp_rot=5.0, amp_pos=0.0, motion_damp=0.0, fade_sec=0.2))
        b = by_frame(bake.bake(keys, seed=1, amp_rot=5.0, amp_pos=0.0, motion_damp=10.0, fade_sec=0.2))
        assert a[30].rotation != pytest.approx(b[30].rotation, abs=1e-4)

    def test_motion_adaptation_responds_to_distance_only(self):
        keys = [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), distance=-50.0),
                kf(60, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), distance=-10.0)]
        a = by_frame(bake.bake(keys, seed=1, amp_rot=5.0, amp_pos=0.0, motion_damp=0.0, fade_sec=0.2))
        b = by_frame(bake.bake(keys, seed=1, amp_rot=5.0, amp_pos=0.0, motion_damp=10.0, fade_sec=0.2))
        assert a[30].rotation != pytest.approx(b[30].rotation, abs=1e-4)

    def _settle_rdev(self, fixture, res, f):
        s = interp.sample_camera(fixture, f)
        return np.array(res[f].rotation) - np.array(s["rotation"])

    def test_settle_adds_decaying_oscillation_in_pan_direction_after_stop(self):
        res = by_frame(bake.bake(PAN_STOP_AT_30, seed=1, amp_rot=0.0, amp_pos=0.0,
                                 settle=5.0, fade_sec=0.1))
        dev = {f: self._settle_rdev(PAN_STOP_AT_30, res, f) for f in range(0, 61)}

        def mag(rng):
            return max(float(np.linalg.norm(dev[f])) for f in rng)

        assert mag(range(33, 50)) > 1e-3
        peak_f = max(range(33, 50), key=lambda f: abs(dev[f][1]))
        assert abs(dev[peak_f][1]) > 5 * max(abs(dev[peak_f][0]), abs(dev[peak_f][2]))
        ry = [dev[f][1] for f in range(31, 56)]
        early_peak_f = max(range(31, 38), key=lambda f: abs(dev[f][1]))
        assert dev[early_peak_f][1] > 0
        assert max(ry) > 1e-3 and min(ry) < -1e-3
        assert 0.01 < mag(range(31, 45)) < 0.3
        assert mag(range(33, 43)) > mag(range(50, 56))
        assert mag(range(50, 56)) < 0.02
        assert mag(range(1, 25)) < 1e-4

    def test_settle_zero_disables_only_settle(self):
        res0 = by_frame(bake.bake(PAN_STOP_AT_30, seed=1, amp_rot=0.0, amp_pos=0.0,
                                  settle=0.0, fade_sec=0.1))
        for f in range(0, 61):
            s = interp.sample_camera(PAN_STOP_AT_30, f)
            assert res0[f].rotation == pytest.approx(s["rotation"], abs=1e-6)
        resn = by_frame(bake.bake(PAN_STOP_AT_30, seed=1, amp_rot=10.0, amp_pos=0.0,
                                  settle=0.0, motion_damp=0.0, fade_sec=0.1))
        s30 = interp.sample_camera(PAN_STOP_AT_30, 30)
        assert resn[30].rotation != pytest.approx(s30["rotation"], abs=1e-3)

    def test_settle_is_additive_to_base_noise(self):
        base = by_frame(bake.bake(PAN_STOP_AT_30, seed=1, amp_rot=8.0, amp_pos=0.0, settle=0.0, fade_sec=0.1))
        only = by_frame(bake.bake(PAN_STOP_AT_30, seed=1, amp_rot=0.0, amp_pos=0.0, settle=5.0, fade_sec=0.1))
        both = by_frame(bake.bake(PAN_STOP_AT_30, seed=1, amp_rot=8.0, amp_pos=0.0, settle=5.0, fade_sec=0.1))
        settle_seen = False
        for f in range(33, 50):
            s = np.array(interp.sample_camera(PAN_STOP_AT_30, f)["rotation"])
            dev_base = np.array(base[f].rotation) - s
            dev_only = np.array(only[f].rotation) - s
            dev_both = np.array(both[f].rotation) - s
            assert dev_both == pytest.approx(dev_base + dev_only, abs=1e-9)
            if float(np.linalg.norm(dev_only)) > 1e-3:
                settle_seen = True
        assert settle_seen

    def test_settle_faded_at_range_end(self):
        res = by_frame(bake.bake(PAN_STOP_AT_50, seed=1, amp_rot=0.0, amp_pos=0.0,
                                 settle=10.0, fade_sec=0.2))

        def mag(rng):
            return max(
                float(np.linalg.norm(
                    np.array(res[f].rotation) - np.array(interp.sample_camera(PAN_STOP_AT_50, f)["rotation"])))
                for f in rng)

        assert mag(range(51, 55)) > 1e-3
        s60 = interp.sample_camera(PAN_STOP_AT_50, 60)
        assert res[60].rotation == pytest.approx(s60["rotation"], abs=1e-4)

    def test_settle_is_angular_not_positional(self):
        res = by_frame(bake.bake(POSITION_PAN_STOP_AT_30, seed=1, amp_rot=0.0, amp_pos=0.0,
                                 settle=10.0, fade_sec=0.1))
        for f in range(0, 61):
            s = interp.sample_camera(POSITION_PAN_STOP_AT_30, f)
            assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-6)

    def test_settle_not_triggered_at_cut(self):
        res = by_frame(bake.bake(PAN_UNTIL_CUT_AT_30_THEN_HOLD, seed=1, amp_rot=0.0, amp_pos=0.0,
                                 settle=10.0, fade_sec=0.1))
        for f in range(30, 45):
            s = interp.sample_camera(PAN_UNTIL_CUT_AT_30_THEN_HOLD, f)
            assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-4)

    def _imp_dev(self, res, f):
        s = interp.sample_camera(STATIC, f)
        return np.array(res[f].rotation) - np.array(s["rotation"])

    def _imp_mag(self, res, f):
        return float(np.linalg.norm(self._imp_dev(res, f)))

    def test_impulse_fires_from_frame_and_decays(self):
        res = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                                 impulses=[(30, 10.0, 0.5)], fade_sec=0.1))
        assert max(self._imp_mag(res, f) for f in range(5, 30)) < 1e-4
        assert max(self._imp_mag(res, f) for f in range(30, 45)) > 1e-3
        peak = max(self._imp_mag(res, f) for f in range(30, 35))
        assert 0.02 < peak < 0.6
        near_F = max(self._imp_mag(res, f) for f in range(30, 35))
        near_FD = max(self._imp_mag(res, f) for f in range(44, 49))
        assert 2.0 < near_F / max(near_FD, 1e-9) < 5.0
        ax = int(np.argmax(np.abs(self._imp_dev(res, 31))))
        comp = [self._imp_dev(res, f)[ax] for f in range(30, 45)]
        signs = [c > 0 for c in comp if abs(c) > 1e-4]
        flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        assert flips >= 3

    def test_omitted_or_empty_impulses_have_no_effect(self):
        for kw in ({}, {"impulses": ()}):
            res = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0,
                                     settle=0.0, fade_sec=0.1, **kw))
            for f in range(0, 61):
                s = interp.sample_camera(STATIC, f)
                assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-9)

    def test_impulse_is_reproducible_per_seed_and_changes_with_seed(self):
        a = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                               impulses=[(30, 10.0, 0.5)], fade_sec=0.1))
        b = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                               impulses=[(30, 10.0, 0.5)], fade_sec=0.1))
        c = by_frame(bake.bake(STATIC, seed=2, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                               impulses=[(30, 10.0, 0.5)], fade_sec=0.1))
        for f in range(30, 45):
            assert a[f].rotation == pytest.approx(b[f].rotation, abs=1e-12)
        assert any(a[f].rotation != pytest.approx(c[f].rotation, abs=1e-6) for f in range(30, 45))

    def test_impulses_additive(self):
        A = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                               impulses=[(30, 6.0, 1.0)], fade_sec=0.1))
        B = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                               impulses=[(36, 6.0, 1.0)], fade_sec=0.1))
        AB = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                                impulses=[(30, 6.0, 1.0), (36, 6.0, 1.0)], fade_sec=0.1))
        overlap_seen = False
        for f in range(36, 50):
            assert self._imp_dev(AB, f) == pytest.approx(
                self._imp_dev(A, f) + self._imp_dev(B, f), abs=1e-9)
            if self._imp_mag(A, f) > 1e-3 and self._imp_mag(B, f) > 1e-3:
                overlap_seen = True
        assert overlap_seen

    def test_impulse_faded_at_range_end(self):
        res = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                                 impulses=[(55, 10.0, 0.5)], fade_sec=0.2))
        assert max(self._imp_mag(res, f) for f in range(55, 58)) > 1e-3
        s60 = interp.sample_camera(STATIC, 60)
        assert res[60].rotation == pytest.approx(s60["rotation"], abs=1e-4)

    def test_impulse_after_range_has_no_effect_and_before_range_leaves_tail(self):
        def mags(impulses):
            res = by_frame(bake.bake(STATIC_KEYED_EVERY_20, ranges=[(20, 40)], seed=1, amp_rot=0.0,
                                     amp_pos=0.0, settle=0.0, impulses=impulses, fade_sec=0.1))
            return [float(np.linalg.norm(
                np.array(res[f].rotation) - np.array(interp.sample_camera(STATIC_KEYED_EVERY_20, f)["rotation"])))
                for f in range(21, 40)]
        assert max(mags([(50, 10.0, 1.0)])) < 1e-4
        assert max(mags([(10, 10.0, 1.0)])) > 1e-3

    def test_impulse_nonpositive_params_no_effect(self):
        for imp in [(30, 0.0, 0.5), (30, 10.0, 0.0), (30, -5.0, 0.5)]:
            res = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                                     impulses=[imp], fade_sec=0.1))
            for f in range(0, 61):
                s = interp.sample_camera(STATIC, f)
                assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-9)

    def test_fov_equals_rounded_sample_when_constant(self):
        res = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=1, amp_rot=20.0, amp_pos=3.0, fade_sec=0.3))
        for f in range(0, 61):
            s = interp.sample_camera(MOVING_CONSTANT_FOV, f)
            assert isinstance(res[f].fov, int)
            assert res[f].fov == bake.round_half_up(s["fov"])

    @pytest.mark.parametrize("value,expected", [
        pytest.param(36.5, 37, id="even_below"),
        pytest.param(37.5, 38, id="odd_below"),
        pytest.param(36.49, 36, id="just_below_half"),
    ])
    def test_round_half_up_rounds_point_five_up(self, value, expected):
        assert bake.round_half_up(value) == expected

    def test_fov_change_over_two_frames_keeps_original_keys(self):
        keys = [
            kf(0, fov=36, interp_block=bake.LINEAR_CAMERA_INTERP),
            kf(2, fov=37, interp_block=bake.LINEAR_CAMERA_INTERP),
        ]
        res = bake.bake(keys, seed=1, amp_rot=5.0, amp_pos=1.0)
        frames = [k.frame for k in res.camera_keys]
        by = {k.frame: k for k in res.camera_keys}
        assert 1 not in by
        assert frames.count(0) == 1 and frames.count(2) == 1
        assert by[0] == keys[0] and by[2] == keys[1]

    def test_perspective_holds_latest_preceding_key(self):
        res = by_frame(bake.bake(MOVING_CONSTANT_FOV, seed=1))
        assert res[10].perspective == 0
        assert res[29].perspective == 0
        assert res[30].perspective == 1
        assert res[45].perspective == 1

    def _fov_30_to_18_between_20_and_50(self):
        L = bake.LINEAR_CAMERA_INTERP
        return [kf(0, fov=30, interp_block=L), kf(20, fov=30, interp_block=L),
                kf(50, fov=18, interp_block=L), kf(70, fov=18, interp_block=L)]

    def test_fov_constant_spans_densely_baked_with_shake(self):
        seq = self._fov_30_to_18_between_20_and_50()
        res = by_frame(bake.bake(seq, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.1))
        for f in range(2, 19):
            assert res[f].fov == 30
            s = interp.sample_camera(seq, f)
            assert res[f].rotation != pytest.approx(s["rotation"], abs=1e-4)
        for f in range(52, 69):
            assert res[f].fov == 18
            s = interp.sample_camera(seq, f)
            assert res[f].rotation != pytest.approx(s["rotation"], abs=1e-4)

    def test_fov_ramp_span_preserves_originals_no_interior_keys(self):
        seq = self._fov_30_to_18_between_20_and_50()
        res = bake.bake(seq, seed=1, amp_rot=20.0, amp_pos=5.0, fade_sec=0.1)
        frames = [k.frame for k in res.camera_keys]
        by = {k.frame: k for k in res.camera_keys}
        assert not any(21 <= f <= 49 for f in by)
        assert frames.count(20) == 1 and frames.count(50) == 1
        orig = {k.frame: k for k in seq}
        assert by[20] == orig[20] and by[50] == orig[50]
        assert all(k.fov in (30, 18) for k in res.camera_keys)

    def test_single_frame_fov_jump_is_baked_not_preserved(self):
        N = NONLINEAR_INTERP
        seq = [kf(0, fov=30, interp_block=N),
               kf(20, fov=30, interp_block=N),
               kf(21, fov=36, interp_block=N),
               kf(40, fov=36, interp_block=N)]
        res = bake.bake(seq, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.1)
        frames = [k.frame for k in res.camera_keys]
        by = {k.frame: k for k in res.camera_keys}
        for f in (20, 21):
            assert frames.count(f) == 1
            assert by[f].interpolation == bake.LINEAR_CAMERA_INTERP
            s = interp.sample_camera(seq, f)
            assert by[f].rotation != pytest.approx(s["rotation"], abs=1e-4)
        assert by[20].fov == 30 and by[21].fov == 36

    def test_default_range_is_full_span(self):
        res = bake.bake(MOVING_CONSTANT_FOV, ranges=None, seed=1)
        frames = sorted(k.frame for k in res.camera_keys)
        assert frames[0] == 0 and frames[-1] == 60

    def test_range_bakes_only_inside(self):
        res = bake.bake(MOVING_CONSTANT_FOV, ranges=[(30, 60)], seed=1)
        baked_frames = sorted(k.frame for k in res.camera_keys if k.frame >= 30)
        assert baked_frames == list(range(30, 61))

    def test_out_of_range_leading_key_preserved(self):
        res = bake.bake(MOVING_CONSTANT_FOV, ranges=[(30, 60)], seed=1)
        out = {k.frame: k for k in res.camera_keys if k.frame < 30}
        assert list(out) == [0]
        assert key_tuple(out[0]) == key_tuple(MOVING_CONSTANT_FOV[0])

    def test_out_of_range_trailing_key_preserved(self):
        res = bake.bake(MOVING_CONSTANT_FOV, ranges=[(0, 30)], seed=1)
        out = {k.frame: k for k in res.camera_keys if k.frame > 30}
        assert list(out) == [60]
        assert key_tuple(out[60]) == key_tuple(MOVING_CONSTANT_FOV[2])

    def test_out_of_range_duplicates_and_order_are_kept_as_original_records(self):
        first = MOVING_CONSTANT_FOV[0]
        duplicate = kf(0, center=(9.0, 9.0, 9.0))
        keys = [MOVING_CONSTANT_FOV[2], first, duplicate, MOVING_CONSTANT_FOV[1]]
        res = bake.bake(keys, ranges=[(30, 60)], seed=1)
        out = [key_tuple(k) for k in res.camera_keys if k.frame < 30]
        assert out == [key_tuple(first), key_tuple(duplicate)]

    def test_duplicates_at_snapped_range_start_all_fall_inside_and_bake_to_one_key(self):
        duplicate = kf(30, center=(9.0, 9.0, 9.0))
        keys = [MOVING_CONSTANT_FOV[2], duplicate, MOVING_CONSTANT_FOV[0], MOVING_CONSTANT_FOV[1]]
        res = bake.bake(keys, ranges=[(25, 60)], seed=1)
        assert res.resolved == [(30, 60)]
        at_start = [k for k in res.camera_keys if k.frame == 30]
        assert len(at_start) == 1
        assert at_start[0].interpolation == bake.LINEAR_CAMERA_INTERP
        assert [key_tuple(k) for k in res.camera_keys if k.frame < 30] == [key_tuple(MOVING_CONSTANT_FOV[0])]

    def test_output_is_frame_ordered(self):
        res = bake.bake(MOVING_CONSTANT_FOV, ranges=[(30, 60)], seed=1)
        frames = [k.frame for k in res.camera_keys]
        assert frames == sorted(frames)
        assert frames[0] == 0

    def test_range_snaps_to_nearest_existing_key(self):
        res = bake.bake(MOVING_CONSTANT_FOV, ranges=[(25, 55)], seed=1)
        frames = sorted(k.frame for k in res.camera_keys)
        assert frames == [0] + list(range(30, 61))
        out0 = next(k for k in res.camera_keys if k.frame == 0)
        assert key_tuple(out0) == key_tuple(MOVING_CONSTANT_FOV[0])

    def test_overlapping_ranges_raise(self):
        with pytest.raises(ValueError):
            bake.bake(MOVING_CONSTANT_FOV, ranges=[(0, 60), (30, 60)], seed=1)

    def test_multiple_ranges_bake_each_and_keep_gap_originals(self):
        seq5 = [
            kf(0, center=(0.0, 0.0, 0.0)),
            kf(15, center=(5.0, 1.0, 0.0)),
            kf(30, center=(10.0, 5.0, 2.0), fov=45, perspective=1),
            kf(45, center=(15.0, 2.0, -1.0)),
            kf(60, center=(20.0, 0.0, -3.0)),
        ]
        res = bake.bake(seq5, ranges=[(0, 15), (45, 60)], seed=1)
        frames = sorted(k.frame for k in res.camera_keys)
        assert frames == list(range(0, 16)) + [30] + list(range(45, 61))
        assert not (set(range(16, 30)) & set(frames))
        assert not (set(range(31, 45)) & set(frames))
        out30 = next(k for k in res.camera_keys if k.frame == 30)
        assert key_tuple(out30) == key_tuple(seq5[2])

    def test_first_frame_after_cut_keeps_shake_without_fade_or_cut_speed(self):
        res = bake.bake(CUT_AT_31, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.2)
        rm = by_frame(res)
        s = interp.sample_camera(CUT_AT_31, 31)
        dev = float(np.linalg.norm(np.array(rm[31].rotation) - np.array(s["rotation"])))
        assert dev > 1e-3

    def test_cut_makes_segment_phase_independent(self):
        with_cut = by_frame(bake.bake(CUT_AT_31, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.2))
        no_cut = by_frame(bake.bake(CUT_AT_31, seed=1, amp_rot=10.0, amp_pos=0.0,
                                    fade_sec=0.2, manual_cuts_remove=[31]))
        assert with_cut[45].rotation != pytest.approx(no_cut[45].rotation, abs=1e-9)

    def test_cut_jump_is_preserved_in_dense_output(self):
        res = by_frame(bake.bake(CUT_AT_31, seed=1, amp_rot=0.0, amp_pos=0.0))
        jump = np.array(res[31].position) - np.array(res[30].position)
        expected = np.array(CUT_AT_31[2].position) - np.array(CUT_AT_31[1].position)
        assert jump == pytest.approx(expected, abs=1e-6)

    def test_cut_segments_shorter_than_two_fades_do_not_shorten_fade(self):
        res = bake.bake(CUT_AT_31, seed=1, fade_sec=0.7)
        assert "fade_shortened" not in {w.code for w in res.warnings}


class TestProfileCrossfadeAndBreathing:
    BREATH_CYCLE_END = 99
    _LIN = bytes([20, 107, 20, 107]) * 6

    def _static_input(self):
        return [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.BREATH_CYCLE_END, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN)]

    def _moving_input(self):
        return [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.BREATH_CYCLE_END, center=(60.0, 0.0, 0.0), rotation=(0.0, 1.5, 0.0), interp_block=self._LIN)]

    def _rot_shake(self, res, src, axis):
        return np.array([res[f].rotation[axis] - interp.sample_camera(src, f)["rotation"][axis]
                         for f in range(1, self.BREATH_CYCLE_END)])

    def _shake_series(self, res, src, kind, axis, end):
        key = "rotation" if kind == "rot" else "position"
        return np.array([getattr(res[f], key)[axis] - interp.sample_camera(src, f)[key][axis]
                         for f in range(end + 1)])

    @staticmethod
    def _hf_ratio(series):
        return float(np.std(np.diff(series)) / (np.std(series) + 1e-12))

    def test_moving_has_more_high_freq_than_still(self):
        src_s, src_m = self._static_input(), self._moving_input()
        rst = by_frame(bake.bake(src_s, seed=1, amp_rot=5.0, amp_pos=0.0, settle=0.0, fade_sec=0.3))
        rmv = by_frame(bake.bake(src_m, seed=1, amp_rot=5.0, amp_pos=0.0, settle=0.0, fade_sec=0.3))
        hf = self._hf_ratio
        for ax in range(3):
            assert (hf(self._rot_shake(rmv, src_m, ax))
                    > hf(self._rot_shake(rst, src_s, ax))), f"rot{ax}"

    def test_per_frame_crossfade_within_single_segment(self):
        L = self._LIN
        decel = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(50, center=(60.0, 0.0, 0.0), interp_block=L),
                 kf(self.BREATH_CYCLE_END, center=(60.0, 0.0, 0.0), interp_block=L)]
        accel = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(50, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(self.BREATH_CYCLE_END, center=(60.0, 0.0, 0.0), interp_block=L)]
        LATE, EARLY = slice(58, 88), slice(12, 42)
        for ax in range(3):
            late_a, late_d, early_a, early_d = [], [], [], []
            for seed in range(3):
                d = by_frame(bake.bake(decel, seed=seed, amp_rot=5.0, amp_pos=0.0,
                                       motion_damp=0.0, settle=0.0, fade_sec=0.3))
                a = by_frame(bake.bake(accel, seed=seed, amp_rot=5.0, amp_pos=0.0,
                                       motion_damp=0.0, settle=0.0, fade_sec=0.3))
                ds = self._shake_series(d, decel, "rot", ax, self.BREATH_CYCLE_END)
                as_ = self._shake_series(a, accel, "rot", ax, self.BREATH_CYCLE_END)
                late_a.append(self._hf_ratio(as_[LATE]))
                late_d.append(self._hf_ratio(ds[LATE]))
                early_a.append(self._hf_ratio(as_[EARLY]))
                early_d.append(self._hf_ratio(ds[EARLY]))
            assert np.mean(late_a) > np.mean(late_d), f"late rot{ax}"
            assert np.mean(early_d) > np.mean(early_a), f"early rot{ax}"

    def test_crossfade_is_speed_proportional_not_binary(self):
        L, REF = self._LIN, 1.0
        s0 = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
              kf(self.BREATH_CYCLE_END, center=(0.0, 0.0, 0.0), interp_block=L)]
        shalf = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(self.BREATH_CYCLE_END, center=(0.5 * self.BREATH_CYCLE_END, 0.0, 0.0), interp_block=L)]
        sfull = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(self.BREATH_CYCLE_END, center=(1.0 * self.BREATH_CYCLE_END, 0.0, 0.0), interp_block=L)]
        W = slice(58, 88)

        def hf_rot(src, seed):
            res = by_frame(bake.bake(src, seed=seed, amp_rot=5.0, amp_pos=0.0,
                                     motion_damp=0.0, speed_ref_world=REF,
                                     settle=0.0, fade_sec=0.3))
            return self._hf_ratio(self._shake_series(res, src, "rot", 0, self.BREATH_CYCLE_END)[W])

        h0 = [hf_rot(s0, s) for s in range(3)]
        hh = [hf_rot(shalf, s) for s in range(3)]
        hful = [hf_rot(sfull, s) for s in range(3)]
        assert np.mean(hful) > np.mean(hh) > np.mean(h0)

    def _breath_energy_axis(self, res, src, kind, axis, end, win=slice(None)):
        series = self._shake_series(res, src, kind, axis, end)[win]
        freqs = np.fft.rfftfreq(series.size, d=1.0 / 30.0)
        k = int(np.argmin(np.abs(freqs - 0.3)))
        return float(np.abs(np.fft.rfft(series))[k])

    def _breath_energy(self, res, src, kind, end, win=slice(None)):
        return float(sum(self._breath_energy_axis(res, src, kind, ax, end, win) for ax in range(3)))

    def _octave_band_ratio(self, res, src, axis, end, kind="pos"):
        series = self._shake_series(res, src, kind, axis, end)
        spec = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(series.size, d=1.0 / 30.0)
        low = spec[(freqs > 0.6) & (freqs <= 1.8)].sum()
        high = spec[freqs > 1.8].sum()
        return float(high / (low + 1e-12))

    def test_crossfade_applies_to_position_via_octave_band_ratio(self):
        src_s, src_m = self._static_input(), self._moving_input()
        st = by_frame(bake.bake(src_s, seed=1, amp_rot=0.0, amp_pos=1.0, freq=1.2, settle=0.0, fade_sec=0.3))
        mv = by_frame(bake.bake(src_m, seed=1, amp_rot=0.0, amp_pos=1.0, freq=1.2, settle=0.0, fade_sec=0.3))
        end = self.BREATH_CYCLE_END
        for ax in range(3):
            assert self._octave_band_ratio(mv, src_m, ax, end) > self._octave_band_ratio(st, src_s, ax, end), f"pos{ax}"

    def test_breathing_appears_in_static_position_only_and_tremor_remains(self):
        src_s, src_m = self._static_input(), self._moving_input()
        pst = by_frame(bake.bake(src_s, seed=1, amp_rot=0.0, amp_pos=1.0, motion_damp=0.0, settle=0.0, fade_sec=0.3))
        pmv = by_frame(bake.bake(src_m, seed=1, amp_rot=0.0, amp_pos=1.0, motion_damp=0.0, settle=0.0, fade_sec=0.3))
        rst = by_frame(bake.bake(src_s, seed=1, amp_rot=5.0, amp_pos=0.0, freq=1.2,
                                 motion_damp=0.0, settle=0.0, fade_sec=0.3))
        rmv = by_frame(bake.bake(src_m, seed=1, amp_rot=5.0, amp_pos=0.0, freq=1.2,
                                 motion_damp=0.0, settle=0.0, fade_sec=0.3))
        end = self.BREATH_CYCLE_END
        assert self._breath_energy(pst, src_s, "pos", end) > 3.0 * self._breath_energy(pmv, src_m, "pos", end)
        assert self._breath_energy(rst, src_s, "rot", end) < 2.0 * self._breath_energy(rmv, src_m, "rot", end)
        for ax in range(3):
            assert self._octave_band_ratio(rst, src_s, ax, end, "rot") > 0.1, f"tremor rot{ax}"

    def test_breathing_scales_with_inverse_speed_in_single_segment(self):
        END = 119
        L, REF = self._LIN, 2.0
        s0 = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
              kf(END, center=(0.0, 0.0, 0.0), interp_block=L)]
        shalf = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(END, center=(1.0 * END, 0.0, 0.0), interp_block=L)]
        sfull = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(END, center=(2.0 * END, 0.0, 0.0), interp_block=L)]
        win = slice(20, END + 1)

        def breath_energy(src):
            res = by_frame(bake.bake(src, seed=5, amp_rot=0.0, amp_pos=2.0,
                                     motion_damp=0.0, speed_ref_world=REF,
                                     settle=0.0, fade_sec=0.3))
            return self._breath_energy(res, src, "pos", END, win)

        assert breath_energy(s0) > breath_energy(shalf) > breath_energy(sfull)


class TestWalkingGait:
    N = 99
    _LIN = bytes([20, 107, 20, 107]) * 6

    def _static(self):
        return [kf(0, center=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.N, center=(0.0, 0.0, 0.0), interp_block=self._LIN)]

    def _pos(self, res, src, axis):
        return np.array([res[f].position[axis] - interp.sample_camera(src, f)["position"][axis]
                         for f in range(self.N + 1)])

    def _rot(self, res, src, axis):
        return np.array([res[f].rotation[axis] - interp.sample_camera(src, f)["rotation"][axis]
                         for f in range(self.N + 1)])

    @staticmethod
    def _dom_freq(series):
        spec = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(series.size, d=1.0 / 30.0)
        return float(freqs[1 + int(np.argmax(spec[1:]))])

    @staticmethod
    def _bin_amp(series, freq):
        spec = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(series.size, d=1.0 / 30.0)
        return float(spec[int(np.argmin(np.abs(freqs - freq)))])

    @staticmethod
    def _maxabs(series):
        return float(np.max(np.abs(series)))

    def test_gait_puts_f_on_lateral_and_2f_on_vertical_only(self):
        src = self._static()
        for f in (1.8, 2.4):
            res = by_frame(bake.bake(src, seed=1, amp_rot=0.0, amp_pos=0.0,
                                     gait_freq=f, gait_amp=1.0, settle=0.0, fade_sec=0.1))
            x, y, z = (self._pos(res, src, ax) for ax in range(3))
            assert abs(self._dom_freq(x) - f) < 0.16, (f, "x")
            assert abs(self._dom_freq(y) - 2 * f) < 0.16, (f, "y")
            assert self._bin_amp(x, f) > 5.0 * self._bin_amp(x, 2 * f), (f, "x-sep")
            assert self._bin_amp(y, 2 * f) > 5.0 * self._bin_amp(y, f), (f, "y-sep")
            assert self._maxabs(z) < 1e-6, (f, "z")

    def test_gait_amplitude_scales_with_gait_amp(self):
        src = self._static()

        def res_for(gait_amp):
            return by_frame(bake.bake(src, seed=1, amp_rot=0.0, amp_pos=0.0,
                                      gait_freq=1.8, gait_amp=gait_amp, settle=0.0, fade_sec=0.1))

        r0, r1, r2 = res_for(0.0), res_for(1.0), res_for(2.0)
        for axis in (0, 1):
            assert self._maxabs(self._pos(r0, src, axis)) < 1e-9, f"axis{axis} zero"
            s1 = float(np.std(self._pos(r1, src, axis)))
            s2 = float(np.std(self._pos(r2, src, axis)))
            assert s1 > 1e-6 and abs(s2 - 2.0 * s1) < 0.005 * s1, f"axis{axis} scale"

    def test_gait_not_applied_to_rotation(self):
        src = self._static()
        res = by_frame(bake.bake(src, seed=1, amp_rot=0.0, amp_pos=0.0,
                                 gait_freq=1.8, gait_amp=1.0, settle=0.0, fade_sec=0.1))
        for ax in range(3):
            assert self._maxabs(self._rot(res, src, ax)) < 1e-9, f"rot{ax}"

    def test_gait_mixed_with_random_noise_not_replacing(self):
        src = self._static()
        kw = dict(amp_rot=2.0, amp_pos=1.0, settle=0.0, fade_sec=0.1)
        on = by_frame(bake.bake(src, seed=2, gait_freq=1.8, gait_amp=0.2, **kw))
        off = by_frame(bake.bake(src, seed=2, gait_freq=0.0, gait_amp=0.2, **kw))
        on_x, off_x = self._pos(on, src, 0), self._pos(off, src, 0)
        on_y, off_y = self._pos(on, src, 1), self._pos(off, src, 1)
        assert abs(self._dom_freq(on_x - off_x) - 1.8) < 0.16, ("x", self._dom_freq(on_x - off_x))
        assert abs(self._dom_freq(on_y - off_y) - 3.6) < 0.16, ("y", self._dom_freq(on_y - off_y))
        assert float(np.corrcoef(on_x, off_x)[0, 1]) > 0.5
        assert float(np.corrcoef(on_y, off_y)[0, 1]) > 0.5
        assert np.allclose(self._pos(on, src, 2), self._pos(off, src, 2), atol=1e-12)
        for ax in range(3):
            assert np.allclose(self._rot(on, src, ax), self._rot(off, src, ax), atol=1e-12), f"rot{ax}"

    def test_gait_off_when_freq_zero(self):
        src = self._static()
        r_default = by_frame(bake.bake(src, seed=1, amp_rot=0.0, amp_pos=0.0,
                                       gait_amp=1.0, settle=0.0, fade_sec=0.1))
        r_explicit = by_frame(bake.bake(src, seed=1, amp_rot=0.0, amp_pos=0.0,
                                        gait_freq=0.0, gait_amp=1.0, settle=0.0, fade_sec=0.1))
        for res in (r_default, r_explicit):
            assert self._maxabs(self._pos(res, src, 0)) < 1e-9
            assert self._maxabs(self._pos(res, src, 1)) < 1e-9


class TestCoreApiTuning:
    N = 99
    _LIN = bytes([20, 107, 20, 107]) * 6

    def _static(self):
        return [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.N, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN)]

    def _moving(self):
        return [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.N, center=(60.0, 0.0, 0.0), rotation=(0.0, 1.5, 0.0), interp_block=self._LIN)]

    def _rot_hf(self, res, src, axis=0):
        r = np.array([res[f].rotation[axis] - interp.sample_camera(src, f)["rotation"][axis]
                      for f in range(12, self.N - 11)])
        return float(np.std(np.diff(r)) / (np.std(r) + 1e-12))

    def _pos_hf(self, res, src, axis=0):
        p = np.array([res[f].position[axis] - interp.sample_camera(src, f)["position"][axis]
                      for f in range(12, self.N - 11)])
        return float(np.std(np.diff(p)) / (np.std(p) + 1e-12))

    def test_still_profile_tunes_static_octave_weights(self):
        src = self._static()
        common = dict(seed=1, amp_rot=5.0, amp_pos=0.0, settle=0.0, fade_sec=0.1,
                      moving_profile=motion.MOVING_PROFILE)
        hi = by_frame(bake.bake(src, still_profile=(1.0, 1.0, 1.0), **common))
        lo = by_frame(bake.bake(src, still_profile=(1.0, 0.05, 0.01), **common))
        for ax in range(3):
            assert self._rot_hf(hi, src, ax) > self._rot_hf(lo, src, ax), ax

    def test_moving_profile_tunes_moving_octave_weights(self):
        src = self._moving()
        common = dict(seed=1, amp_rot=5.0, amp_pos=0.0, motion_damp=0.0, settle=0.0, fade_sec=0.1,
                      still_profile=motion.STILL_PROFILE)
        hi = by_frame(bake.bake(src, moving_profile=(1.0, 1.0, 1.0), **common))
        lo = by_frame(bake.bake(src, moving_profile=(1.0, 0.05, 0.01), **common))
        for ax in range(3):
            assert self._rot_hf(hi, src, ax) > self._rot_hf(lo, src, ax), ax

    def test_profiles_apply_to_position_channels(self, monkeypatch):
        monkeypatch.setattr(motion, "BREATHING_AMP_FACTOR", 0.0)
        for src, vary in ((self._static(), "still_profile"), (self._moving(), "moving_profile")):
            common = dict(seed=1, amp_rot=0.0, amp_pos=1.0, settle=0.0, fade_sec=0.1)
            hi_kw = {"still_profile": motion.STILL_PROFILE, "moving_profile": motion.MOVING_PROFILE}
            lo_kw = dict(hi_kw)
            hi_kw[vary] = (1.0, 1.0, 1.0)
            lo_kw[vary] = (1.0, 0.05, 0.01)
            hi = by_frame(bake.bake(src, **common, **hi_kw))
            lo = by_frame(bake.bake(src, **common, **lo_kw))
            for ax in range(3):
                assert self._pos_hf(hi, src, ax) > self._pos_hf(lo, src, ax), (vary, ax)

    def test_octave_count_follows_profile_length(self):
        common = dict(seed=1, amp_rot=5.0, amp_pos=0.0, freq=1.2, settle=0.0, fade_sec=0.1)
        for src in (self._static(), self._moving()):
            o2 = by_frame(bake.bake(src, still_profile=(1.0, 1.0), moving_profile=(1.0, 1.0), **common))
            o3 = by_frame(bake.bake(src, still_profile=(1.0, 1.0, 1.0),
                                    moving_profile=(1.0, 1.0, 1.0), **common))
            for ax in range(3):
                assert self._rot_hf(o3, src, ax) > self._rot_hf(o2, src, ax), ax

    def test_settle_time_tunes_decay_rate_not_gain(self):
        def late_over_early(settle_time):
            res = by_frame(bake.bake(PAN_STOP_AT_30, seed=1, amp_rot=0.0, amp_pos=0.0,
                                     settle=2.0, settle_time=settle_time, fade_sec=0.1))
            dev = [res[f].rotation[1] - interp.sample_camera(PAN_STOP_AT_30, f)["rotation"][1]
                   for f in range(31, 58)]
            early = float(np.sum(np.array(dev[:13]) ** 2))
            late = float(np.sum(np.array(dev[14:]) ** 2))
            return late / (early + 1e-15)
        assert late_over_early(2.0) > late_over_early(0.3)

    def test_defaults_match_internal_constants(self):
        a = bake.bake(PAN_STOP_AT_30, seed=1, settle=2.0).camera_keys
        b = bake.bake(PAN_STOP_AT_30, seed=1, settle=2.0, still_profile=motion.STILL_PROFILE,
                      moving_profile=motion.MOVING_PROFILE,
                      settle_time=motion.DEFAULT_SETTLE_TIME_SEC).camera_keys
        assert a == b
        sig = inspect.signature(bake.bake).parameters
        assert sig["still_profile"].default is motion.STILL_PROFILE
        assert sig["moving_profile"].default is motion.MOVING_PROFILE
        assert sig["settle_time"].default == motion.DEFAULT_SETTLE_TIME_SEC

    def test_profile_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            bake.bake(self._static(), still_profile=(1.0, 0.5), moving_profile=(1.0, 0.5, 0.25))
        with pytest.raises(ValueError):
            bake.bake(self._static(), still_profile=(1.0, 0.5, 0.25), moving_profile=(1.0, 0.5))


class TestNaiveRotation:
    N = 60
    _LIN = bytes([20, 107, 20, 107]) * 6
    CENTER = (5.0, 3.0, 2.0)

    def _src(self):
        return [kf(0, distance=-30.0, center=self.CENTER, rotation=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.N, distance=-30.0, center=self.CENTER, rotation=(0.0, 0.0, 0.0), interp_block=self._LIN)]

    def test_naive_skips_center_rederivation(self):
        src = self._src()
        common = dict(seed=1, amp_rot=8.0, amp_pos=0.0, settle=0.0, fade_sec=0.1)
        nv = by_frame(bake.bake(src, naive_rotation=True, **common))
        df = by_frame(bake.bake(src, naive_rotation=False, **common))
        interior = range(12, self.N - 11)
        for f in interior:
            assert nv[f].position == pytest.approx(self.CENTER, abs=1e-6), f
            assert nv[f].rotation == pytest.approx(df[f].rotation, abs=1e-9), f
        moved = max(sum(abs(df[f].position[i] - self.CENTER[i]) for i in range(3)) for f in interior)
        assert moved > 1e-3

    def test_naive_still_applies_position_noise(self):
        src = self._src()
        common = dict(seed=1, amp_rot=0.0, amp_pos=0.5, settle=0.0, fade_sec=0.1)
        nv = by_frame(bake.bake(src, naive_rotation=True, **common))
        df = by_frame(bake.bake(src, naive_rotation=False, **common))
        interior = range(12, self.N - 11)
        for f in interior:
            assert nv[f].position == pytest.approx(df[f].position, abs=1e-6), f
        moved = max(sum(abs(nv[f].position[i] - self.CENTER[i]) for i in range(3)) for f in interior)
        assert moved > 1e-3

    def test_naive_rotation_off_by_default(self):
        src = self._src()
        common = dict(seed=1, amp_rot=8.0, amp_pos=0.0, settle=0.0, fade_sec=0.1)
        assert bake.bake(src, **common).camera_keys == bake.bake(src, naive_rotation=False, **common).camera_keys


class TestBakeResolvedRanges:
    def test_resolved_full_range_when_no_ranges(self):
        res = bake.bake(MOVING_CONSTANT_FOV, seed=1)
        assert res.resolved == [(0, 60)]

    def test_resolved_snaps_explicit_range_to_nearest_keys(self):
        res = bake.bake(MOVING_CONSTANT_FOV, seed=1, ranges=[(28, 58)])
        assert res.resolved == [(30, 60)]

    def test_resolved_swaps_when_snapped_ends_reverse(self):
        res = bake.bake(MOVING_CONSTANT_FOV, seed=1, ranges=[(58, 28)])
        assert res.resolved == [(30, 60)]

    def test_resolved_sorted_for_multiple_ranges(self):
        res = bake.bake(MOVING_CONSTANT_FOV, seed=1, ranges=[(60, 60), (0, 0)])
        assert res.resolved == [(0, 0), (60, 60)]

    def test_resolved_snap_tie_breaks_to_smaller_frame(self):
        res = bake.bake(MOVING_CONSTANT_FOV, seed=1, ranges=[(15, 45)])
        assert res.resolved == [(0, 30)]

    def test_resolved_uses_normalized_working_view(self):
        shuffled = [MOVING_CONSTANT_FOV[2], MOVING_CONSTANT_FOV[0], MOVING_CONSTANT_FOV[1]]
        assert bake.bake(shuffled, seed=1).resolved == [(0, 60)]

    def test_resolved_rejects_ranges_touching_only_after_snap(self):
        with pytest.raises(ValueError):
            bake.bake(MOVING_CONSTANT_FOV, seed=1, ranges=[(0, 16), (17, 60)])
