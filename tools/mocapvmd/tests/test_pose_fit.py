import math

import pytest

from mocapvmd.markers import extract_markers
from mocapvmd.model_profile import load_mocap_profile
from mocapvmd.pose_fit import (
    DEFAULT_FIT_PARAMS,
    FitParams,
    FitResult,
    fit,
)
from pmx.pose import evaluate_fk, sample_local_poses


def _dense(profile, tracks, frames):
    return [sample_local_poses(profile.model, tracks, f) for f in frames]


def _fk_markers(profile, dense):
    world = [evaluate_fk(profile.model, lp) for lp in dense]
    return extract_markers(profile, world).markers


def _marker_error(profile, dense, target):
    got = _fk_markers(profile, dense)
    total = 0.0
    for name, series in target.items():
        for g, t in zip(got[name], series, strict=True):
            total += sum((a - b) ** 2 for a, b in zip(g, t, strict=True)) ** 0.5
    return total


def _rotation_angle_rad(q1, q0):
    dot = min(1.0, abs(sum(a * b for a, b in zip(q1, q0, strict=True))))
    return 2.0 * math.acos(dot)


def _unchanged_targets(profile, dense):
    return {n: list(s) for n, s in _fk_markers(profile, dense).items()}


def _shift_head_marker_x(target, dx):
    hx = target["head"][0]
    target["head"][0] = (hx[0] + dx, hx[1], hx[2])


def test_targets_equal_to_fk_markers_keep_original_pose():
    profile = load_mocap_profile(None)
    frames = range(0, 2)
    dense = _dense(profile, {}, frames)
    target = _fk_markers(profile, dense)
    res = fit(profile, dense, target)
    assert isinstance(res, FitResult)
    for fitted_frame, orig_frame in zip(res.poses, dense, strict=True):
        for fb, ob in zip(fitted_frame, orig_frame, strict=True):
            assert fb.position == pytest.approx(ob.position, abs=1e-9)
            assert fb.rotation == pytest.approx(ob.rotation, abs=1e-9)


def test_marker_displacement_below_fit_threshold_keeps_original_pose():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _unchanged_targets(profile, dense)
    _shift_head_marker_x(target, DEFAULT_FIT_PARAMS.skip_threshold * 0.5)
    res = fit(profile, dense, target)
    for fitted_frame, orig_frame in zip(res.poses, dense, strict=True):
        for fb, ob in zip(fitted_frame, orig_frame, strict=True):
            assert fb.position == pytest.approx(ob.position, abs=1e-9)
            assert fb.rotation == pytest.approx(ob.rotation, abs=1e-9)


def test_fit_reduces_error_of_leaf_head_marker():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _unchanged_targets(profile, dense)
    _shift_head_marker_x(target, 0.05)
    before = _marker_error(profile, dense, target)
    res = fit(profile, dense, target)
    after = _marker_error(profile, res.poses, target)
    assert after < before


def test_unreachable_target_does_not_worsen_error():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _unchanged_targets(profile, dense)
    _shift_head_marker_x(target, 100.0)
    before = _marker_error(profile, dense, target)
    res = fit(profile, dense, target)
    after = _marker_error(profile, res.poses, target)
    assert after <= before + 1e-9


def test_large_displacement_corrections_stay_within_absolute_limits():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _unchanged_targets(profile, dense)
    for n in target:
        p = target[n][0]
        target[n][0] = (p[0] + 5.0, p[1], p[2])
    res = fit(profile, dense, target)
    max_rot = math.radians(DEFAULT_FIT_PARAMS.max_rot_deg)
    for fitted_frame, orig_frame in zip(res.poses, dense, strict=True):
        for fb, ob in zip(fitted_frame, orig_frame, strict=True):
            dp = sum((a - b) ** 2 for a, b in zip(fb.position, ob.position, strict=True)) ** 0.5
            assert dp <= DEFAULT_FIT_PARAMS.max_pos + 1e-6
            assert _rotation_angle_rad(fb.rotation, ob.rotation) <= max_rot + 1e-6


def test_small_displacement_correction_is_capped_by_displacement_proportional_limit():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _unchanged_targets(profile, dense)
    d = DEFAULT_FIT_PARAMS.skip_threshold * 3.0
    pos_limit = DEFAULT_FIT_PARAMS.k_pos * d
    rot_limit = DEFAULT_FIT_PARAMS.k_rot * d
    assert pos_limit < DEFAULT_FIT_PARAMS.max_pos
    assert rot_limit < math.radians(DEFAULT_FIT_PARAMS.max_rot_deg)
    _shift_head_marker_x(target, d)
    before = _marker_error(profile, dense, target)
    res = fit(profile, dense, target)
    after = _marker_error(profile, res.poses, target)
    assert after < before
    max_rot = 0.0
    for fitted_frame, orig_frame in zip(res.poses, dense, strict=True):
        for fb, ob in zip(fitted_frame, orig_frame, strict=True):
            dp = sum((a - b) ** 2 for a, b in zip(fb.position, ob.position, strict=True)) ** 0.5
            assert dp <= pos_limit + 1e-6
            angle = _rotation_angle_rad(fb.rotation, ob.rotation)
            assert angle <= rot_limit + 1e-6
            max_rot = max(max_rot, angle)
    assert max_rot > 0.0


def test_default_fit_params_are_positive():
    assert isinstance(DEFAULT_FIT_PARAMS, FitParams)
    assert DEFAULT_FIT_PARAMS.skip_threshold > 0.0
    assert DEFAULT_FIT_PARAMS.max_rot_deg > 0.0
    assert DEFAULT_FIT_PARAMS.max_pos > 0.0
    assert DEFAULT_FIT_PARAMS.k_rot > 0.0
    assert DEFAULT_FIT_PARAMS.k_pos > 0.0


def test_fallback_frame_count_is_reported():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _fk_markers(profile, dense)
    res = fit(profile, dense, target)
    assert isinstance(res.fallback_frames, int)
    assert res.fallback_frames >= 0
