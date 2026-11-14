import math

import pytest

from vmd import fit, interp
from vmd.reduce import Tolerances, reduce_camera_track, verify_camera_track
from vmd.types import CameraKey

EASE = (96, 0, 96, 30)
EARLY_EXIT_ERR = 0.01
REPRESENTABLE_CURVES = [(20, 40, 107, 100), (0, 0, 127, 64), (40, 10, 90, 118), (96, 0, 96, 30)]
FULL_SEARCH_ONLY_CURVES = [(20, 40, 107, 100)]
EARLY_EXIT_CURVES = [(20, 40, 107, 100), (0, 0, 127, 64), (40, 10, 90, 118)]


def _curve_ys(curve, n=5):
    xs = [(i + 1) / (n + 1) for i in range(n)]
    return xs, [interp._solve_factor(*curve, x) for x in xs]


def _hill_xy():
    xs = [(i + 1) / 10 for i in range(9)]
    return xs, [math.sin(math.pi * x) for x in xs]


def test_full_search_preserves_known_curve_cp():
    xs, ys = _curve_ys(EASE, n=9)
    cp, err = fit.fit_bezier_curve(xs, ys)
    assert err < 0.01
    assert cp == EASE


@pytest.mark.parametrize("curve", FULL_SEARCH_ONLY_CURVES)
def test_full_search_quality_multiple_shapes(curve):
    xs, ys = _curve_ys(curve, n=9)
    cp, err = fit.fit_bezier_curve(xs, ys)
    assert err < 0.05


def test_scalar_early_exit_identical_to_full_search_for_ease():
    xs, ys = _curve_ys(EASE, n=9)
    cp_full, err_full = fit.fit_bezier_curve(xs, ys)
    cp_fast, err_fast = fit.fit_bezier_curve(xs, ys, early_exit_err=EARLY_EXIT_ERR)
    assert cp_fast == cp_full
    assert err_fast == err_full


@pytest.mark.parametrize("curve", EARLY_EXIT_CURVES)
def test_scalar_early_exit_error_within_threshold_of_full_search(curve):
    xs, ys = _curve_ys(curve, n=9)
    _cp_full, err_full = fit.fit_bezier_curve(xs, ys)
    _cp_fast, err_fast = fit.fit_bezier_curve(xs, ys, early_exit_err=EARLY_EXIT_ERR)
    assert err_fast <= err_full + EARLY_EXIT_ERR + 1e-12


def test_unrepresentable_data_early_exit_identical_to_full_search():
    xs, ys = _hill_xy()
    cp_full, err_full = fit.fit_bezier_curve(xs, ys)
    cp_fast, err_fast = fit.fit_bezier_curve(xs, ys, early_exit_err=EARLY_EXIT_ERR)
    assert cp_fast == cp_full
    assert err_fast == err_full


@pytest.mark.parametrize("curve", REPRESENTABLE_CURVES)
def test_noisy_curve_early_exit_error_within_threshold_of_full_search(curve):
    xs = [(i + 1) / 10 for i in range(9)]
    base = [interp._solve_factor(*curve, x) for x in xs]
    ys = [b + 0.003 * math.sin(31.0 * (i + 1)) for i, b in enumerate(base)]
    _cp_full, err_full = fit.fit_bezier_curve(xs, ys)
    _cp_fast, err_fast = fit.fit_bezier_curve(xs, ys, early_exit_err=EARLY_EXIT_ERR)
    assert err_fast <= err_full + EARLY_EXIT_ERR + 1e-12


@pytest.mark.parametrize("curve", [(96, 0, 96, 30), (40, 10, 90, 118)])
def test_coeff_early_exit_output_identical_to_full_search(curve):
    xs = [(i + 1) / 6 for i in range(5)]
    targets = [interp._solve_factor(*curve, x) for x in xs]

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets, strict=True)]

    cp_full = fit._fit_coeff_curve(xs, resid_at)
    cp_fast = fit._fit_coeff_curve(xs, resid_at, early_exit_err=EARLY_EXIT_ERR)
    assert cp_fast == cp_full


def test_coeff_unrepresentable_data_early_exit_identical_to_full_search():
    xs, targets = _hill_xy()

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets, strict=True)]

    cp_full = fit._fit_coeff_curve(xs, resid_at)
    cp_fast = fit._fit_coeff_curve(xs, resid_at, early_exit_err=EARLY_EXIT_ERR)
    assert cp_fast == cp_full


def _camera_source_moving_fov_position_rotation():
    lin = bytes([20, 107, 20, 107]) * 6
    tols = Tolerances(
        bone_pos=0.01, bone_rot=0.10,
        camera_pos=0.02, camera_rot=0.05, camera_distance=0.02, camera_fov=0.50,
    )
    n = 11

    def eased(v0, v1):
        return [v0 + (v1 - v0) * interp._solve_factor(*EASE, i / (n - 1)) for i in range(n)]

    posy = eased(0.0, 5.0)
    dist = eased(-30.0, -20.0)
    fov = eased(30.0, 50.0)
    ry = eased(0.0, math.radians(15))
    source = [
        CameraKey(f, dist[f], (0.0, posy[f], 0.0), (0.0, ry[f], 0.0), lin, fov[f], 0)
        for f in range(n)
    ]
    return source, tols, n


def _camera_track(source, tols, n):
    return reduce_camera_track(
        source, [(0, n - 1)], tols,
        cut_thresholds=(5.0, 20.0, 5.0), keep_frames=[], no_cut_detect=True,
        min_seg=1, max_seg=180, strict=False, curve_mode="bezier",
    )


def test_reduce_bezier_within_tol_fov_position_rotation():
    source, tols, n = _camera_source_moving_fov_position_rotation()
    out = _camera_track(source, tols, n)
    assert verify_camera_track(source, out, [(0, n - 1)], tols) == []
