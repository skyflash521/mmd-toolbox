from vmd import fit, interp
from vmd.reduce import (
    BONE_LINEAR_INTERP,
    CAMERA_LINEAR_INTERP,
    Tolerances,
    build_bone_tolerances,
    reduce_bone_track,
    reduce_camera_track,
)
from vmd.types import BoneKey, CameraKey

STRONG_EASE = (96, 0, 96, 30)
COUNTER_KEYS = {"fit_calls", "lsq_calls", "fastpath_linear", "cheap_accept"}


def _bone(frame, pos):
    return BoneKey(b"bone".ljust(15, b"\x00"), frame, pos, (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)


def _bone_keys_eased_pos_x(n=21, v1=10.0):
    span = n - 1
    return [
        _bone(f, (v1 * interp._solve_factor(*STRONG_EASE, f / span), 0.0, 0.0)) for f in range(n)
    ]


def _bone_keys_linear_pos_x(n=21, v1=10.0):
    span = n - 1
    return [_bone(f, (v1 * f / span, 0.0, 0.0)) for f in range(n)]


def _cam(frame, pos):
    return CameraKey(frame, -30.0, pos, (0.0, 0.0, 0.0), CAMERA_LINEAR_INTERP, 30, 0)


def _camera_keys_eased_center_x(n=21, v1=10.0):
    span = n - 1
    return [_cam(f, (v1 * interp._solve_factor(*STRONG_EASE, f / span), 0.0, 0.0)) for f in range(n)]


def _cam_tols():
    return Tolerances(
        bone_pos=0.0, bone_rot=0.0, camera_pos=0.5, camera_rot=5.0,
        camera_distance=0.5, camera_fov=1.0,
    )


def _bone_tols():
    return build_bone_tolerances(bone_pos=0.5, bone_rot=5.0)


def _reduce_bone(src, diagnostics):
    return reduce_bone_track(
        src,
        [(0, 20)],
        _bone_tols(),
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
        diagnostics=diagnostics,
    )


def _reduce_camera(src, diagnostics):
    return reduce_camera_track(
        src,
        [(0, 20)],
        _cam_tols(),
        cut_thresholds=(5.0, 20.0, 5.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
        diagnostics=diagnostics,
    )


def test_reset_zeroes_counters():
    fit.reset_fit_counters()
    c = fit.read_fit_counters()
    assert c == {"fit_calls": 0, "lsq_calls": 0, "fastpath_linear": 0, "cheap_accept": 0}


def test_nonlinear_increments_lsq_calls():
    fit.reset_fit_counters()
    xs = [(i + 1) / 12 for i in range(11)]
    ys = [interp._solve_factor(*STRONG_EASE, x) for x in xs]
    fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    c = fit.read_fit_counters()
    assert c["fit_calls"] == 1
    assert c["lsq_calls"] >= 1
    assert c["fastpath_linear"] == 0


def test_linear_increments_fastpath_not_lsq():
    fit.reset_fit_counters()
    xs = [(i + 1) / 12 for i in range(11)]
    ys = list(xs)
    fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    c = fit.read_fit_counters()
    assert c["fit_calls"] == 1
    assert c["lsq_calls"] == 0
    assert c["fastpath_linear"] == 1


def test_coeff_curve_linear_increments_fastpath():
    fit.reset_fit_counters()
    xs = [(i + 1) / 6 for i in range(5)]

    def resid_at(coeff):
        return [coeff(x) - x for x in xs]

    fit._fit_coeff_curve(xs, resid_at, early_exit_err=0.01)
    c = fit.read_fit_counters()
    assert c["fit_calls"] == 1
    assert c["lsq_calls"] == 0
    assert c["fastpath_linear"] == 1


def test_coeff_curve_nonlinear_increments_lsq_calls():
    fit.reset_fit_counters()
    xs = [(i + 1) / 12 for i in range(11)]

    def resid_at(coeff):
        return [coeff(x) - interp._solve_factor(*STRONG_EASE, x) for x in xs]

    fit._fit_coeff_curve(xs, resid_at, early_exit_err=0.01)
    c = fit.read_fit_counters()
    assert c["fit_calls"] == 1
    assert c["lsq_calls"] >= 1
    assert c["fastpath_linear"] == 0


def test_cheap_accept_fires_for_fixed_ease_curve():
    fit.reset_fit_counters()
    xs = [(i + 1) / 12 for i in range(11)]
    ease_in = fit._CHEAP_EASE_CPS[0]
    ys = [interp._solve_factor(*ease_in, x) for x in xs]
    cp, _err = fit.fit_bezier_curve(xs, ys, early_exit_err=0.02)
    c = fit.read_fit_counters()
    assert c["cheap_accept"] == 1
    assert c["lsq_calls"] == 0
    assert c["fastpath_linear"] == 0
    assert cp == ease_in


def test_diagnostics_records_fit_counts_for_curved_bone_track():
    diag = {}
    _reduce_bone(_bone_keys_eased_pos_x(), diag)
    fc = diag["fit_counts"]
    assert set(fc) == COUNTER_KEYS
    assert all(isinstance(v, int) for v in fc.values())
    assert fc["fit_calls"] >= 1
    assert fc["lsq_calls"] >= 1


def test_diagnostics_records_fit_counts_for_curved_camera_track():
    diag = {}
    _reduce_camera(_camera_keys_eased_center_x(), diag)
    fc = diag["fit_counts"]
    assert set(fc) == COUNTER_KEYS
    assert all(isinstance(v, int) for v in fc.values())
    assert fc["fit_calls"] >= 1
    assert fc["lsq_calls"] >= 1


def test_diagnostics_linear_track_uses_fastpath_only():
    diag = {}
    _reduce_bone(_bone_keys_linear_pos_x(), diag)
    fc = diag["fit_counts"]
    assert fc["fastpath_linear"] >= 1
    assert fc["lsq_calls"] == 0


def test_by_category_counts_only_when_category_given():
    fit.reset_fit_counters()
    xs = [(i + 1) / 12 for i in range(11)]
    ys = [interp._solve_factor(*STRONG_EASE, x) for x in xs]
    fit.fit_bezier_curve(xs, ys, early_exit_err=0.01, category="position")
    by = fit.read_fit_counters_by_category()
    assert by["position"]["fit_calls"] == 1
    assert by["position"]["lsq_calls"] >= 1

    fit.reset_fit_counters()

    def resid_at(coeff):
        return [coeff(x) - interp._solve_factor(*STRONG_EASE, x) for x in xs]

    fit._fit_coeff_curve(xs, resid_at, early_exit_err=0.01, category="rotation")
    by = fit.read_fit_counters_by_category()
    assert by["rotation"]["fit_calls"] == 1
    assert by["rotation"]["lsq_calls"] >= 1

    fit.reset_fit_counters()
    fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    assert fit.read_fit_counters_by_category() == {}


def test_by_channel_attributes_position_vs_rotation():
    diag = {}
    _reduce_bone(_bone_keys_eased_pos_x(), diag)
    by = diag["fit_counts_by_channel"]
    assert by["position"]["lsq_calls"] >= 1
    assert by["rotation"]["lsq_calls"] == 0


def test_by_channel_sums_equal_total_for_every_counter():
    diag = {}
    _reduce_bone(_bone_keys_eased_pos_x(), diag)
    total = diag["fit_counts"]
    by = diag["fit_counts_by_channel"]
    for key in total:
        assert sum(c[key] for c in by.values()) == total[key]


def test_by_channel_camera_has_position():
    diag = {}
    _reduce_camera(_camera_keys_eased_center_x(), diag)
    by = diag["fit_counts_by_channel"]
    assert by["position"]["lsq_calls"] >= 1
    assert all(isinstance(v, int) for c in by.values() for v in c.values())


def test_diagnostics_omitted_still_reduces():
    out = _reduce_bone(_bone_keys_eased_pos_x(), None)
    assert out
