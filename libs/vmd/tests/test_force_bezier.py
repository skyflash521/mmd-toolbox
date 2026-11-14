import math

from vmd import interp
from vmd.fit import (
    CameraRotationChannel,
    EuclideanVectorChannel,
    FovChannel,
    LinearScalarChannel,
    read_fit_counters,
    reset_fit_counters,
)
from vmd.reduce import (
    CAMERA_LINEAR_INTERP,
    Tolerances,
    camera_interp_bytes,
    reduce_camera_track,
)
from vmd.types import CameraKey

EASE = (96, 0, 96, 30)
LIN = (20, 20, 107, 107)
LINEAR_CHANNEL_BYTES = bytes([20, 107, 20, 107])
CAM_LINEAR = LINEAR_CHANNEL_BYTES * 6
CAM_CUT = (5.0, 20.0, 5.0)
_HUGE_TOL = 1e6
HUGE_TOLS = Tolerances(
    bone_pos=_HUGE_TOL, bone_rot=_HUGE_TOL, camera_pos=_HUGE_TOL, camera_rot=_HUGE_TOL,
    camera_distance=_HUGE_TOL, camera_fov=_HUGE_TOL,
)


def _eased(v0, v1, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*EASE, f / span) for f in range(n)]


def _coeff(n=11):
    span = n - 1
    return [interp._solve_factor(*EASE, f / span) for f in range(n)]


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), interp_block=None):
    return CameraKey(frame, dist, center, rot, interp_block or CAM_LINEAR, 30, 0)


def camera_track(source, ranges, tols=HUGE_TOLS, **kw):
    opts = dict(
        cut_thresholds=CAM_CUT, keep_frames=[], no_cut_detect=True,
        min_seg=1, max_seg=180, strict=False, curve_mode="bezier",
    )
    opts.update(kw)
    return reduce_camera_track(source, ranges, tols, **opts)


def _assert_only_least_squares_fits(counts):
    assert counts["lsq_calls"] > 0
    assert counts["fastpath_linear"] == 0
    assert counts["cheap_accept"] == 0


def test_scalar_channel_force_bezier_opts_out_fastpath():
    vals = _eased(0.0, 100.0)
    assert LinearScalarChannel(0, vals, tol=_HUGE_TOL, mode="bezier").curve(0, 10) == LIN
    reset_fit_counters()
    ch = LinearScalarChannel(0, vals, tol=_HUGE_TOL, mode="bezier", force_bezier=True)
    assert ch.curve(0, 10) != LIN
    _assert_only_least_squares_fits(read_fit_counters())


def test_fov_channel_force_bezier_opts_out_fastpath():
    vals = _eased(30.0, 80.0)
    assert FovChannel(0, vals, tol=_HUGE_TOL, mode="bezier").curve(0, 10) == LIN
    reset_fit_counters()
    ch = FovChannel(0, vals, tol=_HUGE_TOL, mode="bezier", force_bezier=True)
    assert ch.curve(0, 10) != LIN
    _assert_only_least_squares_fits(read_fit_counters())


def test_euclidean_channel_force_bezier_opts_out_fastpath():
    ys = _eased(0.0, 100.0)
    vecs = [(0.0, y, 0.0) for y in ys]
    assert EuclideanVectorChannel(0, vecs, tol=_HUGE_TOL, mode="bezier").curve(0, 10)[1] == LIN
    reset_fit_counters()
    ch = EuclideanVectorChannel(0, vecs, tol=_HUGE_TOL, mode="bezier", force_bezier=True)
    assert ch.curve(0, 10)[1] != LIN
    _assert_only_least_squares_fits(read_fit_counters())


def test_camera_rotation_channel_force_bezier_opts_out_fastpath():
    c = _coeff()
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    eulers = [(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f]) for f in range(11)]
    assert CameraRotationChannel(0, eulers, tol=_HUGE_TOL, mode="bezier").curve(0, 10) == LIN
    reset_fit_counters()
    ch = CameraRotationChannel(0, eulers, tol=_HUGE_TOL, mode="bezier", force_bezier=True)
    assert ch.curve(0, 10) != LIN
    _assert_only_least_squares_fits(read_fit_counters())


def test_reduce_camera_force_bezier_stores_nonlinear_rotation_and_distance_curves():
    c = _coeff()
    dist = _eased(-10.0, -110.0)
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    source = [
        cam(f, dist=dist[f], rot=(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f]))
        for f in range(11)
    ]
    base = camera_track(source, [(0, 10)])
    assert all(k.interpolation == CAMERA_LINEAR_INTERP for k in base)
    reset_fit_counters()
    fb = camera_track(source, [(0, 10)], force_bezier=True)
    assert fb[-1].interpolation != CAMERA_LINEAR_INTERP
    assert fb[-1].interpolation[12:16] != LINEAR_CHANNEL_BYTES
    assert fb[-1].interpolation[16:20] != LINEAR_CHANNEL_BYTES
    _assert_only_least_squares_fits(read_fit_counters())


def test_reduce_camera_force_bezier_false_same_as_default():
    dist = _eased(-10.0, -110.0)
    source = [cam(f, dist=dist[f]) for f in range(11)]
    out = camera_track(source, [(0, 10)])
    assert all(k.interpolation == CAMERA_LINEAR_INTERP for k in out)
    out_false = camera_track(source, [(0, 10)], force_bezier=False)
    assert [k.interpolation for k in out_false] == [k.interpolation for k in out]


def _interp_with_eased_pos_x():
    return camera_interp_bytes(EASE, LIN, LIN, LIN, LIN, LIN)


def test_seam_refit_force_bezier_opts_out_fastpath():
    src = [
        cam(0, center=(0.0, 0.0, 0.0)),
        cam(10, center=(100.0, 0.0, 0.0), interp_block=_interp_with_eased_pos_x()),
        cam(20, center=(100.0, 0.0, 0.0)),
    ]
    base = camera_track(src, [(10, 20)])
    k10 = next(k for k in base if k.frame == 10)
    assert k10.interpolation[0:4] == LINEAR_CHANNEL_BYTES
    reset_fit_counters()
    fb = camera_track(src, [(10, 20)], force_bezier=True)
    k10fb = next(k for k in fb if k.frame == 10)
    assert k10fb.interpolation[0:4] != LINEAR_CHANNEL_BYTES
    _assert_only_least_squares_fits(read_fit_counters())
