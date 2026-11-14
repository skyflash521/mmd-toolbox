from vmd import interp
from vmd.reduce import (
    Tolerances,
    camera_interp_bytes,
    reduce_camera_track,
)
from vmd.types import CameraKey

TOLS = Tolerances(
    bone_pos=0.01,
    bone_rot=0.10,
    camera_pos=0.02,
    camera_rot=0.05,
    camera_distance=0.02,
    camera_fov=0.50,
)
CAM_CUT = (5.0, 20.0, 5.0)
EASE = (96, 0, 96, 30)
LIN = (20, 20, 107, 107)
CAM_LINEAR = bytes([20, 107, 20, 107]) * 6


def _interp_with_pos_x(pos_x_cp):
    return camera_interp_bytes(pos_x_cp, LIN, LIN, LIN, LIN, LIN)


def cam(frame, x, eased=False):
    return CameraKey(frame, -30.0, (x, 0.0, 0.0), (0.0, 0.0, 0.0), _interp_with_pos_x(EASE if eased else LIN), 30, 0)


def camera_track(source, ranges, **kw):
    opts = dict(
        cut_thresholds=CAM_CUT,
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
    )
    opts.update(kw)
    return reduce_camera_track(source, ranges, TOLS, **opts)


def _max_pos_x_deviation(out, src, lo, hi):
    return max(
        abs(interp.sample(out, "pos_x", f) - interp.sample(src, "pos_x", f))
        for f in range(lo, hi + 1)
    )


def test_lower_seam_curve_refit_to_follow_source_before_range():
    src = [cam(0, 0.0), cam(10, 100.0, eased=True), cam(20, 100.0)]
    out = camera_track(src, [(10, 20)])
    assert _max_pos_x_deviation(out, src, 0, 10) <= TOLS.camera_pos
    k10 = next(k for k in out if k.frame == 10)
    assert k10.interpolation[0:4] != bytes([20, 107, 20, 107])


def test_upper_seam_out_of_range_key_unchanged():
    src = [cam(0, 0.0), cam(10, 0.0), cam(30, 100.0, eased=True)]
    out = camera_track(src, [(0, 20)])
    k30 = next(k for k in out if k.frame == 30)
    assert k30 == src[2]


def test_no_seam_rewrite_when_range_covers_track():
    src = [cam(0, 0.0), cam(10, 100.0, eased=True), cam(20, 100.0)]
    out = camera_track(src, [(0, 20)])
    k0 = next(k for k in out if k.frame == 0)
    assert k0.interpolation == CAM_LINEAR


def test_linear_mode_no_seam_rewrite():
    src = [cam(0, 0.0), cam(10, 100.0, eased=True), cam(20, 100.0)]
    out = camera_track(src, [(10, 20)], curve_mode="linear")
    k10 = next(k for k in out if k.frame == 10)
    assert k10.interpolation == CAM_LINEAR
