import pytest

from vmd import interp
from vmd.reduce import (
    StrictError,
    Tolerances,
    reduce_camera_track,
    verify_bone_track,
    verify_camera_track,
)
from vmd.types import BoneKey, CameraKey

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
EASE = (96, 0, 96, 30)


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()
TOLS = Tolerances(
    bone_pos=0.01,
    bone_rot=0.10,
    camera_pos=0.02,
    camera_rot=0.05,
    camera_distance=0.02,
    camera_fov=0.50,
)
CAM_CUT = (5.0, 20.0, 5.0)


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, persp)


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, rot, BL)


def _eased(v0, v1, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*EASE, f / span) for f in range(n)]


def test_verify_camera_faithful_linear_no_violation():
    source = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(11)]
    output = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(10.0, 0.0, 0.0))]
    assert verify_camera_track(source, output, [(0, 10)], TOLS) == []


def test_verify_camera_detects_position_violation():
    xs = _eased(0.0, 100.0)
    source = [cam(f, center=(xs[f], 0.0, 0.0)) for f in range(11)]
    output = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(100.0, 0.0, 0.0))]
    bad = verify_camera_track(source, output, [(0, 10)], TOLS)
    assert bad
    assert all(0 < f < 10 for f in bad)
    assert bad == sorted(bad)


def test_verify_camera_fov_step_vs_ramp_within_half_degree_not_flagged():
    source = [cam(f, fov=30 if f < 5 else 31) for f in range(11)]
    output = [cam(0, fov=30), cam(10, fov=31)]
    assert verify_camera_track(source, output, [(0, 10)], TOLS) == []


def test_verify_camera_fov_large_error_flagged():
    fovs = [round(30 + 30 * interp._solve_factor(*EASE, f / 10)) for f in range(11)]
    source = [cam(f, fov=fovs[f]) for f in range(11)]
    output = [cam(0, fov=fovs[0]), cam(10, fov=fovs[10])]
    bad = verify_camera_track(source, output, [(0, 10)], TOLS)
    assert bad
    assert all(0 < f < 10 for f in bad)


def test_verify_bone_faithful_no_violation():
    source = [bone("c", f, pos=(float(f), 0.0, 0.0)) for f in range(11)]
    output = [bone("c", 0, pos=(0.0, 0.0, 0.0)), bone("c", 10, pos=(10.0, 0.0, 0.0))]
    assert verify_bone_track(source, output, [(0, 10)], TOLS) == []


def test_verify_bone_detects_position_violation():
    xs = _eased(0.0, 100.0)
    source = [bone("c", f, pos=(xs[f], 0.0, 0.0)) for f in range(11)]
    output = [bone("c", 0, pos=(0.0, 0.0, 0.0)), bone("c", 10, pos=(100.0, 0.0, 0.0))]
    bad = verify_bone_track(source, output, [(0, 10)], TOLS)
    assert bad
    assert all(0 < f < 10 for f in bad)
    assert bad == sorted(bad)


def _force_linear_output_curves(monkeypatch):
    import vmd.fit as fit

    linear_cp = (20, 20, 107, 107)
    for cls in (
        fit.LinearScalarChannel,
        fit.FovChannel,
        fit.CameraRotationChannel,
        fit.BoneRotationChannel,
    ):
        monkeypatch.setattr(cls, "curve", lambda self, a, b: linear_cp)
    monkeypatch.setattr(
        fit.EuclideanVectorChannel, "curve", lambda self, a, b: (linear_cp,) * 3
    )


def _opts(**kw):
    o = dict(
        cut_thresholds=CAM_CUT,
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
    )
    o.update(kw)
    return o


def test_reduce_camera_strict_raises_on_output_violation(monkeypatch):
    xs = _eased(-10.0, -110.0)
    source = [cam(f, dist=xs[f]) for f in range(11)]
    _force_linear_output_curves(monkeypatch)
    with pytest.raises(StrictError):
        reduce_camera_track(source, [(0, 10)], TOLS, **_opts(strict=True))


def test_reduce_camera_nonstrict_densifies_on_output_violation(monkeypatch):
    xs = _eased(-10.0, -110.0)
    source = [cam(f, dist=xs[f]) for f in range(11)]
    _force_linear_output_curves(monkeypatch)
    keys = reduce_camera_track(source, [(0, 10)], TOLS, **_opts(strict=False))
    assert len(keys) > 2
    assert verify_camera_track(source, keys, [(0, 10)], TOLS) == []


def test_reduce_camera_normal_passes_verification():
    xs = _eased(-10.0, -110.0)
    source = [cam(f, dist=xs[f]) for f in range(11)]
    keys = reduce_camera_track(source, [(0, 10)], TOLS, **_opts())
    assert verify_camera_track(source, keys, [(0, 10)], TOLS) == []
