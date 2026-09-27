import math

import numpy as np
import pytest

from sparsevmd.sample import build_tracks, sample_rotation, sample_scalar
from vmd.reduce import BONE_LINEAR_INTERP, CAMERA_LINEAR_INTERP, bone_interp_bytes, camera_interp_bytes
from vmd.types import BoneKey, CameraKey, VmdDocument

_LINEAR_CP = (20, 20, 107, 107)
_EASE_CP = (40, 10, 90, 118)

CAM_LINEAR = CAMERA_LINEAR_INTERP
BONE_LINEAR = BONE_LINEAR_INTERP
CAM_POSX_EASE = camera_interp_bytes(_EASE_CP, *([_LINEAR_CP] * 5))
BONE_EASE_ROT = bone_interp_bytes(_LINEAR_CP, _LINEAR_CP, _LINEAR_CP, _EASE_CP)
_ROT_Z_90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
_ROT_Z_45 = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, persp)


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, rot, BONE_LINEAR)


def test_camera_single_track():
    doc = VmdDocument(camera=[cam(0), cam(30), cam(60)])
    tracks = build_tracks(doc, "camera")
    assert len(tracks) == 1
    t = tracks[0]
    assert t.kind == "camera"
    assert t.first == 0 and t.last == 60


def test_bone_split_by_name():
    doc = VmdDocument(
        bone=[
            bone("センター", 0),
            bone("頭", 0),
            bone("センター", 30),
            bone("頭", 60),
        ]
    )
    tracks = build_tracks(doc, "bone")
    assert len(tracks) == 2
    by_name = {t.name: t for t in tracks}
    assert set(by_name) == {"センター", "頭"}
    assert by_name["センター"].first == 0 and by_name["センター"].last == 30
    assert by_name["頭"].first == 0 and by_name["頭"].last == 60


def test_normalization_sort_and_dedup_last_wins():
    doc = VmdDocument(camera=[cam(60), cam(0), cam(30, dist=-10.0), cam(30, dist=-99.0)])
    t = build_tracks(doc, "camera")[0]
    frames = [k.frame for k in t.keys]
    assert frames == [0, 30, 60]
    k30 = [k for k in t.keys if k.frame == 30][0]
    assert k30.distance == pytest.approx(-99.0)


def test_bone_normalization_sort_and_dedup_per_track():
    doc = VmdDocument(
        bone=[
            bone("センター", 30, pos=(0.0, 0.0, 0.0)),
            bone("センター", 0),
            bone("センター", 30, pos=(9.0, 0.0, 0.0)),
        ]
    )
    t = build_tracks(doc, "bone")[0]
    frames = [k.frame for k in t.keys]
    assert frames == [0, 30]
    k30 = [k for k in t.keys if k.frame == 30][0]
    assert k30.position[0] == pytest.approx(9.0)


def test_target_all_includes_both():
    doc = VmdDocument(camera=[cam(0), cam(30)], bone=[bone("センター", 0), bone("センター", 30)])
    kinds = {t.kind for t in build_tracks(doc, "all")}
    assert kinds == {"camera", "bone"}


def test_target_camera_excludes_bone():
    doc = VmdDocument(camera=[cam(0), cam(30)], bone=[bone("センター", 0), bone("センター", 30)])
    tracks = build_tracks(doc, "camera")
    assert [t.kind for t in tracks] == ["camera"]


def test_target_bone_excludes_camera():
    doc = VmdDocument(camera=[cam(0), cam(30)], bone=[bone("センター", 0), bone("センター", 30)])
    tracks = build_tracks(doc, "bone")
    assert [t.kind for t in tracks] == ["bone"]


def test_single_key_track_first_equals_last():
    doc = VmdDocument(bone=[bone("センター", 42)])
    t = build_tracks(doc, "bone")[0]
    assert t.first == 42 and t.last == 42


def test_sample_scalar_linear_camera_pos():
    keys = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(10.0, 0.0, 0.0))]
    arr = sample_scalar(keys, "pos_x", 0, 10)
    assert isinstance(arr, np.ndarray)
    assert len(arr) == 11
    np.testing.assert_allclose(arr, np.linspace(0.0, 10.0, 11), atol=1e-6)


def test_sample_scalar_linear_bone_pos():
    keys = [bone("センター", 0, pos=(0.0, 0.0, 0.0)), bone("センター", 10, pos=(0.0, 5.0, 0.0))]
    arr = sample_scalar(keys, "pos_y", 0, 10)
    np.testing.assert_allclose(arr, np.linspace(0.0, 5.0, 11), atol=1e-6)


def test_sample_scalar_subrange():
    keys = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(10.0, 0.0, 0.0))]
    arr = sample_scalar(keys, "pos_x", 2, 5)
    assert len(arr) == 4
    np.testing.assert_allclose(arr, [2.0, 3.0, 4.0, 5.0], atol=1e-6)


def test_sample_scalar_honors_interpolation_curve():
    keys = [
        CameraKey(0, -30.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), CAM_POSX_EASE, 30, 0),
        CameraKey(10, -30.0, (10.0, 0.0, 0.0), (0.0, 0.0, 0.0), CAM_POSX_EASE, 30, 0),
    ]
    arr = sample_scalar(keys, "pos_x", 0, 10)
    assert arr[3] == pytest.approx(2.3752295941041206, abs=1e-6)
    assert arr[3] < 2.9


def test_sample_rotation_camera_euler():
    keys = [cam(0, rot=(0.0, 0.0, 0.0)), cam(10, rot=(0.0, 1.0, 0.0))]
    rots = sample_rotation(keys, 0, 10)
    assert len(rots) == 11
    assert all(len(r) == 3 for r in rots)
    assert rots[0] == pytest.approx((0.0, 0.0, 0.0))
    assert rots[10] == pytest.approx((0.0, 1.0, 0.0))
    assert rots[5] == pytest.approx((0.0, 0.5, 0.0))


def test_sample_rotation_bone_quaternion_slerp():
    q1 = _ROT_Z_90
    keys = [
        bone("センター", 0, rot=(0.0, 0.0, 0.0, 1.0)),
        bone("センター", 10, rot=q1),
    ]
    rots = sample_rotation(keys, 0, 10)
    assert len(rots) == 11
    assert all(len(r) == 4 for r in rots)
    assert rots[0] == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert rots[10] == pytest.approx(q1)
    for r in rots:
        assert sum(c * c for c in r) == pytest.approx(1.0, abs=1e-6)
    assert rots[5] == pytest.approx(_ROT_Z_45)


def test_sample_rotation_bone_honors_interpolation_curve():
    q1 = _ROT_Z_90
    keys = [
        BoneKey(b"c".ljust(15, b"\x00"), 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_EASE_ROT),
        BoneKey(b"c".ljust(15, b"\x00"), 10, (0.0, 0.0, 0.0), q1, BONE_EASE_ROT),
    ]
    rots = sample_rotation(keys, 0, 10)
    assert rots[3] == pytest.approx(
        (0.0, 0.0, 0.18546995755930634, 0.9826499350444946), abs=1e-6
    )
