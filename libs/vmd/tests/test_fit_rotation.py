import math

import pytest

from vmd.fit import BoneRotationChannel, CameraRotationChannel, _quat_angle_deg


def quat_z(deg):
    a = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(a), math.cos(a))


def test_quat_angle_deg_small_angle_precision():
    base = quat_z(0.0)
    for deg in (0.05, 0.1, 0.2, 0.5, 1.0):
        assert _quat_angle_deg(base, quat_z(deg)) == pytest.approx(deg, abs=1e-9)


def test_quat_angle_deg_sign_invariant_exact_zero():
    q = quat_z(33.0)
    assert _quat_angle_deg(q, tuple(-c for c in q)) == pytest.approx(0.0, abs=1e-12)


def test_quat_angle_deg_identical_exact_zero():
    q = quat_z(17.0)
    assert _quat_angle_deg(q, q) == 0.0


def test_camera_rotation_linear_zero_error():
    eulers = [(0.0, math.radians(i), 0.0) for i in range(11)]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0)
    assert frame is None


def test_camera_rotation_deviation_in_degrees():
    eulers = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, math.radians(30), 0.0),
              (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(30.0)
    assert frame == 2


def test_camera_rotation_max_over_axes():
    eulers = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
              (math.radians(2), 0.0, math.radians(30)), (0.0, 0.0, 0.0)]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    err, frame = ch.residual(0, 3)
    assert err == pytest.approx(30.0)
    assert frame == 2


def test_camera_rotation_prefers_axis_reversal_over_max_error():
    ys = [0.0, 30.0, 10.0, 15.0, 90.0]
    eulers = [(0.0, math.radians(y), 0.0) for y in ys]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(52.5)
    assert frame == 2


def test_camera_rotation_unwrap_no_false_error():
    eulers = [
        (math.radians(170), 0.0, 0.0),
        (math.radians(175), 0.0, 0.0),
        (math.radians(-179), 0.0, 0.0),
    ]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    err, frame = ch.residual(0, 2)
    assert err == pytest.approx(0.5)
    assert frame == 1


def test_bone_rotation_on_slerp_path_zero_error():
    quats = [quat_z(i * 9.0) for i in range(11)]
    ch = BoneRotationChannel(0, quats, tol=0.1)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0, abs=1e-6)
    assert frame is None


def test_bone_rotation_off_path_detected():
    quats = [quat_z(i * 9.0) for i in range(11)]
    quats[5] = (0.0, 0.0, 0.0, 1.0)
    ch = BoneRotationChannel(0, quats, tol=0.1)
    err, frame = ch.residual(0, 10)
    assert frame == 5
    assert err == pytest.approx(45.0, abs=1e-6)


def test_bone_rotation_sign_invariant():
    quats = [quat_z(i * 9.0) for i in range(11)]
    quats[10] = tuple(-c for c in quats[10])
    quats[3] = tuple(-c for c in quats[3])
    ch = BoneRotationChannel(0, quats, tol=0.1)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_bone_rotation_prefers_direction_reversal_over_max_error():
    degs = [0.0, 30.0, 10.0, 15.0, 90.0]
    quats = [quat_z(d) for d in degs]
    ch = BoneRotationChannel(0, quats, tol=0.1)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(52.5, abs=1e-4)
    assert frame == 2


def test_bone_rotation_normalized_and_zero_tol():
    quats = [quat_z(i * 9.0) for i in range(11)]
    quats[5] = (0.0, 0.0, 0.0, 1.0)
    ch = BoneRotationChannel(0, quats, tol=9.0)
    nerr, frame = ch.normalized(0, 10)
    assert nerr == pytest.approx(5.0, abs=1e-3)
    assert frame == 5
    ch0 = BoneRotationChannel(0, quats, tol=0.0)
    nerr0, frame0 = ch0.normalized(0, 10)
    assert nerr0 == math.inf and frame0 == 5


def test_bone_rotation_zero_tol_zero_error():
    quats = [quat_z(i * 9.0) for i in range(11)]
    ch = BoneRotationChannel(0, quats, tol=0.0)
    nerr, frame = ch.normalized(0, 10)
    assert nerr == 0.0 and frame is None
