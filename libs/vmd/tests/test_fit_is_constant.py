import math

from vmd.fit import (
    BoneRotationChannel,
    CameraRotationChannel,
    EuclideanVectorChannel,
    FovChannel,
    LinearScalarChannel,
)

_BELOW_CONST_EPS = 1e-11
_ABOVE_CONST_EPS = 1e-3


def quat_z(deg):
    a = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(a), math.cos(a))


def test_scalar_constant_true_over_long_span():
    ch = LinearScalarChannel(0, [5.0] * 200, tol=0.01)
    assert ch.is_constant(0, 199) is True


def test_scalar_varying_false():
    ch = LinearScalarChannel(0, [float(i) for i in range(11)], tol=0.01)
    assert ch.is_constant(0, 10) is False


def test_scalar_below_threshold_true_above_false():
    ch_small = LinearScalarChannel(0, [5.0, 5.0 + _BELOW_CONST_EPS, 5.0], tol=0.01)
    assert ch_small.is_constant(0, 2) is True
    ch_big = LinearScalarChannel(0, [5.0, 5.0 + _ABOVE_CONST_EPS, 5.0], tol=0.01)
    assert ch_big.is_constant(0, 2) is False


def test_scalar_deviation_at_span_end_detected():
    vals = [3.0] * 50
    vals[-1] = 9.0
    ch = LinearScalarChannel(0, vals, tol=0.01)
    assert ch.is_constant(0, 49) is False


def test_scalar_endpoint_a_included_in_subrange():
    ch = LinearScalarChannel(0, [9.0, 9.0, 0.0, 5.0, 5.0, 5.0], tol=0.01)
    assert ch.is_constant(2, 5) is False


def test_scalar_single_sample_span_is_constant():
    ch = LinearScalarChannel(0, [5.0, 999.0], tol=0.01)
    assert ch.is_constant(0, 0) is True


def test_vector_constant_true():
    vecs = [(1.0, 2.0, 3.0)] * 100
    ch = EuclideanVectorChannel(0, vecs, tol=0.01)
    assert ch.is_constant(0, 99) is True


def test_vector_single_axis_varies_false():
    vecs = [(1.0, 2.0, 3.0)] * 5
    vecs[3] = (1.0, 2.0, 3.5)
    ch = EuclideanVectorChannel(0, vecs, tol=0.01)
    assert ch.is_constant(0, 4) is False


def test_vector_below_threshold_true_above_false():
    small = [(1.0, 2.0, 3.0), (1.0 + _BELOW_CONST_EPS, 2.0, 3.0), (1.0, 2.0, 3.0)]
    assert EuclideanVectorChannel(0, small, tol=0.01).is_constant(0, 2) is True
    big = [(1.0, 2.0, 3.0), (1.0 + _ABOVE_CONST_EPS, 2.0, 3.0), (1.0, 2.0, 3.0)]
    assert EuclideanVectorChannel(0, big, tol=0.01).is_constant(0, 2) is False


def test_vector_deviation_at_span_end_detected():
    vecs = [(1.0, 2.0, 3.0)] * 50
    vecs[-1] = (1.0, 2.0, 9.0)
    ch = EuclideanVectorChannel(0, vecs, tol=0.01)
    assert ch.is_constant(0, 49) is False


def test_vector_subrange_constant_true_despite_change_outside():
    vecs = [(0.0, 0.0, 0.0)] * 10
    vecs[0] = (9.0, 9.0, 9.0)
    ch = EuclideanVectorChannel(0, vecs, tol=0.01)
    assert ch.is_constant(2, 9) is True


def test_fov_constant_true():
    ch = FovChannel(0, [30.0] * 100, tol=0.5)
    assert ch.is_constant(0, 99) is True


def test_fov_non_integer_constant_true():
    ch = FovChannel(0, [30.4] * 50, tol=0.5)
    assert ch.is_constant(0, 49) is True


def test_fov_varying_false():
    ch = FovChannel(0, [30.0, 30.0, 35.0, 30.0, 30.0], tol=0.5)
    assert ch.is_constant(0, 4) is False


def test_fov_uses_raw_samples_not_rounded():
    ch = FovChannel(0, [30.1, 30.3, 30.1], tol=0.5)
    assert ch.is_constant(0, 2) is False


def test_fov_below_threshold_true():
    ch = FovChannel(0, [30.0, 30.0 + _BELOW_CONST_EPS, 30.0], tol=0.5)
    assert ch.is_constant(0, 2) is True


def test_fov_deviation_at_span_end_detected():
    vals = [30.0] * 50
    vals[-1] = 35.0
    ch = FovChannel(0, vals, tol=0.5)
    assert ch.is_constant(0, 49) is False


def test_camera_rotation_constant_true():
    eulers = [(0.0, 0.5, 0.0)] * 100
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    assert ch.is_constant(0, 99) is True


def test_camera_rotation_varying_false():
    eulers = [(0.0, math.radians(i), 0.0) for i in range(11)]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    assert ch.is_constant(0, 10) is False


def test_camera_rotation_below_threshold_true_above_false():
    small = [(0.0, 0.5, 0.0), (0.0, 0.5 + _BELOW_CONST_EPS, 0.0), (0.0, 0.5, 0.0)]
    assert CameraRotationChannel(0, small, tol=0.05).is_constant(0, 2) is True
    big = [(0.0, 0.5, 0.0), (0.0, 0.5 + _ABOVE_CONST_EPS, 0.0), (0.0, 0.5, 0.0)]
    assert CameraRotationChannel(0, big, tol=0.05).is_constant(0, 2) is False


def test_camera_rotation_alternating_plus_minus_pi_is_constant():
    eulers = [(0.0, math.pi, 0.0), (0.0, -math.pi, 0.0), (0.0, math.pi, 0.0), (0.0, -math.pi, 0.0)]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    assert ch.is_constant(0, 3) is True


def test_camera_rotation_deviation_at_span_end_detected():
    eulers = [(0.0, 0.5, 0.0)] * 50
    eulers[-1] = (0.0, 1.5, 0.0)
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    assert ch.is_constant(0, 49) is False


def test_bone_rotation_constant_true():
    quats = [quat_z(33.0)] * 100
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(0, 99) is True


def test_bone_rotation_sign_flip_is_constant():
    q = quat_z(33.0)
    quats = [q, tuple(-c for c in q), q, tuple(-c for c in q)]
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(0, 3) is True


def test_bone_rotation_varying_false():
    quats = [quat_z(float(i)) for i in range(11)]
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(0, 10) is False


def test_bone_rotation_below_threshold_true_above_false():
    small = [quat_z(33.0), quat_z(33.0 + _BELOW_CONST_EPS), quat_z(33.0)]
    assert BoneRotationChannel(0, small, tol=0.1).is_constant(0, 2) is True
    big = [quat_z(33.0), quat_z(33.0 + _ABOVE_CONST_EPS), quat_z(33.0)]
    assert BoneRotationChannel(0, big, tol=0.1).is_constant(0, 2) is False


def test_bone_rotation_deviation_at_span_end_detected():
    quats = [quat_z(33.0)] * 50
    quats[-1] = quat_z(40.0)
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(0, 49) is False


def test_bone_rotation_endpoint_a_included_in_subrange():
    quats = [quat_z(90.0)] * 6
    quats[2] = quat_z(33.0)
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(2, 5) is False


def test_bone_rotation_subrange_constant_true_despite_change_outside():
    quats = [quat_z(33.0)] * 10
    quats[0] = quat_z(90.0)
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(2, 9) is True
