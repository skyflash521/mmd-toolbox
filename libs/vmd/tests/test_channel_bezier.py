import math

import pytest

from vmd import interp
from vmd.fit import (
    BoneRotationChannel,
    CameraRotationChannel,
    EuclideanVectorChannel,
    FovChannel,
    LinearScalarChannel,
)


def _eased_scalar(v0, v1, curve, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*curve, f / span) for f in range(n)]


EASE = (96, 0, 96, 30)


def test_scalar_bezier_low_error_on_eased_curve():
    vals = _eased_scalar(0.0, 100.0, EASE)
    lin = LinearScalarChannel(0, vals, tol=1.0, mode="linear")
    bez = LinearScalarChannel(0, vals, tol=1.0, mode="bezier")
    lin_err, _ = lin.residual(0, 10)
    bez_err, _ = bez.residual(0, 10)
    assert lin_err > 5.0
    assert bez_err < 0.5
    assert bez_err < lin_err * 0.2


def test_scalar_bezier_worst_frame_from_bezier_profile():
    vals = _eased_scalar(0.0, 100.0, EASE)
    vals[8] += 20.0
    bez_err, bez_frame = LinearScalarChannel(0, vals, tol=1.0, mode="bezier").residual(0, 10)
    lin_err, lin_frame = LinearScalarChannel(0, vals, tol=1.0, mode="linear").residual(0, 10)
    assert bez_frame == 8
    assert lin_frame == 7
    assert bez_frame != lin_frame


def test_scalar_bezier_linear_data_zero_error():
    vals = [float(i) for i in range(11)]
    bez = LinearScalarChannel(0, vals, tol=1.0, mode="bezier")
    err, _ = bez.residual(0, 10)
    assert err < 1e-6


def test_scalar_bezier_default_mode_is_linear():
    vals = _eased_scalar(0.0, 100.0, EASE)
    ch = LinearScalarChannel(0, vals, tol=1.0)
    err, _ = ch.residual(0, 10)
    assert err > 5.0


def test_scalar_bezier_equal_endpoints_error_is_deviation_from_flat():
    vals = [0.0, 3.0, 5.0, 3.0, 0.0]
    bez = LinearScalarChannel(0, vals, tol=1.0, mode="bezier")
    err, _ = bez.residual(0, 4)
    assert err == pytest.approx(5.0)


def test_fov_bezier_low_error_on_eased_curve():
    vals = _eased_scalar(30.0, 80.0, EASE)
    lin = FovChannel(0, vals, tol=1.0, mode="linear")
    bez = FovChannel(0, vals, tol=1.0, mode="bezier")
    bez_err = bez.residual(0, 10)[0]
    assert bez_err < lin.residual(0, 10)[0]
    assert bez_err < 1.0


def test_vector_bezier_error_is_euclidean_not_per_axis_max():
    bump = [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vx = [(bump[i], 0.0, 0.0) for i in range(11)]
    vxz = [(bump[i], 0.0, bump[i]) for i in range(11)]
    ex = EuclideanVectorChannel(0, vx, tol=1.0, mode="bezier").residual(0, 10)[0]
    exz = EuclideanVectorChannel(0, vxz, tol=1.0, mode="bezier").residual(0, 10)[0]
    assert ex == pytest.approx(5.0)
    assert exz == pytest.approx(5.0 * math.sqrt(2), abs=1e-6)
    assert exz > ex * 1.3


def test_vector_bezier_per_axis_low_error():
    ys = _eased_scalar(0.0, 100.0, EASE)
    vecs = [(0.0, y, 0.0) for y in ys]
    lin = EuclideanVectorChannel(0, vecs, tol=1.0, mode="linear")
    bez = EuclideanVectorChannel(0, vecs, tol=1.0, mode="bezier")
    assert lin.residual(0, 10)[0] > 5.0
    assert bez.residual(0, 10)[0] < 0.5


def test_vector_bezier_two_axes_with_different_eases_low_error():
    ay = _eased_scalar(0.0, 50.0, EASE)
    az = _eased_scalar(0.0, 50.0, (0, 30, 96, 96))
    vecs = [(0.0, ay[i], az[i]) for i in range(11)]
    bez = EuclideanVectorChannel(0, vecs, tol=1.0, mode="bezier")
    err, _ = bez.residual(0, 10)
    assert err < 1.0


def _eased_coeff(curve, n=11):
    span = n - 1
    return [interp._solve_factor(*curve, f / span) for f in range(n)]


def test_camera_rotation_bezier_shared_curve_low_error():
    c = _eased_coeff(EASE)
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    eulers = [(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f]) for f in range(11)]
    lin = CameraRotationChannel(0, eulers, tol=0.1, mode="linear")
    bez = CameraRotationChannel(0, eulers, tol=0.1, mode="bezier")
    assert lin.residual(0, 10)[0] > 1.0
    assert bez.residual(0, 10)[0] < 0.1


def test_camera_rotation_bezier_default_mode_is_linear():
    c = _eased_coeff(EASE)
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    eulers = [(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f]) for f in range(11)]
    ch = CameraRotationChannel(0, eulers, tol=0.1)
    assert ch.residual(0, 10)[0] > 1.0


def test_camera_rotation_bezier_linear_data_zero_error():
    eulers = [(0.01 * f, -0.02 * f, 0.005 * f) for f in range(11)]
    bez = CameraRotationChannel(0, eulers, tol=0.1, mode="bezier")
    assert bez.residual(0, 10)[0] < 1e-3


def _z_quat(deg):
    h = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(h), math.cos(h))


def test_bone_rotation_bezier_slerp_coeff_low_error():
    c = _eased_coeff(EASE)
    quats = [_z_quat(90.0 * c[f]) for f in range(11)]
    lin = BoneRotationChannel(0, quats, tol=0.1, mode="linear")
    bez = BoneRotationChannel(0, quats, tol=0.1, mode="bezier")
    assert lin.residual(0, 10)[0] > 1.0
    assert bez.residual(0, 10)[0] < 0.1


def test_bone_rotation_bezier_default_mode_is_linear():
    c = _eased_coeff(EASE)
    quats = [_z_quat(90.0 * c[f]) for f in range(11)]
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.residual(0, 10)[0] > 1.0


def test_bone_rotation_bezier_linear_data_zero_error():
    quats = [_z_quat(90.0 * (f / 10.0)) for f in range(11)]
    bez = BoneRotationChannel(0, quats, tol=0.1, mode="bezier")
    assert bez.residual(0, 10)[0] < 1e-3
