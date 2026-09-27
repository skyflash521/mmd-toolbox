import math

import pytest

from vmd.fit import EuclideanVectorChannel, FovChannel, _round_half_up


def test_vector_linear_zero_error():
    vecs = [(float(i), float(2 * i), float(-i)) for i in range(11)]
    ch = EuclideanVectorChannel(0, vecs, tol=0.01)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0)
    assert frame is None


def test_vector_error_is_euclidean_over_axes():
    vecs = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (0.0, 0.0, 0.0)]
    ch = EuclideanVectorChannel(0, vecs, tol=1.0)
    err, frame = ch.residual(0, 2)
    assert err == pytest.approx(5.0)
    assert frame == 1


def test_vector_normalized_and_zero_tol():
    vecs = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (0.0, 0.0, 0.0)]
    ch = EuclideanVectorChannel(0, vecs, tol=2.0)
    nerr, frame = ch.normalized(0, 2)
    assert nerr == pytest.approx(2.5)
    assert frame == 1
    ch0 = EuclideanVectorChannel(0, vecs, tol=0.0)
    nerr0, frame0 = ch0.normalized(0, 2)
    assert nerr0 == math.inf and frame0 == 1


def test_vector_split_frame_at_max_distance():
    vecs = [(0.0, 0.0, 0.0)] * 5
    vecs[3] = (0.0, 0.0, 10.0)
    ch = EuclideanVectorChannel(0, vecs, tol=1.0)
    err, frame = ch.residual(0, 4)
    assert frame == 3
    assert err == pytest.approx(10.0)


def test_vector_prefers_velocity_reversal_over_max_error():
    vecs = [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0), (0.0, 0.0, 2.0), (0.0, 0.0, 4.0), (0.0, 0.0, 20.0)]
    ch = EuclideanVectorChannel(0, vecs, tol=1.0)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(11.0)
    assert frame == 2


def test_fov_integer_linear_zero_error():
    vals = [float(30 + i) for i in range(11)]
    ch = FovChannel(0, vals, tol=0.5)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0)


def test_fov_error_includes_rounding_to_integer_degrees():
    vals = [30.0 + 0.1 * i for i in range(11)]
    ch = FovChannel(0, vals, tol=0.5)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.5)


def test_fov_half_integer_quantization_is_half_degree():
    vals = [30.0, 30.5, 31.0, 31.5, 32.0]
    ch = FovChannel(0, vals, tol=0.5)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(0.5)


def test_fov_round_down_below_half():
    vals = [30.0, 30.4, 30.8]
    ch = FovChannel(0, vals, tol=0.5)
    err, frame = ch.residual(0, 2)
    assert err == pytest.approx(0.4)


def test_round_half_up_is_not_bankers():
    assert _round_half_up(0.5) == 1
    assert _round_half_up(1.5) == 2
    assert _round_half_up(2.5) == 3
    assert _round_half_up(30.4) == 30
    assert _round_half_up(30.6) == 31


def test_fov_prefers_velocity_reversal_over_max_error():
    ch = FovChannel(0, [30.0, 40.0, 32.0, 34.0, 50.0], tol=1.0)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(11.0)
    assert frame == 2


def test_fov_normalized_and_zero_tol():
    vals = [30.0, 30.0, 35.0, 30.0, 30.0]
    ch = FovChannel(0, vals, tol=2.0)
    nerr, frame = ch.normalized(0, 4)
    assert nerr == pytest.approx(2.5)
    assert frame == 2
    ch0 = FovChannel(0, vals, tol=0.0)
    nerr0, frame0 = ch0.normalized(0, 4)
    assert nerr0 == math.inf and frame0 == 2


def test_fov_large_deviation_detected():
    vals = [30.0, 30.0, 35.0, 30.0, 30.0]
    ch = FovChannel(0, vals, tol=0.5)
    err, frame = ch.residual(0, 4)
    assert frame == 2
    assert err == pytest.approx(5.0)
