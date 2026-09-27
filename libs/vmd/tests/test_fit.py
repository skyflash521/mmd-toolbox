import math

import pytest

from vmd.fit import LinearScalarChannel


def test_linear_values_zero_error():
    ch = LinearScalarChannel(0, [float(i) for i in range(11)], tol=0.01)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0)
    assert frame is None


def test_tent_peak_detected():
    vals = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ch = LinearScalarChannel(0, [float(v) for v in vals], tol=1.0)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(5.0)
    assert frame == 5


def test_residual_adjacent_no_interior():
    ch = LinearScalarChannel(0, [0.0, 9.0], tol=1.0)
    err, frame = ch.residual(0, 1)
    assert err == pytest.approx(0.0)
    assert frame is None


def test_residual_subsegment_uses_its_own_endpoints():
    vals = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ch = LinearScalarChannel(0, [float(v) for v in vals], tol=1.0)
    err, frame = ch.residual(5, 10)
    assert err == pytest.approx(0.0)
    assert frame is None


def test_residual_frames_are_absolute_from_frame_start():
    vals = [0, 5, 0]
    ch = LinearScalarChannel(100, [float(v) for v in vals], tol=1.0)
    err, frame = ch.residual(100, 102)
    assert err == pytest.approx(5.0)
    assert frame == 101


def test_residual_prefers_velocity_reversal_over_max_error():
    ch = LinearScalarChannel(0, [0.0, 10.0, 2.0, 4.0, 20.0], tol=1.0)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(11.0)
    assert frame == 2


def test_normalized_divides_by_tol():
    vals = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ch = LinearScalarChannel(0, [float(v) for v in vals], tol=2.0)
    nerr, frame = ch.normalized(0, 10)
    assert nerr == pytest.approx(2.5)
    assert frame == 5


def test_normalized_zero_tol_positive_error_is_inf():
    ch = LinearScalarChannel(0, [0.0, 5.0, 0.0], tol=0.0)
    nerr, frame = ch.normalized(0, 2)
    assert nerr == math.inf
    assert frame == 1


def test_normalized_zero_tol_zero_error_is_zero():
    ch = LinearScalarChannel(0, [0.0, 1.0, 2.0], tol=0.0)
    nerr, frame = ch.normalized(0, 2)
    assert nerr == 0.0
    assert frame is None
