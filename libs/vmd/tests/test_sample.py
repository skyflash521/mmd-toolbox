import pytest

from vmd.sample import perspective_series
from vmd.types import CameraKey

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, persp)


def test_perspective_series_holds_latest_key_value_until_next_key():
    keys = [cam(0, persp=0), cam(30, persp=1)]
    series = perspective_series(keys, 0, 31)
    assert series[0] == 0
    assert series[29] == 0
    assert series[30] == 1
    assert series[31] == 1


def test_perspective_series_empty_raises():
    with pytest.raises(ValueError):
        perspective_series([], 0, 10)


def test_perspective_before_first_key_uses_first():
    keys = [cam(10, persp=1), cam(20, persp=0)]
    series = perspective_series(keys, 0, 20)
    assert series[0] == 1
    assert series[20] == 0
