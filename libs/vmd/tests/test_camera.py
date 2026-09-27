import math

import numpy as np
import pytest

from vmd import camera, normalize, read
from vmd.types import CameraKey


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _matrix(rx, ry, rz):
    return _ry(-ry) @ _rx(-rx) @ _rz(-rz)


def _fk(center, distance, rotation):
    rx, ry, rz = rotation
    v = np.array([0.0, 0.0, distance])
    v = _rz(-rz) @ v
    v = _rx(-rx) @ v
    v = _ry(-ry) @ v
    pos = np.array(center, dtype=float) + v
    R = _matrix(rx, ry, rz)
    forward = R @ np.array([0.0, 0.0, 1.0])
    up = R @ np.array([0.0, 1.0, 0.0])
    return pos, forward, up


def cam_key(center=(0.0, 0.0, 0.0), distance=-30.0, rotation=(0.0, 0.0, 0.0),
            fov=30, perspective=0):
    return CameraKey(0, distance, center, rotation, bytes(24), fov, perspective)


def _approx(a, b, abs=1e-6):
    return np.asarray(a) == pytest.approx(np.asarray(b), abs=abs)


GRID = [
    ((0.0, 0.0, 0.0), -30.0, (0.0, 0.0, 0.0)),
    ((1.0, 2.0, 3.0), -45.0, (0.3, 0.0, 0.0)),
    ((-5.0, 10.0, 2.0), -20.0, (0.0, 0.7, 0.0)),
    ((0.0, 8.0, 0.0), -50.0, (0.0, 0.0, 0.5)),
    ((3.0, -2.0, 7.0), -15.0, (0.2, -0.6, 0.4)),
    ((0.0, 0.0, 0.0), -100.0, (-0.4, 1.2, -0.3)),
]


class TestToWorldMatchesReference:
    @pytest.mark.parametrize("center,distance,rotation", GRID)
    def test_synthetic(self, center, distance, rotation):
        pose = camera.to_world(cam_key(center, distance, rotation))
        pos, fwd, up = _fk(center, distance, rotation)
        assert _approx(pose.position, pos)
        assert _approx(pose.forward, fwd)
        assert _approx(pose.up, up)

    def test_forward_points_from_camera_to_center_when_distance_negative(self):
        center = np.array([1.0, 2.0, 3.0])
        key = cam_key(tuple(center), -25.0, (0.3, 0.5, 0.0))
        pose = camera.to_world(key)
        to_center = center - np.array(pose.position)
        to_center /= np.linalg.norm(to_center)
        assert _approx(pose.forward, to_center, abs=1e-6)

    def test_real_file(self, camera_basic_bytes):
        doc, _ = read(camera_basic_bytes)
        doc, _ = normalize(doc, ["camera"])
        for k in doc.camera:
            pose = camera.to_world(k)
            pos, fwd, up = _fk(k.position, k.distance, k.rotation)
            assert _approx(pose.position, pos, abs=1e-4)
            assert _approx(pose.forward, fwd, abs=1e-6)
            assert pose.fov == k.fov
            assert pose.perspective == k.perspective


class TestRoundTrip:
    @pytest.mark.parametrize("center,distance,rotation", GRID)
    def test_recovers_center_and_rotation(self, center, distance, rotation):
        pose = camera.to_world(cam_key(center, distance, rotation))
        got = camera.from_world(pose, distance, prev_rotation=rotation)
        assert _approx(got["position"], center, abs=1e-5)
        assert _approx(got["rotation"], rotation, abs=1e-5)

    def test_recovered_orientation_matches(self):
        center, distance, rotation = (2.0, 1.0, -3.0), -40.0, (0.5, -0.8, 0.6)
        pose = camera.to_world(cam_key(center, distance, rotation))
        got = camera.from_world(pose, distance, prev_rotation=rotation)
        pos2, fwd2, up2 = _fk(got["position"], distance, got["rotation"])
        assert _approx(pose.position, pos2, abs=1e-5)
        assert _approx(pose.forward, fwd2, abs=1e-5)
        assert _approx(pose.up, up2, abs=1e-5)


class TestDistanceZero:
    def test_position_equals_center(self):
        center = (1.0, 2.0, 3.0)
        pose = camera.to_world(cam_key(center, 0.0, (0.3, 0.5, 0.7)))
        assert _approx(pose.position, center, abs=1e-9)

    def test_from_world_recovers_center(self):
        center = (4.0, -1.0, 2.0)
        rotation = (0.2, 0.4, -0.1)
        pose = camera.to_world(cam_key(center, 0.0, rotation))
        got = camera.from_world(pose, 0.0, prev_rotation=rotation)
        assert _approx(got["position"], center, abs=1e-9)

    def test_orientation_still_follows_rotation(self):
        rotation = (0.3, 0.5, 0.7)
        pose = camera.to_world(cam_key((1.0, 2.0, 3.0), 0.0, rotation))
        _, forward, up = _fk((1.0, 2.0, 3.0), 0.0, rotation)
        assert _approx(pose.forward, forward, abs=1e-9)
        assert _approx(pose.up, up, abs=1e-9)


class TestAngleContinuity:
    def test_yaw_sweep_across_pi_is_continuous(self):
        prev = None
        recovered = []
        for ry in np.linspace(2.9, 3.4, 12):
            pose = camera.to_world(cam_key((0.0, 0.0, 0.0), -30.0, (0.0, float(ry), 0.0)))
            got = camera.from_world(pose, -30.0, prev_rotation=prev)
            recovered.append(got["rotation"][1])
            prev = got["rotation"]
        diffs = np.diff(recovered)
        assert np.all(np.abs(diffs) < math.pi), f"不連続なジャンプ: {diffs}"


class TestGimbal:
    @pytest.mark.parametrize("pitch", [math.pi / 2 - 1e-4, -math.pi / 2 + 1e-4,
                                       math.pi / 2, -math.pi / 2])
    def test_roundtrip_orientation_stable_near_gimbal(self, pitch):
        center, distance, rotation = (1.0, 0.0, 0.0), -30.0, (pitch, 0.6, 0.3)
        pose = camera.to_world(cam_key(center, distance, rotation))
        got = camera.from_world(pose, distance, prev_rotation=rotation)
        pos2, fwd2, up2 = _fk(got["position"], distance, got["rotation"])
        assert _approx(pose.position, pos2, abs=1e-4)
        assert _approx(pose.forward, fwd2, abs=1e-4)
        assert _approx(pose.up, up2, abs=1e-4)

    @pytest.mark.parametrize("pitch", [math.pi / 2, -math.pi / 2])
    def test_exact_gimbal_keeps_prev_rotation_z(self, pitch):
        rotation = (pitch, 0.6, 0.3)
        pose = camera.to_world(cam_key((1.0, 0.0, 0.0), -30.0, rotation))
        got = camera.from_world(pose, -30.0, prev_rotation=rotation)
        assert got["rotation"][2] == pytest.approx(rotation[2], abs=1e-9)
