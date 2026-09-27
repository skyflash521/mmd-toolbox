import math
from dataclasses import dataclass

import numpy as np

_GIMBAL_COS_EPS = 1e-6


@dataclass
class CameraPose:
    """position はカメラ本体のワールド位置(カメラ中心ではない)。forward と up は単位ベクトル。"""

    position: tuple
    forward: tuple
    up: tuple
    fov: float
    perspective: int


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    return _ry(-ry) @ _rx(-rx) @ _rz(-rz)


def to_world(camera_key) -> CameraPose:
    rotation_matrix = _rotation_matrix(*camera_key.rotation)
    center = np.array(camera_key.position, dtype=float)
    forward = rotation_matrix @ np.array([0.0, 0.0, 1.0])
    up = rotation_matrix @ np.array([0.0, 1.0, 0.0])
    position = center + rotation_matrix @ np.array([0.0, 0.0, camera_key.distance])
    return CameraPose(
        tuple(position), tuple(forward), tuple(up),
        camera_key.fov, camera_key.perspective,
    )


def _unwrap_near(angle: float, ref: float) -> float:
    return angle + 2.0 * math.pi * round((ref - angle) / (2.0 * math.pi))


def _decompose(rotation_matrix: np.ndarray, prev_rotation):
    sin_neg_rx = max(-1.0, min(1.0, -rotation_matrix[1, 2]))
    neg_rx = math.asin(sin_neg_rx)
    if abs(math.cos(neg_rx)) > _GIMBAL_COS_EPS:
        neg_ry = math.atan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
        neg_rz = math.atan2(rotation_matrix[1, 0], rotation_matrix[1, 1])
    else:
        neg_rz = -prev_rotation[2] if prev_rotation is not None else 0.0
        if sin_neg_rx > 0.0:
            neg_ry = math.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0]) + neg_rz
        else:
            neg_ry = math.atan2(-rotation_matrix[0, 1], rotation_matrix[0, 0]) - neg_rz

    rx, ry, rz = -neg_rx, -neg_ry, -neg_rz
    if prev_rotation is not None:
        rx = _unwrap_near(rx, prev_rotation[0])
        ry = _unwrap_near(ry, prev_rotation[1])
        rz = _unwrap_near(rz, prev_rotation[2])
    return (rx, ry, rz)


def from_world(pose: CameraPose, distance: float, prev_rotation=None) -> dict:
    """{"position": カメラ中心, "rotation": 角度(ラジアン)} を返す。

    prev_rotation を渡すと、角度はそれに最も近い 2π 等価な表現で返り、ジンバル位置では
    prev_rotation の Z 成分を保つ。
    """
    fwd = np.array(pose.forward, dtype=float)
    fwd /= np.linalg.norm(fwd)
    up = np.array(pose.up, dtype=float)
    up = up - np.dot(up, fwd) * fwd
    up /= np.linalg.norm(up)
    right = np.cross(up, fwd)

    rotation_matrix = np.column_stack([right, up, fwd])
    center = np.array(pose.position, dtype=float) - distance * fwd
    rotation = _decompose(rotation_matrix, prev_rotation)
    return {"position": tuple(center), "rotation": rotation}
