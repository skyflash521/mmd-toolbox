"""カメラモデルの座標変換(vmd-camera.md)。

「カメラ中心 + 距離 + 角度」表現とワールド座標表現(カメラ位置 + 前方軸 + 上方向)の
相互変換。回転規約は vmd-camera.md §2・§4 で確定:
  R = Ry(-ry) · Rx(-rx) · Rz(-rz)
  カメラワールド位置 = カメラ中心 + R · (0, 0, distance)
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraPose:
    """カメラのワールド姿勢。"""

    position: tuple  # カメラワールド位置 (x, y, z)
    forward: tuple   # 前方軸 R·(0,0,1)(単位ベクトル)
    up: tuple        # 上方向 R·(0,1,0)(単位ベクトル)
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
    """R = Ry(-ry) · Rx(-rx) · Rz(-rz)(vmd-camera.md §2)。"""
    return _ry(-ry) @ _rx(-rx) @ _rz(-rz)


def to_world(camera_key) -> CameraPose:
    """カメラキー(中心・距離・角度)からワールド姿勢を算出する(§2)。"""
    R = _rotation_matrix(*camera_key.rotation)
    center = np.array(camera_key.position, dtype=float)
    forward = R @ np.array([0.0, 0.0, 1.0])
    up = R @ np.array([0.0, 1.0, 0.0])
    position = center + R @ np.array([0.0, 0.0, camera_key.distance])
    return CameraPose(
        tuple(position), tuple(forward), tuple(up),
        camera_key.fov, camera_key.perspective,
    )


def _unwrap(angle: float, ref: float) -> float:
    """angle を ref に最も近い 2π 等価表現へ移す(±180°ジャンプ防止)。"""
    return angle + 2.0 * math.pi * round((ref - angle) / (2.0 * math.pi))


def _decompose(M: np.ndarray, prev_rotation):
    """R = Ry(-ry)·Rx(-rx)·Rz(-rz) から (rx, ry, rz) を取り出す。

    内部では A=-ry, B=-rx, C=-rz の Y-X-Z 分解を解く。
    ピッチ B=±90°(ジンバル)では A,C が縮退するため prev_rotation で C を固定する。
    """
    sB = max(-1.0, min(1.0, -M[1, 2]))
    B = math.asin(sB)
    cB = math.cos(B)
    if abs(cB) > 1e-6:
        A = math.atan2(M[0, 2], M[2, 2])
        C = math.atan2(M[1, 0], M[1, 1])
    else:
        C = -prev_rotation[2] if prev_rotation is not None else 0.0
        if sB > 0.0:  # B = +90°: A - C = atan2(M01, M00)
            A = math.atan2(M[0, 1], M[0, 0]) + C
        else:         # B = -90°: A + C = atan2(-M01, M00)
            A = math.atan2(-M[0, 1], M[0, 0]) - C

    rx, ry, rz = -B, -A, -C
    if prev_rotation is not None:
        rx = _unwrap(rx, prev_rotation[0])
        ry = _unwrap(ry, prev_rotation[1])
        rz = _unwrap(rz, prev_rotation[2])
    return (rx, ry, rz)


def from_world(pose: CameraPose, distance: float, prev_rotation=None) -> dict:
    """ワールド姿勢から「カメラ中心・角度」を逆算する(§5)。

    戻り値: {"position": (cx, cy, cz), "rotation": (rx, ry, rz)}
    prev_rotation: 直前フレームの角度。指定時はオイラー角をこれに最も近い表現へ
      アンラップし、±180°ジャンプとジンバル縮退を解消する。
    """
    fwd = np.array(pose.forward, dtype=float)
    fwd /= np.linalg.norm(fwd)
    up = np.array(pose.up, dtype=float)
    # 数値誤差・揺れ適用による非直交を除去(前方軸に対して上方向を再直交化)
    up = up - np.dot(up, fwd) * fwd
    up /= np.linalg.norm(up)
    right = np.cross(up, fwd)  # 右手系: e0 = e1 × e2

    R = np.column_stack([right, up, fwd])  # 列 = [right, up, forward] = 元のR
    center = np.array(pose.position, dtype=float) - distance * fwd
    rotation = _decompose(R, prev_rotation)
    return {"position": tuple(center), "rotation": rotation}
