"""カメラモデルの座標変換(vmd-camera.md)。

「カメラ中心 + 距離 + 角度」表現とワールド座標表現(カメラ位置 + 前方軸 + 上方向)の
相互変換。回転規約は vmd-camera.md §2・§4 で確定:
  R = Ry(-ry) · Rx(-rx) · Rz(-rz)
  カメラワールド位置 = カメラ中心 + R · (0, 0, distance)
"""

from dataclasses import dataclass


@dataclass
class CameraPose:
    """カメラのワールド姿勢。"""

    position: tuple  # カメラワールド位置 (x, y, z)
    forward: tuple   # 前方軸 R·(0,0,1)(単位ベクトル)
    up: tuple        # 上方向 R·(0,1,0)(単位ベクトル)
    fov: float
    perspective: int


def to_world(camera_key) -> CameraPose:
    """カメラキー(中心・距離・角度)からワールド姿勢を算出する(§2)。"""
    raise NotImplementedError


def from_world(pose: CameraPose, distance: float, prev_rotation=None) -> dict:
    """ワールド姿勢から「カメラ中心・角度」を逆算する(§5)。

    戻り値: {"position": (cx, cy, cz), "rotation": (rx, ry, rz)}
    prev_rotation: 直前フレームの角度 (rx, ry, rz)。指定時はオイラー角を
      これに最も近い表現へアンラップし、±180°ジャンプを防ぐ(§5)。
    """
    raise NotImplementedError
