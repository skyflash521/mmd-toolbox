"""補間曲線評価・サンプリング(vmd-interp.md)。

入力キー列はフレーム昇順前提(vmd-io.md §5 で正規化済みであること)。
補間ブロックのバイト配置: docs/specs/vmd/VMD_file_format.md
"""

from .types import BoneKey, CameraKey

# カメラ補間ブロック(24バイト)内の各チャンネルのオフセット。
# 各チャンネルは ax, bx, ay, by の順で4バイト(始点X, 終点X, 始点Y, 終点Y)。
CAMERA_CHANNEL_OFFSET = {
    "pos_x": 0,
    "pos_y": 4,
    "pos_z": 8,
    "rot": 12,
    "distance": 16,
    "fov": 20,
}


def sample(keys, channel: str, frame: int):
    """1チャンネルを1フレームで評価する(vmd-interp.md §3・§4)。

    channel:
      カメラキー列 -> "pos_x" / "pos_y" / "pos_z" / "rot" / "distance" / "fov"
      ボーンキー列 -> "pos_x" / "pos_y" / "pos_z" / "rot"
    戻り値:
      スカラーチャンネル -> float
      カメラ "rot" -> オイラー角 (rx, ry, rz)
      ボーン "rot" -> クォータニオン (x, y, z, w)
    """
    raise NotImplementedError


def sample_range(keys, channel: str, frame_start: int, frame_end: int):
    """[frame_start, frame_end] を1フレーム間隔(両端含む)で評価する。

    ベイク用途の一括評価。numpy配列(スカラーチャンネル)または
    各フレームの値のリストを返す。
    """
    raise NotImplementedError


def sample_camera(keys, frame: int) -> dict:
    """カメラ全チャンネルを1フレームで評価する。

    戻り値: {"distance": float, "position": (x, y, z),
             "rotation": (rx, ry, rz), "fov": float}
    """
    raise NotImplementedError
