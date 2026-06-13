"""ベイクループ・視線揺れ変換(shakevmd.md §3, §4)。

サブステップA(本コミット)は §4.2 の視線揺れ変換 apply_gaze_shake のみを実装対象とする。
ベイクループ本体 bake() はサブステップBで実装する(現状スタブ)。

§4.2 の実現方針: 揺れ角度は「元の角度 + ノイズ」のオイラー加算とし、カメラの
ワールド位置が固定される(位置ノイズ分だけシフトする)ように新しいカメラ中心を
逆算する。これにより距離0では自動的に素朴な角度加算と一致し、姿勢の再分解
(camera.from_world)やジンバル対策は不要になる。
"""

from dataclasses import dataclass

import numpy as np

from mmd_toolbox.vmd import camera
from mmd_toolbox.vmd.types import CameraKey


def apply_gaze_shake(camera_key, rot_noise, pos_noise=(0.0, 0.0, 0.0)) -> dict:
    """視線揺れ変換(§4.2)。

    - 揺れ角度 = camera_key.rotation + rot_noise(成分ごとのオイラー加算)
    - カメラのワールド位置は固定(pos_noise 指定時はその分だけワールドでシフト)
    - 新しい角度・距離からカメラ中心を逆算: center = cam_pos - R(新角度)·(0,0,distance)
    rot_noise=(drx,dry,drz) ラジアン、pos_noise=(dx,dy,dz)。
    戻り値: {"position": (cx,cy,cz), "rotation": (rx,ry,rz)}
    """
    raise NotImplementedError


@dataclass
class BakeResult:
    """bake() の結果(サブステップBで具体化)。"""

    camera_keys: list
    warnings: list


def bake(camera_keys, ranges=None, **options) -> BakeResult:
    """ベイクループ本体(§3, §4, §5)。サブステップBで実装。"""
    raise NotImplementedError
