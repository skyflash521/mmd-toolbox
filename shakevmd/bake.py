"""ベイクループ・視線揺れ変換(shakevmd.md §3, §4)。

サブステップA(本コミット)は §4.2 の視線揺れ変換 apply_gaze_shake のみを実装対象とする。
ベイクループ本体 bake() はサブステップBで実装する(現状スタブ)。

§4.2 の実現方針: 揺れ角度は「元の角度 + ノイズ」のオイラー加算とし、カメラの
ワールド位置が固定される(位置ノイズ分だけシフトする)ように新しいカメラ中心を
逆算する。これにより距離0では自動的に素朴な角度加算と一致し、姿勢の再分解
(camera.from_world)やジンバル対策は不要になる。
"""

import math
from dataclasses import dataclass

import numpy as np

from mmd_toolbox.vmd import camera
from mmd_toolbox.vmd.types import CameraKey


def round_half_up(x) -> int:
    """整数度への丸め(四捨五入)。視野角の丸め既定は四捨五入(vmd-interp.md §3/§5)。

    Python 組み込み round() は銀行丸め(round half to even)で .5 境界が
    四捨五入と食い違うため使わない。視野角は正なので floor(x+0.5) で四捨五入になる。
    """
    return int(math.floor(float(x) + 0.5))


def apply_gaze_shake(camera_key, rot_noise, pos_noise=(0.0, 0.0, 0.0)) -> dict:
    """視線揺れ変換(§4.2)。

    - 揺れ角度 = camera_key.rotation + rot_noise(成分ごとのオイラー加算)
    - カメラのワールド位置は固定(pos_noise 指定時はその分だけワールドでシフト)
    - 新しい角度・距離からカメラ中心を逆算: center = cam_pos - R(新角度)·(0,0,distance)
    rot_noise=(drx,dry,drz) ラジアン、pos_noise=(dx,dy,dz)。
    戻り値: {"position": (cx,cy,cz), "rotation": (rx,ry,rz)}
    """
    new_rot = tuple(camera_key.rotation[i] + rot_noise[i] for i in range(3))
    # カメラワールド位置(位置ノイズ分シフト)
    cam_pos = np.array(camera.to_world(camera_key).position, dtype=float)
    cam_pos = cam_pos + np.array(pos_noise, dtype=float)
    # offset = R(new_rot)·(0,0,distance) を中心0のプローブから取得(camera 規約を再利用)
    probe = CameraKey(
        0, camera_key.distance, (0.0, 0.0, 0.0), new_rot, bytes(24),
        camera_key.fov, camera_key.perspective,
    )
    offset = np.array(camera.to_world(probe).position, dtype=float)
    center = cam_pos - offset
    return {"position": tuple(center), "rotation": new_rot}


# MMDデフォルトの線形補間ブロック(カメラ24バイト = 6チャンネル × (20,107,20,107))
LINEAR_CAMERA_INTERP = bytes([20, 107, 20, 107]) * 6

# ベイクは30fps・1フレーム間隔固定(§4.1)。間隔変更オプションは持たない。
FPS = 30.0


@dataclass
class BakeResult:
    """bake() の結果。"""

    camera_keys: list   # ベイク後の全カメラキー(範囲内=高密度、範囲外=原本)
    warnings: list


def bake(
    camera_keys,
    ranges=None,
    *,
    seed: int = 1,
    amp_rot: float = 0.8,        # 度
    amp_pos: float = 0.05,       # MMD距離単位
    rot_weights=(1.0, 1.0, 0.3),  # Pitch/Yaw/Roll = rx/ry/rz 個別重み
    freq: float = 1.2,
    motion_scale: float = 0.5,
    fade_sec: float = 0.7,
    cut_pos_threshold: float = 5.0,
    cut_rot_threshold: float = 20.0,
    manual_cuts_add=(),
    manual_cuts_remove=(),
) -> BakeResult:
    """ベイクループ本体(§3, §4, §5)。

    camera_keys(原本、順不同可)を受け取り、ranges(各 (start,end)、None=全範囲)の
    範囲を30fps・1フレーム間隔でベイクした高密度キーを生成し、範囲外は原本を
    バイト保持して結合した全カメラキー列を返す。
    - 正規化作業ビュー(ソート・重複後勝ち)でサンプリング/範囲解決/カット検出
    - フェードは範囲レベル(範囲端で揺れ0、内部カット境界では非ゼロ)
    - ノイズ位相はセグメントごとに独立(derive_seed)
    - 視野角は四捨五入、パースは直前キーをホールド
    """
    raise NotImplementedError
