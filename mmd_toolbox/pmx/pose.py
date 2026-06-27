"""FK評価(pmx-read-fk-plan.md §5)。

VMDボーンキーを指定フレームでサンプルしてローカル姿勢を作り、PMXボーン
階層に沿って前方運動学でワールド姿勢を評価する。IK・付与親・物理演算は
扱わない近似FK。
"""

import math
from dataclasses import dataclass

import numpy as np

from ..vmd.interp import sample
from .types import PmxModel

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


@dataclass
class LocalBonePose:
    position: tuple[float, float, float]  # VMDローカル移動量
    rotation: tuple[float, float, float, float]  # VMDローカル回転 (x,y,z,w)


@dataclass
class WorldBonePose:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # (x,y,z,w)


def sample_local_poses(model, bone_tracks, frame):
    """フレーム frame の各ボーンのローカル姿勢を返す(model.bones 順)。

    キーの無いボーンは位置 (0,0,0)・単位クォータニオン。
    """
    poses = []
    for bone in model.bones:
        keys = bone_tracks.get(bone.name)
        if keys:
            position = (
                sample(keys, "pos_x", frame),
                sample(keys, "pos_y", frame),
                sample(keys, "pos_z", frame),
            )
            rotation = tuple(sample(keys, "rot", frame))
        else:
            position = (0.0, 0.0, 0.0)
            rotation = _IDENTITY_QUAT
        poses.append(LocalBonePose(position=position, rotation=rotation))
    return tuple(poses)


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_to_quat(m) -> tuple[float, float, float, float]:
    """回転行列からクォータニオン (x,y,z,w)(Shepperd法)。"""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def evaluate_fk(model, local_poses):
    """ローカル姿勢列から各ボーンのワールド姿勢を評価する。

    world = parent_world * translate(base_offset + 移動量) * rotate(回転量)
    親なしボーンはモデル原点を親とする。親が子より後ろのindexでも解決できる
    ようワールド行列をメモ化再帰で求める。
    """
    bones = model.bones
    world_mats: list[np.ndarray | None] = [None] * len(bones)

    def world_of(i: int) -> np.ndarray:
        cached = world_mats[i]
        if cached is not None:
            return cached
        bone = bones[i]
        lp = local_poses[i]
        if bone.parent is None:
            parent_mat = np.eye(4)
            parent_pos = (0.0, 0.0, 0.0)
        else:
            parent_mat = world_of(bone.parent)
            parent_pos = bones[bone.parent].position
        base_offset = (
            bone.position[0] - parent_pos[0],
            bone.position[1] - parent_pos[1],
            bone.position[2] - parent_pos[2],
        )
        if bone.movable:
            translation = (
                base_offset[0] + lp.position[0],
                base_offset[1] + lp.position[1],
                base_offset[2] + lp.position[2],
            )
        else:
            translation = base_offset
        rotation = lp.rotation if bone.rotatable else _IDENTITY_QUAT
        local_mat = np.eye(4)
        local_mat[:3, :3] = _quat_to_matrix(rotation)
        local_mat[:3, 3] = translation
        world_mat = parent_mat @ local_mat
        world_mats[i] = world_mat
        return world_mat

    results = []
    for i in range(len(bones)):
        wm = world_of(i)
        results.append(
            WorldBonePose(
                position=(
                    float(wm[0, 3]),
                    float(wm[1, 3]),
                    float(wm[2, 3]),
                ),
                rotation=_matrix_to_quat(wm[:3, :3]),
            )
        )
    return tuple(results)


def evaluate_fk_range(model, bone_tracks, frames):
    """frames の各フレームについて FK 評価結果を順に返す。"""
    return [
        evaluate_fk(model, sample_local_poses(model, bone_tracks, f))
        for f in frames
    ]
