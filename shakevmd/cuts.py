"""カット検出・セグメント分割(shakevmd.md §5.3)。

カット = 「フレームFから新しいショットが始まる」境界。セグメント境界は F-1 と F の
間に置く。bake/noise/motion はここが返すセグメント列を入力として受け取り、
カットの波及(位相独立・速度解析の分割・settle抑制・フェード不要)を構造で保証する。

検出はフレーム差1の隣接カメラキー間で行う:
  - カメラ中心位置のジャンプ、または
  - カメラワールド位置のジャンプ(距離のみ急変するズームカット検出のため)が
    pos_threshold(MMD距離単位)を超える、または
  - 角度ジャンプが rot_threshold(度)を超える
手動指定(add/remove)は自動検出に優先する(コアAPI)。
"""

import math
from dataclasses import dataclass

import numpy as np

from mmd_toolbox.vmd import camera


@dataclass(frozen=True)
class Segment:
    """ベイク範囲を分割した1セグメント [start, end](両端含む、フレーム番号)。"""

    start: int
    end: int


def _geodesic_deg(pose0, pose1) -> float:
    """2つのカメラ姿勢の間の測地角(度)。to_world の前方/上から直交基底を作り、
    trace(R0^T R1) = 1 + 2cosθ の関係から角度を求める(§5.3)。"""

    def basis(p):
        f = np.asarray(p.forward, dtype=float)
        u = np.asarray(p.up, dtype=float)
        r = np.cross(u, f)
        return r, u, f

    r0, u0, f0 = basis(pose0)
    r1, u1, f1 = basis(pose1)
    c = (float(np.dot(r0, r1) + np.dot(u0, u1) + np.dot(f0, f1)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def detect_cuts(keys, pos_threshold: float, rot_threshold: float) -> list[int]:
    """フレーム差1の隣接カメラキー間でカットを検出し、カットフレーム(後側)を昇順で返す。

    keys はフレーム昇順前提(vmd-io.md §5 正規化済み)。
    各隣接ペア (k0, k1) で k1.frame - k0.frame == 1 のとき:
      max(カメラ中心位置ジャンプ, カメラワールド位置ジャンプ) > pos_threshold、または
      角度ジャンプ(度) > rot_threshold なら k1.frame をカットとする(いずれも厳密超過)。
    位置はユークリッド距離、角度は姿勢間の測地角。
    """
    poses = [camera.to_world(k) for k in keys]
    result: list[int] = []
    for i in range(len(keys) - 1):
        k0, k1 = keys[i], keys[i + 1]
        if k1.frame - k0.frame != 1:
            continue
        center_jump = math.dist(k0.position, k1.position)
        world_jump = math.dist(poses[i].position, poses[i + 1].position)
        angle_jump = _geodesic_deg(poses[i], poses[i + 1])
        if max(center_jump, world_jump) > pos_threshold or angle_jump > rot_threshold:
            result.append(k1.frame)
    return result


def resolve_cuts(
    detected: list[int], add: list[int] | None = None, remove: list[int] | None = None
) -> list[int]:
    """自動検出に手動指定を反映する(手動優先)。add は強制追加、remove は打ち消し。

    同一フレームが add と remove 双方にある場合は remove を優先する。
    戻り値は昇順・重複なしのカットフレーム列。
    """
    frames = set(detected) | set(add or [])
    frames -= set(remove or [])
    return sorted(frames)


def segment_bounds(frame_start: int, frame_end: int, cuts: list[int]) -> list[Segment]:
    """[frame_start, frame_end] をカットで分割したセグメント列を返す。

    カットフレーム F は境界を F-1 と F の間に置く(F は次セグメントの先頭)。
    範囲内(frame_start < F <= frame_end)のカットのみ作用する。
    カットがなければ単一セグメント。
    """
    points = sorted({c for c in cuts if frame_start < c <= frame_end})
    segments: list[Segment] = []
    prev = frame_start
    for c in points:
        segments.append(Segment(prev, c - 1))
        prev = c
    segments.append(Segment(prev, frame_end))
    return segments
