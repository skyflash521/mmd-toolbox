import math
from dataclasses import dataclass

import numpy as np

from vmd import camera


@dataclass(frozen=True)
class Segment:
    """end もセグメントに含む。"""

    start: int
    end: int


def _geodesic_deg(pose0, pose1) -> float:
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
    """keys はフレーム昇順で重複が無いこと。pos_threshold は MMD の距離単位、rot_threshold は度。"""
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
    frames = set(detected) | set(add or [])
    frames -= set(remove or [])
    return sorted(frames)


def segment_bounds(frame_start: int, frame_end: int, cuts: list[int]) -> list[Segment]:
    points = sorted({c for c in cuts if frame_start < c <= frame_end})
    segments: list[Segment] = []
    prev = frame_start
    for c in points:
        segments.append(Segment(prev, c - 1))
        prev = c
    segments.append(Segment(prev, frame_end))
    return segments
