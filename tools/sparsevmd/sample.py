from dataclasses import dataclass
from typing import Literal

from vmd import interp, io
from vmd.sample import perspective_series  # noqa: F401


@dataclass
class Track:
    """name は kind が camera なら None、bone ならボーン表示名。"""

    kind: Literal["camera", "bone"]
    name: str | None
    keys: list
    first: int
    last: int


def build_tracks(doc, target: Literal["camera", "bone", "all"]) -> list[Track]:
    sections = {"camera": ["camera"], "bone": ["bone"], "all": ["camera", "bone"]}[target]
    normalized, _warnings = io.normalize(doc, sections=sections)

    tracks = []
    if "camera" in sections and normalized.camera:
        keys = list(normalized.camera)
        tracks.append(
            Track("camera", None, keys, keys[0].frame, keys[-1].frame)
        )
    if "bone" in sections:
        tracks.extend(_split_bone_tracks(normalized.bone))
    return tracks


def _split_bone_tracks(bone_keys):
    tracks = []
    group = []
    current_raw = None
    for k in bone_keys:
        if current_raw is None or k.name_raw == current_raw:
            group.append(k)
            current_raw = k.name_raw
        else:
            tracks.append(_bone_track(group))
            group = [k]
            current_raw = k.name_raw
    if group:
        tracks.append(_bone_track(group))
    return tracks


def _bone_track(group):
    return Track("bone", group[0].name, group, group[0].frame, group[-1].frame)


def sample_scalar(keys, channel, frame_start, frame_end):
    return interp.sample_range(keys, channel, frame_start, frame_end)


def sample_rotation(keys, frame_start, frame_end):
    """戻り値の各要素は、camera なら Euler 角の3要素、bone ならクォータニオン。"""
    return interp.sample_range(keys, "rot", frame_start, frame_end)
