import math

from vmd.cuts import (  # noqa: F401
    _axis_angle_deg,
    _quat_angle_deg,
    assemble_boundaries,
    detect_cuts_bone,
    detect_cuts_camera,
    perspective_cut_frames,
)

_CAMERA_THRESHOLD_COUNT = 3
_BONE_THRESHOLD_COUNT = 2


def parse_cut_threshold_camera(text):
    """戻り値は (pos, rot, dist)。"""
    return _parse_thresholds(text, _CAMERA_THRESHOLD_COUNT)


def parse_cut_threshold_bone(text):
    """戻り値は (pos, rot)。"""
    return _parse_thresholds(text, _BONE_THRESHOLD_COUNT)


def _parse_thresholds(text, n):
    parts = text.split(",")
    if len(parts) != n:
        raise ValueError(f"閾値は {n} 個のカンマ区切り: {text!r}")
    vals = []
    for p in parts:
        if p == "" or p != p.strip():
            raise ValueError(f"閾値に空要素・空白は不可: {text!r}")
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"閾値は数値: {text!r}") from None
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"閾値は0以上の有限値: {text!r}")
        vals.append(v)
    return tuple(vals)
