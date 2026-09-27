import math
from typing import NamedTuple


class _CleaningBase(NamedTuple):
    pos_window: int
    rot_window: int
    pos_blend: float
    rot_blend: float


_CLEANING_BASE = {
    "root": _CleaningBase(3, 3, 0.15, 0.10),
    "center": _CleaningBase(7, 5, 0.45, 0.25),
    "torso": _CleaningBase(5, 5, 0.25, 0.35),
    "arms": _CleaningBase(5, 5, 0.25, 0.25),
    "fingers": _CleaningBase(3, 3, 0.10, 0.15),
    "legs": _CleaningBase(5, 5, 0.30, 0.25),
    "foot_ik": _CleaningBase(7, 3, 0.65, 0.20),
    "toe_ik": _CleaningBase(5, 3, 0.50, 0.20),
    "unknown": _CleaningBase(3, 3, 0.10, 0.10),
}


class _BySuppression(NamedTuple):
    at_s0: float
    at_s1: float

    def at(self, suppression):
        return self.at_s0 + (self.at_s1 - self.at_s0) * suppression


_FOOT_XZ_LOCK_CENTER = _BySuppression(at_s0=0.0, at_s1=0.97)
_FOOT_XZ_LOCK_EDGE = _BySuppression(at_s0=0.0, at_s1=0.25)
_GROUNDING_HORIZ_VEL_THRESH = _BySuppression(at_s0=0.08, at_s1=1.0)
_LOCK_MAX_DISPLACEMENT_PER_FRAME = _BySuppression(at_s0=0.5, at_s1=2.0)
_FOOT_LOCK_FADE_WIDTH = 3


def _validate_suppression(suppression):
    if not math.isfinite(suppression) or not 0.0 <= suppression <= 1.0:
        raise ValueError(f"横滑り抑制 S は 0〜1 の有限値である必要があります: {suppression!r}")


def resolve_foot_lock(suppression, category):
    _validate_suppression(suppression)
    if category == "foot_ik":
        return {
            "xz_center": _FOOT_XZ_LOCK_CENTER.at(suppression),
            "xz_edge": _FOOT_XZ_LOCK_EDGE.at(suppression),
            "y_center": 0.50,
            "y_edge": 0.10,
            "fade_width": _FOOT_LOCK_FADE_WIDTH,
        }
    if category == "toe_ik":
        return {
            "xz_center": 0.30,
            "xz_edge": 0.10,
            "y_center": 0.30,
            "y_edge": 0.10,
            "fade_width": _FOOT_LOCK_FADE_WIDTH,
        }
    raise ValueError(f"接地ロックの対象外の種別: {category!r}(foot_ik / toe_ik のみ)")


def resolve_foot_detection(suppression):
    _validate_suppression(suppression)
    return {
        "horiz_vel_thresh": _GROUNDING_HORIZ_VEL_THRESH.at(suppression),
        "max_displacement": _LOCK_MAX_DISPLACEMENT_PER_FRAME.at(suppression),
    }


def resolve_cleaning(strength, category):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError(f"クリーニング強度は有限の非負値である必要があります: {strength!r}")
    if category not in _CLEANING_BASE:
        raise ValueError(f"未知の種別: {category!r}")

    base = _CLEANING_BASE[category]
    return {
        "pos_window": base.pos_window,
        "rot_window": base.rot_window,
        "pos_strength": min(1.0, base.pos_blend * strength),
        "rot_strength": min(1.0, base.rot_blend * strength),
    }


class ReductionTolerance(NamedTuple):
    pos: float
    rot_deg: float


class _ReductionScale(NamedTuple):
    pos: float
    rot: float


PRESET_NAMES = ("slower", "slow", "medium", "fast", "faster")
_REDUCTION_BASE = {
    "slower": ReductionTolerance(0.05, 0.40),
    "slow": ReductionTolerance(0.10, 0.75),
    "medium": ReductionTolerance(0.20, 1.50),
    "fast": ReductionTolerance(0.80, 6.0),
    "faster": ReductionTolerance(1.60, 12.0),
}
_REDUCTION_SCALE = {
    "root": _ReductionScale(1.0, 1.0),
    "center": _ReductionScale(0.7, 0.8),
    "torso": _ReductionScale(0.8, 0.9),
    "arms": _ReductionScale(1.0, 1.0),
    "fingers": _ReductionScale(1.5, 1.5),
    "legs": _ReductionScale(1.0, 1.0),
    "foot_ik": _ReductionScale(0.7, 1.0),
    "toe_ik": _ReductionScale(1.0, 0.8),
    "unknown": _ReductionScale(1.0, 1.0),
}


def _validate_tolerance(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} は有限の非負値である必要があります: {value!r}")


def resolve_reduction_tolerances(preset, category, override_pos=None, override_rot=None):
    if preset not in _REDUCTION_BASE:
        raise ValueError(
            f"未知の疎化プリセット: {preset!r}(有効: {', '.join(PRESET_NAMES)})"
        )
    if category not in _REDUCTION_SCALE:
        raise ValueError(f"未知の種別: {category!r}")

    base_pos, base_rot = _REDUCTION_BASE[preset]
    if override_pos is not None:
        _validate_tolerance(override_pos, "override_pos")
        base_pos = override_pos
    if override_rot is not None:
        _validate_tolerance(override_rot, "override_rot")
        base_rot = override_rot

    scale = _REDUCTION_SCALE[category]
    return {"bone_pos": base_pos * scale.pos, "bone_rot": base_rot * scale.rot}


def reduction_base(name) -> ReductionTolerance:
    if name not in _REDUCTION_BASE:
        raise ValueError(f"未知の疎化プリセット: {name!r}(有効: {', '.join(PRESET_NAMES)})")
    return _REDUCTION_BASE[name]
