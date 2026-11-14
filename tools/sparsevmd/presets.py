from dataclasses import fields

from vmd.reduce import Tolerances as Tolerances

PRESET_NAMES = ("precise", "balanced", "aggressive")


_PRESETS = {
    "precise": dict(
        bone_pos=0.005,
        bone_rot=0.05,
        camera_pos=0.01,
        camera_rot=0.02,
        camera_distance=0.01,
        camera_fov=0.50,
    ),
    "balanced": dict(
        bone_pos=0.01,
        bone_rot=0.10,
        camera_pos=0.02,
        camera_rot=0.05,
        camera_distance=0.02,
        camera_fov=0.50,
    ),
    "aggressive": dict(
        bone_pos=0.05,
        bone_rot=0.50,
        camera_pos=0.10,
        camera_rot=0.25,
        camera_distance=0.10,
        camera_fov=1.00,
    ),
}

_FIELD_NAMES = frozenset(f.name for f in fields(Tolerances))


def resolve_tolerances(preset_name, overrides=None):
    """overrides は {Tolerances のフィールド名: 検証済みの値}。未知のプリセット・フィールド名は ValueError。"""
    if preset_name not in _PRESETS:
        raise ValueError(
            f"未知のプリセット: {preset_name!r}(有効: {', '.join(PRESET_NAMES)})"
        )

    values = dict(_PRESETS[preset_name])
    if overrides:
        for key, val in overrides.items():
            if key not in _FIELD_NAMES:
                raise ValueError(f"未知の許容誤差フィールド: {key!r}")
            values[key] = val

    return Tolerances(**values)
