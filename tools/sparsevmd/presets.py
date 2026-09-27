import math
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

# VMD は視野角を整数度で保存するため、丸めだけで最大0.5度の誤差が出る。
_CAMERA_FOV_MIN = 0.5


def resolve_tolerances(preset_name, overrides=None):
    """overrides は {Tolerances のフィールド名: 値}。検証に失敗すると ValueError。"""
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

    for key, val in values.items():
        values[key] = _validate(key, val)

    return Tolerances(**values)


def _validate(name, value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"許容誤差は数値が必要: {name}={value!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"許容誤差は有限値が必要: {name}={value!r}")
    if v < 0.0:
        raise ValueError(f"許容誤差は0以上が必要: {name}={value!r}")
    if name == "camera_fov" and v < _CAMERA_FOV_MIN:
        raise ValueError(
            f"--camera-fov-tol は {_CAMERA_FOV_MIN} 以上が必要(整数度保存のため): {value!r}"
        )
    return v
