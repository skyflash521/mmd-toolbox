"""品質プリセットと許容誤差の解決。

`--preset NAME` から既定許容誤差を引き、個別オプションの明示値があれば
それを優先する。値の検証もここで集中管理する:
FOV以外は0以上の有限値(0は整数フレーム上の完全一致を要求する)、FOVは0.5以上。
未知プリセット名・未知フィールド・NaN/Inf/負値・FOV<0.5 は ValueError。
CLI 層はこの ValueError を引数エラー(終了コード2)に対応づける。
"""

import math
from dataclasses import fields

# Tolerances(許容誤差の器)の実体は共通ライブラリ vmd.reduce にある。
# preset 名 → 値の表と検証は sparsevmd の運用ポリシーとして本モジュールに残す。
# 旧 import パス(sparsevmd.presets.Tolerances)維持のため再公開する。同名への別名付けは、
# 本モジュール内の使用が将来消えても未使用 import と見なされないようにする再公開の明示。
from vmd.reduce import Tolerances as Tolerances

# 品質プリセット名。
PRESET_NAMES = ("precise", "balanced", "aggressive")


# プリセットごとの許容誤差表(precise / balanced / aggressive)。
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

# 視野角はVMDが整数度保存のため、丸めだけで最大0.5度の誤差が出る。
_CAMERA_FOV_MIN = 0.5


def resolve_tolerances(preset_name, overrides=None):
    """プリセット名と個別上書きから `Tolerances` を構築する。

    overrides は {フィールド名: 値} の dict。明示値はプリセット値より優先する。
    検証に失敗した場合は ValueError を送出する。
    """
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

    # 検証し、変換後の float を格納する(数値文字列等が str のまま残らないように)。
    for key, val in values.items():
        values[key] = _validate(key, val)

    return Tolerances(**values)


def _validate(name, value):
    """単一の許容誤差値を検証し、float へ変換して返す。不正なら ValueError。

    float() 可能な値(int/float/数値文字列)は変換して受理する。None や非数値
    オブジェクト、単位付き文字列など float() できない値は ValueError とする
    (TypeError を漏らさない)。
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"許容誤差は数値が必要: {name}={value!r}")
    if not math.isfinite(v):
        raise ValueError(f"許容誤差は有限値が必要: {name}={value!r}")
    if v < 0.0:
        raise ValueError(f"許容誤差は0以上が必要: {name}={value!r}")
    if name == "camera_fov" and v < _CAMERA_FOV_MIN:
        raise ValueError(
            f"--camera-fov-tol は {_CAMERA_FOV_MIN} 以上が必要(整数度保存のため): {value!r}"
        )
    return v
