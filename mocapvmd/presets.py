"""プリセットと種別別パラメータの解決(mocapvmd.md §5.2 / §5.3 / §5.4)。

クリーニング強度(§5.2): 種別ごとの基準パラメータ(位置窓・回転窓・位置強度・回転強度)を balanced
基準で持ち、`--preset` の強度倍率を**強度のみ**に掛けて解決する(窓幅は倍率で変えない)。位置強度・
回転強度は元値と平滑化値のブレンド係数(0で元値保持、1で平滑化値採用)。

接地ロック強度(§5.4)は resolve_foot_lock、疎化の許容誤差(§5.3。`--reduce-preset` の基準値 × 種別
スケール)は resolve_reduction_tolerances で解決する。疎化プリセット値は mocapvmd 独自で sparsevmd と
共有しない。
"""

import math

# クリーニング強度プリセット名(§5.2)。
PRESET_NAMES = ("light", "balanced", "stable-foot", "strong")

# §5.2 初期パラメータ表(balanced 基準)。
# 種別 -> (位置窓, 回転窓, 位置強度, 回転強度)。
_BASE = {
    "root": (3, 3, 0.15, 0.10),
    "center": (7, 5, 0.45, 0.25),
    "torso": (5, 5, 0.25, 0.35),
    "arms": (5, 5, 0.25, 0.25),
    "fingers": (3, 3, 0.10, 0.15),
    "legs": (5, 5, 0.30, 0.25),
    "foot_ik": (7, 3, 0.65, 0.20),
    "toe_ik": (5, 3, 0.50, 0.20),
    "unknown": (3, 3, 0.10, 0.10),  # 分類不能。保守的に弱く処理する
}


# §5.4 接地ロック強度。アンカーへのブレンド係数(0〜1)で倍率でなく直接値。
# foot_ik の X/Z 接地中央のみプリセット別(stable-foot が最強・light が最弱)。
_FOOT_XZ_CENTER = {"light": 0.70, "balanced": 0.90, "strong": 0.93, "stable-foot": 0.97}
_FOOT_LOCK_FADE_WIDTH = 3  # 接地端のフェード幅(端からのフレーム数)


def resolve_foot_lock(preset, category):
    """プリセットと種別(foot_ik / toe_ik)から接地ロック係数 dict を返す(§5.4)。

    返す dict: xz_center / xz_edge / y_center / y_edge / fade_width。係数は接地アンカーへの
    ブレンド係数(0〜1。1に近いほど強く固定)で、倍率でなく直接値。foot_ik の X/Z 接地中央のみ
    プリセット別、foot_ik の Y と toe_ik の各チャンネルはプリセット非依存。接地ロックは
    foot_ik / toe_ik のみ対象で、他種別・未知プリセットは ValueError。
    """
    if preset not in PRESET_NAMES:
        raise ValueError(
            f"未知のプリセット: {preset!r}(有効: {', '.join(PRESET_NAMES)})"
        )
    if category == "foot_ik":
        return {
            "xz_center": _FOOT_XZ_CENTER[preset],
            "xz_edge": 0.25,
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


def _strength_multiplier(preset, category):
    """§5.2 強度倍率。stable-foot は foot_ik のみ 1.5、他は 1.0。"""
    if preset == "stable-foot":
        return 1.5 if category == "foot_ik" else 1.0
    multipliers = {"light": 0.5, "balanced": 1.0, "strong": 1.4}
    return multipliers[preset]


def resolve_cleaning(preset, category):
    """プリセットと種別から、クリーニングパラメータ dict を返す(§5.2)。

    返す dict: pos_window / rot_window / pos_strength / rot_strength。倍率は強度のみに掛け、
    窓幅は据え置く。未知のプリセット名・未知の種別は ValueError。
    """
    if preset not in PRESET_NAMES:
        raise ValueError(
            f"未知のプリセット: {preset!r}(有効: {', '.join(PRESET_NAMES)})"
        )
    if category not in _BASE:
        raise ValueError(f"未知の種別: {category!r}")

    pos_window, rot_window, pos_strength, rot_strength = _BASE[category]
    m = _strength_multiplier(preset, category)
    return {
        "pos_window": pos_window,
        "rot_window": rot_window,
        "pos_strength": pos_strength * m,
        "rot_strength": rot_strength * m,
    }


# §5.3 疎化トレランス。プリセット基準値(位置 MMD単位 / 回転 度)と種別スケール(位置, 回転)。
# mocapvmd 独自値で sparsevmd と共有しない。
REDUCTION_PRESET_NAMES = ("precise", "balanced", "aggressive")
_REDUCTION_BASE = {
    "precise": (0.01, 0.10),
    "balanced": (0.02, 0.20),
    "aggressive": (0.05, 0.40),
}
_REDUCTION_SCALE = {
    "root": (1.0, 1.0),
    "center": (0.7, 0.8),
    "torso": (0.8, 0.9),
    "arms": (1.0, 1.0),
    "fingers": (1.5, 1.5),
    "legs": (1.0, 1.0),
    "foot_ik": (0.7, 1.0),
    "toe_ik": (1.0, 0.8),
    "unknown": (1.0, 1.0),
}


def _validate_tolerance(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} は有限の非負値である必要があります: {value!r}")


def resolve_reduction_tolerances(preset, category, override_pos=None, override_rot=None):
    """プリセットと種別から疎化の許容誤差 dict を返す(§5.3)。

    各ボーンの許容誤差 = プリセット基準値 × その種別のスケール。返す dict: bone_pos / bone_rot。
    override_pos / override_rot を渡すとプリセット基準値を上書きし(種別スケールは引き続き掛ける)、
    基準値より優先する。未知の疎化プリセット名・種別、非有限・負の上書き値は ValueError。
    """
    if preset not in _REDUCTION_BASE:
        raise ValueError(
            f"未知の疎化プリセット: {preset!r}(有効: {', '.join(REDUCTION_PRESET_NAMES)})"
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

    scale_pos, scale_rot = _REDUCTION_SCALE[category]
    return {"bone_pos": base_pos * scale_pos, "bone_rot": base_rot * scale_rot}
