"""クリーニング強度プリセットと種別別パラメータの解決(mocapvmd.md §5.2)。

種別ごとの基準パラメータ(位置窓・回転窓・位置強度・回転強度)を balanced 基準で持ち、
`--preset` の強度倍率を**強度のみ**に掛けて解決する(窓幅は倍率で変えない)。位置強度・
回転強度は元値と平滑化値のブレンド係数(0で元値保持、1で平滑化値採用)。

疎化の許容誤差プリセット(`--reduce-preset`)とその種別スケールは運用ポリシーが別なので、
本モジュールではなく疎化側で持つ(§5.3)。
"""

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
