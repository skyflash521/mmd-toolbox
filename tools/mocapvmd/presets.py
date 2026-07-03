"""プリセットと種別別パラメータの解決(mocapvmd.md §5.2 / §5.3 / §5.4)。

クリーニング強度(§5.2): 種別ごとの基準パラメータ(位置の窓幅・回転の窓幅・位置のブレンド率・回転のブレンド率)を
基準で持ち、`--clean-strength` の倍率を**ブレンド率のみ**に掛けて解決する(窓幅は倍率で変えない)。位置のブレンド率・
回転のブレンド率は元値と平滑化値のブレンド係数(0で元値保持、1で平滑化値採用)で、倍率適用後は 0〜1 にクランプする。

名前付きプリセット(§5.3)は疎化の許容誤差 `--preset`(PRESET_NAMES)1つに集約する。接地ロック強度(§5.4)は
resolve_foot_lock、疎化の許容誤差(§5.3。`--preset` の基準値 × 種別スケール)は resolve_reduction_tolerances で
解決する。疎化プリセット値は mocapvmd 独自で sparsevmd と共有しない。
"""

import math

# §5.2 種別別クリーニング基準パラメータ表(`--clean-strength` 1.0 基準)。
# 種別 -> (位置の窓幅, 回転の窓幅, 位置のブレンド率, 回転のブレンド率)。
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


# §4.3 / §5.4 / §5.5 横滑り抑制 S(0〜1)→ 接地ロック・接地検出パラメータ(初期値。実データで調整)。
# S=0 で従来の検出のまま X/Z をアンカーへ寄せず滑りを保持、S=1(既定)で高さ主導検出(水平速度ゲートを
# 緩める)＋強い X/Z 固定＋補正上限緩和により接地中の横滑りを除去する。各値は S で線形に写像する。
_SLIDE_FOOT_XZ_MAX = 0.97              # S=1 の foot_ik X/Z 接地中央(S=0 は 0.0)
_SLIDE_FOOT_XZ_EDGE_MAX = 0.25        # S=1 の foot_ik X/Z 接地端(S=0 は 0.0)。端値も S 連動させ、
                                      # S が小さいとき端>中央でフェードが逆転するのを防ぐ(常に端≤中央)。
_SLIDE_HORIZ_VEL_THRESH = (0.08, 1.0)  # 接地検出の水平速度許容 (S=0, S=1)
_SLIDE_MAX_CORRECTION = (0.5, 2.0)     # 接地ロックの1フレーム最大変位上限 MMD単位 (S=0, S=1)
_FOOT_LOCK_FADE_WIDTH = 3              # 接地端のフェード幅(端からのフレーム数)


def _lerp(a, b, t):
    return a + (b - a) * t


def _validate_suppression(suppression):
    if not math.isfinite(suppression) or not 0.0 <= suppression <= 1.0:
        raise ValueError(f"横滑り抑制 S は 0〜1 の有限値である必要があります: {suppression!r}")


def resolve_foot_lock(suppression, category):
    """横滑り抑制 S(0〜1)と種別(foot_ik / toe_ik)から接地ロック係数 dict を返す(§5.4)。

    返す dict: xz_center / xz_edge / y_center / y_edge / fade_width。係数は接地アンカーへの
    ブレンド係数(0〜1。1に近いほど強く固定)で倍率でなく直接値。foot_ik の X/Z 接地中央のみ S で
    決まり(S=0→0.0 / S=1→0.97 を線形写像)、foot_ik の Y と toe_ik の各チャンネルは S 非依存の
    固定値。接地ロックは foot_ik / toe_ik のみ対象で、他種別は ValueError。S が範囲外は ValueError。
    """
    _validate_suppression(suppression)
    if category == "foot_ik":
        return {
            "xz_center": _SLIDE_FOOT_XZ_MAX * suppression,
            "xz_edge": _SLIDE_FOOT_XZ_EDGE_MAX * suppression,
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
    """横滑り抑制 S(0〜1)から接地検出・補正の S 連動パラメータ dict を返す(§4.3 / §5.5)。

    返す dict: horiz_vel_thresh(接地候補の水平速度許容。S が大きいほど高くし、接地高さで水平に
    滑る足も接地として拾う。S=0 は従来の 0.08)/ max_displacement(接地区間内の1フレーム最大変位
    上限。S が大きいほど緩め、大きな滑りもアンカーへ引き戻す。S=0 は 0.5)。いずれも S で線形に
    写像する(初期値。実データで調整)。S が範囲外は ValueError。
    """
    _validate_suppression(suppression)
    return {
        "horiz_vel_thresh": _lerp(_SLIDE_HORIZ_VEL_THRESH[0], _SLIDE_HORIZ_VEL_THRESH[1], suppression),
        "max_displacement": _lerp(_SLIDE_MAX_CORRECTION[0], _SLIDE_MAX_CORRECTION[1], suppression),
    }


def resolve_cleaning(strength, category):
    """クリーニング強度の倍率と種別から、クリーニングパラメータ dict を返す(§5.2)。

    返す dict: pos_window / rot_window / pos_strength / rot_strength。倍率(`--clean-strength`)は
    ブレンド率のみに掛け、窓幅は据え置く。倍率適用後のブレンド率は 0〜1 にクランプする(1.0=完全平滑化を
    超えない)。倍率は有限の非負値のみ。非有限・負の倍率、未知の種別は ValueError。
    """
    if not math.isfinite(strength) or strength < 0:
        raise ValueError(f"クリーニング強度は有限の非負値である必要があります: {strength!r}")
    if category not in _BASE:
        raise ValueError(f"未知の種別: {category!r}")

    pos_window, rot_window, pos_strength, rot_strength = _BASE[category]
    return {
        "pos_window": pos_window,
        "rot_window": rot_window,
        "pos_strength": min(1.0, pos_strength * strength),
        "rot_strength": min(1.0, rot_strength * strength),
    }


# §5.3 疎化トレランス。プリセット基準値(位置 MMD単位 / 回転 度)と種別スケール(位置, 回転)。
# mocapvmd 独自値で sparsevmd と共有しない。
# 速度の観点で命名(遅い=高忠実・キー多・処理遅、速い=高圧縮・キー少・処理速)。既定は中央の
# medium(0.20 / 1.50)。許容を緩めるほどキーが減り疎化処理も速い(fast / faster)。位置と回転は連動して粗くする。
# 本ツール唯一の名前付きプリセット(`--preset`)。
PRESET_NAMES = ("slower", "slow", "medium", "fast", "faster")
_REDUCTION_BASE = {
    "slower": (0.05, 0.40),
    "slow": (0.10, 0.75),
    "medium": (0.20, 1.50),
    "fast": (0.80, 6.0),
    "faster": (1.60, 12.0),
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

    scale_pos, scale_rot = _REDUCTION_SCALE[category]
    return {"bone_pos": base_pos * scale_pos, "bone_rot": base_rot * scale_rot}


def reduction_base(name):
    """疎化プリセット名の基準位置許容・基準回転許容 (pos, rot) を返す(§5.3。--describe の自己記述用)。

    種別スケールを掛ける前のプリセット基準値。未知の疎化プリセット名は ValueError。
    """
    if name not in _REDUCTION_BASE:
        raise ValueError(f"未知の疎化プリセット: {name!r}(有効: {', '.join(PRESET_NAMES)})")
    return _REDUCTION_BASE[name]
