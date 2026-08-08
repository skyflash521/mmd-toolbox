"""クリーニング強度解決のテスト。

種別ごとの基準パラメータ(位置の窓幅・回転の窓幅・位置のブレンド率・回転のブレンド率)に、--clean-strength の
数値倍率をブレンド率のみへ掛けてクリーニングパラメータを解決する。窓幅は倍率で変えない。倍率適用後の
ブレンド率は 0〜1 にクランプ(1.0=完全平滑化を超えない)。倍率は有限の非負値のみ。
"""

import pytest

from mocapvmd import presets


def test_preset_names_are_reduction_levels():
    # 本ツール唯一の名前付きプリセット(--preset)は疎化トレランスの速度軸。
    assert presets.PRESET_NAMES == ("slower", "slow", "medium", "fast", "faster")


# 初期パラメータ表(倍率 1.0 基準): category -> (pos_window, rot_window, pos_strength, rot_strength)。
# 期待値オラクル。resolve_cleaning の strength=1.0 出力と一致するべき基準。
_BASE = {
    "root": (3, 3, 0.15, 0.10),
    "center": (7, 5, 0.45, 0.25),
    "torso": (5, 5, 0.25, 0.35),
    "arms": (5, 5, 0.25, 0.25),
    "fingers": (3, 3, 0.10, 0.15),
    "legs": (5, 5, 0.30, 0.25),
    "foot_ik": (7, 3, 0.65, 0.20),
    "toe_ik": (5, 3, 0.50, 0.20),
    "unknown": (3, 3, 0.10, 0.10),
}


@pytest.mark.parametrize("category", list(_BASE))
def test_unit_strength_base_values(category):
    # strength=1.0 は基準値そのもの。
    pw, rw, ps, rs = _BASE[category]
    p = presets.resolve_cleaning(1.0, category)
    assert p["pos_window"] == pw
    assert p["rot_window"] == rw
    assert p["pos_strength"] == pytest.approx(ps)
    assert p["rot_strength"] == pytest.approx(rs)


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, 1.4])
@pytest.mark.parametrize("category", list(_BASE))
def test_strength_multiplier_applies_to_blend_only(strength, category):
    # 全倍率×全種別で、倍率は位置・回転の両ブレンド率のみに掛かり(0〜1 クランプ)、窓幅は不変。
    # 種別ごとの適用漏れ・回転のブレンド率への誤適用・窓幅の誤変更を一括して捕捉する。
    pw, rw, ps, rs = _BASE[category]
    p = presets.resolve_cleaning(strength, category)
    assert p["pos_window"] == pw       # 窓幅は倍率で変えない
    assert p["rot_window"] == rw
    assert p["pos_strength"] == pytest.approx(min(1.0, ps * strength))
    assert p["rot_strength"] == pytest.approx(min(1.0, rs * strength))


def test_strength_clamps_blend_to_one():
    # 大きな倍率でもブレンド率は 1.0 を超えない(平滑化値を逸脱する外挿を防ぐ)。
    p = presets.resolve_cleaning(10.0, "foot_ik")  # 0.65×10=6.5 → 1.0 にクランプ
    assert p["pos_strength"] == pytest.approx(1.0)
    assert p["rot_strength"] == pytest.approx(1.0)


def test_zero_strength_keeps_original():
    # 倍率 0 はブレンド率 0(元値保持=無加工相当)。
    p = presets.resolve_cleaning(0.0, "center")
    assert p["pos_strength"] == pytest.approx(0.0)
    assert p["rot_strength"] == pytest.approx(0.0)


def test_unknown_category_conservative():
    # 分類不能 unknown は保守的な弱設定(窓3/3・強度0.10/0.10)。
    p = presets.resolve_cleaning(1.0, "unknown")
    assert (p["pos_window"], p["rot_window"]) == (3, 3)
    assert p["pos_strength"] == pytest.approx(0.10)
    assert p["rot_strength"] == pytest.approx(0.10)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.01])
def test_invalid_strength_raises(bad):
    # 非有限・負の倍率は ValueError(CLI で終了コード2へ変換される)。
    with pytest.raises(ValueError):
        presets.resolve_cleaning(bad, "center")


def test_invalid_category_raises():
    with pytest.raises(ValueError):
        presets.resolve_cleaning(1.0, "nonexistent")


# --- 疎化の許容誤差解決 ------------------------------------------------------
# 各ボーンの許容誤差 = プリセット基準値 × 種別スケール。--reduce-error-* 明示時は基準値を上書き
# (種別スケールは引き続き掛ける)。slower/slow/medium/fast/faster(速度観点)と種別スケールは mocapvmd 独自値。

# プリセット基準値(位置 MMD単位 / 回転 度)。既定は中央の medium(0.20 / 1.50)。
_REDUCE_BASE = {
    "slower": (0.05, 0.40), "slow": (0.10, 0.75), "medium": (0.20, 1.50),
    "fast": (0.80, 6.0), "faster": (1.60, 12.0),
}

# 種別スケール(位置, 回転)。
_REDUCE_SCALE = {
    "root": (1.0, 1.0), "center": (0.7, 0.8), "torso": (0.8, 0.9), "arms": (1.0, 1.0),
    "fingers": (1.5, 1.5), "legs": (1.0, 1.0), "foot_ik": (0.7, 1.0),
    "toe_ik": (1.0, 0.8), "unknown": (1.0, 1.0),
}


@pytest.mark.parametrize("preset", list(_REDUCE_BASE))
@pytest.mark.parametrize("category", list(_REDUCE_SCALE))
def test_reduction_tolerance_base_times_scale(preset, category):
    base_pos, base_rot = _REDUCE_BASE[preset]
    spos, srot = _REDUCE_SCALE[category]
    t = presets.resolve_reduction_tolerances(preset, category)
    assert t["bone_pos"] == pytest.approx(base_pos * spos)
    assert t["bone_rot"] == pytest.approx(base_rot * srot)


def test_reduction_tolerance_override_replaces_base_then_scales():
    # --reduce-error-* の明示はプリセット基準値を上書きし、種別スケールは引き続き掛かる。
    t = presets.resolve_reduction_tolerances("medium", "center", override_pos=0.1, override_rot=2.0)
    assert t["bone_pos"] == pytest.approx(0.1 * 0.7)   # center 位置スケール 0.7
    assert t["bone_rot"] == pytest.approx(2.0 * 0.8)   # center 回転スケール 0.8


def test_reduction_tolerance_partial_override_pos_only():
    # 位置のみ上書き。回転はプリセット基準値のまま種別スケールが掛かる。
    t = presets.resolve_reduction_tolerances("slower", "fingers", override_pos=0.2)
    assert t["bone_pos"] == pytest.approx(0.2 * 1.5)    # override × fingers位置1.5
    assert t["bone_rot"] == pytest.approx(0.40 * 1.5)   # slower回転0.40 × fingers回転1.5


def test_reduction_tolerance_partial_override_rot_only():
    # 回転のみ上書き。位置はプリセット基準値のまま種別スケールが掛かる(回転側だけの処理漏れを弾く)。
    t = presets.resolve_reduction_tolerances("slower", "fingers", override_rot=2.0)
    assert t["bone_pos"] == pytest.approx(0.05 * 1.5)   # slower位置0.05 × fingers位置1.5
    assert t["bone_rot"] == pytest.approx(2.0 * 1.5)    # override × fingers回転1.5


@pytest.mark.parametrize("kw", [{"override_pos": 0.0}, {"override_rot": 0.0}])
def test_reduction_tolerance_zero_override_is_valid(kw):
    # 非負を許容するので 0 は有効(ValueError にしない)。0 は全キー保持の設定。
    t = presets.resolve_reduction_tolerances("medium", "center", **kw)
    if "override_pos" in kw:
        assert t["bone_pos"] == pytest.approx(0.0)
        assert t["bone_rot"] == pytest.approx(1.50 * 0.8)
    else:
        assert t["bone_rot"] == pytest.approx(0.0)
        assert t["bone_pos"] == pytest.approx(0.20 * 0.7)


def test_reduction_tolerance_invalid_preset_raises():
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("turbo", "center")


def test_reduction_tolerance_invalid_category_raises():
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("medium", "nonexistent")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.01])
def test_reduction_tolerance_invalid_override_raises(bad):
    # 非有限(符号問わず)・負の上書き値は ValueError(CLI で終了コード2 へ変換される)。
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("medium", "center", override_pos=bad)
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("medium", "center", override_rot=bad)


# --- 接地ロック強度・接地検出 ------------------------------------------------
# 接地ロック・検出は横滑り抑制 S(0〜1)で連動する。foot_ik の X/Z 接地中央は S=0→0.0 / S=1→0.97 の
# 線形写像、他チャンネル(foot_ik の Y、toe_ik の全チャンネル)は S 非依存の固定値。接地検出の水平
# 速度許容(S=0→0.08 / S=1→1.0)と最大補正量上限(S=0→0.5 / S=1→2.0)も S で線形に動く。


@pytest.mark.parametrize("s,xz", [(0.0, 0.0), (0.5, 0.485), (1.0, 0.97)])
def test_foot_lock_foot_ik_xz_center_by_suppression(s, xz):
    # foot_ik の X/Z 接地中央は 0.97×S、接地端は 0.25×S(端≤中央を保ち S=0 で完全にロックを外す)。
    p = presets.resolve_foot_lock(s, "foot_ik")
    assert p["xz_center"] == pytest.approx(xz)
    assert p["xz_edge"] == pytest.approx(0.25 * s)
    assert p["xz_edge"] <= p["xz_center"] + 1e-9


def test_foot_lock_default_suppression_removes_slide():
    # 既定 S=1.0 は X/Z 接地中央が最強(0.97)=規定「横滑りなし」。
    assert presets.resolve_foot_lock(1.0, "foot_ik")["xz_center"] == pytest.approx(0.97)


@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_foot_lock_foot_ik_y_is_suppression_independent(s):
    # foot_ik の Y は中央0.50・端0.10 で S 非依存。
    p = presets.resolve_foot_lock(s, "foot_ik")
    assert p["y_center"] == pytest.approx(0.50)
    assert p["y_edge"] == pytest.approx(0.10)


@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_foot_lock_toe_ik_is_suppression_independent(s):
    # toe_ik は X/Z・Y とも中央0.30・端0.10 で S 非依存(つま先の動き・回転を保つ)。
    p = presets.resolve_foot_lock(s, "toe_ik")
    assert p["xz_center"] == pytest.approx(0.30)
    assert p["xz_edge"] == pytest.approx(0.10)
    assert p["y_center"] == pytest.approx(0.30)
    assert p["y_edge"] == pytest.approx(0.10)


def test_foot_lock_foot_ik_xz_center_monotonic_in_suppression():
    # 接地固定の強さは S が大きいほど強い(単調増加)。
    centers = [presets.resolve_foot_lock(s, "foot_ik")["xz_center"]
               for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert centers == sorted(centers)
    assert len(set(centers)) == 5


@pytest.mark.parametrize("category", ["foot_ik", "toe_ik"])
@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_foot_lock_fade_width_default_3(s, category):
    # フェード幅は端から既定3フレーム。
    assert presets.resolve_foot_lock(s, category)["fade_width"] == 3


@pytest.mark.parametrize("category", ["center", "legs", "toe", "unknown"])
def test_foot_lock_non_footik_category_raises(category):
    # 接地ロックは foot_ik / toe_ik のみ対象。他種別は対象外で ValueError。
    with pytest.raises(ValueError):
        presets.resolve_foot_lock(1.0, category)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01])
def test_foot_lock_invalid_suppression_raises(bad):
    # 横滑り抑制 S は 0〜1 の有限値のみ。範囲外・非有限は ValueError(CLI で終了コード2)。
    with pytest.raises(ValueError):
        presets.resolve_foot_lock(bad, "foot_ik")


# --- 接地検出の S 連動パラメータ ---------------------------------------------


@pytest.mark.parametrize("s,horiz,maxd", [(0.0, 0.08, 0.5), (0.5, 0.54, 1.25), (1.0, 1.0, 2.0)])
def test_foot_detection_by_suppression(s, horiz, maxd):
    # 水平速度許容・最大補正量上限はともに S で線形に写像する。
    d = presets.resolve_foot_detection(s)
    assert d["horiz_vel_thresh"] == pytest.approx(horiz)
    assert d["max_displacement"] == pytest.approx(maxd)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01])
def test_foot_detection_invalid_suppression_raises(bad):
    with pytest.raises(ValueError):
        presets.resolve_foot_detection(bad)
