"""クリーニングプリセット解決のテスト(mocapvmd.md §5.2)。

種別ごとの基準パラメータ(位置窓・回転窓・位置強度・回転強度)に、--preset の強度倍率を
掛けてクリーニングパラメータを解決する。窓幅は倍率で変えない。light=0.5 / balanced=1.0 /
strong=1.4(全種別)/ stable-foot=foot_ik のみ 1.5・他 1.0。
"""

import pytest

from mocapvmd import presets


def test_preset_names():
    assert presets.PRESET_NAMES == ("light", "balanced", "stable-foot", "strong")


# §5.2 初期パラメータ表(balanced 基準): category -> (pos_window, rot_window, pos_strength, rot_strength)。
# 期待値オラクル。resolve_cleaning の balanced 出力と一致するべき基準。
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


def _expected_multiplier(preset, category):
    # §5.2 強度倍率表。stable-foot は foot_ik のみ 1.5、他は 1.0。
    if preset == "stable-foot":
        return 1.5 if category == "foot_ik" else 1.0
    return {"light": 0.5, "balanced": 1.0, "strong": 1.4}[preset]


@pytest.mark.parametrize("category", list(_BASE))
def test_balanced_base_values(category):
    # balanced は基準値そのもの(倍率1.0)。
    pw, rw, ps, rs = _BASE[category]
    p = presets.resolve_cleaning("balanced", category)
    assert p["pos_window"] == pw
    assert p["rot_window"] == rw
    assert p["pos_strength"] == pytest.approx(ps)
    assert p["rot_strength"] == pytest.approx(rs)


@pytest.mark.parametrize("preset", ["light", "balanced", "stable-foot", "strong"])
@pytest.mark.parametrize("category", list(_BASE))
def test_multiplier_applies_to_strength_only(preset, category):
    # 全プリセット×全種別で、倍率は位置・回転の両強度のみに掛かり、窓幅は不変であることを検証する。
    # これにより種別ごとの適用漏れ・回転強度への誤適用・窓幅の誤変更を一括して捕捉する。
    pw, rw, ps, rs = _BASE[category]
    m = _expected_multiplier(preset, category)
    p = presets.resolve_cleaning(preset, category)
    assert p["pos_window"] == pw       # 窓幅は倍率で変えない
    assert p["rot_window"] == rw
    assert p["pos_strength"] == pytest.approx(ps * m)
    assert p["rot_strength"] == pytest.approx(rs * m)


def test_unknown_category_conservative():
    # 分類不能 unknown は保守的な弱設定(窓3/3・強度0.10/0.10)。
    p = presets.resolve_cleaning("balanced", "unknown")
    assert (p["pos_window"], p["rot_window"]) == (3, 3)
    assert p["pos_strength"] == pytest.approx(0.10)
    assert p["rot_strength"] == pytest.approx(0.10)


def test_invalid_preset_raises():
    with pytest.raises(ValueError):
        presets.resolve_cleaning("turbo", "center")


def test_invalid_category_raises():
    with pytest.raises(ValueError):
        presets.resolve_cleaning("balanced", "nonexistent")


# --- 疎化の許容誤差解決(§5.3)-----------------------------------------------
# 各ボーンの許容誤差 = プリセット基準値 × 種別スケール。--reduce-error-* 明示時は基準値を上書き
# (種別スケールは引き続き掛ける)。precise/balanced/aggressive と種別スケールは mocapvmd 独自値。

# §5.3 プリセット基準値(位置 MMD単位 / 回転 度)。
_REDUCE_BASE = {"precise": (0.01, 0.10), "balanced": (0.02, 0.20), "aggressive": (0.05, 0.40)}

# §5.3 種別スケール(位置, 回転)。
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
    t = presets.resolve_reduction_tolerances("balanced", "center", override_pos=0.1, override_rot=2.0)
    assert t["bone_pos"] == pytest.approx(0.1 * 0.7)   # center 位置スケール 0.7
    assert t["bone_rot"] == pytest.approx(2.0 * 0.8)   # center 回転スケール 0.8


def test_reduction_tolerance_partial_override_pos_only():
    # 位置のみ上書き。回転はプリセット基準値のまま種別スケールが掛かる。
    t = presets.resolve_reduction_tolerances("precise", "fingers", override_pos=0.2)
    assert t["bone_pos"] == pytest.approx(0.2 * 1.5)    # override × fingers位置1.5
    assert t["bone_rot"] == pytest.approx(0.10 * 1.5)   # precise回転0.10 × fingers回転1.5


def test_reduction_tolerance_partial_override_rot_only():
    # 回転のみ上書き。位置はプリセット基準値のまま種別スケールが掛かる(回転側だけの処理漏れを弾く)。
    t = presets.resolve_reduction_tolerances("precise", "fingers", override_rot=2.0)
    assert t["bone_pos"] == pytest.approx(0.01 * 1.5)   # precise位置0.01 × fingers位置1.5
    assert t["bone_rot"] == pytest.approx(2.0 * 1.5)    # override × fingers回転1.5


@pytest.mark.parametrize("kw", [{"override_pos": 0.0}, {"override_rot": 0.0}])
def test_reduction_tolerance_zero_override_is_valid(kw):
    # 非負を許容するので 0 は有効(ValueError にしない)。0 は全キー保持の設定。
    t = presets.resolve_reduction_tolerances("balanced", "center", **kw)
    if "override_pos" in kw:
        assert t["bone_pos"] == pytest.approx(0.0)
        assert t["bone_rot"] == pytest.approx(0.20 * 0.8)
    else:
        assert t["bone_rot"] == pytest.approx(0.0)
        assert t["bone_pos"] == pytest.approx(0.02 * 0.7)


def test_reduction_tolerance_invalid_preset_raises():
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("turbo", "center")


def test_reduction_tolerance_invalid_category_raises():
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("balanced", "nonexistent")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.01])
def test_reduction_tolerance_invalid_override_raises(bad):
    # 非有限(符号問わず)・負の上書き値は ValueError(CLI で終了コード2 へ変換される)。
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("balanced", "center", override_pos=bad)
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("balanced", "center", override_rot=bad)


# --- 接地ロック強度(§5.4)---------------------------------------------------
# 接地ロックは接地中に足IK・つま先IKを接地アンカーへ寄せるブレンド係数(0〜1)。倍率でなく直接値。
# foot_ik の X/Z 接地中央はプリセット別、他(foot_ik の Y、toe_ik の全チャンネル)はプリセット非依存。

# §5.4 foot_ik X/Z 接地中央のプリセット別確定値。stable-foot が最強、light が最弱。
_FOOT_XZ_CENTER = {"light": 0.70, "balanced": 0.90, "strong": 0.93, "stable-foot": 0.97}


@pytest.mark.parametrize("preset,xz_center", list(_FOOT_XZ_CENTER.items()))
def test_foot_lock_foot_ik_xz_center_by_preset(preset, xz_center):
    # foot_ik の X/Z 接地中央はプリセット別、接地端は 0.25 固定(§5.4)。
    p = presets.resolve_foot_lock(preset, "foot_ik")
    assert p["xz_center"] == pytest.approx(xz_center)
    assert p["xz_edge"] == pytest.approx(0.25)


@pytest.mark.parametrize("preset", ["light", "balanced", "stable-foot", "strong"])
def test_foot_lock_foot_ik_y_is_preset_independent(preset):
    # foot_ik の Y は中央0.50・端0.10 でプリセット非依存(§5.4)。
    p = presets.resolve_foot_lock(preset, "foot_ik")
    assert p["y_center"] == pytest.approx(0.50)
    assert p["y_edge"] == pytest.approx(0.10)


@pytest.mark.parametrize("preset", ["light", "balanced", "stable-foot", "strong"])
def test_foot_lock_toe_ik_is_preset_independent(preset):
    # toe_ik は X/Z・Y とも中央0.30・端0.10 でプリセット非依存(つま先の動き・回転を保つ。§5.4)。
    p = presets.resolve_foot_lock(preset, "toe_ik")
    assert p["xz_center"] == pytest.approx(0.30)
    assert p["xz_edge"] == pytest.approx(0.10)
    assert p["y_center"] == pytest.approx(0.30)
    assert p["y_edge"] == pytest.approx(0.10)


def test_foot_lock_foot_ik_xz_center_ordering():
    # 接地固定の強さは stable-foot > strong > balanced > light(§5.4)。
    centers = [presets.resolve_foot_lock(p, "foot_ik")["xz_center"]
               for p in ("light", "balanced", "strong", "stable-foot")]
    assert centers == sorted(centers)
    assert len(set(centers)) == 4


@pytest.mark.parametrize("category", ["foot_ik", "toe_ik"])
@pytest.mark.parametrize("preset", ["light", "balanced", "stable-foot", "strong"])
def test_foot_lock_fade_width_default_3(preset, category):
    # フェード幅は端から既定3フレーム(§5.4)。
    assert presets.resolve_foot_lock(preset, category)["fade_width"] == 3


def test_foot_lock_invalid_preset_raises():
    with pytest.raises(ValueError):
        presets.resolve_foot_lock("turbo", "foot_ik")


@pytest.mark.parametrize("category", ["center", "legs", "toe", "unknown"])
def test_foot_lock_non_footik_category_raises(category):
    # 接地ロックは foot_ik / toe_ik のみ対象。他種別は対象外で ValueError。
    with pytest.raises(ValueError):
        presets.resolve_foot_lock("balanced", category)
