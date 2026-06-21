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
