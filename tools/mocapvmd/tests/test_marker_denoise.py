"""マーカー軌跡のロバスト平滑化のテスト。

カット境界で分割し、median filter でスパイクを抑え、ゼロ位相低域化と強度
ブレンド・最大変位クランプで部位別に平滑化する。前後変位を診断に残す。
"""

import pytest

from mocapvmd.model_profile import load_mocap_profile

from mocapvmd.marker_denoise import (
    DEFAULT_PRESET,
    SmoothParams,
    SmoothResult,
    smooth,
)

_CATEGORIES = {"center", "torso", "head", "arms", "wrists", "legs", "feet"}


def _profile_categories():
    profile = load_mocap_profile(None)
    return {m: mb.category for m, mb in profile.marker_bindings.items()}


def _flat(value, n):
    return [(value, 0.0, 0.0) for _ in range(n)]


# ---------------------------------------------------------------------------
# プリセット
# ---------------------------------------------------------------------------


# 部位別既定値 (window, strength, max_disp)。
_EXPECTED_PRESET = {
    "center": (7, 0.35, 0.15),
    "torso": (5, 0.30, 0.20),
    "head": (5, 0.25, 0.15),
    "arms": (5, 0.25, 0.25),
    "wrists": (5, 0.35, 0.30),
    "legs": (5, 0.25, 0.25),
    "feet": (5, 0.20, 0.20),
}


def test_default_preset_covers_all_categories():
    assert set(DEFAULT_PRESET) == _CATEGORIES
    for cat, p in DEFAULT_PRESET.items():
        assert isinstance(p, SmoothParams)
        assert p.window >= 3 and p.window % 2 == 1  # 奇数窓
        assert 0.0 < p.strength <= 1.0
        assert p.max_disp > 0.0


def test_default_preset_matches_spec_values():
    for cat, (window, strength, max_disp) in _EXPECTED_PRESET.items():
        p = DEFAULT_PRESET[cat]
        assert p.window == window
        assert p.strength == pytest.approx(strength)
        assert p.max_disp == pytest.approx(max_disp)


# ---------------------------------------------------------------------------
# 平滑化
# ---------------------------------------------------------------------------


def test_constant_series_unchanged():
    series = {"head": _flat(3.0, 10)}
    cats = {"head": "head"}
    res = smooth(series, cats)
    assert isinstance(res, SmoothResult)
    for got, orig in zip(res.markers["head"], series["head"]):
        assert got == pytest.approx(orig, abs=1e-9)
    assert res.displacement["head"] == pytest.approx(0.0, abs=1e-9)


def test_single_spike_is_reduced():
    # 最大変位内の単発スパイクは外れ値置換＋ブレンドで近傍へ寄る。
    series = {"wrist_l": _flat(0.0, 9)}
    series["wrist_l"][4] = (0.2, 0.0, 0.0)  # wrists の max_disp(0.30)内
    cats = {"wrist_l": "wrists"}
    res = smooth(series, cats)
    # スパイクが減衰する(強度<1なので完全には消えない)。
    assert abs(res.markers["wrist_l"][4][0]) < 0.2
    # 近傍は不変。
    assert res.markers["wrist_l"][0][0] == pytest.approx(0.0, abs=1e-9)


def test_max_displacement_clamped():
    # 大ノイズでも各フレームの変位は部位別 max_disp 以下。
    series = {"wrist_l": [(float(i % 2) * 10.0, 0.0, 0.0) for i in range(20)]}
    cats = {"wrist_l": "wrists"}
    res = smooth(series, cats)
    md = DEFAULT_PRESET["wrists"].max_disp
    for got, orig in zip(res.markers["wrist_l"], series["wrist_l"]):
        disp = sum((g - o) ** 2 for g, o in zip(got, orig)) ** 0.5
        assert disp <= md + 1e-9


def test_no_average_across_cut():
    # カット境界をまたいで平均しない: 境界での段差が保たれる。
    series = {"center": _flat(0.0, 6) + _flat(10.0, 6)}
    cats = {"center": "center"}
    res = smooth(series, cats, cuts=(6,))
    # 前半末尾は0付近、後半先頭は10付近(境界をまたいで混ざらない)。
    assert res.markers["center"][5][0] == pytest.approx(0.0, abs=1e-9)
    assert res.markers["center"][6][0] == pytest.approx(10.0, abs=1e-9)


def test_linear_ramp_preserved():
    # 連続運動(線形ランプ)は平滑化で鈍らない。
    series = {"wrist_l": [(float(i), 0.0, 0.0) for i in range(12)]}
    cats = {"wrist_l": "wrists"}
    res = smooth(series, cats)
    for got, orig in zip(res.markers["wrist_l"], series["wrist_l"]):
        assert got[0] == pytest.approx(orig[0], abs=1e-6)


def test_direction_change_is_protected():
    # 方向転換(ターン)は鈍らせすぎない: ピークは max_disp 以内に保たれ局所最大のまま。
    series = {"wrist_l": [(float(v), 0.0, 0.0) for v in (0, 1, 2, 3, 2, 1, 0)]}
    cats = {"wrist_l": "wrists"}
    res = smooth(series, cats)
    md = DEFAULT_PRESET["wrists"].max_disp
    peak = res.markers["wrist_l"][3][0]
    assert peak >= 3.0 - md  # max_disp を超えて丸めない
    assert peak > res.markers["wrist_l"][2][0]
    assert peak > res.markers["wrist_l"][4][0]


def test_displacement_is_max_change():
    series = {"head": _flat(0.0, 9)}
    series["head"][4] = (2.0, 0.0, 0.0)
    cats = {"head": "head"}
    res = smooth(series, cats)
    expected = max(
        sum((g - o) ** 2 for g, o in zip(got, orig)) ** 0.5
        for got, orig in zip(res.markers["head"], series["head"])
    )
    assert res.displacement["head"] == pytest.approx(expected, abs=1e-9)


def test_empty_series_is_handled():
    # フレーム0(空系列)でも例外を出さず空を返す。
    res = smooth({"head": []}, {"head": "head"})
    assert res.markers["head"] == []
    assert res.displacement["head"] == pytest.approx(0.0)


def test_smooths_with_profile_categories():
    # profile のカテゴリ割当で全マーカーを平滑化できる。
    cats = _profile_categories()
    series = {m: _flat(1.0, 8) for m in cats}
    res = smooth(series, cats)
    assert set(res.markers) == set(cats)
    assert set(res.displacement) == set(cats)
