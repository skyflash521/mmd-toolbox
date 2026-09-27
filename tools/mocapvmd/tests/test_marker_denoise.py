import pytest

from mocapvmd.marker_denoise import (
    DEFAULT_PRESET,
    SmoothParams,
    SmoothResult,
    smooth,
)
from mocapvmd.model_profile import load_mocap_profile

_CATEGORIES = {"center", "torso", "head", "arms", "wrists", "legs", "feet"}


def _profile_categories():
    profile = load_mocap_profile(None)
    return {m: mb.category for m, mb in profile.marker_bindings.items()}


def _flat(value, n):
    return [(value, 0.0, 0.0) for _ in range(n)]


_EXPECTED_PRESET = {
    "center": SmoothParams(window=7, strength=0.35, max_disp=0.15),
    "torso": SmoothParams(window=5, strength=0.30, max_disp=0.20),
    "head": SmoothParams(window=5, strength=0.25, max_disp=0.15),
    "arms": SmoothParams(window=5, strength=0.25, max_disp=0.25),
    "wrists": SmoothParams(window=5, strength=0.35, max_disp=0.30),
    "legs": SmoothParams(window=5, strength=0.25, max_disp=0.25),
    "feet": SmoothParams(window=5, strength=0.20, max_disp=0.20),
}


def test_default_preset_covers_all_categories_with_odd_windows():
    assert set(DEFAULT_PRESET) == _CATEGORIES
    for p in DEFAULT_PRESET.values():
        assert isinstance(p, SmoothParams)
        assert p.window >= 3 and p.window % 2 == 1
        assert 0.0 < p.strength <= 1.0
        assert p.max_disp > 0.0


def test_default_preset_values():
    for cat, expected in _EXPECTED_PRESET.items():
        p = DEFAULT_PRESET[cat]
        assert p.window == expected.window
        assert p.strength == pytest.approx(expected.strength)
        assert p.max_disp == pytest.approx(expected.max_disp)


def test_constant_series_unchanged():
    series = {"head": _flat(3.0, 10)}
    cats = {"head": "head"}
    res = smooth(series, cats)
    assert isinstance(res, SmoothResult)
    for got, orig in zip(res.markers["head"], series["head"], strict=True):
        assert got == pytest.approx(orig, abs=1e-9)
    assert res.displacement["head"] == pytest.approx(0.0, abs=1e-9)


def test_single_spike_within_max_disp_is_reduced_and_neighbors_kept():
    series = {"wrist_l": _flat(0.0, 9)}
    spike_below_wrists_max_disp = 0.2
    series["wrist_l"][4] = (spike_below_wrists_max_disp, 0.0, 0.0)
    cats = {"wrist_l": "wrists"}
    res = smooth(series, cats)
    assert abs(res.markers["wrist_l"][4][0]) < spike_below_wrists_max_disp
    assert res.markers["wrist_l"][0][0] == pytest.approx(0.0, abs=1e-9)


def test_every_frame_displacement_is_clamped_to_category_max_disp():
    series = {"wrist_l": [(float(i % 2) * 10.0, 0.0, 0.0) for i in range(20)]}
    cats = {"wrist_l": "wrists"}
    res = smooth(series, cats)
    md = DEFAULT_PRESET["wrists"].max_disp
    for got, orig in zip(res.markers["wrist_l"], series["wrist_l"], strict=True):
        disp = sum((g - o) ** 2 for g, o in zip(got, orig, strict=True)) ** 0.5
        assert disp <= md + 1e-9


def test_step_at_cut_is_not_averaged_across():
    series = {"center": _flat(0.0, 6) + _flat(10.0, 6)}
    cats = {"center": "center"}
    res = smooth(series, cats, cuts=(6,))
    assert res.markers["center"][5][0] == pytest.approx(0.0, abs=1e-9)
    assert res.markers["center"][6][0] == pytest.approx(10.0, abs=1e-9)


def test_linear_ramp_preserved():
    series = {"wrist_l": [(float(i), 0.0, 0.0) for i in range(12)]}
    cats = {"wrist_l": "wrists"}
    res = smooth(series, cats)
    for got, orig in zip(res.markers["wrist_l"], series["wrist_l"], strict=True):
        assert got[0] == pytest.approx(orig[0], abs=1e-6)


def test_direction_change_peak_stays_local_max_within_max_disp():
    series = {"wrist_l": [(float(v), 0.0, 0.0) for v in (0, 1, 2, 3, 2, 1, 0)]}
    cats = {"wrist_l": "wrists"}
    res = smooth(series, cats)
    md = DEFAULT_PRESET["wrists"].max_disp
    peak = res.markers["wrist_l"][3][0]
    assert peak >= 3.0 - md
    assert peak > res.markers["wrist_l"][2][0]
    assert peak > res.markers["wrist_l"][4][0]


def test_displacement_is_largest_smoothing_shift_over_frames():
    series = {"head": _flat(0.0, 9)}
    series["head"][4] = (2.0, 0.0, 0.0)
    cats = {"head": "head"}
    res = smooth(series, cats)
    expected = max(
        sum((g - o) ** 2 for g, o in zip(got, orig, strict=True)) ** 0.5
        for got, orig in zip(res.markers["head"], series["head"], strict=True)
    )
    assert res.displacement["head"] == pytest.approx(expected, abs=1e-9)


def test_empty_series_returns_empty():
    res = smooth({"head": []}, {"head": "head"})
    assert res.markers["head"] == []
    assert res.displacement["head"] == pytest.approx(0.0)


def test_smooths_every_marker_with_profile_categories():
    cats = _profile_categories()
    series = {m: _flat(1.0, 8) for m in cats}
    res = smooth(series, cats)
    assert set(res.markers) == set(cats)
    assert set(res.displacement) == set(cats)
