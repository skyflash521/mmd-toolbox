from typing import NamedTuple

import pytest

from mocapvmd import presets


class _CleaningAtUnitStrength(NamedTuple):
    pos_window: int
    rot_window: int
    pos_strength: float
    rot_strength: float


class _ReductionBase(NamedTuple):
    pos: float
    rot_deg: float


class _ReductionScale(NamedTuple):
    pos: float
    rot: float


_CLEANING_AT_UNIT_STRENGTH = {
    "root": _CleaningAtUnitStrength(3, 3, 0.15, 0.10),
    "center": _CleaningAtUnitStrength(7, 5, 0.45, 0.25),
    "torso": _CleaningAtUnitStrength(5, 5, 0.25, 0.35),
    "arms": _CleaningAtUnitStrength(5, 5, 0.25, 0.25),
    "fingers": _CleaningAtUnitStrength(3, 3, 0.10, 0.15),
    "legs": _CleaningAtUnitStrength(5, 5, 0.30, 0.25),
    "foot_ik": _CleaningAtUnitStrength(7, 3, 0.65, 0.20),
    "toe_ik": _CleaningAtUnitStrength(5, 3, 0.50, 0.20),
    "unknown": _CleaningAtUnitStrength(3, 3, 0.10, 0.10),
}

_REDUCE_BASE = {
    "slower": _ReductionBase(0.05, 0.40),
    "slow": _ReductionBase(0.10, 0.75),
    "medium": _ReductionBase(0.20, 1.50),
    "fast": _ReductionBase(0.80, 6.0),
    "faster": _ReductionBase(1.60, 12.0),
}

_REDUCE_SCALE = {
    "root": _ReductionScale(1.0, 1.0),
    "center": _ReductionScale(0.7, 0.8),
    "torso": _ReductionScale(0.8, 0.9),
    "arms": _ReductionScale(1.0, 1.0),
    "fingers": _ReductionScale(1.5, 1.5),
    "legs": _ReductionScale(1.0, 1.0),
    "foot_ik": _ReductionScale(0.7, 1.0),
    "toe_ik": _ReductionScale(1.0, 0.8),
    "unknown": _ReductionScale(1.0, 1.0),
}

_FOOT_IK_XZ_EDGE_AT_FULL_SUPPRESSION = 0.25


def test_preset_names_are_reduction_levels():
    assert presets.PRESET_NAMES == ("slower", "slow", "medium", "fast", "faster")


@pytest.mark.parametrize("category", list(_CLEANING_AT_UNIT_STRENGTH))
def test_unit_strength_returns_category_base_values(category):
    expected = _CLEANING_AT_UNIT_STRENGTH[category]
    p = presets.resolve_cleaning(1.0, category)
    assert p["pos_window"] == expected.pos_window
    assert p["rot_window"] == expected.rot_window
    assert p["pos_strength"] == pytest.approx(expected.pos_strength)
    assert p["rot_strength"] == pytest.approx(expected.rot_strength)


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, 1.4])
@pytest.mark.parametrize("category", list(_CLEANING_AT_UNIT_STRENGTH))
def test_strength_multiplies_both_blends_clamped_and_keeps_windows(strength, category):
    base = _CLEANING_AT_UNIT_STRENGTH[category]
    p = presets.resolve_cleaning(strength, category)
    assert p["pos_window"] == base.pos_window
    assert p["rot_window"] == base.rot_window
    assert p["pos_strength"] == pytest.approx(min(1.0, base.pos_strength * strength))
    assert p["rot_strength"] == pytest.approx(min(1.0, base.rot_strength * strength))


def test_strength_clamps_blend_to_one():
    strength_saturating_every_blend = 10.0
    p = presets.resolve_cleaning(strength_saturating_every_blend, "foot_ik")
    assert p["pos_strength"] == pytest.approx(1.0)
    assert p["rot_strength"] == pytest.approx(1.0)


def test_zero_strength_gives_zero_blend():
    p = presets.resolve_cleaning(0.0, "center")
    assert p["pos_strength"] == pytest.approx(0.0)
    assert p["rot_strength"] == pytest.approx(0.0)


def test_unknown_category_uses_windows_3_and_blends_0_10():
    p = presets.resolve_cleaning(1.0, "unknown")
    assert (p["pos_window"], p["rot_window"]) == (3, 3)
    assert p["pos_strength"] == pytest.approx(0.10)
    assert p["rot_strength"] == pytest.approx(0.10)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.01])
def test_invalid_strength_raises(bad):
    with pytest.raises(ValueError):
        presets.resolve_cleaning(bad, "center")


def test_invalid_category_raises():
    with pytest.raises(ValueError):
        presets.resolve_cleaning(1.0, "nonexistent")


@pytest.mark.parametrize("preset", list(_REDUCE_BASE))
@pytest.mark.parametrize("category", list(_REDUCE_SCALE))
def test_reduction_tolerance_base_times_scale(preset, category):
    base = _REDUCE_BASE[preset]
    scale = _REDUCE_SCALE[category]
    t = presets.resolve_reduction_tolerances(preset, category)
    assert t["bone_pos"] == pytest.approx(base.pos * scale.pos)
    assert t["bone_rot"] == pytest.approx(base.rot_deg * scale.rot)


def test_reduction_tolerance_override_replaces_base_then_scales():
    center = _REDUCE_SCALE["center"]
    t = presets.resolve_reduction_tolerances("medium", "center", override_pos=0.1, override_rot=2.0)
    assert t["bone_pos"] == pytest.approx(0.1 * center.pos)
    assert t["bone_rot"] == pytest.approx(2.0 * center.rot)


def test_reduction_tolerance_partial_override_pos_only():
    fingers = _REDUCE_SCALE["fingers"]
    t = presets.resolve_reduction_tolerances("slower", "fingers", override_pos=0.2)
    assert t["bone_pos"] == pytest.approx(0.2 * fingers.pos)
    assert t["bone_rot"] == pytest.approx(_REDUCE_BASE["slower"].rot_deg * fingers.rot)


def test_reduction_tolerance_partial_override_rot_only():
    fingers = _REDUCE_SCALE["fingers"]
    t = presets.resolve_reduction_tolerances("slower", "fingers", override_rot=2.0)
    assert t["bone_pos"] == pytest.approx(_REDUCE_BASE["slower"].pos * fingers.pos)
    assert t["bone_rot"] == pytest.approx(2.0 * fingers.rot)


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"override_pos": 0.0}, id="pos"),
        pytest.param({"override_rot": 0.0}, id="rot"),
    ],
)
def test_reduction_tolerance_zero_override_is_accepted_and_zeroes_that_channel(kw):
    medium = _REDUCE_BASE["medium"]
    center = _REDUCE_SCALE["center"]
    t = presets.resolve_reduction_tolerances("medium", "center", **kw)
    if "override_pos" in kw:
        assert t["bone_pos"] == pytest.approx(0.0)
        assert t["bone_rot"] == pytest.approx(medium.rot_deg * center.rot)
    else:
        assert t["bone_rot"] == pytest.approx(0.0)
        assert t["bone_pos"] == pytest.approx(medium.pos * center.pos)


def test_reduction_tolerance_invalid_preset_raises():
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("turbo", "center")


def test_reduction_tolerance_invalid_category_raises():
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("medium", "nonexistent")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.01])
def test_reduction_tolerance_invalid_override_raises(bad):
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("medium", "center", override_pos=bad)
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("medium", "center", override_rot=bad)


@pytest.mark.parametrize("s,xz", [(0.0, 0.0), (0.5, 0.485), (1.0, 0.97)])
def test_foot_lock_foot_ik_xz_is_linear_in_suppression_with_edge_not_above_center(s, xz):
    p = presets.resolve_foot_lock(s, "foot_ik")
    assert p["xz_center"] == pytest.approx(xz)
    assert p["xz_edge"] == pytest.approx(_FOOT_IK_XZ_EDGE_AT_FULL_SUPPRESSION * s)
    assert p["xz_edge"] <= p["xz_center"] + 1e-9


def test_foot_lock_full_suppression_xz_center_is_0_97():
    assert presets.resolve_foot_lock(1.0, "foot_ik")["xz_center"] == pytest.approx(0.97)


@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_foot_lock_foot_ik_y_is_suppression_independent(s):
    p = presets.resolve_foot_lock(s, "foot_ik")
    assert p["y_center"] == pytest.approx(0.50)
    assert p["y_edge"] == pytest.approx(0.10)


@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_foot_lock_toe_ik_is_suppression_independent(s):
    p = presets.resolve_foot_lock(s, "toe_ik")
    assert p["xz_center"] == pytest.approx(0.30)
    assert p["xz_edge"] == pytest.approx(0.10)
    assert p["y_center"] == pytest.approx(0.30)
    assert p["y_edge"] == pytest.approx(0.10)


def test_foot_lock_foot_ik_xz_center_strictly_increases_with_suppression():
    suppressions = (0.0, 0.25, 0.5, 0.75, 1.0)
    centers = [presets.resolve_foot_lock(s, "foot_ik")["xz_center"] for s in suppressions]
    assert centers == sorted(centers)
    assert len(set(centers)) == len(suppressions)


@pytest.mark.parametrize("category", ["foot_ik", "toe_ik"])
@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_foot_lock_fade_width_is_3(s, category):
    assert presets.resolve_foot_lock(s, category)["fade_width"] == 3


@pytest.mark.parametrize("category", ["center", "legs", "toe", "unknown"])
def test_foot_lock_non_footik_category_raises(category):
    with pytest.raises(ValueError):
        presets.resolve_foot_lock(1.0, category)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01])
def test_foot_lock_invalid_suppression_raises(bad):
    with pytest.raises(ValueError):
        presets.resolve_foot_lock(bad, "foot_ik")


@pytest.mark.parametrize("s,horiz,maxd", [(0.0, 0.08, 0.5), (0.5, 0.54, 1.25), (1.0, 1.0, 2.0)])
def test_foot_detection_is_linear_in_suppression(s, horiz, maxd):
    d = presets.resolve_foot_detection(s)
    assert d["horiz_vel_thresh"] == pytest.approx(horiz)
    assert d["max_displacement"] == pytest.approx(maxd)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01])
def test_foot_detection_invalid_suppression_raises(bad):
    with pytest.raises(ValueError):
        presets.resolve_foot_detection(bad)
