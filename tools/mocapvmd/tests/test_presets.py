import pytest

from mocapvmd import presets


def _category_scale(preset, category):
    base = presets.reduction_base(preset)
    t = presets.resolve_reduction_tolerances(preset, category)
    return t["bone_pos"] / base.pos, t["bone_rot"] / base.rot_deg


def test_preset_names_are_reduction_levels():
    assert presets.PRESET_NAMES == ("slower", "slow", "medium", "fast", "faster")


def test_strength_scales_each_blend_of_unit_strength():
    category_with_distinct_pos_and_rot_blend = "root"
    unit = presets.resolve_cleaning(1.0, category_with_distinct_pos_and_rot_blend)
    half = presets.resolve_cleaning(0.5, category_with_distinct_pos_and_rot_blend)
    assert half["pos_strength"] == pytest.approx(0.5 * unit["pos_strength"])
    assert half["rot_strength"] == pytest.approx(0.5 * unit["rot_strength"])


def test_strength_clamps_blend_to_one():
    strength_saturating_every_blend = 10.0
    p = presets.resolve_cleaning(strength_saturating_every_blend, "foot_ik")
    assert p["pos_strength"] == pytest.approx(1.0)
    assert p["rot_strength"] == pytest.approx(1.0)


def test_zero_strength_gives_zero_blend():
    p = presets.resolve_cleaning(0.0, "center")
    assert p["pos_strength"] == pytest.approx(0.0)
    assert p["rot_strength"] == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [float("nan"), -0.01])
def test_invalid_strength_raises(bad):
    with pytest.raises(ValueError):
        presets.resolve_cleaning(bad, "center")


def test_invalid_category_raises():
    with pytest.raises(ValueError):
        presets.resolve_cleaning(1.0, "nonexistent")


def test_reduction_tolerance_override_replaces_base_then_scales():
    pos_scale, rot_scale = _category_scale("medium", "center")
    t = presets.resolve_reduction_tolerances("medium", "center", override_pos=0.1, override_rot=2.0)
    assert t["bone_pos"] == pytest.approx(0.1 * pos_scale)
    assert t["bone_rot"] == pytest.approx(2.0 * rot_scale)


def test_reduction_tolerance_partial_override_pos_only():
    pos_scale, _ = _category_scale("slower", "fingers")
    t = presets.resolve_reduction_tolerances("slower", "fingers", override_pos=0.2)
    assert t["bone_pos"] == pytest.approx(0.2 * pos_scale)
    assert t["bone_rot"] == presets.resolve_reduction_tolerances("slower", "fingers")["bone_rot"]


def test_reduction_tolerance_partial_override_rot_only():
    _, rot_scale = _category_scale("slower", "fingers")
    t = presets.resolve_reduction_tolerances("slower", "fingers", override_rot=2.0)
    assert t["bone_pos"] == presets.resolve_reduction_tolerances("slower", "fingers")["bone_pos"]
    assert t["bone_rot"] == pytest.approx(2.0 * rot_scale)


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"override_pos": 0.0}, id="pos"),
        pytest.param({"override_rot": 0.0}, id="rot"),
    ],
)
def test_reduction_tolerance_zero_override_is_accepted_and_zeroes_that_channel(kw):
    unchanged = presets.resolve_reduction_tolerances("medium", "center")
    t = presets.resolve_reduction_tolerances("medium", "center", **kw)
    if "override_pos" in kw:
        assert t["bone_pos"] == pytest.approx(0.0)
        assert t["bone_rot"] == unchanged["bone_rot"]
    else:
        assert t["bone_rot"] == pytest.approx(0.0)
        assert t["bone_pos"] == unchanged["bone_pos"]


def test_reduction_tolerance_invalid_preset_raises():
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("turbo", "center")


def test_reduction_tolerance_invalid_category_raises():
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("medium", "nonexistent")


@pytest.mark.parametrize("bad", [float("nan"), -0.01])
def test_reduction_tolerance_invalid_override_raises(bad):
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("medium", "center", override_pos=bad)
    with pytest.raises(ValueError):
        presets.resolve_reduction_tolerances("medium", "center", override_rot=bad)


@pytest.mark.parametrize("category", ["toe"])
def test_foot_lock_non_footik_category_raises(category):
    with pytest.raises(ValueError):
        presets.resolve_foot_lock(1.0, category)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_foot_lock_invalid_suppression_raises(bad):
    with pytest.raises(ValueError):
        presets.resolve_foot_lock(bad, "foot_ik")


@pytest.mark.parametrize("bad", [1.01])
def test_foot_detection_invalid_suppression_raises(bad):
    with pytest.raises(ValueError):
        presets.resolve_foot_detection(bad)
