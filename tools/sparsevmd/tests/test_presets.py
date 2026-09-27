import math

import pytest

from sparsevmd import presets


def test_preset_names():
    assert presets.PRESET_NAMES == ("precise", "balanced", "aggressive")


def test_balanced_values():
    t = presets.resolve_tolerances("balanced")
    assert t.bone_pos == pytest.approx(0.01)
    assert t.bone_rot == pytest.approx(0.10)
    assert t.camera_pos == pytest.approx(0.02)
    assert t.camera_rot == pytest.approx(0.05)
    assert t.camera_distance == pytest.approx(0.02)
    assert t.camera_fov == pytest.approx(0.50)


def test_precise_values():
    t = presets.resolve_tolerances("precise")
    assert t.bone_pos == pytest.approx(0.005)
    assert t.bone_rot == pytest.approx(0.05)
    assert t.camera_pos == pytest.approx(0.01)
    assert t.camera_rot == pytest.approx(0.02)
    assert t.camera_distance == pytest.approx(0.01)
    assert t.camera_fov == pytest.approx(0.50)


def test_aggressive_values():
    t = presets.resolve_tolerances("aggressive")
    assert t.bone_pos == pytest.approx(0.05)
    assert t.bone_rot == pytest.approx(0.50)
    assert t.camera_pos == pytest.approx(0.10)
    assert t.camera_rot == pytest.approx(0.25)
    assert t.camera_distance == pytest.approx(0.10)
    assert t.camera_fov == pytest.approx(1.00)


def test_override_precedence():
    t = presets.resolve_tolerances("balanced", {"camera_fov": 0.75, "bone_pos": 0.001})
    assert t.camera_fov == pytest.approx(0.75)
    assert t.bone_pos == pytest.approx(0.001)
    assert t.camera_pos == pytest.approx(0.02)


def test_tolerances_fields():
    t = presets.resolve_tolerances("balanced")
    assert set(vars(t)) == {
        "bone_pos",
        "bone_rot",
        "camera_pos",
        "camera_rot",
        "camera_distance",
        "camera_fov",
    }


@pytest.mark.parametrize(
    "field", ["bone_pos", "bone_rot", "camera_pos", "camera_rot", "camera_distance"]
)
def test_zero_allowed_for_non_fov(field):
    t = presets.resolve_tolerances("balanced", {field: 0.0})
    assert getattr(t, field) == 0.0


def test_override_stored_as_float():
    t = presets.resolve_tolerances(
        "balanced", {"bone_pos": 0, "camera_fov": 1, "camera_pos": "0.03"}
    )
    assert isinstance(t.bone_pos, float) and t.bone_pos == 0.0
    assert isinstance(t.camera_fov, float) and t.camera_fov == 1.0
    assert isinstance(t.camera_pos, float) and t.camera_pos == pytest.approx(0.03)


@pytest.mark.parametrize("bad", [None, object(), "0.5deg", "abc"])
def test_non_numeric_override_rejected(bad):
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"bone_pos": bad})


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        presets.resolve_tolerances("ultra")


def test_fov_floor():
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"camera_fov": 0.4})
    assert presets.resolve_tolerances("balanced", {"camera_fov": 0.5}).camera_fov == 0.5


def test_negative_rejected():
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"bone_pos": -0.01})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_rejected(bad):
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"camera_pos": bad})


def test_unknown_override_key_rejected():
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"bone_position": 0.01})
