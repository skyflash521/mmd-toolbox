import pytest

from sparsevmd import presets


def test_preset_names():
    assert presets.PRESET_NAMES == ("precise", "balanced", "aggressive")


def test_override_precedence():
    t = presets.resolve_tolerances("balanced", {"camera_fov": 0.75, "bone_pos": 0.001})
    assert t.camera_fov == pytest.approx(0.75)
    assert t.bone_pos == pytest.approx(0.001)
    assert t.camera_pos == presets.resolve_tolerances("balanced").camera_pos


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        presets.resolve_tolerances("ultra")


def test_unknown_override_key_rejected():
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"bone_position": 0.01})
