import pytest

from mocapvmd.default_profile import ROLE_TO_INDEX as _DEFAULT_ROLE_TO_INDEX
from mocapvmd.default_profile import build_default_model
from mocapvmd.model_profile import (
    STANDARD_BONE_NAMES,
    MocapModelProfile,
    MocapModelProfileError,
    load_mocap_profile,
    validate_required_roles,
)
from pmx.types import PmxFormatError, PmxModel

from .helpers import build_standard_pmx

_EXPECTED_MARKERS = {
    "center", "pelvis", "chest", "head",
    "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
    "wrist_l", "wrist_r", "knee_l", "knee_r",
    "ankle_l", "ankle_r", "toe_l", "toe_r",
}
_EXPECTED_FEATURES = {
    "shoulder_line", "hip_line", "torso_axis",
    "forearm_l", "forearm_r", "shin_l", "shin_r",
}
_CATEGORIES = {"center", "torso", "head", "arms", "wrists", "legs", "feet"}


def _write_pmx(tmp_path, names):
    p = tmp_path / "model.pmx"
    p.write_bytes(build_standard_pmx(names))
    return str(p)


def test_default_profile_loads():
    p = load_mocap_profile(None)
    assert isinstance(p, MocapModelProfile)
    assert p.source == "default"
    assert isinstance(p.model, PmxModel)
    assert set(p.required_bones) >= set(STANDARD_BONE_NAMES)
    assert set(p.marker_bindings) == _EXPECTED_MARKERS
    assert set(p.feature_bindings) == _EXPECTED_FEATURES


def test_default_marker_bindings_valid():
    p = load_mocap_profile(None)
    n = len(p.model.bones)
    for marker, mb in p.marker_bindings.items():
        assert mb.marker == marker
        assert 0 <= mb.bone < n
        assert mb.offset == (0.0, 0.0, 0.0)
        assert mb.category in _CATEGORIES
        assert 0.0 < mb.weight <= 1.0


def test_default_feature_bindings_valid():
    p = load_mocap_profile(None)
    for feature, fb in p.feature_bindings.items():
        assert fb.feature == feature
        assert fb.kind == "vector"
        assert fb.a in p.required_bones
        assert fb.b in p.required_bones
        assert 0.0 < fb.weight <= 1.0


def test_validate_required_roles_accepts_default_profile():
    validate_required_roles(load_mocap_profile(None))


def test_default_role_index_matches_standard_names():
    model = build_default_model()
    for role, name in STANDARD_BONE_NAMES.items():
        assert _DEFAULT_ROLE_TO_INDEX[role] == model.name_to_index[name]


def test_pmx_profile_has_same_shape_as_default_and_resolves_names(tmp_path):
    path = _write_pmx(tmp_path, list(STANDARD_BONE_NAMES.values()))
    p = load_mocap_profile(path)
    assert isinstance(p, MocapModelProfile)
    assert p.source == "pmx"
    assert isinstance(p.model, PmxModel)
    assert set(p.required_bones) >= set(STANDARD_BONE_NAMES)
    assert set(p.marker_bindings) == _EXPECTED_MARKERS
    assert set(p.feature_bindings) == _EXPECTED_FEATURES
    n = len(p.model.bones)
    for marker, mb in p.marker_bindings.items():
        assert mb.marker == marker
        assert 0 <= mb.bone < n
        assert mb.offset == (0.0, 0.0, 0.0)
        assert mb.category in _CATEGORIES
        assert 0.0 < mb.weight <= 1.0
    for feature, fb in p.feature_bindings.items():
        assert fb.feature == feature
        assert fb.kind == "vector"
        assert fb.a in p.required_bones
        assert fb.b in p.required_bones
        assert 0.0 < fb.weight <= 1.0
    for role, name in STANDARD_BONE_NAMES.items():
        assert p.model.bones[p.required_bones[role]].name == name


def test_missing_required_role_raises(tmp_path):
    names = [n for n in STANDARD_BONE_NAMES.values() if n != STANDARD_BONE_NAMES["wrist_r"]]
    path = _write_pmx(tmp_path, names)
    with pytest.raises(MocapModelProfileError):
        load_mocap_profile(path)


def test_pmx_format_error_propagates(tmp_path):
    bad = tmp_path / "bad.pmx"
    bad.write_bytes(b"NOTPMX")
    with pytest.raises(PmxFormatError):
        load_mocap_profile(str(bad))


def test_standard_bone_names_cover_exactly_the_default_roles():
    assert set(STANDARD_BONE_NAMES) == set(_DEFAULT_ROLE_TO_INDEX)
