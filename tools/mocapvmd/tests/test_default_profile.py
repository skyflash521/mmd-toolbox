from mocapvmd.default_profile import (
    FEATURE_BINDINGS,
    MARKER_BINDINGS,
    ROLE_TO_INDEX,
    build_default_model,
)
from pmx.pose import evaluate_fk, sample_local_poses
from pmx.types import PmxModel

_REQUIRED_ROLES = {
    "center",
    "groove",
    "lower_body",
    "upper_body",
    "upper_body2",
    "neck",
    "head",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "toe_l",
    "toe_r",
}

_CATEGORIES = {"center", "torso", "head", "arms", "wrists", "legs", "feet"}

_EXPECTED_MARKERS = {
    "center",
    "pelvis",
    "chest",
    "head",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "toe_l",
    "toe_r",
}

_EXPECTED_FEATURES = {
    "shoulder_line",
    "hip_line",
    "torso_axis",
    "forearm_l",
    "forearm_r",
    "shin_l",
    "shin_r",
}


def test_build_default_model_is_pmxmodel():
    model = build_default_model()
    assert isinstance(model, PmxModel)
    assert len(model.bones) >= len(_REQUIRED_ROLES)


def test_name_to_index_points_back_to_every_bone():
    model = build_default_model()
    n = len(model.bones)
    for name, idx in model.name_to_index.items():
        assert 0 <= idx < n
        assert model.bones[idx].name == name
    for b in model.bones:
        assert b.name in model.name_to_index


def test_hierarchy_is_self_contained():
    model = build_default_model()
    n = len(model.bones)
    roots = 0
    for b in model.bones:
        if b.parent is None:
            roots += 1
        else:
            assert 0 <= b.parent < n
    assert roots >= 1


def test_required_roles_resolve_to_valid_bones():
    model = build_default_model()
    n = len(model.bones)
    assert _REQUIRED_ROLES <= set(ROLE_TO_INDEX)
    for role in _REQUIRED_ROLES:
        idx = ROLE_TO_INDEX[role]
        assert 0 <= idx < n


def test_required_roles_map_to_distinct_bones():
    indices = [ROLE_TO_INDEX[role] for role in _REQUIRED_ROLES]
    assert len(set(indices)) == len(_REQUIRED_ROLES)


def test_center_and_groove_are_movable():
    model = build_default_model()
    for role in ("center", "groove"):
        assert model.bones[ROLE_TO_INDEX[role]].movable is True


def test_fk_runs_on_default_model():
    model = build_default_model()
    world = evaluate_fk(model, sample_local_poses(model, {}, 0))
    assert len(world) == len(model.bones)


def test_marker_bindings_cover_expected_markers_and_every_category():
    assert set(MARKER_BINDINGS) == _EXPECTED_MARKERS
    used_categories = set()
    for marker, (role, category, weight) in MARKER_BINDINGS.items():
        assert role in ROLE_TO_INDEX, f"未知ロール: {marker} -> {role}"
        assert category in _CATEGORIES, f"未知カテゴリ: {marker} -> {category}"
        assert 0.0 < weight <= 1.0
        used_categories.add(category)
    assert used_categories == _CATEGORIES


def test_feature_bindings_cover_derived_features():
    assert set(FEATURE_BINDINGS) == _EXPECTED_FEATURES
    for feature, (head_role, tail_role, weight) in FEATURE_BINDINGS.items():
        assert head_role in ROLE_TO_INDEX, f"未知ロール head: {feature} -> {head_role}"
        assert tail_role in ROLE_TO_INDEX, f"未知ロール tail: {feature} -> {tail_role}"
        assert head_role != tail_role
        assert 0.0 < weight <= 1.0
