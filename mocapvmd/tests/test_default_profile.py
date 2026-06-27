"""既定モデルプロファイル(内蔵縮約骨格)のテスト。

PMX未指定時に使う既定モデルプロファイルが、FK評価とマーカー解決に必要な
情報(縮約ボーン階層・標準ロール対応・マーカー/派生特徴のbinding)を
共通 PmxModel 相当として提供できることを検証する。
"""

from mmd_toolbox.pmx.pose import evaluate_fk, sample_local_poses
from mmd_toolbox.pmx.types import PmxModel

from mocapvmd.default_profile import (
    FEATURE_BINDINGS,
    MARKER_BINDINGS,
    ROLE_TO_INDEX,
    build_default_model,
)

# markers.py が必要とする必須標準ロール(初期マーカー表)。
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

# 計画§6.1 の初期マーカー表。
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

# 計画§6.2 の派生特徴(肩線・腰線・胴体軸・前腕方向・下腿方向)。
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


def test_name_to_index_is_consistent():
    model = build_default_model()
    n = len(model.bones)
    # 各ボーン名から有効なindexを引け、その並びは bones と矛盾しない。
    for name, idx in model.name_to_index.items():
        assert 0 <= idx < n
        assert model.bones[idx].name == name
    # 全ボーンが索引から引ける(同名は先勝ちで少なくとも1つ)。
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
    # 各標準ロールは別々のボーンを指す(wrist_l と elbow_l が同骨に潰れない)。
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


def test_marker_bindings_cover_initial_table():
    # 初期マーカー表を過不足なく提供する。
    assert set(MARKER_BINDINGS) == _EXPECTED_MARKERS
    used_categories = set()
    for marker, (role, category, weight) in MARKER_BINDINGS.items():
        assert role in ROLE_TO_INDEX, f"未知ロール: {marker} -> {role}"
        assert category in _CATEGORIES, f"未知カテゴリ: {marker} -> {category}"
        assert 0.0 < weight <= 1.0
        used_categories.add(category)
    # 部位別カテゴリを全て被覆する。
    assert used_categories == _CATEGORIES


def test_feature_bindings_cover_derived_features():
    assert set(FEATURE_BINDINGS) == _EXPECTED_FEATURES
    for feature, (a_role, b_role, weight) in FEATURE_BINDINGS.items():
        assert a_role in ROLE_TO_INDEX, f"未知ロール a: {feature} -> {a_role}"
        assert b_role in ROLE_TO_INDEX, f"未知ロール b: {feature} -> {b_role}"
        assert a_role != b_role
        assert 0.0 < weight <= 1.0
