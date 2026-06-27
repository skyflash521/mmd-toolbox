"""マーカー抽出と共通FK接続のテスト。

mocap側は独自FKを持たず、共通 mmd_toolbox.pmx.pose で profile.model のワールド
姿勢を評価し、そこから標準マーカーのワールド軌跡と派生特徴ベクトル列を作る。
"""

import math

import pytest

from mmd_toolbox.pmx.pose import WorldBonePose, evaluate_fk_range

from mocapvmd.model_profile import STANDARD_BONE_NAMES, load_mocap_profile

from .helpers import bone, build_standard_pmx

try:
    from mocapvmd.markers import (
        MarkerTrajectories,
        evaluate_world_poses,
        extract_markers,
    )

    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False

# impl pending: pose-denoise Step3+4 markers
pytestmark = (
    []
    if _IMPORT_OK
    else pytest.mark.skip(reason="impl pending: pose-denoise Step3+4 markers")
)

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

_RZ45 = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))


def _approx(v):
    return pytest.approx(v, abs=1e-4)


# ---------------------------------------------------------------------------
# 共通FK接続(独自FKを持たない)
# ---------------------------------------------------------------------------


def test_evaluate_world_poses_uses_common_fk():
    profile = load_mocap_profile(None)
    world = evaluate_world_poses(profile, {}, range(0, 3))
    assert len(world) == 3
    for per_frame in world:
        assert len(per_frame) == len(profile.model.bones)
        assert all(isinstance(w, WorldBonePose) for w in per_frame)


def test_evaluate_world_poses_delegates_to_common_fk():
    # mocap側に独自FKを持たず、共通 evaluate_fk_range と同一結果を返す。
    profile = load_mocap_profile(None)
    tracks = {"上半身": [bone("上半身", 0, rot=_RZ45)]}
    frames = range(0, 3)
    got = evaluate_world_poses(profile, tracks, frames)
    expected = evaluate_fk_range(profile.model, tracks, frames)
    assert got == expected


def test_evaluate_world_poses_with_pmx_profile(tmp_path):
    # PMX指定由来の profile.model でも共通FKから world pose を取得できる(§10.3)。
    path = tmp_path / "model.pmx"
    path.write_bytes(build_standard_pmx(list(STANDARD_BONE_NAMES.values())))
    profile = load_mocap_profile(str(path))
    assert profile.source == "pmx"
    world = evaluate_world_poses(profile, {}, range(0, 2))
    assert len(world) == 2
    assert all(len(pf) == len(profile.model.bones) for pf in world)
    assert all(isinstance(w, WorldBonePose) for pf in world for w in pf)


def test_base_pose_world_matches_profile_positions():
    # キー無し(基準姿勢)では各ボーンのワールド位置が縮約骨格の基準位置に一致。
    profile = load_mocap_profile(None)
    world = evaluate_world_poses(profile, {}, range(0, 1))
    head_idx = profile.required_bones["head"]
    assert world[0][head_idx].position == _approx(profile.model.bones[head_idx].position)


# ---------------------------------------------------------------------------
# マーカー抽出
# ---------------------------------------------------------------------------


def test_extract_returns_all_markers_and_features():
    profile = load_mocap_profile(None)
    world = evaluate_world_poses(profile, {}, range(0, 2))
    traj = extract_markers(profile, world)
    assert isinstance(traj, MarkerTrajectories)
    assert set(traj.markers) == _EXPECTED_MARKERS
    assert set(traj.features) == _EXPECTED_FEATURES
    for series in traj.markers.values():
        assert len(series) == 2
        assert all(len(p) == 3 for p in series)
    for series in traj.features.values():
        assert len(series) == 2
        assert all(len(p) == 3 for p in series)


def test_marker_positions_are_bone_world_positions():
    # 全マーカーが対応ボーンのワールド位置(offset(0,0,0))に一致する。
    profile = load_mocap_profile(None)
    world = evaluate_world_poses(profile, {}, range(0, 2))
    traj = extract_markers(profile, world)
    for marker, mb in profile.marker_bindings.items():
        for f in range(2):
            assert traj.markers[marker][f] == _approx(world[f][mb.bone].position)


def test_feature_vectors_are_a_minus_b():
    # 全派生特徴がベクトル a-b(始点ロール - 終点ロール)に一致する。
    profile = load_mocap_profile(None)
    world = evaluate_world_poses(profile, {}, range(0, 2))
    traj = extract_markers(profile, world)
    for feature, fb in profile.feature_bindings.items():
        ai = profile.required_bones[fb.a]
        bi = profile.required_bones[fb.b]
        for f in range(2):
            a = world[f][ai].position
            b = world[f][bi].position
            expected = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
            assert traj.features[feature][f] == _approx(expected)


def test_parent_rotation_moves_marker():
    # 上半身を回すと頭マーカーのワールド位置が基準から動く。
    profile = load_mocap_profile(None)
    tracks = {"上半身": [bone("上半身", 0, rot=_RZ45)]}
    base = extract_markers(profile, evaluate_world_poses(profile, {}, range(0, 1)))
    moved = extract_markers(profile, evaluate_world_poses(profile, tracks, range(0, 1)))
    assert moved.markers["head"][0] != _approx(base.markers["head"][0])
