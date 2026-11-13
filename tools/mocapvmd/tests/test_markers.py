import pytest

from mocapvmd.markers import (
    evaluate_world_poses,
    extract_markers,
)
from mocapvmd.model_profile import load_mocap_profile


def _approx(v):
    return pytest.approx(v, abs=1e-4)


def test_base_pose_world_matches_profile_positions():
    profile = load_mocap_profile(None)
    world = evaluate_world_poses(profile, {}, range(0, 1))
    head_idx = profile.required_bones["head"]
    assert world[0][head_idx].position == _approx(profile.model.bones[head_idx].position)


def test_marker_positions_are_bone_world_positions():
    profile = load_mocap_profile(None)
    world = evaluate_world_poses(profile, {}, range(0, 2))
    traj = extract_markers(profile, world)
    for marker, mb in profile.marker_bindings.items():
        for f in range(2):
            assert traj.markers[marker][f] == _approx(world[f][mb.bone].position)
