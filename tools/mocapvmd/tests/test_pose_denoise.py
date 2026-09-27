import math

import pytest

from mocapvmd.markers import extract_markers
from mocapvmd.model_profile import STANDARD_BONE_NAMES, load_mocap_profile
from mocapvmd.pose_denoise import apply_pose_denoise
from pmx.pose import evaluate_fk, sample_local_poses
from vmd.reduce import BONE_LINEAR_INTERP
from vmd.types import BoneKey

from .helpers import BONE_NONLINEAR, bone, build_standard_pmx


def _marker_trajectory_via_fk(profile, bone_keys, marker):
    tracks = {}
    for k in bone_keys:
        tracks.setdefault(k.name, []).append(k)
    for v in tracks.values():
        v.sort(key=lambda k: k.frame)
    f0 = min(k.frame for k in bone_keys)
    f1 = max(k.frame for k in bone_keys)
    traj = []
    for f in range(f0, f1 + 1):
        world = evaluate_fk(profile.model, sample_local_poses(profile.model, tracks, f))
        traj.append(extract_markers(profile, [world]).markers[marker][0])
    return traj


def test_empty_input_returns_empty():
    assert apply_pose_denoise([], pmx_path=None) == []


def test_only_input_model_bones_are_densified_over_whole_frame_range():
    keys = [
        bone("センター", 2, pos=(0.2, 0.0, 0.0)),
        bone("センター", 5, pos=(0.5, 0.0, 0.0)),
        bone("頭", 0),
        bone("頭", 10, rot=(0.0, 0.0, 0.05, 0.99875)),
    ]
    whole_frame_range = list(range(0, 11))
    out = apply_pose_denoise(keys, pmx_path=None)
    assert {k.name for k in out} == {"センター", "頭"}
    center_frames = sorted(k.frame for k in out if k.name == "センター")
    assert center_frames == whole_frame_range
    head_frames = sorted(k.frame for k in out if k.name == "頭")
    assert head_frames == whole_frame_range


def test_dense_keys_use_linear_interpolation():
    keys = [bone("センター", 0), bone("センター", 6, pos=(0.5, 0.0, 0.0))]
    out = apply_pose_denoise(keys, pmx_path=None)
    center = [k for k in out if k.name == "センター"]
    assert center
    for k in center:
        assert k.interpolation == BONE_LINEAR_INTERP


def test_non_model_bone_key_passes_through_identical():
    thumb_key = bone(
        "左親指１", 0, pos=(0.2, 0.1, 0.0), rot=(0.0, 0.0, 0.1, 0.995), interp=BONE_NONLINEAR
    )
    keys = [bone("センター", 0), bone("センター", 5, pos=(0.3, 0.0, 0.0)), thumb_key]
    out = apply_pose_denoise(keys, pmx_path=None)
    thumb = [k for k in out if k.name == "左親指１"]
    assert thumb == [thumb_key]


def test_static_motion_is_preserved():
    keys = [bone("センター", 0), bone("センター", 8)]
    out = apply_pose_denoise(keys, pmx_path=None)
    for k in out:
        if k.name == "センター":
            assert k.position == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
            assert k.rotation == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-6)


def test_linear_center_motion_keeps_head_marker_trajectory():
    profile = load_mocap_profile(None)
    keys = [bone("センター", f, pos=(0.5 * f / 20.0, 0.0, 0.0)) for f in range(21)]
    before = _marker_trajectory_via_fk(profile, keys, "head")
    after = _marker_trajectory_via_fk(profile, apply_pose_denoise(keys, pmx_path=None), "head")
    min_head_travel = 0.1
    assert math.dist(before[0], before[-1]) > min_head_travel
    for a, b in zip(before, after, strict=True):
        assert a == pytest.approx(b, abs=1e-6)


def test_output_keys_are_valid_bonekeys():
    keys = [bone("頭", 0), bone("頭", 4, rot=(0.0, 0.0, 0.05, 0.9987))]
    out = apply_pose_denoise(keys, pmx_path=None)
    assert out
    for k in out:
        assert isinstance(k, BoneKey)
        assert isinstance(k.frame, int)
        assert len(k.position) == 3
        assert len(k.rotation) == 4
        assert isinstance(k.interpolation, bytes)
        assert len(k.interpolation) == 64


def test_diagnostics_out_populated_without_worsening_marker_error():
    keys = [
        bone("センター", 0, pos=(0.2, 0.0, 0.0)),
        bone("センター", 5, pos=(0.5, 0.0, 0.0)),
        bone("頭", 0),
        bone("頭", 10, rot=(0.0, 0.0, 0.05, 0.99875)),
    ]
    whole_frame_count = 11
    diag = {}
    out = apply_pose_denoise(keys, pmx_path=None, diagnostics_out=diag)
    assert out
    assert diag["enabled"] is True
    assert diag["pmx"] is None
    assert diag["frames"] == whole_frame_count
    assert diag["markers"]["available"] > 0
    assert diag["markers"]["required_bones_ok"] is True
    md = diag["marker_displacement"]
    assert md["max"] >= md["mean"] >= 0.0
    assert isinstance(md["by_marker"], dict) and md["by_marker"]
    for stats in md["by_marker"].values():
        assert stats["max"] >= stats["mean"] >= 0.0
    fit = diag["fit"]
    assert fit["frames"] == whole_frame_count
    assert fit["fallback_frames"] >= 0
    assert fit["mean_error_after"] <= fit["mean_error_before"] + 1e-9
    assert fit["max_bone_delta_deg"] >= 0.0
    assert fit["max_center_delta"] >= 0.0


def test_diagnostics_records_pmx_path(tmp_path):
    pmx = tmp_path / "model.pmx"
    pmx.write_bytes(build_standard_pmx(list(STANDARD_BONE_NAMES.values())))
    keys = [bone("センター", 0), bone("センター", 5, pos=(0.3, 0.0, 0.0))]
    diag = {}
    apply_pose_denoise(keys, pmx_path=str(pmx), diagnostics_out=diag)
    assert diag["pmx"] == str(pmx)


def test_no_diagnostics_arg_returns_list():
    out = apply_pose_denoise([bone("頭", 0), bone("頭", 4, rot=(0.0, 0.0, 0.05, 0.9987))])
    assert isinstance(out, list)
