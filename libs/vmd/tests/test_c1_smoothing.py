import math

from vmd import reduce as reducer
from vmd.reduce import (
    BONE_LINEAR_INTERP,
    build_bone_tolerances,
    reduce_bone_track,
    verify_bone_track,
)
from vmd.types import BoneKey

NAME = b"bone".ljust(15, b"\x00")


def _sine_position_src(n=41, amp=5.0, w=0.4, rot_deg_per_frame=0.0):
    keys = []
    for f in range(n):
        half = math.radians(rot_deg_per_frame * f) / 2.0
        rot = (0.0, 0.0, math.sin(half), math.cos(half))
        keys.append(BoneKey(NAME, f, (amp * math.sin(w * f), 0.0, 0.0), rot, BONE_LINEAR_INTERP))
    return keys


def _reduce(src, tols, *, strict, c1, monkeypatch, curve_mode="bezier"):
    monkeypatch.setattr(reducer, "_C1_SMOOTHING", c1)
    return reduce_bone_track(
        src,
        [(0, src[-1].frame)],
        tols,
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=strict,
        curve_mode=curve_mode,
    )


def test_c1_keeps_key_frames_and_changes_some_interior_interpolation(monkeypatch):
    src = _sine_position_src()
    tols = build_bone_tolerances(bone_pos=0.5, bone_rot=5.0)
    off = _reduce(src, tols, strict=False, c1=False, monkeypatch=monkeypatch)
    on = _reduce(src, tols, strict=False, c1=True, monkeypatch=monkeypatch)
    assert [k.frame for k in off] == [k.frame for k in on]
    assert any(off[i].interpolation != on[i].interpolation for i in range(1, len(on) - 1))


def test_c1_leaves_rotation_control_points_unchanged(monkeypatch):
    src = _sine_position_src(rot_deg_per_frame=2.0)
    tols = build_bone_tolerances(bone_pos=0.5, bone_rot=5.0)
    base = _reduce(src, tols, strict=False, c1=False, monkeypatch=monkeypatch)
    smoothed = reducer._apply_c1_bone(base, src)
    assert [k.control_points()["R"] for k in smoothed] == [k.control_points()["R"] for k in base]


def test_c1_not_applied_in_linear_mode(monkeypatch):
    src = _sine_position_src()
    tols = build_bone_tolerances(bone_pos=0.5, bone_rot=5.0)
    keys = _reduce(src, tols, strict=False, c1=True, monkeypatch=monkeypatch, curve_mode="linear")
    assert all(k.interpolation == BONE_LINEAR_INTERP for k in keys)


def test_c1_output_within_tolerance_nonstrict(monkeypatch):
    src = _sine_position_src()
    tols = build_bone_tolerances(bone_pos=0.5, bone_rot=5.0)
    keys = _reduce(src, tols, strict=False, c1=True, monkeypatch=monkeypatch)
    assert verify_bone_track(src, keys, [(0, src[-1].frame)], tols) == []


def test_c1_pos_control_points_within_0_127_and_x_monotonic(monkeypatch):
    src = _sine_position_src()
    tols = build_bone_tolerances(bone_pos=0.5, bone_rot=5.0)
    keys = _reduce(src, tols, strict=False, c1=True, monkeypatch=monkeypatch)
    for k in keys:
        cps = k.control_points()
        for ch in ("X", "Y", "Z"):
            x1, y1, x2, y2 = cps[ch]
            assert 0 <= x1 <= 127 and 0 <= y1 <= 127
            assert 0 <= x2 <= 127 and 0 <= y2 <= 127
            assert x1 <= x2


def test_c1_strict_keeps_original_fit_when_c1_breaks_tolerance(monkeypatch):
    src = _sine_position_src()
    rng = [(0, src[-1].frame)]
    tols = build_bone_tolerances(bone_pos=0.2, bone_rot=5.0)
    base = _reduce(src, tols, strict=True, c1=False, monkeypatch=monkeypatch)
    assert verify_bone_track(src, base, rng, tols) == []
    candidate = reducer._apply_c1_bone(base, src)
    assert verify_bone_track(src, candidate, rng, tols) != []
    keys = _reduce(src, tols, strict=True, c1=True, monkeypatch=monkeypatch)
    assert verify_bone_track(src, keys, rng, tols) == []
    assert [k.frame for k in keys] == [k.frame for k in base]
    assert [k.interpolation for k in keys] == [k.interpolation for k in base]
