import math

import pytest

from mocapvmd import footik, presets

_DEFAULT_SUPPRESSION = 1.0
_DET = presets.resolve_foot_detection(_DEFAULT_SUPPRESSION)
_SWING_STEP_ABOVE_HORIZ_VEL_THRESH = 1.5


def _detect(positions, paired_positions=None):
    return footik.detect_grounding_segments(
        positions, paired_positions=paired_positions, horiz_vel_thresh=_DET["horiz_vel_thresh"]
    )


def _lock(positions, segments, category):
    return footik.apply_foot_lock(
        positions, segments, presets.resolve_foot_lock(_DEFAULT_SUPPRESSION, category),
        max_displacement=_DET["max_displacement"],
    )


def _stabilize(tracks):
    return footik.stabilize_foot_ik(tracks, _DEFAULT_SUPPRESSION)


def _drifting_stance_then_swing(n_still=8, n_move=4):
    drift = [0.0, 0.03, 0.06, 0.03, 0.0, 0.03, 0.06, 0.03][:n_still]
    base = drift[-1] if drift else 0.0
    xs = drift + [base + _SWING_STEP_ABOVE_HORIZ_VEL_THRESH * (i + 1) for i in range(n_move)]
    return [(x, 0.0, 0.0) for x in xs]


def _euclid(a, b):
    return math.dist(a, b)


def test_unpaired_track_matches_building_blocks():
    pos = _drifting_stance_then_swing()
    frames = list(range(len(pos)))
    result = _stabilize({"右足ＩＫ": ("foot_ik", frames, pos)})
    ts = result["右足ＩＫ"]

    det = _detect(pos)
    locked, locks = _lock(pos, det.segments, "foot_ik")

    assert ts.category == "foot_ik"
    assert ts.side == "right"
    assert ts.paired is False
    assert ts.grounding == det
    assert tuple(ts.locked_positions) == tuple(locked)
    assert tuple(ts.locks) == tuple(locks)


def test_paired_foot_and_toe_use_each_other_as_relative_reference():
    foot = [(0.0, 0.0, 0.0)] * 12
    relative_spike_frame = 6
    toe_x = [1.0, 1.03, 1.06, 1.03, 1.0, 1.03, 1.3, 1.0, 1.03, 1.06, 1.03, 1.0]
    toe = [(x, 0.0, 0.0) for x in toe_x]
    frames = list(range(12))
    result = _stabilize(
        {"右足ＩＫ": ("foot_ik", frames, foot), "右つま先ＩＫ": ("toe_ik", frames, toe)},
    )
    foot_ts = result["右足ＩＫ"]
    expected_foot = _detect(foot, paired_positions=toe)
    assert foot_ts.paired is True
    assert foot_ts.side == "right"
    assert foot_ts.grounding == expected_foot
    assert relative_spike_frame in foot_ts.grounding.relative_rejected_frames

    toe_ts = result["右つま先ＩＫ"]
    assert toe_ts.category == "toe_ik"
    assert toe_ts.paired is True
    assert toe_ts.side == "right"
    expected_toe = _detect(toe, paired_positions=foot)
    assert toe_ts.grounding == expected_toe
    toe_locked, _ = _lock(toe, expected_toe.segments, "toe_ik")
    assert tuple(toe_ts.locked_positions) == tuple(toe_locked)


def test_paired_alignment_uses_absolute_frames_both_directions():
    foot = [(x, 0.0, 0.0) for x in [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]]
    foot_frames = [0, 1, 2, 3, 4, 5]
    toe = [(x, 0.0, 0.0) for x in [1.5, 1.0, 1.0, 1.0, 1.0, 1.0]]
    toe_frames = [2, 3, 4, 5, 6, 7]
    result = _stabilize(
        {"右足ＩＫ": ("foot_ik", foot_frames, foot),
         "右つま先ＩＫ": ("toe_ik", toe_frames, toe)},
    )
    toe_aligned_to_foot_frames = [
        None, None, (1.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    ]
    assert result["右足ＩＫ"].grounding == _detect(foot, paired_positions=toe_aligned_to_foot_frames)
    foot_aligned_to_toe_frames = [
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), None, None,
    ]
    assert result["右つま先ＩＫ"].grounding == _detect(toe, paired_positions=foot_aligned_to_toe_frames)


def test_ambiguous_tracks_processed_without_pairing_or_relative_rejection():
    foot1 = _drifting_stance_then_swing()
    foot2 = _drifting_stance_then_swing()
    toe = [(1.0, 0.0, 0.0)] * len(foot1)
    toe[3] = (1.5, 0.0, 0.0)
    frames = list(range(len(foot1)))
    result = _stabilize({
        "右足ＩＫ": ("foot_ik", frames, foot1),
        "右足ＩＫ先": ("foot_ik", frames, foot2),
        "右つま先ＩＫ": ("toe_ik", frames, toe),
    })
    for name in ("右足ＩＫ", "右足ＩＫ先", "右つま先ＩＫ"):
        assert result[name].paired is False
    assert result["右足ＩＫ"].grounding == _detect(foot1)
    assert result["右足ＩＫ"].grounding.relative_rejected_frames == frozenset()


def test_diagnostics_change_and_ratio():
    pos = _drifting_stance_then_swing()
    frames = list(range(len(pos)))
    ts = _stabilize({"右足ＩＫ": ("foot_ik", frames, pos)})["右足ＩＫ"]

    changes = [_euclid(p, q) for p, q in zip(pos, ts.locked_positions, strict=True)]
    in_seg = sum(s.end - s.start + 1 for s in ts.grounding.segments)
    assert ts.max_change == pytest.approx(max(changes))
    assert ts.mean_change == pytest.approx(sum(changes) / len(changes))
    assert ts.lock_applied_ratio == pytest.approx(in_seg / len(pos))


def test_warning_lists_clamped_segments():
    long_sliding_stance = [(round(0.3 * i, 6), 0.0, 0.0) for i in range(24)]
    frames = list(range(24))
    ts = _stabilize({"右足ＩＫ": ("foot_ik", frames, long_sliding_stance)})["右足ＩＫ"]
    clamped = tuple(s for s in ts.locks if s.clamped)
    assert len(clamped) >= 1
    assert tuple(ts.warnings) == clamped


def test_non_grounding_track_is_unchanged():
    pos = [(_SWING_STEP_ABOVE_HORIZ_VEL_THRESH * i, 0.0, 0.0) for i in range(10)]
    frames = list(range(10))
    ts = _stabilize({"右足ＩＫ": ("foot_ik", frames, pos)})["右足ＩＫ"]
    assert ts.grounding.segments == ()
    assert tuple(ts.locked_positions) == tuple(pos)
    assert ts.lock_applied_ratio == pytest.approx(0.0)
    assert ts.max_change == pytest.approx(0.0)
