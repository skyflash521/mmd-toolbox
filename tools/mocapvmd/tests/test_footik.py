import pytest

from mocapvmd import footik

_MIN_GROUND_LEN = 4
_HORIZ_STEP_JUST_BELOW_THRESH = 0.079
_HORIZ_STEP_JUST_ABOVE_THRESH = 0.081
_VERT_STEP_AT_THRESH = 0.04
_VERT_STEP_JUST_ABOVE_THRESH = 0.041
_FAST_STEP = 0.5


def detect(positions, **kw):
    return footik.detect_grounding_segments(positions, **kw)


def segs(det):
    return [(s.start, s.end) for s in det.segments]


def _x_ramp(step, n=8):
    return [(round(step * i, 6), 0.0, 0.0) for i in range(n)]


def _xz_diagonal_ramp(step_per_axis, n=8):
    return [(round(step_per_axis * i, 6), 0.0, round(step_per_axis * i, 6)) for i in range(n)]


def _y_alternating(dy, n=8):
    return [(0.0, dy if i % 2 else 0.0, 0.0) for i in range(n)]


def _still_block_between_fast_steps(block):
    return (
        [(0.0, 0.0, 0.0), (_FAST_STEP, 0.0, 0.0)]
        + [(1.0, 0.0, 0.0)] * block
        + [(1.5, 0.0, 0.0), (2.0, 0.0, 0.0)]
    )


def test_empty_track_has_no_segments_or_candidates():
    det = detect([])
    assert det.segments == ()
    assert det.candidate_frames == frozenset()


@pytest.mark.parametrize("n", range(1, _MIN_GROUND_LEN))
def test_still_track_shorter_than_min_ground_len_has_no_segment(n):
    det = detect([(0.0, 0.0, 0.0)] * n)
    assert det.segments == ()


def test_full_still_low_track_is_one_segment():
    det = detect([(0.0, 0.0, 0.0)] * 8)
    assert segs(det) == [(0, 7)]
    assert det.candidate_frames == frozenset(range(8))


def test_still_block_of_min_ground_len_is_segment_and_shorter_keeps_only_candidates():
    det3 = detect(_still_block_between_fast_steps(_MIN_GROUND_LEN - 1))
    assert det3.segments == ()
    assert det3.candidate_frames == frozenset({2, 3, 4})
    assert segs(detect(_still_block_between_fast_steps(_MIN_GROUND_LEN))) == [(2, 5)]


def test_horizontal_velocity_below_threshold_is_grounded():
    det = detect(_x_ramp(_HORIZ_STEP_JUST_BELOW_THRESH))
    assert det.candidate_frames == frozenset(range(8))
    assert segs(det) == [(0, 7)]


def test_horizontal_velocity_above_threshold_is_not_grounded():
    det = detect(_x_ramp(_HORIZ_STEP_JUST_ABOVE_THRESH))
    assert det.candidate_frames == frozenset()
    assert det.segments == ()


def test_horizontal_velocity_is_euclidean_not_per_axis():
    g = detect(_xz_diagonal_ramp(0.05))
    assert g.candidate_frames == frozenset(range(8))
    assert segs(g) == [(0, 7)]
    ng = detect(_xz_diagonal_ramp(0.06))
    assert ng.candidate_frames == frozenset()
    assert ng.segments == ()


def test_vertical_velocity_at_threshold_is_grounded_and_just_above_is_not():
    g = detect(_y_alternating(_VERT_STEP_AT_THRESH))
    assert g.candidate_frames == frozenset(range(8))
    assert segs(g) == [(0, 7)]
    ng = detect(_y_alternating(_VERT_STEP_JUST_ABOVE_THRESH))
    assert ng.candidate_frames == frozenset()
    assert ng.segments == ()


def test_y_tolerance_boundary_is_local_min_plus_0_08():
    ys = [0.081, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.05, 0.079]
    det = detect([(0.0, y, 0.0) for y in ys])
    assert det.candidate_frames == frozenset(range(1, 11))
    assert segs(det) == [(1, 10)]


def test_y_local_window_radius_is_5():
    valley_frame = 7
    ys = [0.20] * 15
    ys[valley_frame] = 0.00
    det = detect([(0.0, y, 0.0) for y in ys])
    assert det.candidate_frames == frozenset({0, 1, 13, 14})
    assert det.segments == ()


def test_stance_swing_stance_splits_into_two_segments():
    stance1 = [(0.0, 0.0, 0.0)] * 6
    swing = [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.4, 0.0, 0.0), (1.7, 0.0, 0.0)]
    stance2 = [(2.0, 0.0, 0.0)] * 6
    det = detect(stance1 + swing + stance2)
    assert segs(det) == [(0, 5), (10, 15)]
