import math

import pytest

from mocapvmd import footik
from mocapvmd.footik import GroundSegment

_XZ_CENTER = 0.90
_Y_CENTER = 0.50
_FIXED_LOCK_STRENGTH = {
    "xz_center": _XZ_CENTER,
    "xz_edge": 0.25,
    "y_center": _Y_CENTER,
    "y_edge": 0.10,
    "fade_width": 3,
}
_MAX_CORRECTION = 0.5
_OFFSET = 0.2


def _track(xs, ys=None, zs=None):
    n = len(xs)
    ys = ys if ys is not None else [0.0] * n
    zs = zs if zs is not None else [0.0] * n
    return [(xs[i], ys[i], zs[i]) for i in range(n)]


def _lock(positions, segments):
    return footik.apply_foot_lock(positions, segments, _FIXED_LOCK_STRENGTH)


def test_anchor_is_median_robust_to_outlier():
    pos = _track([1.0, 1.0, 5.0, 1.0, 1.0])
    a = footik.compute_ground_anchor(pos, GroundSegment(0, 4))
    assert a == pytest.approx((1.0, 0.0, 0.0))


def test_anchor_per_axis_median():
    pos = [(0.0, 10.0, 100.0), (2.0, 12.0, 100.0), (4.0, 14.0, 130.0)]
    a = footik.compute_ground_anchor(pos, GroundSegment(0, 2))
    assert a == pytest.approx((2.0, 12.0, 100.0))


@pytest.mark.parametrize("offset,new_x", [
    pytest.param(0, 0.15, id="start_edge"),
    pytest.param(1, 0.10666667, id="start_edge_dist1"),
    pytest.param(2, 0.06333333, id="start_edge_dist2"),
    pytest.param(3, 0.02, id="center"),
    pytest.param(4, 0.06333333, id="end_edge_dist2"),
    pytest.param(5, 0.10666667, id="end_edge_dist1"),
    pytest.param(6, 0.15, id="end_edge"),
])
def test_fade_curve_from_edge_to_center_is_symmetric(offset, new_x):
    xs = [0.0] * 7
    xs[offset] = _OFFSET
    locked, _ = _lock(_track(xs), [GroundSegment(0, 6)])
    assert locked[offset][0] == pytest.approx(new_x)


@pytest.mark.parametrize("offset", [
    pytest.param(1, id="start_edge_dist1"),
    pytest.param(2, id="end_edge_dist1"),
])
def test_fade_width_halves_segment_len_for_short_segment(offset):
    xs = [0.0, 0.0, 0.0, 0.0]
    xs[offset] = _OFFSET
    locked, _ = _lock(_track(xs), [GroundSegment(0, 3)])
    assert locked[offset][0] == pytest.approx(0.085)


def test_xz_and_y_use_separate_coefficients():
    center = 3
    xs = [0.0] * 7
    ys = [0.0] * 7
    zs = [0.0] * 7
    xs[center] = _OFFSET
    ys[center] = _OFFSET
    zs[center] = _OFFSET
    locked, _ = _lock(_track(xs, ys, zs), [GroundSegment(0, 6)])
    assert locked[center][0] == pytest.approx(0.02)
    assert locked[center][1] == pytest.approx(0.10)
    assert locked[center][2] == pytest.approx(0.02)


def test_frames_outside_segments_unchanged():
    xs = [9.0, 9.0, 9.0, 0.0, 0.5, 0.0, 0.0, 7.0, 7.0, 7.0]
    pos = _track(xs)
    locked, _ = _lock(pos, [GroundSegment(3, 6)])
    assert locked[0] == pos[0] and locked[1] == pos[1] and locked[2] == pos[2]
    assert locked[7] == pos[7] and locked[8] == pos[8] and locked[9] == pos[9]


def test_no_clamp_when_within_limit():
    xs = [0.0] * 7
    xs[3] = _OFFSET
    locked, locks = _lock(_track(xs), [GroundSegment(0, 6)])
    assert len(locks) == 1
    assert locks[0].clamped is False
    assert locks[0].coef_scale == pytest.approx(1.0)
    assert locks[0].max_displacement == pytest.approx(0.18)


def test_clamp_is_3d_euclidean_and_uniform_across_segment():
    violating = 0.6
    other_inner = 0.3
    xs = [0.0] * 9
    ys = [0.0] * 9
    zs = [0.0] * 9
    xs[3], ys[3], zs[3] = violating, violating, violating
    xs[4] = other_inner
    locked, locks = _lock(_track(xs, ys, zs), [GroundSegment(0, 8)])
    raw_max = math.hypot(_XZ_CENTER * violating, _Y_CENTER * violating, _XZ_CENTER * violating)
    scale = _MAX_CORRECTION / raw_max
    assert locks[0].clamped is True
    assert locks[0].coef_scale == pytest.approx(scale)
    assert locks[0].max_displacement == pytest.approx(_MAX_CORRECTION)
    assert locked[3][0] == pytest.approx(violating * (1 - _XZ_CENTER * scale))
    assert locked[3][1] == pytest.approx(violating * (1 - _Y_CENTER * scale))
    assert locked[3][2] == pytest.approx(violating * (1 - _XZ_CENTER * scale))
    assert locked[4][0] == pytest.approx(other_inner * (1 - _XZ_CENTER * scale))


def test_segment_anchor_is_per_segment():
    xs = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]
    _, locks = _lock(_track(xs), [GroundSegment(0, 3), GroundSegment(4, 7)])
    assert [lk.segment for lk in locks] == [GroundSegment(0, 3), GroundSegment(4, 7)]
    assert locks[0].anchor == pytest.approx((0.0, 0.0, 0.0))
    assert locks[1].anchor == pytest.approx((5.0, 0.0, 0.0))


def test_clamp_is_independent_per_segment():
    clamped_inner = 3
    unclamped_inner = 10
    xs = [0.0] * 14
    xs[clamped_inner] = 1.0
    xs[unclamped_inner] = _OFFSET
    locked, locks = _lock(_track(xs), [GroundSegment(0, 6), GroundSegment(7, 13)])
    assert locks[0].clamped is True
    assert locks[0].coef_scale == pytest.approx(_MAX_CORRECTION / _XZ_CENTER)
    assert locks[1].clamped is False
    assert locks[1].coef_scale == pytest.approx(1.0)
    assert locked[unclamped_inner][0] == pytest.approx(_OFFSET * (1 - _XZ_CENTER))
