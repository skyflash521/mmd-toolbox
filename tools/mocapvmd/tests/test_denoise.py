import math

import pytest

from mocapvmd import denoise

IDENT = (0.0, 0.0, 0.0, 1.0)

_N = 11
_SPIKE_FRAME = 5
_POS_SPIKE_THRESHOLD = 0.3
_POS_CUT_JUMP = 2.0
_RAMP_STEPS = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0]
_RAMP_ACCENT_FRAMES = {3, 4, 5, 6, 7}


def quat_y(deg):
    """Y軸まわり deg 度回転の quaternion (x,y,z,w)。"""
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def still(n, pos=(0.0, 0.0, 0.0)):
    return [tuple(pos) for _ in range(n)]


def idents(n):
    return [IDENT] * n


def detect(positions, rotations, pos_window=5, rot_window=5):
    return denoise.detect_noise_events(
        positions, rotations, pos_window=pos_window, rot_window=rot_window
    )


def _axis(events, axis):
    return {f for (f, a) in events if a == axis}


def _x_step_at_frame6():
    return [(0.0, 0.0, 0.0)] * 6 + [(_POS_CUT_JUMP, 0.0, 0.0)] * 5


def test_empty_track_has_no_spikes():
    r = detect([], [])
    assert r.pos_spikes == set()
    assert r.rot_spikes == set()


@pytest.mark.parametrize("n", [1, 2])
def test_track_shorter_than_three_frames_has_no_candidates_or_spikes(n):
    r = detect(still(n), idents(n))
    assert r.pos_spikes == set()
    assert r.rot_spikes == set()
    assert r.pos_candidates == set()
    assert r.rot_candidates == set()


def test_all_same_value_no_events():
    r = detect(still(_N), idents(_N))
    assert r.pos_candidates == set()
    assert r.rot_candidates == set()
    assert r.pos_spikes == set()
    assert r.rot_spikes == set()
    assert r.pos_accent == set()
    assert r.rot_accent == set()


def test_position_single_frame_round_trip_is_spike_on_that_axis_only():
    pos = still(_N)
    pos[_SPIKE_FRAME] = (0.5, 0.0, 0.0)
    r = detect(pos, idents(_N))
    assert (_SPIKE_FRAME, 0) in r.pos_candidates
    assert (_SPIKE_FRAME, 0) in r.pos_spikes
    assert (_SPIKE_FRAME, 1) not in r.pos_spikes
    assert (_SPIKE_FRAME, 2) not in r.pos_spikes


def test_position_spike_threshold_exact_is_not_candidate():
    pos = still(_N)
    pos[_SPIKE_FRAME] = (_POS_SPIKE_THRESHOLD, 0.0, 0.0)
    r = detect(pos, idents(_N))
    assert (_SPIKE_FRAME, 0) not in r.pos_candidates
    assert (_SPIKE_FRAME, 0) not in r.pos_spikes


def test_position_spike_threshold_over_is_spike():
    pos = still(_N)
    pos[_SPIKE_FRAME] = (0.31, 0.0, 0.0)
    r = detect(pos, idents(_N))
    assert (_SPIKE_FRAME, 0) in r.pos_spikes


def test_position_jump_that_does_not_return_is_not_spike():
    pos = still(_N)
    pos[_SPIKE_FRAME] = (0.5, 0.0, 0.0)
    pos[_SPIKE_FRAME + 1] = (0.5, 0.0, 0.0)
    r = detect(pos, idents(_N))
    assert (_SPIKE_FRAME, 0) not in r.pos_spikes


def test_position_same_direction_velocity_run_is_accent_and_not_candidate():
    pos = [(x, 0.0, 0.0) for x in _RAMP_STEPS]
    r = detect(pos, idents(_N))
    assert _axis(r.pos_accent, 0) == _RAMP_ACCENT_FRAMES
    assert _axis(r.pos_candidates, 0) == set()
    assert _axis(r.pos_spikes, 0) == set()


def test_accent_priority_over_spike_at_run_peak():
    xs = [0.0, 0.0, 0.5, 1.0, 1.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    peak = 4
    rising_and_falling_run_frames = {1, 2, 3, 4, 5, 6, 7}
    pos = [(x, 0.0, 0.0) for x in xs]
    r = detect(pos, idents(_N))
    assert (peak, 0) in r.pos_candidates
    assert (peak, 0) in r.pos_accent
    assert (peak, 0) not in r.pos_spikes
    assert _axis(r.pos_accent, 0) == rising_and_falling_run_frames
    assert _axis(r.pos_spikes, 0) == set()


def test_axis_independence_x_spike_and_y_accent():
    pos = [(0.0, y, 0.0) for y in _RAMP_STEPS]
    pos[_SPIKE_FRAME] = (0.6, pos[_SPIKE_FRAME][1], 0.0)
    r = detect(pos, idents(_N))
    assert (_SPIKE_FRAME, 0) in r.pos_spikes
    assert (_SPIKE_FRAME, 0) not in r.pos_accent
    assert (_SPIKE_FRAME, 1) in r.pos_accent
    assert (_SPIKE_FRAME, 1) not in r.pos_spikes


def test_range_edges_are_boundaries():
    r = detect(still(_N), idents(_N))
    assert 0 in r.boundaries
    assert 10 in r.boundaries


def test_position_cut_marks_both_adjacent_frames_as_boundaries():
    r = detect(_x_step_at_frame6(), idents(_N))
    assert 5 in r.boundaries
    assert 6 in r.boundaries


def test_position_jump_at_cut_is_not_spike():
    r = detect(_x_step_at_frame6(), idents(_N))
    assert (6, 0) not in r.pos_spikes


def test_cuts_reported_separately_from_boundaries():
    r = detect(_x_step_at_frame6(), idents(_N))
    assert r.cuts == {6}


def test_rotation_single_frame_round_trip_is_spike():
    rots = idents(_N)
    rots[_SPIKE_FRAME] = quat_y(10.0)
    r = detect(still(_N), rots)
    assert _SPIKE_FRAME in r.rot_candidates
    assert _SPIKE_FRAME in r.rot_spikes


def test_rotation_jump_below_threshold_is_not_candidate():
    rots = idents(_N)
    rots[_SPIKE_FRAME] = quat_y(3.0)
    r = detect(still(_N), rots)
    assert _SPIKE_FRAME not in r.rot_candidates
    assert _SPIKE_FRAME not in r.rot_spikes


def test_rotation_quaternion_sign_flip_is_not_candidate():
    rots = idents(_N)
    rots[_SPIKE_FRAME] = (0.0, 0.0, 0.0, -1.0)
    r = detect(still(_N), rots)
    assert _SPIKE_FRAME not in r.rot_candidates
    assert _SPIKE_FRAME not in r.rot_spikes


def test_non_unit_quaternion_is_normalized_before_detection():
    scale = lambda q: tuple(2.0 * c for c in q)  # noqa: E731
    rots = [scale(IDENT)] * _N
    rots[_SPIKE_FRAME] = scale(quat_y(10.0))
    r = detect(still(_N), rots)
    assert _SPIKE_FRAME in r.rot_spikes


def test_rotation_same_direction_velocity_run_is_accent():
    degs = [0.0, 0.0, 0.0, 0.0, 8.0, 16.0, 24.0, 32.0, 32.0, 32.0, 32.0]
    rots = [quat_y(d) for d in degs]
    r = detect(still(_N), rots)
    assert r.rot_accent == _RAMP_ACCENT_FRAMES
    assert r.rot_spikes == set()


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        detect(still(5), idents(4))


def test_non_finite_position_raises():
    pos = still(_N)
    pos[_SPIKE_FRAME] = (float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError):
        detect(pos, idents(_N))


def test_infinite_position_raises():
    pos = still(_N)
    pos[_SPIKE_FRAME] = (float("inf"), 0.0, 0.0)
    with pytest.raises(ValueError):
        detect(pos, idents(_N))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_quaternion_raises(bad):
    rots = idents(_N)
    rots[_SPIKE_FRAME] = (bad, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        detect(still(_N), rots)


def test_zero_norm_quaternion_raises():
    rots = idents(_N)
    rots[_SPIKE_FRAME] = (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        detect(still(_N), rots)


@pytest.mark.parametrize("pw,rw", [(4, 5), (5, 4)])
def test_even_window_raises(pw, rw):
    with pytest.raises(ValueError):
        denoise.detect_noise_events(still(_N), idents(_N), pos_window=pw, rot_window=rw)


@pytest.mark.parametrize(
    "pw,rw",
    [
        pytest.param(-3, 5, id="negative_odd_pos_window"),
        pytest.param(5, -3, id="negative_odd_rot_window"),
        pytest.param(0, 5, id="zero_pos_window"),
    ],
)
def test_non_positive_window_raises(pw, rw):
    with pytest.raises(ValueError):
        denoise.detect_noise_events(still(_N), idents(_N), pos_window=pw, rot_window=rw)
