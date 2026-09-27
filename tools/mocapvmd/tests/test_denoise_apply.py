import math

import pytest

from mocapvmd import denoise

IDENT = (0.0, 0.0, 0.0, 1.0)

_N = 11
_POS_CLAMP_PER_AXIS = 0.3
_ROT_CLAMP_DEG = 5.0
_ZERO_POS = [(0.0, 0.0, 0.0)] * _N
_MICRO_JITTER_X = [0.0, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 0.0]


def quat_y(deg):
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def _alternating_2deg_rotations():
    return [quat_y(2.0 * ((-1) ** i)) for i in range(_N)]


def apply(positions, rotations, pos_window=5, rot_window=5, pos_strength=0.5, rot_strength=0.5):
    return denoise.apply_denoise(
        positions,
        rotations,
        pos_window=pos_window,
        rot_window=rot_window,
        pos_strength=pos_strength,
        rot_strength=rot_strength,
    )


def _norm(q):
    return math.sqrt(sum(c * c for c in q))


def _unit(q):
    n = _norm(q)
    return tuple(c / n for c in q)


def _quat_angle_deg(a, b):
    a, b = _unit(a), _unit(b)
    d = abs(sum(x * y for x, y in zip(a, b, strict=True)))
    d = min(1.0, d)
    return math.degrees(2.0 * math.acos(d))


def _assert_unit_quaternions(quats):
    for q in quats:
        assert _norm(q) == pytest.approx(1.0, abs=1e-6)


def _x_variation(positions):
    xs = [p[0] for p in positions]
    return sum(abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1))


def test_zero_strength_is_identity():
    xs = [0.0, 0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0.0, 0.05, -0.05, 0.0]
    pos = [(x, 0.0, 0.0) for x in xs]
    rots = _alternating_2deg_rotations()
    out_pos, out_rot = apply(pos, rots, pos_strength=0.0, rot_strength=0.0)
    for o, i in zip(out_pos, pos, strict=True):
        assert o == pytest.approx(i)
    for o, i in zip(out_rot, rots, strict=True):
        assert _quat_angle_deg(o, i) == pytest.approx(0.0, abs=1e-6)
    _assert_unit_quaternions(out_rot)


def test_position_micro_jitter_total_variation_reduced():
    pos = [(x, 0.0, 0.0) for x in _MICRO_JITTER_X]
    out_pos, _ = apply(pos, [IDENT] * _N, pos_strength=1.0)
    assert _x_variation(out_pos) < _x_variation(pos)


def test_output_lengths_match_input():
    pos = [(0.1 * i, 0.0, 0.0) for i in range(_N)]
    out_pos, out_rot = apply(pos, [IDENT] * _N)
    assert len(out_pos) == _N
    assert len(out_rot) == _N


def test_position_spike_suppressed_below_half():
    spike = 0.5
    pos = list(_ZERO_POS)
    pos[5] = (spike, 0.0, 0.0)
    out_pos, _ = apply(pos, [IDENT] * _N, pos_strength=1.0)
    assert abs(out_pos[5][0]) < spike / 2


def test_position_accent_run_preserved():
    xs = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0]
    pos = [(x, 0.0, 0.0) for x in xs]
    out_pos, _ = apply(pos, [IDENT] * _N, pos_strength=1.0)
    for f in (3, 4, 5, 6, 7):
        assert out_pos[f][0] == pytest.approx(xs[f])


def test_position_range_edges_preserved():
    xs = _MICRO_JITTER_X
    pos = [(x, 0.0, 0.0) for x in xs]
    out_pos, _ = apply(pos, [IDENT] * _N, pos_strength=1.0)
    assert out_pos[0][0] == pytest.approx(xs[0])
    assert out_pos[10][0] == pytest.approx(xs[10])


def test_position_cut_adjacent_frames_preserved():
    xs = [0.0] * 6 + [2.0] * 5
    pos = [(x, 0.0, 0.0) for x in xs]
    out_pos, _ = apply(pos, [IDENT] * _N, pos_strength=1.0)
    assert out_pos[5][0] == pytest.approx(0.0)
    assert out_pos[6][0] == pytest.approx(2.0)


def test_position_change_clamped_per_axis_and_reaches_limit():
    alternating_below_cut = [0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0]
    pos = [(v, v, 0.0) for v in alternating_below_cut]
    out_pos, _ = apply(pos, [IDENT] * _N, pos_strength=1.0)
    dx = [abs(o[0] - i[0]) for o, i in zip(out_pos, pos, strict=True)]
    dy = [abs(o[1] - i[1]) for o, i in zip(out_pos, pos, strict=True)]
    assert max(dx) <= _POS_CLAMP_PER_AXIS + 1e-9
    assert max(dy) <= _POS_CLAMP_PER_AXIS + 1e-9
    assert max(dx) == pytest.approx(_POS_CLAMP_PER_AXIS, abs=1e-6)
    assert max(dy) == pytest.approx(_POS_CLAMP_PER_AXIS, abs=1e-6)


def test_rotation_change_clamped_and_reaches_limit():
    alternating_below_cut_deg = [0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0]
    rots = [quat_y(d) for d in alternating_below_cut_deg]
    _, out_rot = apply(_ZERO_POS, rots, rot_strength=1.0)
    changes = [_quat_angle_deg(o, i) for o, i in zip(out_rot, rots, strict=True)]
    assert max(changes) <= _ROT_CLAMP_DEG + 1e-6
    assert max(changes) == pytest.approx(_ROT_CLAMP_DEG, abs=1e-3)
    _assert_unit_quaternions(out_rot)


def test_rotation_jitter_adjacent_angle_sum_reduced():
    rots = _alternating_2deg_rotations()
    _, out_rot = apply(_ZERO_POS, rots, rot_strength=1.0)
    before = sum(_quat_angle_deg(rots[i], rots[i + 1]) for i in range(10))
    after = sum(_quat_angle_deg(out_rot[i], out_rot[i + 1]) for i in range(10))
    assert after < before
    _assert_unit_quaternions(out_rot)


def test_rotation_accent_and_edges_preserved():
    degs = [0.0, 0.0, 0.0, 0.0, 8.0, 16.0, 24.0, 32.0, 32.0, 32.0, 32.0]
    rots = [quat_y(d) for d in degs]
    _, out_rot = apply(_ZERO_POS, rots, rot_strength=1.0)
    for f in (3, 4, 5, 6, 7):
        assert _quat_angle_deg(out_rot[f], rots[f]) == pytest.approx(0.0, abs=1e-6)
    assert _quat_angle_deg(out_rot[0], rots[0]) == pytest.approx(0.0, abs=1e-6)
    assert _quat_angle_deg(out_rot[10], rots[10]) == pytest.approx(0.0, abs=1e-6)


def test_rotation_cut_adjacent_frames_preserved():
    degs = [0.0] * 6 + [40.0] * 5
    rots = [quat_y(d) for d in degs]
    _, out_rot = apply(_ZERO_POS, rots, rot_strength=1.0)
    assert _quat_angle_deg(out_rot[5], rots[5]) == pytest.approx(0.0, abs=1e-6)
    assert _quat_angle_deg(out_rot[6], rots[6]) == pytest.approx(0.0, abs=1e-6)


def test_rotation_zero_norm_input_raises():
    rots = [IDENT] * _N
    rots[5] = (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        apply(_ZERO_POS, rots)
