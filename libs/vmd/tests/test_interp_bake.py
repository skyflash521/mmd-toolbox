import numpy as np
import pytest

from vmd import interp
from vmd.types import BoneKey

LINEAR = (20, 20, 107, 107)


def bone_interp(x=LINEAR, y=LINEAR, z=LINEAR, r=LINEAR) -> bytes:
    b = bytearray(64)
    b[0], b[4], b[8], b[12] = x
    b[1], b[5], b[9], b[13] = y
    b[17], b[6], b[10], b[14] = z
    b[18], b[7], b[11], b[15] = r
    return bytes(b)


def bkey(frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), curves=None):
    return BoneKey(b"\x00" * 15, frame, tuple(pos), tuple(rot),
                   bone_interp(**(curves or {})))


def assert_matches_sample(keys, frame_start, frame_end, abs_tol=1e-8):
    positions, rotations = interp.bake_bone_track(keys, frame_start, frame_end)
    n = frame_end - frame_start + 1
    assert len(positions) == n
    assert len(rotations) == n
    for i, f in enumerate(range(frame_start, frame_end + 1)):
        expected_pos = (interp.sample(keys, "pos_x", f),
                        interp.sample(keys, "pos_y", f),
                        interp.sample(keys, "pos_z", f))
        assert tuple(positions[i]) == pytest.approx(expected_pos, abs=abs_tol), f"pos at {f}"
        expected_rot = np.array(interp.sample(keys, "rot", f), dtype=float)
        got_rot = np.array(rotations[i], dtype=float)
        if float(np.dot(got_rot, expected_rot)) < 0:
            expected_rot = -expected_rot
        assert got_rot == pytest.approx(expected_rot, abs=abs_tol), f"rot at {f}"


class TestDenseIdentity:
    def test_positions_exact_on_dense_keys(self):
        keys = [bkey(f, pos=(f * 1.5, -float(f), f * f * 0.25)) for f in range(6)]
        positions, _ = interp.bake_bone_track(keys, 0, 5)
        assert len(positions) == 6
        for f in range(6):
            assert tuple(positions[f]) == keys[f].position

    def test_rotations_exact_with_unit_norm_data(self):
        quats = [(0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 0.5),
                 (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)]
        keys = [bkey(f, rot=quats[f]) for f in range(4)]
        _, rotations = interp.bake_bone_track(keys, 0, 3)
        for f in range(4):
            assert tuple(rotations[f]) == quats[f]

    def test_rotations_normalized_for_non_unit_data(self):
        keys = [bkey(0, rot=(0.0, 0.0, 0.0, 2.0)), bkey(1, rot=(0.0, 0.0, 2.0, 0.0))]
        _, rotations = interp.bake_bone_track(keys, 0, 1)
        for f in (0, 1):
            assert np.linalg.norm(rotations[f]) == pytest.approx(1.0, abs=1e-12)
        assert_matches_sample(keys, 0, 1, abs_tol=1e-12)


MIXED_GAP_FRAMES = [0, 1, 3, 13, 73, 75]


def _mixed_gap_keys(curves_list):
    keys = []
    for i, f in enumerate(MIXED_GAP_FRAMES):
        c = curves_list[i % len(curves_list)]
        q = (0.0, 0.0, np.sin(0.2 * i), np.cos(0.2 * i))
        keys.append(bkey(f, pos=(f * 0.1, 5.0 - i, (i % 3) * 2.0), rot=q, curves=c))
    return keys


class TestUnequalGaps:
    def test_all_frames_match_sample_linear(self):
        keys = _mixed_gap_keys([None])
        assert_matches_sample(keys, MIXED_GAP_FRAMES[0], MIXED_GAP_FRAMES[-1])

    def test_all_frames_match_sample_with_distinct_nonlinear_curve_per_channel(self):
        curves = [
            {"x": (0, 127, 127, 0), "y": (5, 122, 122, 5), "z": (0, 64, 127, 64), "r": (10, 117, 117, 10)},
            {"x": (64, 0, 127, 64), "y": (0, 0, 127, 127), "z": (40, 10, 80, 120), "r": (5, 122, 122, 5)},
        ]
        keys = _mixed_gap_keys(curves)
        assert_matches_sample(keys, MIXED_GAP_FRAMES[0], MIXED_GAP_FRAMES[-1], abs_tol=1e-6)


class TestLinearFastpath:
    @pytest.mark.parametrize("cp", [LINEAR, (30, 30, 90, 90), (0, 0, 127, 127)])
    def test_fastpath_matches_newton_and_identity(self, cp):
        keys = [bkey(0, pos=(0.0, 0.0, 0.0)),
                bkey(16, pos=(1.0, 1.0, 1.0), curves={"x": cp, "y": cp, "z": cp})]
        positions, _ = interp.bake_bone_track(keys, 0, 16)
        for f in range(17):
            expected = interp.sample(keys, "pos_x", f)
            assert positions[f][0] == pytest.approx(expected, abs=1e-8)
            assert positions[f][0] == pytest.approx(f / 16.0, abs=1e-8)


class TestConstantFastpath:
    def test_constant_channel_with_nonlinear_curve(self):
        cp = (0, 127, 127, 0)
        keys = [bkey(0, pos=(0.0, 4.0, -2.0)),
                bkey(10, pos=(10.0, 4.0, -2.0),
                     curves={"x": cp, "y": cp, "z": cp})]
        positions, _ = interp.bake_bone_track(keys, 0, 10)
        for f in range(11):
            assert positions[f][1] == 4.0
            assert positions[f][2] == -2.0

    def test_constant_rotation_returns_normalized_value(self):
        q = (0.0, 0.0, 0.0, 2.0)
        keys = [bkey(0, rot=q), bkey(10, rot=q, curves={"r": (0, 127, 127, 0)})]
        _, rotations = interp.bake_bone_track(keys, 0, 10)
        for f in range(11):
            assert tuple(rotations[f]) == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)


class TestBoundaries:
    def test_single_key_constant_fill(self):
        keys = [bkey(7, pos=(1.0, 2.0, 3.0), rot=(0.5, 0.5, 0.5, 0.5))]
        positions, rotations = interp.bake_bone_track(keys, 0, 10)
        assert len(positions) == 11
        for i in range(11):
            assert tuple(positions[i]) == (1.0, 2.0, 3.0)
            assert tuple(rotations[i]) == (0.5, 0.5, 0.5, 0.5)

    def test_gap1_value_jump_preserved(self):
        cp = (0, 127, 127, 0)
        keys = [bkey(10, pos=(0.0, 0.0, 0.0)),
                bkey(11, pos=(5.0, -5.0, 1.0), curves={"x": cp, "y": cp, "z": cp})]
        positions, _ = interp.bake_bone_track(keys, 10, 11)
        assert tuple(positions[0]) == (0.0, 0.0, 0.0)
        assert tuple(positions[1]) == (5.0, -5.0, 1.0)

    def test_range_extends_beyond_keys(self):
        keys = [bkey(5, pos=(1.0, 0.0, 0.0)), bkey(8, pos=(4.0, 0.0, 0.0))]
        assert_matches_sample(keys, 0, 12)

    def test_partial_range_inside_segment(self):
        keys = [bkey(0, pos=(0.0, 0.0, 0.0)),
                bkey(20, pos=(10.0, 0.0, 0.0), curves={"x": (5, 122, 122, 5)})]
        assert_matches_sample(keys, 6, 13)
