import math

from vmd.cuts import (
    assemble_boundaries,
    detect_cuts_bone,
    detect_cuts_camera,
    perspective_cut_frames,
)


def quat_z(deg):
    a = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(a), math.cos(a))


def test_camera_position_jump_detected():
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0)] * 3
    distances = [-30.0] * 3
    cut = detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0))
    assert cut == {2}


def test_camera_position_uses_euclidean_distance():
    positions = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0)] * 2
    distances = [-30.0] * 2
    assert detect_cuts_camera(0, positions, rotations, distances, (4.9, 20.0, 5.0)) == {1}
    assert detect_cuts_camera(0, positions, rotations, distances, (5.1, 20.0, 5.0)) == set()


def test_camera_rotation_jump_detected_in_degrees():
    positions = [(0.0, 0.0, 0.0)] * 3
    rotations = [(0.0, 0.0, 0.0), (0.0, math.radians(30), 0.0), (0.0, math.radians(30), 0.0)]
    distances = [-30.0] * 3
    cut = detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0))
    assert cut == {1}


def test_camera_rotation_wraparound_not_detected():
    positions = [(0.0, 0.0, 0.0)] * 2
    rotations = [(0.0, math.radians(179), 0.0), (0.0, math.radians(-179), 0.0)]
    distances = [-30.0] * 2
    assert detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0)) == set()


def test_camera_rotation_uses_max_axis():
    positions = [(0.0, 0.0, 0.0)] * 2
    rotations = [(0.0, 0.0, 0.0), (math.radians(2), 0.0, math.radians(30))]
    distances = [-30.0] * 2
    assert detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0)) == {1}


def test_camera_distance_jump_detected():
    positions = [(0.0, 0.0, 0.0)] * 3
    rotations = [(0.0, 0.0, 0.0)] * 3
    distances = [-30.0, -30.0, -10.0]
    cut = detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0))
    assert cut == {2}


def test_camera_below_threshold_no_cut():
    positions = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0)] * 3
    distances = [-30.0, -31.0, -32.0]
    cut = detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0))
    assert cut == set()


def test_camera_cut_frame_offset_by_start():
    positions = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0)] * 2
    distances = [-30.0] * 2
    cut = detect_cuts_camera(100, positions, rotations, distances, (5.0, 20.0, 5.0))
    assert cut == {101}


def test_camera_difference_equal_to_threshold_is_not_cut():
    positions = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0)] * 2
    distances = [-30.0] * 2
    assert detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0)) == set()


def test_camera_empty_and_single_no_cut():
    assert detect_cuts_camera(0, [], [], [], (5.0, 20.0, 5.0)) == set()
    assert detect_cuts_camera(0, [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [-30.0], (5.0, 20.0, 5.0)) == set()


def test_bone_position_jump_detected():
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0, 1.0)] * 3
    cut = detect_cuts_bone(0, positions, rotations, (1.0, 30.0))
    assert cut == {2}


def test_bone_rotation_jump_detected():
    positions = [(0.0, 0.0, 0.0)] * 3
    rotations = [(0.0, 0.0, 0.0, 1.0), quat_z(60), quat_z(60)]
    cut = detect_cuts_bone(0, positions, rotations, (1.0, 30.0))
    assert cut == {1}


def test_bone_rotation_below_threshold():
    positions = [(0.0, 0.0, 0.0)] * 2
    rotations = [(0.0, 0.0, 0.0, 1.0), quat_z(10)]
    cut = detect_cuts_bone(0, positions, rotations, (1.0, 30.0))
    assert cut == set()


def test_bone_position_uses_euclidean_distance():
    positions = [(0.0, 0.0, 0.0), (0.6, 0.8, 0.0)]
    rotations = [(0.0, 0.0, 0.0, 1.0)] * 2
    assert detect_cuts_bone(0, positions, rotations, (0.9, 30.0)) == {1}
    assert detect_cuts_bone(0, positions, rotations, (1.1, 30.0)) == set()


def test_bone_position_difference_equal_to_threshold_is_not_cut():
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0, 1.0)] * 2
    assert detect_cuts_bone(0, positions, rotations, (1.0, 30.0)) == set()


def test_bone_cut_frame_offset_by_start():
    positions = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0, 1.0)] * 2
    assert detect_cuts_bone(50, positions, rotations, (1.0, 30.0)) == {51}


def test_bone_empty_and_single_no_cut():
    assert detect_cuts_bone(0, [], [], (1.0, 30.0)) == set()
    assert detect_cuts_bone(0, [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0, 1.0)], (1.0, 30.0)) == set()


def test_perspective_change_is_boundary():
    persp = [0, 0, 1, 1]
    assert perspective_cut_frames(0, persp) == {2}


def test_perspective_no_change():
    assert perspective_cut_frames(0, [1, 1, 1]) == set()


def test_perspective_offset_by_start():
    assert perspective_cut_frames(100, [0, 0, 1]) == {102}


def test_perspective_empty_and_single():
    assert perspective_cut_frames(0, []) == set()
    assert perspective_cut_frames(0, [1]) == set()


def test_assemble_includes_range_ends():
    b = assemble_boundaries(0, 60, cuts={30}, perspective_frames=set(), keep_frames=[], no_cut_detect=False)
    assert b == [0, 30, 60]


def test_assemble_keep_frames_in_range():
    b = assemble_boundaries(0, 60, cuts=set(), perspective_frames=set(), keep_frames=[15, 45], no_cut_detect=False)
    assert b == [0, 15, 45, 60]


def test_assemble_keep_frames_out_of_range_dropped():
    b = assemble_boundaries(10, 50, cuts=set(), perspective_frames=set(), keep_frames=[5, 30, 99], no_cut_detect=False)
    assert b == [10, 30, 50]


def test_assemble_no_cut_detect_drops_cuts_but_keeps_perspective_and_keep():
    b = assemble_boundaries(
        0, 60, cuts={20}, perspective_frames={40}, keep_frames=[10], no_cut_detect=True
    )
    assert b == [0, 10, 40, 60]


def test_assemble_dedup_and_sorted():
    b = assemble_boundaries(
        0, 60, cuts={30}, perspective_frames={30}, keep_frames=[0, 30, 60], no_cut_detect=False
    )
    assert b == [0, 30, 60]
