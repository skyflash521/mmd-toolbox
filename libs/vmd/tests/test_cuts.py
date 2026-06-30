"""不連続検出・必須境界のテスト(vmd-reduce.md §7.2, §7.1, §7.2)。

cuts は隣接サンプル F-1 と F の差が閾値を超えたフレーム F を不連続境界として返す。
camera は 中心位置(POS,ユークリッド)・回転(ROT,軸別最小角の最大,度)・距離(DIST,絶対値)、
bone は 位置(POS,ユークリッド)・回転(ROT,quaternion角度距離,度)を対象にする。
perspective の切り替えフレームは常に境界。assemble_boundaries は範囲端・keep-frame・
perspective境界・(no-cut-detect でなければ)検出cut を統合する。
"""

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


# --- camera 検出 ------------------------------------------------------------


def test_camera_position_jump_detected():
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0)] * 3
    distances = [-30.0] * 3
    # frame 2 で 19 のジャンプ。POS=5 超過。frame は start+index=2。
    cut = detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0))
    assert cut == {2}


def test_camera_position_euclidean_diagonal():
    # 斜め成分(3,4,0)→距離5。POS=4.9 超過、POS=5.1 は非超過。
    positions = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0)] * 2
    distances = [-30.0] * 2
    assert detect_cuts_camera(0, positions, rotations, distances, (4.9, 20.0, 5.0)) == {1}
    assert detect_cuts_camera(0, positions, rotations, distances, (5.1, 20.0, 5.0)) == set()


def test_camera_rotation_jump_detected():
    positions = [(0.0, 0.0, 0.0)] * 3
    # frame1 で Y軸 +30度(ラジアン)。ROT=20 超過。
    rotations = [(0.0, 0.0, 0.0), (0.0, math.radians(30), 0.0), (0.0, math.radians(30), 0.0)]
    distances = [-30.0] * 3
    cut = detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0))
    assert cut == {1}


def test_camera_rotation_wraparound_not_detected():
    # 179度→-179度 は見かけ上 358度差だが実角度差は2度。最小角差で誤検出しない(vmd-reduce.md §7.1)。
    positions = [(0.0, 0.0, 0.0)] * 2
    rotations = [(0.0, math.radians(179), 0.0), (0.0, math.radians(-179), 0.0)]
    distances = [-30.0] * 2
    assert detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0)) == set()


def test_camera_rotation_uses_max_axis():
    # X軸2度・Z軸30度 → 最大軸(30度)で判定。ROT=20 超過。
    positions = [(0.0, 0.0, 0.0)] * 2
    rotations = [(0.0, 0.0, 0.0), (math.radians(2), 0.0, math.radians(30))]
    distances = [-30.0] * 2
    assert detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0)) == {1}


def test_camera_distance_jump_detected():
    positions = [(0.0, 0.0, 0.0)] * 3
    rotations = [(0.0, 0.0, 0.0)] * 3
    distances = [-30.0, -30.0, -10.0]  # frame2 で 20 のジャンプ。DIST=5 超過。
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
    assert cut == {101}  # start=100 + index1


def test_camera_threshold_exact_value_no_cut():
    # 「超えた場合」=厳密に超過(vmd-reduce.md §7.1)。ちょうど閾値ならカットしない。
    positions = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]  # 距離ちょうど5.0
    rotations = [(0.0, 0.0, 0.0)] * 2
    distances = [-30.0] * 2
    assert detect_cuts_camera(0, positions, rotations, distances, (5.0, 20.0, 5.0)) == set()


def test_camera_empty_and_single_no_cut():
    assert detect_cuts_camera(0, [], [], [], (5.0, 20.0, 5.0)) == set()
    assert detect_cuts_camera(0, [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [-30.0], (5.0, 20.0, 5.0)) == set()


# --- bone 検出 --------------------------------------------------------------


def test_bone_position_jump_detected():
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0, 1.0)] * 3
    cut = detect_cuts_bone(0, positions, rotations, (1.0, 30.0))
    assert cut == {2}


def test_bone_rotation_jump_detected():
    positions = [(0.0, 0.0, 0.0)] * 3
    # frame1 で Z軸 60度回転。quaternion角度距離 60度 > ROT=30。
    rotations = [(0.0, 0.0, 0.0, 1.0), quat_z(60), quat_z(60)]
    cut = detect_cuts_bone(0, positions, rotations, (1.0, 30.0))
    assert cut == {1}


def test_bone_rotation_below_threshold():
    positions = [(0.0, 0.0, 0.0)] * 2
    rotations = [(0.0, 0.0, 0.0, 1.0), quat_z(10)]  # 10度 < 30
    cut = detect_cuts_bone(0, positions, rotations, (1.0, 30.0))
    assert cut == set()


def test_bone_position_euclidean_diagonal():
    positions = [(0.0, 0.0, 0.0), (0.6, 0.8, 0.0)]  # 距離1.0
    rotations = [(0.0, 0.0, 0.0, 1.0)] * 2
    assert detect_cuts_bone(0, positions, rotations, (0.9, 30.0)) == {1}
    assert detect_cuts_bone(0, positions, rotations, (1.1, 30.0)) == set()


def test_bone_position_exact_threshold_no_cut():
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]  # ちょうど1.0
    rotations = [(0.0, 0.0, 0.0, 1.0)] * 2
    assert detect_cuts_bone(0, positions, rotations, (1.0, 30.0)) == set()


def test_bone_cut_frame_offset_by_start():
    positions = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0, 1.0)] * 2
    assert detect_cuts_bone(50, positions, rotations, (1.0, 30.0)) == {51}


def test_bone_empty_and_single_no_cut():
    assert detect_cuts_bone(0, [], [], (1.0, 30.0)) == set()
    assert detect_cuts_bone(0, [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0, 1.0)], (1.0, 30.0)) == set()


# --- perspective 境界 -------------------------------------------------------


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


# --- assemble_boundaries ----------------------------------------------------


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
    # 閾値検出cut(20)は無効化、perspective(40)とkeep(10)と範囲端は残る。
    assert b == [0, 10, 40, 60]


def test_assemble_dedup_and_sorted():
    b = assemble_boundaries(
        0, 60, cuts={30}, perspective_frames={30}, keep_frames=[0, 30, 60], no_cut_detect=False
    )
    assert b == [0, 30, 60]
