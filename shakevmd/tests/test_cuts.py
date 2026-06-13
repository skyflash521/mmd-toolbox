"""cuts のテスト(shakevmd.md §5.3)。

検出はフレーム差1の隣接キー間。カメラ中心位置/カメラワールド位置/角度のジャンプ。
セグメント境界はカットフレーム F の F-1 と F の間。手動指定は自動検出に優先。
"""

import math

import pytest

from shakevmd import cuts
from mmd_toolbox.vmd.types import CameraKey

POS_TH = 5.0
ROT_TH = 20.0


def cam(frame, *, distance=-30.0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
        fov=30, perspective=0):
    return CameraKey(frame, distance, center, rotation, bytes(24), fov, perspective)


# ---------------------------------------------------------------------------
# detect_cuts
# ---------------------------------------------------------------------------


class TestDetectCuts:
    def test_no_cut_when_static(self):
        keys = [cam(0), cam(1), cam(2)]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_center_position_jump(self):
        # フレーム差1でカメラ中心が大きく飛ぶ → カット(後側フレーム)
        keys = [cam(10), cam(11, center=(100.0, 0.0, 0.0))]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == [11]

    def test_below_threshold_no_cut(self):
        keys = [cam(10), cam(11, center=(1.0, 0.0, 0.0))]  # 1.0 < 5.0
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_position_jump_is_euclidean(self):
        # 各軸は閾値未満でもユークリッド距離が閾値超ならカット(軸別判定ではない)
        keys = [cam(10), cam(11, center=(4.0, 4.0, 0.0))]  # 各軸4<5、距離≈5.66>5
        assert math.dist((0.0, 0.0, 0.0), (4.0, 4.0, 0.0)) > POS_TH
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == [11]

    def test_zoom_cut_via_world_position(self):
        # 距離のみ急変(中心・角度は同一)。カメラ中心位置・角度には現れないが
        # カメラワールド位置(= 中心 + R·(0,0,distance))が動く → 検出
        keys = [cam(20, distance=-30.0), cam(21, distance=-10.0)]  # |Δ|=20 > 5
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == [21]
        # 中心位置・角度差は 0 なので、ワールド位置併用が無ければ見逃す
        # (位置閾値を大きくし、ワールド位置由来でなければ拾えないことを確認)
        assert cuts.detect_cuts(keys, 1000.0, ROT_TH) == []

    def test_center_position_branch_isolated(self):
        # 中心位置ブランチを独立に固定する。中心差>閾値だが、回転で相殺して
        # ワールド位置差≤閾値・角度差<閾値にした構成。「ワールド位置だけ見る実装」では
        # 検出できず、中心位置条件が独立に効いていることを要求する。
        from mmd_toolbox.vmd import camera

        k0 = cam(40, center=(0.0, 0.0, 0.0), distance=-50.0, rotation=(0.0, 0.0, 0.0))
        k1 = cam(41, center=(0.0, 8.0, 0.0), distance=-50.0,
                 rotation=(math.radians(10.0), 0.0, 0.0))
        assert math.dist(k0.position, k1.position) > POS_TH          # 中心差 8 > 5
        w0 = camera.to_world(k0).position
        w1 = camera.to_world(k1).position
        assert math.dist(w0, w1) <= POS_TH                           # ワールド差 ≤ 5
        # 角度差 10° < 20°
        assert cuts.detect_cuts([k0, k1], POS_TH, ROT_TH) == [41]

    def test_world_position_requires_camera_transform(self):
        # 中心差・角度差はどちらも閾値未満だが、ワールド位置(中心+R·(0,0,distance))の
        # ジャンプは閾値超になる構成。生distance差や中心差だけの実装では検出できず、
        # mmd_toolbox.vmd.camera の座標変換を要する。
        from mmd_toolbox.vmd import camera

        k0 = cam(40, center=(0.0, 0.0, 0.0), distance=-50.0, rotation=(0.0, 0.0, 0.0))
        k1 = cam(41, center=(3.0, 0.0, 0.0), distance=-50.0,
                 rotation=(math.radians(10.0), 0.0, 0.0))
        # 前提: 中心差 3 < 5、角度差 10° < 20°、ワールド位置差は 5 超
        assert math.dist(k0.position, k1.position) < POS_TH
        w0 = camera.to_world(k0).position
        w1 = camera.to_world(k1).position
        assert math.dist(w0, w1) > POS_TH
        assert cuts.detect_cuts([k0, k1], POS_TH, ROT_TH) == [41]

    def test_exact_threshold_is_not_a_cut(self):
        # 「超える」は strict >。ちょうど閾値はカットにしない(位置)
        keys = [cam(10), cam(11, center=(POS_TH, 0.0, 0.0))]  # 中心差 = 5.0 == POS_TH
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_angle_exact_threshold_is_not_a_cut(self):
        # 角度もちょうど閾値はカットにしない(strict >)。distance=0 で角度のみに分離
        keys = [cam(10, distance=0.0),
                cam(11, distance=0.0, rotation=(math.radians(ROT_TH), 0.0, 0.0))]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_fov_perspective_change_not_a_cut(self):
        # 検出条件は中心位置・ワールド位置・角度のみ。視野角/パースの変更はカットにしない
        keys = [cam(10, fov=30, perspective=0), cam(11, fov=90, perspective=1)]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    @pytest.mark.parametrize(
        "rotation",
        [
            (math.radians(45.0), 0.0, 0.0),
            (0.0, math.radians(45.0), 0.0),
            (0.0, 0.0, math.radians(45.0)),
        ],
    )
    def test_angle_jump_any_axis(self, rotation):
        # 角度ジャンプは特定軸に限定しない。X/Y/Z いずれの軸でも検出する。
        # distance=0 で角度のみに分離(ワールド位置=中心で不動)
        keys = [cam(30, distance=0.0), cam(31, distance=0.0, rotation=rotation)]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == [31]

    def test_angle_below_threshold_no_cut(self):
        small = math.radians(5.0)  # 5度 < 20度
        keys = [cam(30, distance=0.0), cam(31, distance=0.0, rotation=(small, 0.0, 0.0))]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_only_adjacent_frame_pairs(self):
        # フレーム差が1でないペアはカット判定の対象外(間に補間があるため)
        keys = [cam(0), cam(10, center=(100.0, 0.0, 0.0))]  # 差10
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_multiple_cuts_sorted(self):
        # 各ショットは内部で静止(意図しないジャンプを作らない)。
        # 0→1 で中心ジャンプ(カット@1)、1→2 は同一(カットなし)、
        # 2→3 で角度ジャンプ(カット@3)。
        big = math.radians(90.0)
        keys = [
            cam(0, center=(0.0, 0.0, 0.0)),
            cam(1, center=(100.0, 0.0, 0.0)),                       # cut @1
            cam(2, center=(100.0, 0.0, 0.0)),                       # 同一ショット
            cam(3, center=(100.0, 0.0, 0.0), rotation=(big, 0.0, 0.0)),  # cut @3
        ]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == [1, 3]


# ---------------------------------------------------------------------------
# resolve_cuts(手動指定が自動検出に優先)
# ---------------------------------------------------------------------------


class TestResolveCuts:
    def test_passthrough_when_no_manual(self):
        assert cuts.resolve_cuts([1, 3]) == [1, 3]

    def test_add_forces_cut(self):
        assert cuts.resolve_cuts([1], add=[5]) == [1, 5]

    def test_remove_cancels_cut(self):
        assert cuts.resolve_cuts([1, 3], remove=[3]) == [1]

    def test_sorted_unique(self):
        assert cuts.resolve_cuts([3, 1, 3], add=[1, 7]) == [1, 3, 7]

    def test_remove_takes_priority_over_add(self):
        # 同フレームを add と remove に指定した場合 remove 優先(打ち消し)
        assert cuts.resolve_cuts([1], add=[5], remove=[5]) == [1]


# ---------------------------------------------------------------------------
# segment_bounds(F-1 と F の間で分割)
# ---------------------------------------------------------------------------


class TestSegmentBounds:
    def test_no_cuts_single_segment(self):
        assert cuts.segment_bounds(0, 100, []) == [cuts.Segment(0, 100)]

    def test_single_cut_splits_at_boundary(self):
        # カット @50 → [0,49] と [50,100]
        assert cuts.segment_bounds(0, 100, [50]) == [
            cuts.Segment(0, 49),
            cuts.Segment(50, 100),
        ]

    def test_multiple_cuts(self):
        assert cuts.segment_bounds(0, 100, [30, 60]) == [
            cuts.Segment(0, 29),
            cuts.Segment(30, 59),
            cuts.Segment(60, 100),
        ]

    def test_cuts_outside_range_ignored(self):
        assert cuts.segment_bounds(10, 50, [5, 70]) == [cuts.Segment(10, 50)]

    def test_cut_at_range_start_ignored(self):
        # 範囲先頭でのカットは空セグメントを作らない(先頭は常に新セグメント)
        assert cuts.segment_bounds(10, 50, [10]) == [cuts.Segment(10, 50)]

    def test_cut_at_range_end(self):
        # 末尾フレームのカットは F-1|F 境界で分割し、末尾は単一フレームセグメント
        assert cuts.segment_bounds(10, 50, [50]) == [
            cuts.Segment(10, 49),
            cuts.Segment(50, 50),
        ]
