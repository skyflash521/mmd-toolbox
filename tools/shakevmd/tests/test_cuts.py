import math

import pytest

from shakevmd import cuts
from vmd import camera
from vmd.types import CameraKey

POS_TH = 5.0
ROT_TH = 20.0


def cam(frame, *, distance=-30.0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
        fov=30, perspective=0):
    return CameraKey(frame, distance, center, rotation, bytes(24), fov, perspective)


class TestDetectCuts:
    def test_no_cut_when_static(self):
        keys = [cam(0), cam(1), cam(2)]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_center_jump_above_threshold_is_cut_at_later_frame(self):
        keys = [cam(10), cam(11, center=(100.0, 0.0, 0.0))]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == [11]

    def test_center_jump_below_threshold_is_not_cut(self):
        keys = [cam(10), cam(11, center=(1.0, 0.0, 0.0))]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_position_jump_is_euclidean(self):
        keys = [cam(10), cam(11, center=(4.0, 4.0, 0.0))]
        assert math.dist((0.0, 0.0, 0.0), (4.0, 4.0, 0.0)) > POS_TH
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == [11]

    def test_distance_only_jump_is_cut_via_world_position(self):
        keys = [cam(20, distance=-30.0), cam(21, distance=-10.0)]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == [21]
        assert cuts.detect_cuts(keys, 1000.0, ROT_TH) == []

    def test_center_jump_is_cut_even_when_world_jump_is_within_threshold(self):
        k0 = cam(40, center=(0.0, 0.0, 0.0), distance=-50.0, rotation=(0.0, 0.0, 0.0))
        k1 = cam(41, center=(0.0, 8.0, 0.0), distance=-50.0,
                 rotation=(math.radians(10.0), 0.0, 0.0))
        assert math.dist(k0.position, k1.position) > POS_TH
        w0 = camera.to_world(k0).position
        w1 = camera.to_world(k1).position
        assert math.dist(w0, w1) <= POS_TH
        assert cuts.detect_cuts([k0, k1], POS_TH, ROT_TH) == [41]

    def test_world_jump_is_cut_even_when_center_and_angle_jumps_are_within_threshold(self):
        k0 = cam(40, center=(0.0, 0.0, 0.0), distance=-50.0, rotation=(0.0, 0.0, 0.0))
        k1 = cam(41, center=(3.0, 0.0, 0.0), distance=-50.0,
                 rotation=(math.radians(10.0), 0.0, 0.0))
        assert math.dist(k0.position, k1.position) < POS_TH
        w0 = camera.to_world(k0).position
        w1 = camera.to_world(k1).position
        assert math.dist(w0, w1) > POS_TH
        assert cuts.detect_cuts([k0, k1], POS_TH, ROT_TH) == [41]

    def test_position_jump_equal_to_threshold_is_not_cut(self):
        keys = [cam(10), cam(11, center=(POS_TH, 0.0, 0.0))]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_angle_jump_equal_to_threshold_is_not_cut(self):
        keys = [cam(10, distance=0.0),
                cam(11, distance=0.0, rotation=(math.radians(ROT_TH), 0.0, 0.0))]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_fov_perspective_change_not_a_cut(self):
        keys = [cam(10, fov=30, perspective=0), cam(11, fov=90, perspective=1)]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    @pytest.mark.parametrize(
        "rotation",
        [
            pytest.param((math.radians(45.0), 0.0, 0.0), id="x"),
            pytest.param((0.0, math.radians(45.0), 0.0), id="y"),
            pytest.param((0.0, 0.0, math.radians(45.0)), id="z"),
        ],
    )
    def test_angle_jump_any_axis(self, rotation):
        keys = [cam(30, distance=0.0), cam(31, distance=0.0, rotation=rotation)]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == [31]

    def test_angle_below_threshold_no_cut(self):
        small = math.radians(5.0)
        keys = [cam(30, distance=0.0), cam(31, distance=0.0, rotation=(small, 0.0, 0.0))]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_keys_more_than_one_frame_apart_are_never_cut(self):
        keys = [cam(0), cam(10, center=(100.0, 0.0, 0.0))]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == []

    def test_multiple_cuts_sorted(self):
        big = math.radians(90.0)
        keys = [
            cam(0, center=(0.0, 0.0, 0.0)),
            cam(1, center=(100.0, 0.0, 0.0)),
            cam(2, center=(100.0, 0.0, 0.0)),
            cam(3, center=(100.0, 0.0, 0.0), rotation=(big, 0.0, 0.0)),
        ]
        assert cuts.detect_cuts(keys, POS_TH, ROT_TH) == [1, 3]


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
        assert cuts.resolve_cuts([1], add=[5], remove=[5]) == [1]


class TestSegmentBounds:
    def test_no_cuts_single_segment(self):
        assert cuts.segment_bounds(0, 100, []) == [cuts.Segment(0, 100)]

    def test_single_cut_splits_at_boundary(self):
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
        assert cuts.segment_bounds(10, 50, [10]) == [cuts.Segment(10, 50)]

    def test_cut_at_range_end_leaves_single_frame_segment(self):
        assert cuts.segment_bounds(10, 50, [50]) == [
            cuts.Segment(10, 49),
            cuts.Segment(50, 50),
        ]
