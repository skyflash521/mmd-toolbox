import pytest

from vmd.fit import LinearScalarChannel
from vmd.reduce import (
    BONE_LINEAR_INTERP,
    StrictError,
    build_bone_tolerances,
    reduce_bone_track,
    reduce_track,
)
from vmd.types import BoneKey


def lin(frame_start, values, tol):
    return LinearScalarChannel(frame_start, [float(v) for v in values], tol)


def test_linear_segment_no_split():
    ch = lin(0, range(11), tol=0.01)
    keys = reduce_track([0, 10], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 10]


def test_peak_splits_at_extremum():
    vals = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ch = lin(0, vals, tol=1.0)
    keys = reduce_track([0, 10], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 5, 10]


def test_valley_splits_at_extremum():
    vals = [0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0]
    ch = lin(0, vals, tol=1.0)
    keys = reduce_track([0, 10], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 5, 10]


def test_max_seg_none_disables_cap():
    ch = lin(0, range(101), tol=0.01)
    assert reduce_track([0, 100], [ch], min_seg=1, max_seg=None, strict=False) == [0, 100]
    capped = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)
    assert capped != [0, 100] and len(capped) > 2


def test_mandatory_boundaries_kept():
    ch = lin(0, [float(i) for i in range(61)], tol=0.01)
    keys = reduce_track([0, 30, 60], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 30, 60]


def test_unsorted_duplicate_boundaries_normalized():
    ch = lin(0, [float(i) for i in range(61)], tol=0.01)
    keys = reduce_track([60, 0, 30, 30, 0], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 30, 60]


def test_max_segment_sliding_cap_uses_minimum_number_of_spans():
    ch = lin(0, [float(i) for i in range(101)], tol=1.0)
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)
    assert keys[0] == 0 and keys[-1] == 100
    gaps = [b - a for a, b in zip(keys, keys[1:], strict=False)]
    assert all(g <= 40 for g in gaps)
    assert len(gaps) == 3


def test_dense_fallback_non_strict():
    vals = [0, 5, 0, 5, 0]
    ch = lin(0, vals, tol=0.5)
    keys = reduce_track([0, 4], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 1, 2, 3, 4]


def test_strict_raises_when_min_seg_blocks():
    vals = [0, 5, 0, 5, 0]
    ch = lin(0, vals, tol=0.5)
    with pytest.raises(StrictError):
        reduce_track([0, 4], [ch], min_seg=4, max_seg=180, strict=True)


def test_min_seg_ignored_and_every_frame_kept_when_non_strict():
    vals = [0, 5, 0, 5, 0]
    ch = lin(0, vals, tol=0.5)
    keys = reduce_track([0, 4], [ch], min_seg=4, max_seg=180, strict=False)
    assert keys == [0, 1, 2, 3, 4]


def test_multichannel_respects_each_channel_tolerance():
    ch_a = lin(0, [float(i) for i in range(11)], tol=1.0)
    ch_b = lin(0, [0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0], tol=0.01)
    keys = reduce_track([0, 10], [ch_a, ch_b], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 4, 5, 6, 10]


def test_split_prefers_extremum_not_max_error_frame():
    ch = lin(0, [0, 10, 2, 4, 20], tol=9.0)
    keys = reduce_track([0, 4], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 2, 4]


def test_adjacent_segment_accepted():
    ch = lin(0, [0.0, 100.0], tol=0.001)
    keys = reduce_track([0, 1], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 1]


def test_constant_span_not_capped():
    ch = lin(0, [5.0] * 101, tol=0.01)
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)
    assert keys == [0, 100]


def test_loose_non_constant_span_capped():
    ch = lin(0, [float(i) for i in range(101)], tol=1.0)
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)
    gaps = [b - a for a, b in zip(keys, keys[1:], strict=False)]
    assert all(g <= 40 for g in gaps)
    assert len(gaps) == 3


def test_mixed_constant_and_varying_span_no_grid_in_flat():
    vals = [5.0] * 61 + [5.0 + float(i + 1) for i in range(40)]
    ch = lin(0, vals, tol=1.0)
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)
    assert keys[0] == 0 and keys[-1] == 100
    assert not any(0 < k < 55 for k in keys)
    assert any(55 <= k <= 65 for k in keys)


def test_maxspan_cap_requires_all_channels_constant():
    const_ch = lin(0, [5.0] * 101, tol=0.01)
    vary_ch = lin(0, [float(i) for i in range(101)], tol=1.0)
    keys = reduce_track([0, 100], [const_ch, vary_ch], min_seg=1, max_seg=40, strict=False)
    gaps = [b - a for a, b in zip(keys, keys[1:], strict=False)]
    assert all(g <= 40 for g in gaps)
    assert len(gaps) == 3


def test_non_constant_fallback_when_channel_lacks_is_constant():
    class StubChannel:
        def normalized(self, a, b):
            return (0.0, None)

    keys = reduce_track([0, 100], [StubChannel()], min_seg=1, max_seg=40, strict=False)
    gaps = [b - a for a, b in zip(keys, keys[1:], strict=False)]
    assert all(g <= 40 for g in gaps)
    assert len(gaps) == 3


def test_constant_span_reduce_deterministic():
    def run():
        ch = lin(0, [5.0] * 101, tol=0.01)
        return reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)

    assert run() == run()


def test_reduce_uses_only_normalized_contract():
    class StubChannel:
        def __init__(self, worst):
            self._worst = worst

        def normalized(self, a, b):
            return self._worst.get((a, b), (0.0, None))

    stub = StubChannel({(0, 10): (3.0, 5), (0, 5): (0.0, None), (5, 10): (0.0, None)})
    keys = reduce_track([0, 10], [stub], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 5, 10]


def test_progress_reaches_total_monotonically():
    vals = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ch = lin(0, vals, tol=1.0)
    calls = []
    reduce_track([0, 10], [ch], min_seg=1, max_seg=180, strict=False, progress=lambda d, t: calls.append((d, t)))
    assert calls
    assert all(t == 10 for _, t in calls)
    done = [d for d, _ in calls]
    assert done == sorted(done)
    assert done[-1] == 10


def test_fitting_long_span_capped_and_recorded():
    ch = lin(0, [float(i) for i in range(101)], tol=1.0)
    caps = []
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False, caps=caps)
    gaps = [b - a for a, b in zip(keys, keys[1:], strict=False)]
    assert all(g <= 40 for g in gaps)
    assert caps and all(0 < c["frame"] < 100 for c in caps)


def test_constant_long_span_not_capped():
    ch = lin(0, [5.0] * 101, tol=0.01)
    caps = []
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False, caps=caps)
    assert keys == [0, 100]
    assert caps == []


def test_error_split_recorded_in_splits_not_caps():
    ch = lin(0, [0, 10, 2, 4, 20], tol=9.0)
    splits, caps = [], []
    reduce_track([0, 4], [ch], min_seg=1, max_seg=180, strict=False, splits=splits, caps=caps)
    assert splits
    assert caps == []


def test_no_mechanical_grid_when_span_within_max_seg():
    ch = lin(0, [float(i) for i in range(31)], tol=1.0)
    caps = []
    keys = reduce_track([0, 30], [ch], min_seg=1, max_seg=180, strict=False, caps=caps)
    assert keys == [0, 30]
    assert caps == []


def test_reduce_bone_track_diagnostics_has_maxspan_caps():
    name = b"bone".ljust(15, b"\x00")
    src = [
        BoneKey(name, f, (float(f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)
        for f in range(101)
    ]
    diag = {}
    reduce_bone_track(
        src,
        [(0, 100)],
        build_bone_tolerances(bone_pos=0.5, bone_rot=5.0),
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=40,
        strict=False,
        curve_mode="bezier",
        diagnostics=diag,
    )
    assert "maxspan_caps" in diag
    assert diag["maxspan_caps"]
