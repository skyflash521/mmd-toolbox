import math

import pytest

from vmd.reduce import (
    BONE_LINEAR_INTERP,
    build_bone_tolerances,
    reduce_bone_track,
)
from vmd.types import BoneKey

NAME = b"bone".ljust(15, b"\x00")
TOLS = build_bone_tolerances(bone_pos=0.01, bone_rot=0.10)
PHASE = "出力後検証"


def _peak_source(n=41):
    keys = []
    for f in range(n):
        y = float(f) if f <= 15 else float(30 - f)
        keys.append(BoneKey(NAME, f, (0.0, y, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP))
    return keys


def _sine_source(n=41):
    return [
        BoneKey(NAME, f, (5.0 * math.sin(0.4 * f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)
        for f in range(n)
    ]


def _reduce(source, ranges, **kw):
    opts = dict(
        cut_thresholds=(1.0, 30.0),
        keep_frames=[],
        no_cut_detect=False,
        min_seg=1,
        max_seg=180,
        strict=False,
    )
    opts.update(kw)
    return reduce_bone_track(source, ranges, TOLS, **opts)


def _collect(source, ranges, **kw):
    calls = []
    _reduce(source, ranges, progress=lambda d, t, note: calls.append((d, t, note)), **kw)
    return calls


def test_progress_is_called_when_given():
    calls = _collect(_peak_source(), [(0, 30)])
    assert calls


def test_progress_done_is_monotonic_and_reaches_total():
    calls = _collect(_peak_source(), [(0, 30)])
    dones = [d for d, _, _ in calls]
    assert dones == sorted(dones)
    assert dones[-1] == 30
    assert all(0 <= d <= 30 for d in dones)


def test_progress_total_equals_sum_of_range_widths():
    calls = _collect(_peak_source(), [(0, 20), (30, 40)])
    assert all(t == 30 for _, t, _ in calls)


def test_progress_reports_verification_phase_with_empty_note_otherwise():
    calls = _collect(_peak_source(), [(0, 20), (30, 40)])
    assert {note for _, _, note in calls} == {"", PHASE}
    assert calls[-1] == (30, 30, PHASE)


def test_progress_verification_phase_reported_once_per_range_at_cumulative_position():
    calls = _collect(_peak_source(), [(0, 20), (30, 40)])
    assert [(d, t) for d, t, note in calls if note == PHASE] == [(20, 30), (30, 30)]


def test_progress_across_multiple_ranges_is_monotonic_in_sum():
    calls = _collect(_peak_source(), [(0, 10), (12, 25), (30, 40)])
    dones = [d for d, _, _ in calls]
    assert dones == sorted(dones)
    assert all(t == 33 for _, t, _ in calls)
    assert dones[-1] == 33
    assert [d for d, _, note in calls if note == PHASE] == [10, 23, 33]


def test_progress_does_not_change_output():
    source = _peak_source()
    ranges = [(0, 20), (30, 40)]
    without = _reduce(source, ranges)
    with_progress = _reduce(source, ranges, progress=lambda d, t, note: None)
    assert [(k.frame, k.position, k.rotation, k.interpolation) for k in without] == [
        (k.frame, k.position, k.rotation, k.interpolation) for k in with_progress
    ]


@pytest.mark.parametrize("c1_smoothing", [True, False])
def test_progress_works_together_with_c1_smoothing_and_bezier(c1_smoothing):
    source = _sine_source()
    calls = _collect(
        source, [(0, 40)], curve_mode="bezier", c1_smoothing=c1_smoothing, no_cut_detect=True
    )
    dones = [d for d, _, _ in calls]
    assert dones == sorted(dones)
    assert all(t == 40 for _, t, _ in calls)
    assert calls[-1] == (40, 40, PHASE)


def test_progress_works_together_with_diagnostics():
    diag = {}
    calls = _collect(_peak_source(), [(0, 30)], diagnostics=diag)
    assert calls[-1][0] == 30
    assert diag["verify"]


def _slow_sine_source(n=41):
    return [
        BoneKey(NAME, f, (5.0 * math.sin(0.1 * f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)
        for f in range(n)
    ]


def test_progress_reports_intermediate_resolution_within_a_single_long_range():
    calls = _collect(_slow_sine_source(), [(0, 40)])
    assert any(note == "" and 0 < d < 40 for d, _, note in calls)
