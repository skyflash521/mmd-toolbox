import math

import pytest

from vmd.reduce import (
    CAMERA_LINEAR_INTERP,
    Tolerances,
    measure_bone_errors,
    measure_camera_errors,
    reduce_bone_track,
    reduce_camera_track,
)
from vmd.types import BoneKey, CameraKey

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()
TOLS = Tolerances(
    bone_pos=0.01,
    bone_rot=0.10,
    camera_pos=0.02,
    camera_rot=0.05,
    camera_distance=0.02,
    camera_fov=0.50,
)
CAM_CUT = (5.0, 20.0, 5.0)
BONE_CUT = (1.0, 30.0)


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, persp)


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, rot, BL)


def camera_track(source, ranges, **kw):
    opts = dict(
        cut_thresholds=CAM_CUT,
        keep_frames=[],
        no_cut_detect=False,
        min_seg=1,
        max_seg=180,
        strict=False,
    )
    opts.update(kw)
    return reduce_camera_track(source, ranges, TOLS, **opts)


def frames(keys):
    return [k.frame for k in keys]


def test_camera_linear_reduces_to_endpoints():
    source = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(31)]
    keys = camera_track(source, [(0, 30)])
    assert frames(keys) == [0, 30]
    assert all(k.interpolation == CAMERA_LINEAR_INTERP for k in keys)


def test_camera_peak_kept():
    source = []
    for f in range(31):
        y = float(f) if f <= 15 else float(30 - f)
        source.append(cam(f, center=(0.0, y, 0.0)))
    keys = camera_track(source, [(0, 30)])
    assert frames(keys) == [0, 15, 30]


def test_camera_cut_detected_preserved_as_adjacent_jump():
    source = []
    for f in range(21):
        x = 0.0 if f < 10 else 50.0
        source.append(cam(f, center=(x, 0.0, 0.0)))
    keys = camera_track(source, [(0, 20)])
    fr = frames(keys)
    assert 9 in fr and 10 in fr
    k9 = next(k for k in keys if k.frame == 9)
    k10 = next(k for k in keys if k.frame == 10)
    assert k9.position[0] == pytest.approx(0.0)
    assert k10.position[0] == pytest.approx(50.0)


def test_camera_perspective_switch_kept_as_adjacent_keys_without_cut_detection():
    source = [cam(f, center=(float(f), 0.0, 0.0), persp=0 if f < 10 else 1) for f in range(21)]
    keys = camera_track(source, [(0, 20)], no_cut_detect=True)
    by_frame = {k.frame: k for k in keys}
    assert by_frame[9].perspective == 0
    assert by_frame[10].perspective == 1


def test_camera_fov_peak_kept():
    source = []
    for f in range(31):
        fov = 30 + (15 - abs(f - 15))
        source.append(cam(f, fov=fov))
    keys = camera_track(source, [(0, 30)])
    assert 15 in frames(keys)


def test_camera_rotation_peak_kept():
    source = []
    for f in range(31):
        ry = math.radians(15 - abs(f - 15))
        source.append(cam(f, rot=(0.0, ry, 0.0)))
    keys = camera_track(source, [(0, 30)])
    assert 15 in frames(keys)


def test_camera_keep_frame_forced():
    source = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(31)]
    keys = camera_track(source, [(0, 30)], keep_frames=[12])
    assert 12 in frames(keys)


def test_camera_range_outside_keys_preserved_verbatim():
    source = [cam(f, center=(float(f), 0.0, 0.0), fov=30 + f) for f in (0, 5, 10, 20, 30)]
    keys = camera_track(source, [(0, 10)])
    fr = frames(keys)
    assert 20 in fr and 30 in fr
    assert fr == sorted(fr)
    k20 = next(k for k in keys if k.frame == 20)
    assert k20.position[0] == pytest.approx(20.0)
    assert k20.fov == 50
    assert k20.interpolation == CAM_LINEAR


def test_camera_multiple_ranges_reduced_independently_and_gap_keys_kept():
    source = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(61)]
    keys = camera_track(source, [(0, 20), (40, 60)])
    fr = frames(keys)
    assert 0 in fr and 20 in fr and 40 in fr and 60 in fr
    assert any(21 <= f <= 39 for f in fr)
    assert fr == sorted(fr)


def test_camera_progress_reports_verification_phase_and_reaches_total():
    source = [cam(f, center=(0.0, float(f) if f <= 15 else float(30 - f), 0.0)) for f in range(41)]
    calls = []
    camera_track(source, [(0, 20), (30, 40)], progress=lambda d, t, note: calls.append((d, t, note)))
    assert all(t == 30 for _, t, _ in calls)
    assert {note for _, _, note in calls} == {"", "出力後検証"}
    assert calls[-1][0] == 30


def test_camera_diagnostics_fields():
    source = [cam(f, fov=30 + (15 - abs(f - 15))) for f in range(31)]
    diag = {}
    camera_track(source, [(0, 30)], curve_mode="linear", diagnostics=diag)
    assert set(diag) == {
        "cuts", "splits", "maxspan_caps", "seam_rewrites", "verify", "fit_counts", "fit_counts_by_channel",
    }
    assert any(s["channel"] == "fov" for s in diag["splits"])
    assert all(set(s) == {"frame", "channel", "norm_error"} for s in diag["splits"])
    (record,) = diag["verify"]
    assert record["range"] == [0, 30]
    assert len(record["bad_counts"]) == len(record["added_counts"]) == record["iterations"]
    assert record["bad_counts"][-1] == 0
    assert record["added_total"] == sum(record["added_counts"])


def test_camera_diagnostics_cuts_include_perspective_switch_even_without_cut_detection():
    source = [cam(f, center=(0.0 if f < 5 else 50.0, 0.0, 0.0), persp=0 if f < 10 else 1) for f in range(21)]
    diag = {}
    camera_track(source, [(0, 20)], no_cut_detect=True, diagnostics=diag)
    assert diag["cuts"] == [10]


def test_camera_bezier_diagnostics_record_lower_seam_rewrite():
    source = [cam(0), cam(10, center=(10.0, 0.0, 0.0)), cam(20, center=(20.0, 0.0, 0.0))]
    diag = {}
    camera_track(source, [(10, 20)], curve_mode="bezier", diagnostics=diag)
    assert diag["seam_rewrites"] == [10]


def test_measure_camera_errors_reports_each_channel_max():
    source = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(11)]
    output = [cam(0, dist=-31.0, center=(0.0, 0.0, 0.0), rot=(0.0, math.radians(2.0), 0.0), fov=32),
              cam(10, dist=-31.0, center=(10.0, 0.0, 0.5), rot=(0.0, math.radians(2.0), 0.0), fov=32)]
    m = measure_camera_errors(source, output, [(0, 10)])
    assert m["pos_x"] == pytest.approx(0.0, abs=1e-6)
    assert m["pos_z"] == pytest.approx(0.5)
    assert m["distance"] == pytest.approx(1.0)
    assert m["fov"] == pytest.approx(2.0)
    assert m["rot_deg"] == pytest.approx(2.0)


def test_measure_bone_errors_reports_axis_max_and_rotation_degrees():
    half = math.radians(3.0) / 2.0
    source = [bone("c", f, pos=(float(f), 0.0, 0.0)) for f in range(11)]
    output = [bone("c", 0, pos=(0.0, 0.25, 0.0), rot=(0.0, 0.0, math.sin(half), math.cos(half))),
              bone("c", 10, pos=(10.0, 0.25, 0.0), rot=(0.0, 0.0, math.sin(half), math.cos(half)))]
    m = measure_bone_errors(source, output, [(0, 10)])
    assert m["pos_x"] == pytest.approx(0.0, abs=1e-6)
    assert m["pos_y"] == pytest.approx(0.25)
    assert m["rot_deg"] == pytest.approx(3.0)


def bone_track(source, ranges, **kw):
    opts = dict(
        cut_thresholds=BONE_CUT,
        keep_frames=[],
        no_cut_detect=False,
        min_seg=1,
        max_seg=180,
        strict=False,
    )
    opts.update(kw)
    return reduce_bone_track(source, ranges, TOLS, **opts)


def test_bone_linear_reduces_to_endpoints():
    source = [bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
    keys = bone_track(source, [(0, 30)])
    assert frames(keys) == [0, 30]
    assert all(k.name == "センター" for k in keys)


def test_bone_peak_kept():
    source = []
    for f in range(31):
        y = float(f) if f <= 15 else float(30 - f)
        source.append(bone("頭", f, pos=(0.0, y, 0.0)))
    keys = bone_track(source, [(0, 30)])
    assert frames(keys) == [0, 15, 30]


def test_bone_cut_detected_preserved_as_adjacent_jump():
    source = [bone("センター", f, pos=(0.0, 0.0 if f < 10 else 5.0, 0.0)) for f in range(21)]
    keys = bone_track(source, [(0, 20)])
    fr = frames(keys)
    assert 9 in fr and 10 in fr


def test_bone_keep_frame_forced():
    source = [bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
    keys = bone_track(source, [(0, 30)], keep_frames=[12])
    assert 12 in frames(keys)


def test_bone_bezier_reduce_is_deterministic():
    source = [bone("c", f, pos=(math.sin(f / 4.0) * 3.0, 0.0, 0.0)) for f in range(31)]

    def run():
        keys = bone_track(source, [(0, 30)], curve_mode="bezier", no_cut_detect=True)
        return [(k.frame, k.position, k.rotation, k.interpolation) for k in keys]

    assert run() == run()
