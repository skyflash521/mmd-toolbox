import dataclasses
import json

import pytest

from shakevmd import bake as bake_mod
from shakevmd import cli
from vmd import interp, io
from vmd.types import CameraKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    return CameraKey(frame, dist, center, rot, LINEAR, 30, 0)


SMOOTH_KEYS = [
    cam(0),
    cam(30, dist=-25.0, center=(10.0, 5.0, 2.0), rot=(0.2, 0.1, 0.0)),
    cam(60, dist=-20.0, center=(20.0, 0.0, -3.0), rot=(-0.1, 0.3, 0.05)),
]

CUT_AT_30_KEYS = [cam(0), cam(29), cam(30, center=(40.0, 0.0, 0.0)), cam(60, center=(40.0, 0.0, 0.0))]


def _expected_max_amplitude(source_keys, result):
    source = sorted(source_keys, key=lambda k: k.frame)
    applied = {f for a, b in result.resolved for f in range(a, b + 1)}
    largest = 0.0
    for k in result.camera_keys:
        if k.frame not in applied:
            continue
        s = interp.sample_camera(source, k.frame)
        for i in range(3):
            largest = max(largest, abs(k.rotation[i] - s["rotation"][i]), abs(k.position[i] - s["position"][i]))
    return largest


def _machine_events(capsysbinary):
    text = capsysbinary.readouterr().out.decode("utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln]


def test_bake_result_has_detected_cuts_and_max_amplitude_fields():
    names = {f.name for f in dataclasses.fields(bake_mod.BakeResult)}
    assert {"detected_cuts", "max_amplitude"} <= names


def test_detected_cuts_lists_the_frames_where_a_cut_starts():
    result = bake_mod.bake(CUT_AT_30_KEYS, seed=1)
    assert result.detected_cuts == [30]


def test_detected_cuts_follow_the_cut_thresholds_given_to_bake():
    result = bake_mod.bake(CUT_AT_30_KEYS, seed=1, cut_pos_threshold=1000.0, cut_rot_threshold=1000.0)
    assert result.detected_cuts == []


def test_detected_cuts_exclude_manual_cuts():
    result = bake_mod.bake(SMOOTH_KEYS, seed=1, manual_cuts_add=[20])
    assert result.detected_cuts == []


def test_max_amplitude_is_the_largest_component_deviation_inside_applied_ranges():
    result = bake_mod.bake(SMOOTH_KEYS, ranges=[(10, 50)], seed=1, amp_rot=2.0, amp_pos=0.3)
    assert isinstance(result.max_amplitude, float)
    assert result.max_amplitude > 0.0
    assert result.max_amplitude == pytest.approx(_expected_max_amplitude(SMOOTH_KEYS, result), rel=1e-9)


def test_max_amplitude_ignores_the_deviation_outside_applied_ranges(monkeypatch):
    keys = [cam(f, center=(0.1 * f, 0.0, 0.0), rot=(0.0, 0.005 * f, 0.0)) for f in range(0, 61, 10)]
    expected = _expected_max_amplitude(
        keys, bake_mod.bake(keys, ranges=[(20, 40)], seed=1, amp_rot=0.5, amp_pos=0.05, fade_sec=0.0))
    real_sample = interp.sample_camera

    def sample_displaced_outside_applied_range(keys, frame):
        s = real_sample(keys, frame)
        if frame in (0, 10, 50, 60):
            s = {**s, "position": (s["position"][0] + 100.0, *s["position"][1:])}
        return s

    monkeypatch.setattr(interp, "sample_camera", sample_displaced_outside_applied_range)
    result = bake_mod.bake(keys, ranges=[(20, 40)], seed=1, amp_rot=0.5, amp_pos=0.05, fade_sec=0.0)
    assert result.resolved == [(20, 40)]
    assert 0.0 < expected < 1.0
    assert result.max_amplitude == pytest.approx(expected, rel=1e-9)


def test_max_amplitude_is_zero_without_any_shake():
    result = bake_mod.bake(SMOOTH_KEYS, seed=1, amp_rot=0.0, amp_pos=0.0)
    assert result.max_amplitude == pytest.approx(0.0, abs=1e-9)


def _stub_bake_result_fields(monkeypatch):
    real_bake = cli.bake

    def stub(camera_keys, *args, **kwargs):
        return dataclasses.replace(real_bake(camera_keys, *args, **kwargs), detected_cuts=[7], max_amplitude=123.0)

    monkeypatch.setattr(cli, "bake", stub)


def _write_input(path, keys=SMOOTH_KEYS):
    io.write_file(VmdDocument(camera=list(keys)), str(path))
    return str(path)


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "bake"])
def test_machine_result_reports_the_cuts_and_amplitude_carried_by_bake_result(
        tmp_path, capsysbinary, monkeypatch, dry_run):
    _stub_bake_result_fields(monkeypatch)
    extra = ["--dry-run"] if dry_run else ["--no-smooth", "-o", str(tmp_path / "out.vmd")]
    assert cli.main([_write_input(tmp_path / "in.vmd"), "--machine", *extra]) == 0
    result = _machine_events(capsysbinary)[-1]
    assert result["detected_cuts"] == [7]
    assert result["max_amplitude"] == 123.0


def test_plain_dry_run_prints_the_cuts_and_amplitude_carried_by_bake_result(tmp_path, capsys, monkeypatch):
    _stub_bake_result_fields(monkeypatch)
    assert cli.main([_write_input(tmp_path / "in.vmd"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "max amplitude: 123" in out
    assert "cuts: [7]" in out
