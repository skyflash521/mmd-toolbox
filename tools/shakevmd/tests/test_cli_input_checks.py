import json
import math

import pytest

from shakevmd import cli
from vmd import io
from vmd.types import BoneKey, CameraKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6
_EXIT_FAILURE = 1


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, LINEAR, fov, persp)


def _write(path, keys, **doc_kwargs):
    io.write_file(VmdDocument(camera=list(keys), **doc_kwargs), str(path))
    return str(path)


def _machine_events(capsysbinary):
    text = capsysbinary.readouterr().out.decode("utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln]


def _bone(frame, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(b"bone".ljust(15, b"\x00"), frame, position, rotation, bytes(64))


_INVALID_CAMERA_KEYS = {
    "nan-position": cam(30, center=(math.nan, 0.0, 0.0)),
    "inf-position": cam(30, center=(0.0, math.inf, 0.0)),
    "nan-distance": cam(30, dist=math.nan),
    "inf-distance": cam(30, dist=-math.inf),
    "nan-rotation": cam(30, rot=(0.0, 0.0, math.nan)),
    "inf-rotation": cam(30, rot=(math.inf, 0.0, 0.0)),
}


@pytest.mark.parametrize("invalid_key", _INVALID_CAMERA_KEYS.values(), ids=list(_INVALID_CAMERA_KEYS))
@pytest.mark.parametrize("extra", [[], ["--dry-run"], ["--no-smooth"]], ids=["default", "dry-run", "no-smooth"])
def test_non_finite_camera_values_are_invalid_camera_values(tmp_path, capsysbinary, invalid_key, extra):
    out = tmp_path / "out.vmd"
    src = _write(tmp_path / "in.vmd", [cam(0), invalid_key, cam(60)])
    rc = cli.main([src, "-o", str(out), "--machine", *extra])
    assert rc == _EXIT_FAILURE
    e = _machine_events(capsysbinary)[-1]
    assert e["type"] == "error" and e["code"] == "invalid_camera_values"
    assert e["field"] == "input" and e["exit_code"] == _EXIT_FAILURE
    assert not out.exists()


def test_non_finite_camera_value_is_reported_in_plain_mode(tmp_path, capsys):
    out = tmp_path / "out.vmd"
    src = _write(tmp_path / "in.vmd", [cam(0), _INVALID_CAMERA_KEYS["nan-position"], cam(60)])
    assert cli.main([src, "-o", str(out)]) == _EXIT_FAILURE
    assert "invalid_camera_values" in capsys.readouterr().err
    assert not out.exists()


def test_non_finite_value_of_a_key_dropped_by_duplicate_removal_is_still_rejected(tmp_path, capsysbinary):
    src = _write(tmp_path / "in.vmd", [cam(0), cam(30, center=(math.nan, 0.0, 0.0)), cam(30), cam(60)])
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == _EXIT_FAILURE
    assert _machine_events(capsysbinary)[-1]["code"] == "invalid_camera_values"


def test_non_finite_values_outside_the_camera_section_are_ignored(tmp_path):
    out = tmp_path / "out.vmd"
    src = _write(tmp_path / "in.vmd", [cam(0), cam(60, center=(5.0, 0.0, 0.0))],
                 bone=[_bone(0), _bone(1, position=(math.nan, 0.0, 0.0)), _bone(2, rotation=(0.0, 0.0, 0.0, 0.0))])
    assert cli.main([src, "-o", str(out)]) == 0
    assert out.exists()


@pytest.mark.parametrize("extra", [[], ["--dry-run"], ["--no-smooth"]], ids=["default", "dry-run", "no-smooth"])
def test_camera_keys_with_zero_angles_are_processed(tmp_path, capsysbinary, extra):
    out = tmp_path / "out.vmd"
    src = _write(tmp_path / "in.vmd", [cam(0), cam(30, center=(1.0, 0.0, 0.0)), cam(60)])
    assert cli.main([src, "-o", str(out), "--machine", *extra]) == 0
    events = _machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    assert not [e for e in events if e["type"] == "error"]
    assert out.exists() == ("--dry-run" not in extra)


def _write_duplicate_frame_input(path):
    return _write(path, [cam(0, center=(9.0, 0.0, 0.0)), cam(0, center=(0.0, 0.0, 0.0)), cam(60)])


def _duplicate_warnings(events):
    return [e for e in events if e["type"] == "warning" and e["code"] == "normalize-duplicate"]


@pytest.mark.parametrize("extra", [[], ["--dry-run"]], ids=["bake", "dry-run"])
def test_machine_surfaces_normalize_duplicate_warning(tmp_path, capsysbinary, extra):
    src = _write_duplicate_frame_input(tmp_path / "in.vmd")
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth", *extra]) == 0
    events = _machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    dup = _duplicate_warnings(events)
    assert len(dup) == 1
    assert isinstance(dup[0]["message"], str) and dup[0]["message"]


def test_machine_reports_the_duplicate_only_as_normalize_duplicate(tmp_path, capsysbinary):
    src = _write_duplicate_frame_input(tmp_path / "in.vmd")
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"]) == 0
    codes = [e["code"] for e in _machine_events(capsysbinary) if e["type"] == "warning"]
    assert "bake_normalize_duplicate" not in codes


def test_plain_mode_prints_normalize_duplicate_warning_line_on_stderr(tmp_path, capsys):
    src = _write_duplicate_frame_input(tmp_path / "in.vmd")
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--no-smooth"]) == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert any(ln.startswith("warning: normalize-duplicate: ") for ln in err.splitlines())


def test_later_key_wins_among_same_frame_duplicates(tmp_path):
    out = tmp_path / "out.vmd"
    src = _write_duplicate_frame_input(tmp_path / "in.vmd")
    assert cli.main([src, "-o", str(out), "--no-smooth", "--amp-rot", "0", "--amp-pos", "0"]) == 0
    first = io.read(str(out))[0].camera[0]
    assert first.frame == 0
    assert first.position == pytest.approx((0.0, 0.0, 0.0), abs=1e-4)


def test_unsorted_keys_without_duplicates_raise_no_warning(tmp_path, capsysbinary):
    src = _write(tmp_path / "in.vmd", [cam(60), cam(0), cam(30)])
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"]) == 0
    assert not [e for e in _machine_events(capsysbinary) if e["type"] == "warning"]


def test_input_without_duplicates_raises_no_normalize_warning(tmp_path, capsysbinary):
    src = _write(tmp_path / "in.vmd", [cam(0), cam(30), cam(60)])
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"]) == 0
    assert not _duplicate_warnings(_machine_events(capsysbinary))
