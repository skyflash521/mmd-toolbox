import json
import math

import pytest

from sparsevmd import cli
from vmd import io
from vmd.types import (
    BoneKey,
    CameraKey,
    IkBone,
    IkPropertyKey,
    LightKey,
    MorphKey,
    SelfShadowKey,
    VmdDocument,
)

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
_EXIT_FAILURE = 1
_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_BONE_NAME = "センター".encode("cp932").ljust(15, b"\x00")


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, 30, 0)


def bone(frame, pos=(0.0, 0.0, 0.0), rot=_IDENTITY):
    return BoneKey(_BONE_NAME, frame, pos, rot, BL)


def _write(path, **sections):
    io.write_file(VmdDocument(**sections), str(path))
    return str(path)


def _ramp_cameras():
    return [cam(f, center=(float(f), 0.0, 0.0)) for f in range(11)]


def _ramp_bones():
    return [bone(f, pos=(0.0, float(f), 0.0)) for f in range(11)]


def _machine_events(capsysbinary):
    text = capsysbinary.readouterr().out.decode("utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln]


def _section_bytes(path, section):
    doc, _ = io.read(str(path))
    return io.write(VmdDocument(**{section: getattr(doc, section)}))


def _assert_sections_unchanged(src, out, sections):
    for section in sections:
        assert _section_bytes(out, section) == _section_bytes(src, section), section


_MODES = {
    "default": [],
    "dry-run": ["--dry-run"],
    "linear-no-cut-detect": ["--curve-mode", "linear", "--no-cut-detect"],
}

_INVALID_BONE_KEYS = {
    "nan-position": bone(5, pos=(math.nan, 0.0, 0.0)),
    "inf-position": bone(5, pos=(0.0, math.inf, 0.0)),
    "nan-rotation": bone(5, rot=(0.0, 0.0, math.nan, 1.0)),
    "inf-rotation": bone(5, rot=(0.0, -math.inf, 0.0, 1.0)),
    "zero-norm-rotation": bone(5, rot=(0.0, 0.0, 0.0, 0.0)),
}

_INVALID_CAMERA_KEYS = {
    "nan-position": cam(5, center=(math.nan, 0.0, 0.0)),
    "inf-position": cam(5, center=(0.0, math.inf, 0.0)),
    "nan-distance": cam(5, dist=math.nan),
    "inf-distance": cam(5, dist=-math.inf),
    "nan-rotation": cam(5, rot=(0.0, 0.0, math.nan)),
    "inf-rotation": cam(5, rot=(math.inf, 0.0, 0.0)),
}


def _assert_input_error(capsysbinary, code):
    e = _machine_events(capsysbinary)[-1]
    assert e["type"] == "error" and e["code"] == code
    assert e["field"] == "input" and e["exit_code"] == _EXIT_FAILURE


@pytest.mark.parametrize("invalid_key", _INVALID_BONE_KEYS.values(), ids=list(_INVALID_BONE_KEYS))
@pytest.mark.parametrize("extra", _MODES.values(), ids=list(_MODES))
def test_invalid_bone_values_are_rejected_even_with_valid_camera(tmp_path, capsysbinary, invalid_key, extra):
    out = tmp_path / "out.vmd"
    bones = [k for k in _ramp_bones() if k.frame != 5] + [invalid_key]
    src = _write(tmp_path / "in.vmd", camera=_ramp_cameras(), bone=bones)
    assert cli.main([src, "-o", str(out), "--machine", *extra]) == _EXIT_FAILURE
    _assert_input_error(capsysbinary, "invalid_bone_values")
    assert not out.exists()


@pytest.mark.parametrize("invalid_key", _INVALID_CAMERA_KEYS.values(), ids=list(_INVALID_CAMERA_KEYS))
@pytest.mark.parametrize("extra", _MODES.values(), ids=list(_MODES))
def test_invalid_camera_values_are_rejected_even_with_valid_bones(tmp_path, capsysbinary, invalid_key, extra):
    out = tmp_path / "out.vmd"
    cameras = [k for k in _ramp_cameras() if k.frame != 5] + [invalid_key]
    src = _write(tmp_path / "in.vmd", camera=cameras, bone=_ramp_bones())
    assert cli.main([src, "-o", str(out), "--machine", *extra]) == _EXIT_FAILURE
    _assert_input_error(capsysbinary, "invalid_camera_values")
    assert not out.exists()


@pytest.mark.parametrize("target, section_keys, code", [
    ("bone", {"bone": [*_ramp_bones(), _INVALID_BONE_KEYS["nan-position"]]}, "invalid_bone_values"),
    ("camera", {"camera": [*_ramp_cameras(), _INVALID_CAMERA_KEYS["nan-position"]]}, "invalid_camera_values"),
], ids=["bone-only", "camera-only"])
def test_invalid_values_are_rejected_when_the_target_is_the_invalid_section(
        tmp_path, capsysbinary, target, section_keys, code):
    out = tmp_path / "out.vmd"
    src = _write(tmp_path / "in.vmd", **section_keys)
    assert cli.main([src, "-o", str(out), "--machine", "--target", target]) == _EXIT_FAILURE
    _assert_input_error(capsysbinary, code)
    assert not out.exists()


def test_invalid_values_in_both_sections_are_still_an_input_error(tmp_path, capsysbinary):
    out = tmp_path / "out.vmd"
    src = _write(tmp_path / "in.vmd",
                 camera=[*_ramp_cameras(), _INVALID_CAMERA_KEYS["nan-distance"]],
                 bone=[*_ramp_bones(), _INVALID_BONE_KEYS["nan-position"]])
    assert cli.main([src, "-o", str(out), "--machine"]) == _EXIT_FAILURE
    e = _machine_events(capsysbinary)[-1]
    assert e["type"] == "error" and e["code"] in ("invalid_bone_values", "invalid_camera_values")
    assert e["field"] == "input" and e["exit_code"] == _EXIT_FAILURE
    assert not out.exists()


@pytest.mark.parametrize("invalid_key", _INVALID_BONE_KEYS.values(), ids=list(_INVALID_BONE_KEYS))
def test_invalid_bone_values_are_ignored_when_target_is_camera(tmp_path, capsysbinary, invalid_key):
    out = tmp_path / "out.vmd"
    bones = [k for k in _ramp_bones() if k.frame != 5] + [invalid_key]
    src = _write(tmp_path / "in.vmd", camera=_ramp_cameras(), bone=bones)
    assert cli.main([src, "-o", str(out), "--machine", "--target", "camera"]) == 0
    events = _machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    assert not [e for e in events if e["type"] == "error"]
    _assert_sections_unchanged(src, out, ["bone"])


@pytest.mark.parametrize("invalid_key", _INVALID_CAMERA_KEYS.values(), ids=list(_INVALID_CAMERA_KEYS))
def test_invalid_camera_values_are_ignored_when_target_is_bone(tmp_path, capsysbinary, invalid_key):
    out = tmp_path / "out.vmd"
    cameras = [k for k in _ramp_cameras() if k.frame != 5] + [invalid_key]
    src = _write(tmp_path / "in.vmd", camera=cameras, bone=_ramp_bones())
    assert cli.main([src, "-o", str(out), "--machine", "--target", "bone"]) == 0
    events = _machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    assert not [e for e in events if e["type"] == "error"]
    _assert_sections_unchanged(src, out, ["camera"])


@pytest.mark.parametrize("section_keys", [
    {"bone": [*_ramp_bones(), _INVALID_BONE_KEYS["nan-rotation"]]},
    {"camera": [*_ramp_cameras(), _INVALID_CAMERA_KEYS["nan-rotation"]]},
], ids=["bone", "camera"])
def test_invalid_values_are_reported_as_one_error_line_in_plain_mode(tmp_path, capsys, section_keys):
    out = tmp_path / "out.vmd"
    src = _write(tmp_path / "in.vmd", **section_keys)
    assert cli.main([src, "-o", str(out)]) == _EXIT_FAILURE
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1
    assert "Traceback" not in err
    assert not out.exists()


def test_non_finite_bone_value_of_a_key_dropped_by_duplicate_removal_is_still_rejected(tmp_path, capsysbinary):
    bones = [bone(0), bone(5, pos=(math.nan, 0.0, 0.0)), bone(5), bone(10)]
    src = _write(tmp_path / "in.vmd", bone=bones)
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"]) == _EXIT_FAILURE
    _assert_input_error(capsysbinary, "invalid_bone_values")


def test_non_finite_camera_value_of_a_key_dropped_by_duplicate_removal_is_still_rejected(tmp_path, capsysbinary):
    cameras = [cam(0), cam(5, center=(math.nan, 0.0, 0.0)), cam(5), cam(10)]
    src = _write(tmp_path / "in.vmd", camera=cameras)
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"]) == _EXIT_FAILURE
    _assert_input_error(capsysbinary, "invalid_camera_values")


@pytest.mark.parametrize("extra", _MODES.values(), ids=list(_MODES))
def test_invalid_values_outside_bone_and_camera_sections_are_ignored(tmp_path, extra):
    out = tmp_path / "out.vmd"
    name = "あ".encode("cp932").ljust(15, b"\x00")
    src = _write(
        tmp_path / "in.vmd",
        camera=_ramp_cameras(), bone=_ramp_bones(),
        morph=[MorphKey(name, 0, 0.0), MorphKey(name, 3, math.nan), MorphKey(name, 6, math.inf)],
        light=[LightKey(0, (math.nan, 0.6, 0.6), (-0.5, math.inf, 0.5))],
        self_shadow=[SelfShadowKey(0, 1, math.nan)],
        ik_property=[IkPropertyKey(0, 1, [IkBone("左足ＩＫ".encode("cp932").ljust(20, b"\x00"), 1)])],
    )
    assert cli.main([src, "-o", str(out), *extra]) == 0
    assert out.exists() == ("--dry-run" not in extra)
    if out.exists():
        _assert_sections_unchanged(src, out, ["morph", "light", "self_shadow", "ik_property"])


@pytest.mark.parametrize("extra", _MODES.values(), ids=list(_MODES))
def test_camera_keys_with_zero_angles_are_processed(tmp_path, capsysbinary, extra):
    out = tmp_path / "out.vmd"
    cameras = _ramp_cameras()
    assert all(k.rotation == (0.0, 0.0, 0.0) for k in cameras)
    src = _write(tmp_path / "in.vmd", camera=cameras)
    assert cli.main([src, "-o", str(out), "--machine", "--target", "camera", *extra]) == 0
    events = _machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    assert not [e for e in events if e["type"] == "error"]
    assert out.exists() == ("--dry-run" not in extra)


def test_zero_angle_camera_is_valid_while_zero_norm_bone_rotation_is_not(tmp_path, capsysbinary):
    zero_angle_cameras = _ramp_cameras()
    assert all(k.rotation == (0.0, 0.0, 0.0) for k in zero_angle_cameras)
    ok = _write(tmp_path / "ok.vmd", camera=zero_angle_cameras, bone=_ramp_bones())
    assert cli.main([ok, "-o", str(tmp_path / "ok_out.vmd"), "--machine"]) == 0
    assert _machine_events(capsysbinary)[-1]["type"] == "result"

    bad_bones = [k for k in _ramp_bones() if k.frame != 5] + [_INVALID_BONE_KEYS["zero-norm-rotation"]]
    bad = _write(tmp_path / "bad.vmd", camera=zero_angle_cameras, bone=bad_bones)
    assert cli.main([bad, "-o", str(tmp_path / "bad_out.vmd"), "--machine"]) == _EXIT_FAILURE
    _assert_input_error(capsysbinary, "invalid_bone_values")


def _duplicate_cameras():
    return [cam(0, center=(9.0, 0.0, 0.0)), cam(0), cam(5, center=(9.0, 0.0, 0.0)), cam(5), cam(10)]


def _duplicate_bones():
    return [bone(0, pos=(9.0, 0.0, 0.0)), bone(0), bone(5, pos=(9.0, 0.0, 0.0)), bone(5), bone(10)]


def _duplicate_warnings(events):
    return [e for e in events if e["type"] == "warning" and e["code"] == "normalize-duplicate"]


@pytest.mark.parametrize("section, keys", [("camera", _duplicate_cameras), ("bone", _duplicate_bones)])
@pytest.mark.parametrize("extra", [[], ["--dry-run"]], ids=["reduce", "dry-run"])
def test_machine_surfaces_one_duplicate_warning_per_section(tmp_path, capsysbinary, section, keys, extra):
    src = _write(tmp_path / "in.vmd", **{section: keys()})
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine", "--target", section, *extra]) == 0
    events = _machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    dup = _duplicate_warnings(events)
    assert len(dup) == 1
    assert dup[0]["section"] == [section]
    assert isinstance(dup[0]["message"], str) and dup[0]["message"]
    assert all(i < len(events) - 1 for i, e in enumerate(events) if e["type"] == "warning")


def test_machine_surfaces_duplicate_warnings_of_both_sections_separately(tmp_path, capsysbinary):
    src = _write(tmp_path / "in.vmd", camera=_duplicate_cameras(), bone=_duplicate_bones())
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"]) == 0
    dup = _duplicate_warnings(_machine_events(capsysbinary))
    assert sorted(tuple(e["section"]) for e in dup) == [("bone",), ("camera",)]


def test_plain_mode_prints_duplicate_warning_line_on_stderr(tmp_path, capsys):
    src = _write(tmp_path / "in.vmd", camera=_duplicate_cameras())
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--target", "camera"]) == 0
    out, err = capsys.readouterr()
    assert out == ""
    lines = [ln for ln in err.splitlines() if ln.startswith("warning: normalize-duplicate: ")]
    assert len(lines) == 1 and lines[0].endswith("(camera)")


def test_unsorted_keys_without_duplicates_raise_no_warning(tmp_path, capsysbinary):
    cameras = list(reversed(_ramp_cameras()))
    bones = list(reversed(_ramp_bones()))
    src = _write(tmp_path / "in.vmd", camera=cameras, bone=bones)
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"]) == 0
    assert not [e for e in _machine_events(capsysbinary) if e["type"] == "warning"]


def test_input_without_duplicates_raises_no_duplicate_warning(tmp_path, capsys):
    src = _write(tmp_path / "in.vmd", camera=_ramp_cameras(), bone=_ramp_bones())
    assert cli.main([src, "-o", str(tmp_path / "out.vmd")]) == 0
    assert "normalize-duplicate" not in capsys.readouterr().err


def test_later_bone_key_wins_among_same_frame_duplicates(tmp_path):
    out = tmp_path / "out.vmd"
    src = _write(tmp_path / "in.vmd", bone=_duplicate_bones())
    assert cli.main([src, "-o", str(out), "--target", "bone", "--curve-mode", "linear"]) == 0
    doc, _ = io.read(str(out))
    assert [k.frame for k in doc.bone] == [0, 10]
    assert all(k.position == pytest.approx((0.0, 0.0, 0.0)) for k in doc.bone)


def test_later_camera_key_wins_among_same_frame_duplicates(tmp_path):
    out = tmp_path / "out.vmd"
    src = _write(tmp_path / "in.vmd", camera=_duplicate_cameras())
    assert cli.main([src, "-o", str(out), "--target", "camera", "--curve-mode", "linear"]) == 0
    doc, _ = io.read(str(out))
    assert [k.frame for k in doc.camera] == [0, 10]
    assert all(k.position == pytest.approx((0.0, 0.0, 0.0)) for k in doc.camera)


def test_machine_result_counts_input_keys_before_duplicate_removal(tmp_path, capsysbinary):
    src = _write(tmp_path / "in.vmd", camera=_duplicate_cameras(), bone=_duplicate_bones())
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"]) == 0
    result = _machine_events(capsysbinary)[-1]
    assert result["type"] == "result"
    assert result["camera"]["input_keys"] == len(_duplicate_cameras())
    assert result["bone"]["input_keys"] == len(_duplicate_bones())


@pytest.mark.parametrize("target, section, keys", [
    ("camera", "bone", _duplicate_bones),
    ("bone", "camera", _duplicate_cameras),
], ids=["camera-target-bone-duplicates", "bone-target-camera-duplicates"])
def test_duplicates_in_sections_outside_the_target_raise_no_warning(
        tmp_path, capsysbinary, target, section, keys):
    other = "camera" if section == "bone" else "bone"
    other_keys = _ramp_cameras() if other == "camera" else _ramp_bones()
    out = tmp_path / "out.vmd"
    src = _write(tmp_path / "in.vmd", **{section: keys(), other: other_keys})
    assert cli.main([src, "-o", str(out), "--machine", "--target", target]) == 0
    events = _machine_events(capsysbinary)
    assert not _duplicate_warnings(events)
    _assert_sections_unchanged(src, out, [section])


@pytest.mark.parametrize("extra", [[], ["--dry-run"]], ids=["reduce", "dry-run"])
def test_machine_counts_input_keys_before_duplicate_removal_in_every_output(tmp_path, capsysbinary, extra):
    src = _write(tmp_path / "in.vmd", camera=_duplicate_cameras(), bone=_duplicate_bones())
    assert cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine", *extra]) == 0
    result = _machine_events(capsysbinary)[-1]
    assert result["type"] == "result"
    assert result["camera"]["input_keys"] == len(_duplicate_cameras())
    if "--dry-run" in extra:
        assert result["keys"] == {"camera": len(_duplicate_cameras()), "bone": len(_duplicate_bones())}
        assert [b["input_keys"] for b in result["bones"]] == [len(_duplicate_bones())]
    else:
        assert result["bone"]["input_keys"] == len(_duplicate_bones())
