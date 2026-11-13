import json
import math

from mocapvmd import cli
from mocapvmd.model_profile import STANDARD_BONE_NAMES

from .helpers import bone, build_standard_pmx, morph, write_vmd

_EXIT_FAILURE = 1
_EXIT_ARG_ERROR = 2
_RAMP_LAST_FRAME = 10


def _machine_events(capsysbinary):
    text = capsysbinary.readouterr().out.decode("utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln]


def _last_event(capsysbinary):
    return _machine_events(capsysbinary)[-1]


def _write_ramp(path):
    write_vmd(path, bone=[bone("センター", f, pos=(0.1 * f, 0.0, 0.0)) for f in range(_RAMP_LAST_FRAME + 1)])


def test_input_without_bone_keys_is_no_bone_keys(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, morph=[morph("あ", 0, 0.0), morph("あ", 5, 1.0)])
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == _EXIT_FAILURE
    e = _last_event(capsysbinary)
    assert e["type"] == "error" and e["code"] == "no_bone_keys"
    assert e["field"] == "input" and e["exit_code"] == _EXIT_FAILURE
    assert not out.exists()


def test_missing_pmx_is_pmx_not_file(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_ramp(src)
    rc = cli.main([str(src), "-o", str(out), "--machine", "--pmx", str(tmp_path / "nope.pmx"),
                   "--denoise-mode", "pose"])
    assert rc == _EXIT_FAILURE
    e = _last_event(capsysbinary)
    assert e["type"] == "error" and e["code"] == "pmx_not_file"
    assert e["field"] == "--pmx" and e["exit_code"] == _EXIT_FAILURE
    assert not out.exists()


def test_pmx_given_as_directory_is_pmx_not_file_in_bone_mode(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_ramp(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine", "--pmx", str(tmp_path)])
    assert rc == _EXIT_FAILURE
    assert _last_event(capsysbinary)["code"] == "pmx_not_file"


def test_existing_pmx_is_accepted_in_bone_mode(tmp_path):
    src = tmp_path / "in.vmd"
    pmx = tmp_path / "model.pmx"
    _write_ramp(src)
    pmx.write_bytes(build_standard_pmx(list(STANDARD_BONE_NAMES.values())))
    assert cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--pmx", str(pmx)]) == 0


def test_pose_mode_rejects_clean_strength_other_than_default(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_ramp(src)
    rc = cli.main([str(src), "-o", str(out), "--machine", "--denoise-mode", "pose",
                   "--clean-strength", "1.5"])
    assert rc == _EXIT_ARG_ERROR
    e = _last_event(capsysbinary)
    assert e["type"] == "error" and e["code"] == "bad_argument"
    assert e["field"] == "--clean-strength" and e["exit_code"] == _EXIT_ARG_ERROR
    assert not out.exists()


def test_pose_mode_accepts_explicit_default_clean_strength(tmp_path):
    src = tmp_path / "in.vmd"
    _write_ramp(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose",
                   "--clean-strength", "1.0"])
    assert rc == 0


def test_bone_mode_accepts_clean_strength_other_than_default(tmp_path):
    src = tmp_path / "in.vmd"
    _write_ramp(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "bone",
                   "--clean-strength", "0.5"])
    assert rc == 0


def test_non_finite_bone_value_is_rejected_as_invalid_bone_values(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    nan_position_key = bone("センター", 1, pos=(math.nan, 0.0, 0.0))
    write_vmd(src, bone=[bone("センター", 0), nan_position_key, bone("センター", 2, pos=(1.0, 0.0, 0.0))])
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == _EXIT_FAILURE
    e = _last_event(capsysbinary)
    assert e["type"] == "error" and e["code"] == "invalid_bone_values"
    assert e["field"] == "input" and e["exit_code"] == _EXIT_FAILURE
    assert not out.exists()


def test_near_zero_norm_rotation_is_invalid_bone_values(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 1, rot=(1e-12, 0.0, 0.0, 0.0)),
                         bone("センター", 2, pos=(1.0, 0.0, 0.0))])
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == _EXIT_FAILURE
    e = _last_event(capsysbinary)
    assert e["type"] == "error" and e["code"] == "invalid_bone_values"
    assert e["field"] == "input" and e["exit_code"] == _EXIT_FAILURE
    assert not out.exists()


def _write_duplicate_frame_input(path):
    write_vmd(path, bone=[
        bone("センター", 0, pos=(9.0, 0.0, 0.0)),
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 4, pos=(2.0, 0.0, 0.0)),
    ])


def test_machine_surfaces_normalize_duplicate_warning(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_duplicate_frame_input(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine", "--no-reduce"])
    assert rc == 0
    events = _machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    dup = [e for e in events if e["type"] == "warning" and e["code"] == "normalize-duplicate"]
    assert len(dup) >= 1
    assert isinstance(dup[0]["message"], str) and dup[0]["message"]


def test_plain_mode_prints_normalize_duplicate_warning_line_on_stderr(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    _write_duplicate_frame_input(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--no-reduce"])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert any(ln.startswith("warning: normalize-duplicate: ") for ln in err.splitlines())


def test_reordering_unsorted_keys_emits_no_warning(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", f, pos=(0.1 * f, 0.0, 0.0)) for f in reversed(range(_RAMP_LAST_FRAME + 1))])
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    assert not [e for e in _machine_events(capsysbinary) if e["type"] == "warning"]
