import json

import pytest

from mocapvmd import cli, presets
from mocapvmd.model_profile import STANDARD_BONE_NAMES
from vmd import io
from vmd.reduce import BONE_LINEAR_INTERP
from vmd.types import BoneKey, VmdDocument

from .helpers import bone, build_standard_pmx, write_vmd

_RAMP_FRAME_COUNT = 11
_RAMP_LAST_FRAME = _RAMP_FRAME_COUNT - 1
_VMD_FPS = 30.0
_CP932_UNDECODABLE_BONE_NAME = b"\x81\x20name".ljust(15, b"\x00")

_META_OPTIONS = ("--describe", "--version", "--help", "--machine")
_NEGATED_FLAGS = ("--no-denoise", "--no-foot-ik-stabilize", "--no-reduce")
_DESCRIBED_OPTION_COUNT = 18
_NON_NEGATIVE_UNBOUNDED = {"min": 0, "max": None, "exclusive_min": False}
_EXPECTED_TYPE_CONSTRAINT_DEFAULT_BY_OPTION = {
    "input": ("str", None, None),
    "--output": ("str", None, None),
    "--overwrite": ("flag", None, False),
    "--preset": ("enum", {"choices": list(presets.PRESET_NAMES)}, "medium"),
    "--clean-strength": ("float", _NON_NEGATIVE_UNBOUNDED, 1.0),
    "--denoise": ("flag", None, True),
    "--denoise-mode": ("enum", {"choices": ["bone", "pose"]}, "bone"),
    "--pmx": ("str", None, None),
    "--foot-ik-stabilize": ("flag", None, True),
    "--foot-slide-suppression": ("float", {"min": 0, "max": 1, "exclusive_min": False}, 1.0),
    "--reduce-error-bone-pos": ("float", _NON_NEGATIVE_UNBOUNDED, None),
    "--reduce-error-bone-rot": ("float", _NON_NEGATIVE_UNBOUNDED, None),
    "--curve-mode": ("enum", {"choices": ["bezier", "linear"]}, "bezier"),
    "--reduce": ("flag", None, True),
    "--list-bones": ("flag", None, False),
    "--dry-run": ("flag", None, False),
    "--quiet": ("flag", None, False),
    "--verbose": ("flag", None, False),
}
_EXPECTED_PRESET_BASE_TOLERANCES = {
    "slower": {"reduce_error_bone_pos": 0.05, "reduce_error_bone_rot": 0.40},
    "slow": {"reduce_error_bone_pos": 0.10, "reduce_error_bone_rot": 0.75},
    "medium": {"reduce_error_bone_pos": 0.20, "reduce_error_bone_rot": 1.50},
    "fast": {"reduce_error_bone_pos": 0.80, "reduce_error_bone_rot": 6.0},
    "faster": {"reduce_error_bone_pos": 1.60, "reduce_error_bone_rot": 12.0},
}


def machine_events(capsysbinary):
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln]


def _count_terminal_events(events):
    return sum(1 for e in events if e["type"] in ("result", "error"))


def machine_error(capsysbinary):
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "error"
    assert _count_terminal_events(events) == 1
    return events[-1]


def _write_center_linear_ramp(path):
    write_vmd(path, bone=[bone("センター", f, pos=(float(f), 0.0, 0.0))
                          for f in range(_RAMP_FRAME_COUNT)])


def _write_right_foot_ik_slow_grounded_drift(path):
    write_vmd(path, bone=[bone("右足ＩＫ", f, pos=(round(0.05 * f, 6), 0.0, 0.0)) for f in range(11)])


def _write_undecodable_name_track(path, frame_count):
    keys = [BoneKey(_CP932_UNDECODABLE_BONE_NAME, f, (float(f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                    BONE_LINEAR_INTERP)
            for f in range(frame_count)]
    io.write_file(VmdDocument(bone=keys), str(path))


def _raise_keyboard_interrupt(*a, **k):
    raise KeyboardInterrupt()


def test_machine_emits_single_process_result_counting_input_bone_keys(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    assert _count_terminal_events(events) == 1
    r = events[-1]
    assert r["mode"] == "process"
    assert r["output"] == str(out)
    assert r["input_keys"] == _RAMP_FRAME_COUNT
    assert isinstance(r["output_keys"], int) and r["output_keys"] >= 1
    assert out.exists()


def test_machine_stdout_is_newline_terminated_json_lines_without_blank_lines(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    text = raw.decode("utf-8")
    assert text.endswith("\n")
    for ln in text.split("\n")[:-1]:
        assert ln != ""
        obj = json.loads(ln)
        assert "type" in obj


def test_machine_stdout_lf_only_no_cr(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n")
    assert b"\r" not in raw


def test_machine_stdout_is_json_only_even_with_verbose(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine", "--verbose"])
    text = capsysbinary.readouterr().out.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln]
    assert lines
    for ln in lines:
        json.loads(ln)


def test_machine_default_emits_denoise_foot_ik_reduce_progress_each_with_one_fixed_start_event(
        tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_right_foot_ik_slow_grounded_drift(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "process"
    progress = [e for e in events if e["type"] == "progress"]
    assert progress
    stages = {p["stage"] for p in progress}
    assert {"denoise", "foot_ik", "reduce"} <= stages
    for p in progress:
        assert set(p) >= {"type", "stage", "done", "total", "note", "elapsed"}
    for stage in ("denoise", "foot_ik", "reduce"):
        starts = [p for p in progress if p["stage"] == stage and p["done"] == 0 and p["total"] is None]
        assert len(starts) == 1, f"{stage} 段の開始イベントは 1 本"
        assert starts[0]["note"] == "" and starts[0]["elapsed"] == 0.0


def test_machine_denoise_and_foot_ik_emit_only_start_event_and_reduce_ends_at_multikey_track_count(
        tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_right_foot_ik_slow_grounded_drift(src)
    multikey_track_count = 1
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    progress = [e for e in machine_events(capsysbinary) if e["type"] == "progress"]
    for stage in ("denoise", "foot_ik"):
        assert len([p for p in progress if p["stage"] == stage]) == 1, stage
    reduce_updates = [p for p in progress if p["stage"] == "reduce" and p["total"] is not None]
    assert (reduce_updates[-1]["done"], reduce_updates[-1]["total"]) == (
        multikey_track_count, multikey_track_count)
    assert all(p["note"] == "" for p in progress)


def test_machine_disabled_stages_emit_no_progress(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine",
                   "--no-denoise", "--no-foot-ik-stabilize"])
    assert rc == 0
    stages = {e["stage"] for e in machine_events(capsysbinary) if e["type"] == "progress"}
    assert "denoise" not in stages and "foot_ik" not in stages
    assert "reduce" in stages


def test_machine_passes_vmd_io_decode_error_through_as_warning_with_single_element_section(
        tmp_path, capsysbinary):
    _write_undecodable_name_track(tmp_path / "in.vmd", frame_count=4)
    rc = cli.main([str(tmp_path / "in.vmd"), "-o", str(tmp_path / "out.vmd"), "--machine", "--no-reduce"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    warns = [e for e in events if e["type"] == "warning"]
    w = next(w for w in warns if w["code"] == "decode-error")
    assert isinstance(w["section"], list) and len(w["section"]) == 1
    assert isinstance(w["message"], str) and w["message"]


def test_machine_dedups_identical_decode_error_warnings_across_frames_into_one(tmp_path, capsysbinary):
    _write_undecodable_name_track(tmp_path / "in.vmd", frame_count=6)
    rc = cli.main([str(tmp_path / "in.vmd"), "-o", str(tmp_path / "out.vmd"), "--machine", "--no-reduce"])
    assert rc == 0
    warns = [e for e in machine_events(capsysbinary) if e["type"] == "warning"]
    decode = [w for w in warns if w["code"] == "decode-error"]
    assert len(decode) == 1


def test_machine_error_unknown_option_field_is_first_unrecognized_token(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "--machine", "--bogus"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2
    assert e["field"] == "--bogus"
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_missing_input_is_bad_argument_with_field_input(capsysbinary):
    rc = cli.main(["--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "input"


def test_machine_error_non_numeric_value_is_bad_argument_with_long_option_field(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "--machine", "--clean-strength", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--clean-strength"


@pytest.mark.parametrize("opt,val", [
    pytest.param("--clean-strength", "-0.1", id="clean-strength-negative"),
    pytest.param("--clean-strength", "nan", id="clean-strength-nan"),
    pytest.param("--foot-slide-suppression", "1.5", id="foot-slide-suppression-above-1"),
    pytest.param("--foot-slide-suppression", "nan", id="foot-slide-suppression-nan"),
    pytest.param("--reduce-error-bone-pos", "-1", id="reduce-error-bone-pos-negative"),
    pytest.param("--reduce-error-bone-rot", "inf", id="reduce-error-bone-rot-inf"),
])
def test_machine_error_parsed_value_out_of_range_or_nonfinite_is_bad_argument_with_option_field(
        tmp_path, capsysbinary, opt, val):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "--machine", opt, val])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == opt and e["exit_code"] == 2


def test_machine_error_input_not_file(tmp_path, capsysbinary):
    rc = cli.main([str(tmp_path / "nope.vmd"), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "input_not_file" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_pmx_not_file(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "--machine", "--denoise-mode", "pose", "--pmx", str(tmp_path / "nope.pmx")])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "pmx_not_file" and e["field"] == "--pmx" and e["exit_code"] == 1


def test_machine_error_output_exists(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "-o", str(src), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output"


def test_machine_error_output_exists_distinct_path(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    out = tmp_path / "out.vmd"
    out.write_bytes(b"old content")
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output"


def test_machine_error_not_vmd(tmp_path, capsysbinary):
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file at all")
    rc = cli.main([str(bad), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "not_vmd" and e["field"] == "input" and e["exit_code"] == 1
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_nonfinite_bone_value_is_invalid_bone_values(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0, pos=(float("nan"), 0.0, 0.0)), bone("センター", 1)])
    rc = cli.main([str(src), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "invalid_bone_values" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_not_pmx(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 10, pos=(0.3, 0.0, 0.0))])
    pmx = tmp_path / "bad.pmx"
    pmx.write_bytes(b"NOTPMX")
    rc = cli.main([str(src), "--machine", "--denoise-mode", "pose", "--pmx", str(pmx)])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "not_pmx" and e["field"] == "--pmx" and e["exit_code"] == 1


def test_machine_error_pmx_missing_standard_bone_is_model_profile_invalid(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 10, pos=(0.3, 0.0, 0.0))])
    pmx = tmp_path / "model.pmx"
    names = [n for n in STANDARD_BONE_NAMES.values() if n != STANDARD_BONE_NAMES["wrist_r"]]
    pmx.write_bytes(build_standard_pmx(names))
    rc = cli.main([str(src), "--machine", "--denoise-mode", "pose", "--pmx", str(pmx)])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "model_profile_invalid" and e["field"] == "--pmx" and e["exit_code"] == 1


def test_machine_error_output_parent_is_file_is_write_failed_with_path(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    clash = tmp_path / "afile"
    clash.write_bytes(b"x")
    out = str(clash / "out.vmd")
    rc = cli.main([str(src), "-o", out, "--machine"])
    assert rc == 3
    e = machine_error(capsysbinary)
    assert e["code"] == "write_failed" and e["field"] == "--output" and e["exit_code"] == 3
    assert e["path"] == out


def test_machine_error_unexpected_exception_in_reduce_is_internal_error(tmp_path, capsysbinary, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(cli.reduce, "reduce_bones", boom)
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "internal_error" and e["exit_code"] == 1


def test_non_machine_error_prints_reason_to_stderr_and_no_json_to_stdout(tmp_path, capsys):
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file")
    rc = cli.main([str(bad)])
    assert rc == 1
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")


def test_machine_dry_run_emits_inspect_result_with_resolved_plan_without_writing(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "-o", str(out), "--machine", "--dry-run"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = events[-1]
    assert r["type"] == "result" and r["mode"] == "inspect"
    assert _count_terminal_events(events) == 1
    assert r["output"] is None and not out.exists()
    assert r["input_kind"] == "bone"
    assert r["keys"] == _RAMP_FRAME_COUNT
    assert r["frame_range"] == [0, _RAMP_LAST_FRAME]
    assert r["duration_sec"] == _RAMP_LAST_FRAME / _VMD_FPS
    assert isinstance(r["sections"], list) and "bone" in r["sections"]
    assert r["preset"] == "medium" and r["clean_strength"] == 1.0
    assert r["denoise"] is True and r["denoise_mode"] == "bone"
    assert r["foot_ik_stabilize"] is True and r["foot_slide_suppression"] == 1.0
    assert r["curve_mode"] == "bezier" and r["reduce"] is True
    assert isinstance(r["bones"], list) and r["bones"]
    b0 = r["bones"][0]
    assert set(b0) == {"name", "category", "keys", "frame_range"}
    assert b0["name"] == "センター" and b0["category"] == "center"
    assert b0["keys"] == _RAMP_FRAME_COUNT and b0["frame_range"] == [0, _RAMP_LAST_FRAME]
    assert isinstance(r["reduction"], dict) and "センター" in r["reduction"]
    red = r["reduction"]["センター"]
    assert set(red) == {"input_keys", "output_keys", "tol_pos", "tol_rot", "cuts", "errors"}
    assert set(red["errors"]) == {"pos_x", "pos_y", "pos_z", "rot_deg"}
    assert r["pose_denoise"] is None


def test_machine_dry_run_inspect_no_reduce_null_reduction(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "--machine", "--dry-run", "--no-reduce"])
    assert rc == 0
    r = machine_events(capsysbinary)[-1]
    assert r["mode"] == "inspect" and r["reduce"] is False and r["reduction"] is None


def test_machine_dry_run_inspect_pose_mode_reports_pose_denoise_with_default_profile(
        tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[
        bone("センター", 0), bone("センター", 10, pos=(0.3, 0.0, 0.0)),
        bone("頭", 0), bone("頭", 10, rot=(0.0, 0.0, 0.05, 0.99875)),
    ])
    rc = cli.main([str(src), "--machine", "--dry-run", "--denoise-mode", "pose"])
    assert rc == 0
    r = machine_events(capsysbinary)[-1]
    assert r["mode"] == "inspect"
    pd = r["pose_denoise"]
    assert set(pd) == {"pmx", "frames", "markers", "marker_displacement", "fit"}
    assert set(pd["markers"]) == {"available", "required_bones_ok"}
    assert set(pd["marker_displacement"]) == {"max", "mean"}
    assert set(pd["fit"]) == {"mean_error_before", "mean_error_after", "fallback_frames",
                              "max_bone_delta_deg", "max_center_delta"}
    assert pd["pmx"] is None


def test_machine_list_bones_lists_each_name_once_in_first_appearance_order_with_category(
        tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("右足ＩＫ", 0), bone("センター", 1)])
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--machine", "--list-bones"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = events[-1]
    assert r["type"] == "result" and r["mode"] == "list_bones"
    assert not out.exists()
    names = [b["name"] for b in r["bones"]]
    assert names == ["センター", "右足ＩＫ"]
    for b in r["bones"]:
        assert set(b) == {"name", "category"}
    cats = {b["name"]: b["category"] for b in r["bones"]}
    assert cats["センター"] == "center" and cats["右足ＩＫ"] == "foot_ik"


def test_machine_list_bones_not_blocked_by_invalid_values(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0, pos=(float("nan"), 0.0, 0.0))])
    rc = cli.main([str(src), "--machine", "--list-bones"])
    assert rc == 0
    r = machine_events(capsysbinary)[-1]
    assert r["mode"] == "list_bones"


def test_machine_list_bones_takes_precedence_over_dry_run(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 1)])
    rc = cli.main([str(src), "--machine", "--list-bones", "--dry-run"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert _count_terminal_events(events) == 1
    assert events[-1]["mode"] == "list_bones"


def test_machine_version_stays_human_text(capsys):
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mocapvmd" in out and not out.lstrip().startswith("{")


def test_machine_help_stays_human(capsys):
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


def test_help_lists_machine_reduce_verbose_flags(capsys):
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "--machine" in text and "--reduce" in text and "--verbose" in text


def test_machine_keyboard_interrupt_is_cancelled_error_130_without_output(tmp_path, capsysbinary, monkeypatch):
    monkeypatch.setattr(cli.reduce, "reduce_bones", _raise_keyboard_interrupt)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 130
    e = machine_error(capsysbinary)
    assert e["code"] == "cancelled" and e["exit_code"] == 130 and e["field"] is None
    assert not out.exists()


def test_non_machine_keyboard_interrupt_exits_130_with_stderr_reason_without_output(
        tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli.reduce, "reduce_bones", _raise_keyboard_interrupt)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_linear_ramp(src)
    rc = cli.main([str(src), "-o", str(out)])
    assert rc == 130
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")
    assert not out.exists()


def test_main_installs_ctrl_break_bridge_once(monkeypatch, capsysbinary):
    installs = []
    monkeypatch.setattr(cli, "install_sigbreak_handler", lambda: installs.append(True))
    assert cli.main(["--describe"]) == 0
    assert installs == [True]


def single_describe_result(capsysbinary):
    events = machine_events(capsysbinary)
    assert len(events) == 1 and events[0]["type"] == "result" and events[0]["mode"] == "describe"
    return events[0]


def test_describe_without_input_emits_options_and_presets_without_run_result_keys(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = single_describe_result(capsysbinary)
    assert isinstance(r["options"], list) and r["options"]
    assert isinstance(r["presets"], list)
    for k in ("output", "keys", "bones", "reduction", "input_kind"):
        assert k not in r


def test_describe_works_without_machine_flag(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    assert single_describe_result(capsysbinary)["mode"] == "describe"


def test_describe_options_exclude_meta_and_negated_forms_and_pin_type_constraint_default(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = single_describe_result(capsysbinary)
    by_name = {o["name"]: o for o in r["options"]}
    for meta in _META_OPTIONS:
        assert meta not in by_name
    for neg in _NEGATED_FLAGS:
        assert neg not in by_name
    assert len(r["options"]) == _DESCRIBED_OPTION_COUNT
    for o in r["options"]:
        assert set(o) == {"name", "type", "constraint", "default", "help"}
        assert isinstance(o["help"], str) and o["help"]
    assert set(by_name) == set(_EXPECTED_TYPE_CONSTRAINT_DEFAULT_BY_OPTION)
    for name, (type_, constraint, default) in _EXPECTED_TYPE_CONSTRAINT_DEFAULT_BY_OPTION.items():
        o = by_name[name]
        assert o["type"] == type_, name
        assert o["constraint"] == constraint, name
        assert o["default"] == default, name


def test_describe_presets_pin_base_pos_and_rot_tolerances(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = single_describe_result(capsysbinary)
    for p in r["presets"]:
        assert set(p) == {"name", "values"}
        assert set(p["values"]) == {"reduce_error_bone_pos", "reduce_error_bone_rot"}
    assert {p["name"]: p["values"] for p in r["presets"]} == _EXPECTED_PRESET_BASE_TOLERANCES


def test_describe_type_table_covers_every_non_meta_parser_arg():
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._DESCRIBED_TYPE_AND_CONSTRAINT_BY_DEST)


def test_describe_without_machine_reports_arg_error_as_terminal_error_event(capsysbinary):
    rc = cli.main(["--describe", "--clean-strength", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--clean-strength" and e["exit_code"] == 2


def test_machine_error_output_is_directory_even_with_overwrite(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _write_center_linear_ramp(src)
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    for extra in ([], ["--overwrite"]):
        rc = cli.main([str(src), "-o", str(outdir), "--machine", *extra])
        assert rc == 2
        e = machine_error(capsysbinary)
        assert e["code"] == "output_is_directory" and e["field"] == "--output"
        assert e["exit_code"] == 2 and e["path"] == str(outdir)


def test_machine_list_bones_ignores_output_is_directory(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0)])
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    rc = cli.main([str(src), "-o", str(outdir), "--machine", "--list-bones"])
    assert rc == 0
    assert machine_events(capsysbinary)[-1]["mode"] == "list_bones"


def test_non_machine_usage_error_stderr_is_exactly_one_argparse_message_line_without_usage(capsys):
    rc = cli.main(["in.vmd", "--bogus"])
    assert rc == 2
    assert capsys.readouterr().err == "error: unrecognized arguments: --bogus\n"
