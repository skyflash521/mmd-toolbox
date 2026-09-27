import json

import pytest

from sparsevmd import cli
from vmd import io
from vmd.types import BoneKey, CameraKey, VmdDocument

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()


def cam(frame, center=(0.0, 0.0, 0.0)):
    return CameraKey(frame, -30.0, center, (0.0, 0.0, 0.0), CAM_LINEAR, 30, 0)


def bone(name, frame, pos=(0.0, 0.0, 0.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, (0.0, 0.0, 0.0, 1.0), BL)


def write_vmd(path, **sections):
    io.write_file(VmdDocument(**sections), str(path))


def linear_camera_doc():
    return [cam(f, center=(float(f), 0.0, 0.0)) for f in range(31)]


def machine_events(capsysbinary):
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln]


def machine_error(capsysbinary):
    events = machine_events(capsysbinary)
    assert events, "stdout に少なくとも 1 イベントが要る"
    assert events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    return events[-1]


def ramp_camera(path):
    write_vmd(path, camera=linear_camera_doc())


def test_version_prints_package_version_and_exits_zero(capsys):
    from sparsevmd import __version__

    rc = cli.main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sparsevmd" in out and __version__ in out


def test_machine_version_stays_human(capsys):
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sparsevmd" in out and not out.lstrip().startswith("{")


def test_machine_help_stays_human(capsys):
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


def test_help_lists_machine_describe_quiet_version_cut_detect(capsys):
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    for flag in ("--machine", "--describe", "--quiet", "--version", "--cut-detect"):
        assert flag in text


def test_machine_error_bad_argument_unknown_option_field_is_first_token(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--bogus"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2
    assert e["field"] == "--bogus"
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_bad_argument_missing_input(capsysbinary):
    rc = cli.main(["--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "input" and e["exit_code"] == 2


def test_machine_error_bad_argument_type_error_field_is_long_option(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--max-segment-frames", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--max-segment-frames" and e["exit_code"] == 2


@pytest.mark.parametrize("opt", ["--min-segment-frames", "--max-segment-frames"])
def test_machine_error_bad_argument_segment_below_one(tmp_path, capsysbinary, opt):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--target", "camera", opt, "0"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == opt and e["exit_code"] == 2


def test_machine_error_segment_bounds_conflict_has_null_field(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--target", "camera",
                   "--min-segment-frames", "10", "--max-segment-frames", "5"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "segment_bounds_conflict" and e["field"] is None and e["exit_code"] == 2


def test_machine_error_bad_tolerance_fov_below_half(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--target", "camera", "--camera-fov-tol", "0.4"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_tolerance" and e["field"] is None and e["exit_code"] == 2
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_target_selection_conflict(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=[bone("センター", 0)])
    rc = cli.main([str(src), "--machine", "--target", "camera", "--bone", "センター"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "target_selection_conflict" and e["field"] is None and e["exit_code"] == 2


def test_machine_error_output_exists(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "-o", str(src), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output" and e["exit_code"] == 2


def test_machine_error_output_exists_distinct_path(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    out = tmp_path / "out.vmd"
    out.write_bytes(b"old content")
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output" and e["exit_code"] == 2


def test_machine_error_bone_selection_invalid_on_sole_unmatched_glob(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("頭", 0), bone("頭", 30)])
    rc = cli.main([str(src), "--machine", "--target", "bone", "--bone-glob", "幻*"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bone_selection_invalid" and e["field"] is None and e["exit_code"] == 2
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_range_invalid_after_open_end_expansion(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--target", "camera", "--range", "999:"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "range_invalid" and e["field"] == "--range" and e["exit_code"] == 2


def test_machine_error_input_not_file(tmp_path, capsysbinary):
    rc = cli.main([str(tmp_path / "nope.vmd"), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "input_not_file" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_bone_file_not_file(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30)])
    rc = cli.main([str(src), "--machine", "--target", "bone",
                   "--bone-file", str(tmp_path / "nope.txt")])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "bone_file_not_file" and e["field"] == "--bone-file" and e["exit_code"] == 1


def test_machine_error_bad_bone_file_on_invalid_utf8(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(11)])
    bf = tmp_path / "bones.txt"
    bf.write_bytes(b"\xff\xfe\x00 invalid utf8")
    rc = cli.main([str(src), "--machine", "--target", "bone", "--bone-file", str(bf)])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_bone_file" and e["field"] == "--bone-file" and e["exit_code"] == 1


def test_machine_error_not_vmd(tmp_path, capsysbinary):
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file at all")
    rc = cli.main([str(bad), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "not_vmd" and e["field"] == "input" and e["exit_code"] == 1
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_no_target_keys_for_camera_target(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30)])
    rc = cli.main([str(src), "--machine", "--target", "camera"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "no_target_keys" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_strict_tolerance_unmet(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    cam_keys = [cam(f, center=(0.0, 0.0 if f % 2 == 0 else 5.0, 0.0)) for f in range(9)]
    write_vmd(src, camera=cam_keys)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine", "--target", "camera",
                   "--curve-mode", "linear", "--strict",
                   "--min-segment-frames", "8", "--max-segment-frames", "180"])
    assert rc == 4
    e = machine_error(capsysbinary)
    assert e["code"] == "strict_tolerance_unmet" and e["field"] is None and e["exit_code"] == 4


def test_machine_error_write_failed_when_output_parent_is_file(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    clash = tmp_path / "afile"
    clash.write_bytes(b"x")
    out = str(clash / "out.vmd")
    rc = cli.main([str(src), "-o", out, "--machine", "--target", "camera", "--curve-mode", "linear"])
    assert rc == 3
    e = machine_error(capsysbinary)
    assert e["code"] == "write_failed" and e["field"] == "--output" and e["exit_code"] == 3
    assert e["path"] == out


def test_machine_error_internal_error_on_unexpected_exception(tmp_path, capsysbinary, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(cli, "reduce_camera_track", boom)
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine", "--target", "camera"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "internal_error" and e["exit_code"] == 1 and e["field"] is None


def test_machine_error_stdout_is_valid_json_lines_lf_only(tmp_path, capsysbinary):
    rc = cli.main([str(tmp_path / "nope.vmd"), "--machine"])
    assert rc == 1
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n") and b"\r" not in raw
    for ln in raw.decode("utf-8").split("\n"):
        if ln:
            obj = json.loads(ln)
            assert "type" in obj


def test_non_machine_error_prints_reason_to_stderr_not_json(tmp_path, capsys):
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file")
    rc = cli.main([str(bad)])
    assert rc == 1
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")


def test_non_machine_missing_input_is_arg_error(capsys):
    rc = cli.main([])
    assert rc == 2
    assert "error:" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("extra", [pytest.param([], id="plain"), pytest.param(["--overwrite"], id="overwrite")])
def test_machine_error_output_is_directory_regardless_of_overwrite(tmp_path, capsysbinary, extra):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    rc = cli.main([str(src), "-o", str(outdir), "--machine", *extra])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_is_directory" and e["field"] == "--output"
    assert e["exit_code"] == 2 and e["path"] == str(outdir)


def test_machine_list_bones_ignores_output_is_directory(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=[bone("センター", 0)])
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    rc = cli.main([str(src), "-o", str(outdir), "--machine", "--list-bones"])
    assert rc == 0
    assert machine_events(capsysbinary)[-1]["mode"] == "list_bones"


def test_non_machine_usage_error_is_single_error_line_without_usage(capsys):
    rc = cli.main(["in.vmd", "--bogus"])
    assert rc == 2
    assert capsys.readouterr().err == "error: unrecognized arguments: --bogus\n"
