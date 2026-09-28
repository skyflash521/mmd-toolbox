import json

from shakevmd import bake as bake_mod
from shakevmd import cli, presets
from vmd import io
from vmd.types import BoneKey, CameraKey, MorphKey, VmdDocument, VmdWarning

LINEAR = bytes([20, 107, 20, 107]) * 6


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, LINEAR, fov, persp)


KEYS = [
    cam(0, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, dist=-25.0, center=(10.0, 5.0, 2.0), rot=(0.2, 0.1, 0.0)),
    cam(60, dist=-20.0, center=(20.0, 0.0, -3.0), rot=(-0.1, 0.3, 0.05)),
]


def write_input(path, keys=KEYS, **doc_kwargs):
    io.write_file(VmdDocument(camera=list(keys), **doc_kwargs), str(path))
    return str(path)


def machine_events(capsysbinary):
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln]


def machine_error(capsysbinary):
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    return events[-1]


def test_machine_emits_result_event(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    result = events[-1]
    assert result["mode"] == "bake"
    assert result["output"] == str(out)
    assert result["keys"] == 61
    assert isinstance(result["applied_ranges"], list)
    assert isinstance(result["max_amplitude"], float)
    assert isinstance(result["detected_cuts"], list)


CUT_AT_30_KEYS = [
    cam(0), cam(29), cam(30, center=(40.0, 0.0, 0.0)), cam(60, center=(40.0, 0.0, 0.0)),
]


def test_machine_result_reports_snapped_ranges_and_cuts_without_describe_fields(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd", keys=CUT_AT_30_KEYS)
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth", "--range", "1:58"])
    assert rc == 0
    result = machine_events(capsysbinary)[-1]
    assert result["applied_ranges"] == [[0, 60]]
    assert result["detected_cuts"] == [30]
    assert "options" not in result and "presets" not in result


def test_machine_result_keys_counts_smoothed_output(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    assert cli.main([inp, "-o", str(out), "--machine"]) == 0
    result = machine_events(capsysbinary)[-1]
    doc, _ = io.read(str(out))
    assert result["keys"] == len(doc.camera) < 61


def test_machine_passes_library_warning_code_through(tmp_path, capsysbinary, monkeypatch):
    real_read = io.read

    def read_with_warning(path):
        doc, warns = real_read(path)
        warns.append(VmdWarning(code="decode-error", message="注入した読込警告", section="bone"))
        return doc, warns

    monkeypatch.setattr(cli.io, "read", read_with_warning)
    inp = write_input(tmp_path / "in.vmd")
    assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"]) == 0
    warns = [e for e in machine_events(capsysbinary) if e["type"] == "warning"]
    assert {"code": "decode-error", "section": ["bone"]}.items() <= warns[0].items()


def test_machine_stdout_is_valid_json_lines(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    text = raw.decode("utf-8")
    assert text.endswith("\n")
    objs = []
    for ln in text.split("\n")[:-1]:
        assert ln != ""
        obj = json.loads(ln)
        assert "type" in obj
        objs.append(obj)
    assert objs[-1]["type"] == "result" and objs[-1]["mode"] == "bake"


def test_machine_no_human_text_on_stdout(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--verbose", "--no-smooth"])
    text = capsysbinary.readouterr().out.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln]
    assert lines
    for ln in lines:
        json.loads(ln)


def test_machine_emits_warning_event_for_non_camera_sections(tmp_path, capsysbinary):
    bone = [BoneKey(name_raw=b"bone".ljust(15, b"\x00"), frame=0,
                    position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                    interpolation=bytes(64))]
    morph = [MorphKey(name_raw=b"morph".ljust(15, b"\x00"), frame=0, weight=0.0)]
    inp = write_input(tmp_path / "in.vmd", bone=bone, morph=morph)
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"
    warns = [e for e in events if e["type"] == "warning"]
    codes = {w["code"] for w in warns}
    assert "non_camera_sections_passthrough" in codes
    w = next(w for w in warns if w["code"] == "non_camera_sections_passthrough")
    assert isinstance(w["section"], list)
    assert set(w["section"]) == {"bone", "morph"}
    assert isinstance(w["message"], str) and w["message"]


def test_machine_emits_warning_event_for_duplicate_frame(tmp_path, capsysbinary):
    dup = [cam(0), cam(30), cam(30, center=(9.0, 9.0, 9.0)), cam(60)]
    inp = write_input(tmp_path / "in.vmd", keys=dup)
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"
    warns = [e for e in events if e["type"] == "warning"]
    w = next(w for w in warns if w["code"] == "bake_normalize_duplicate")
    assert w["section"] == ["camera"]
    assert isinstance(w["message"], str) and w["message"]


def test_machine_emits_warning_event_for_octave_clamp(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth", "--freq", "3.0"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"
    warns = [e for e in events if e["type"] == "warning"]
    w = next(w for w in warns if w["code"] == "octave_clamped")
    assert w["section"] is None
    assert isinstance(w["message"], str) and w["message"]


def test_machine_emits_warning_event_for_fade_shortened(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth", "--range", "0:30"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"
    warns = [e for e in events if e["type"] == "warning"]
    w = next(w for w in warns if w["code"] == "fade_shortened")
    assert w["section"] is None
    assert isinstance(w["message"], str) and w["message"]


def test_machine_emits_progress_events(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"
    progress = [e for e in events if e["type"] == "progress"]
    assert progress, "progress イベントが少なくとも 1 本出る"
    stages = {p["stage"] for p in progress}
    assert "bake" in stages
    for p in progress:
        assert set(p) >= {"type", "stage", "done", "total", "note", "elapsed"}
    first = progress[0]
    assert (first["stage"], first["done"], first["total"], first["note"], first["elapsed"]) == \
        ("bake", 0, None, "", 0.0)


def test_machine_smooth_emits_smooth_progress(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"
    progress = [e for e in events if e["type"] == "progress"]
    assert "smooth" in {e["stage"] for e in progress}
    first_smooth = next(e for e in progress if e["stage"] == "smooth")
    assert (first_smooth["done"], first_smooth["total"], first_smooth["note"],
            first_smooth["elapsed"]) == (0, None, "", 0.0)


def test_bake_progress_events_have_empty_note(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"]) == 0
    bake_progress = [e for e in machine_events(capsysbinary) if e["type"] == "progress" and e["stage"] == "bake"]
    assert len(bake_progress) > 1
    assert all(e["note"] == "" for e in bake_progress)


def test_non_machine_run_prints_no_json(tmp_path, capsys):
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--no-smooth"])
    assert rc == 0
    assert out.exists()
    stdout = capsys.readouterr().out
    assert stdout.strip() == "" or not stdout.lstrip().startswith("{")


def test_machine_version_help_stay_human(capsys):
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "shakevmd" in out and not out.lstrip().startswith("{")
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


def test_help_lists_machine_flag(capsys):
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "--machine" in text


def test_machine_error_bad_argument_unknown_option(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--bogus"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_bad_argument_missing_input(capsysbinary):
    rc = cli.main(["--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "input"


def test_machine_error_bad_argument_invalid_value_field(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "--machine", "--seed", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--seed"


def test_machine_error_not_vmd(tmp_path, capsysbinary):
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file")
    rc = cli.main([str(bad), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "not_vmd" and e["field"] == "input" and e["exit_code"] == 1
    assert isinstance(e["message"], str) and e["message"]


class _UnreadableInputError(Exception):
    pass


def test_machine_error_not_vmd_names_exception_type(tmp_path, capsysbinary, monkeypatch):
    def failing_read(path):
        raise _UnreadableInputError("broken")

    monkeypatch.setattr(cli.io, "read", failing_read)
    inp = write_input(tmp_path / "in.vmd")
    assert cli.main([inp, "--machine"]) == 1
    assert "_UnreadableInputError" in machine_error(capsysbinary)["message"]


def test_machine_error_no_camera_keys(tmp_path, capsysbinary):
    p = str(tmp_path / "nocam.vmd")
    io.write_file(VmdDocument(camera=[]), p)
    rc = cli.main([p, "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "no_camera_keys" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_output_exists(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", inp, "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output"


def test_machine_error_output_exists_distinct_path(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    out.write_bytes(b"old content")
    rc = cli.main([inp, "-o", str(out), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output"


def test_machine_error_range_reversed(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--range", "999:"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "range_reversed" and e["field"] == "--range"


def test_machine_error_range_overlap(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine",
                   "--range", "0:30", "--range", "30:60"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "range_overlap" and e["field"] == "--range"


def test_machine_error_value_overflow(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--fade", "1e308"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "value_overflow" and e["field"] is None and e["exit_code"] == 2


def test_machine_error_non_finite_output(tmp_path, capsysbinary, monkeypatch):
    inf_key = CameraKey(0, -30.0, (float("inf"), 0.0, 0.0), (0.0, 0.0, 0.0), LINEAR, 30, 0)

    def bad_bake(camera_keys, *a, **k):
        return bake_mod.BakeResult(camera_keys=[inf_key], warnings=[], resolved=[(0, 0)])

    monkeypatch.setattr(cli, "bake", bad_bake)
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "non_finite_output" and e["field"] is None


def test_machine_error_write_failed(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    clash = tmp_path / "afile"
    clash.write_bytes(b"x")
    out = str(clash / "out.vmd")
    rc = cli.main([inp, "-o", out, "--machine", "--no-smooth"])
    assert rc == 3
    e = machine_error(capsysbinary)
    assert e["code"] == "write_failed" and e["field"] == "--output" and e["exit_code"] == 3
    assert e["path"] == out


def test_machine_error_internal_error(tmp_path, capsysbinary, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(cli, "bake", boom)
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "internal_error" and e["exit_code"] == 1


def test_non_machine_error_prints_reason_to_stderr(tmp_path, capsys):
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file")
    rc = cli.main([str(bad)])
    assert rc == 1
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")


def describe_result(capsysbinary):
    events = machine_events(capsysbinary)
    assert len(events) == 1 and events[0]["type"] == "result" and events[0]["mode"] == "describe"
    return events[0]


def test_describe_emits_result_without_input(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    assert isinstance(r["options"], list) and r["options"]
    assert isinstance(r["presets"], list)
    for k in ("output", "keys", "applied_ranges", "max_amplitude", "detected_cuts"):
        assert k not in r


def test_describe_works_without_machine_flag(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    assert describe_result(capsysbinary)["mode"] == "describe"


def test_describe_options_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    by_name = {o["name"]: o for o in r["options"]}
    for meta in ("--describe", "--version", "--help", "--machine"):
        assert meta not in by_name
    for o in r["options"]:
        assert set(o) == {"name", "type", "constraint", "default", "help"}
        assert isinstance(o["help"], str) and o["help"]
    assert by_name["input"]["type"] == "str" and by_name["input"]["constraint"] is None
    assert by_name["--amp-rot"]["type"] == "float"
    assert by_name["--amp-rot"]["constraint"] == {"min": 0, "max": None, "exclusive_min": False}
    assert by_name["--amp-rot"]["default"] == 0.8
    assert by_name["--freq"]["constraint"]["exclusive_min"] is True
    assert by_name["--seed"]["type"] == "int" and by_name["--seed"]["constraint"] is None
    assert by_name["--seed"]["default"] == 1
    assert by_name["--preset"]["type"] == "enum"
    assert set(by_name["--preset"]["constraint"]["choices"]) == set(presets.PRESET_NAMES)
    assert by_name["--preset"]["default"] is None
    assert by_name["--smooth"]["type"] == "flag" and by_name["--smooth"]["default"] is True
    assert by_name["--overwrite"]["type"] == "flag" and by_name["--overwrite"]["default"] is False
    rw = by_name["--rot-weights"]
    assert rw["type"] == "compound" and rw["constraint"]["format"] == "P,Y,R"
    assert [f["name"] for f in rw["constraint"]["fields"]] == ["P", "Y", "R"]
    assert rw["default"] == [1.0, 1.0, 0.3]
    assert by_name["--output"]["default"] is None
    assert by_name["--range"]["default"] is None and by_name["--impulse"]["default"] is None


def test_describe_lists_exactly_the_processing_options(capsysbinary):
    assert cli.main(["--describe"]) == 0
    names = {o["name"] for o in describe_result(capsysbinary)["options"]}
    assert names == {
        "input", "--output", "--overwrite", "--range", "--amp-rot", "--amp-pos", "--rot-weights", "--freq",
        "--seed", "--fade", "--motion-damp", "--settle", "--cut-threshold", "--impulse", "--preset",
        "--dry-run", "--verbose", "--smooth", "--quiet",
    }


def test_describe_defaults_of_shake_parameters_and_flags(capsysbinary):
    assert cli.main(["--describe"]) == 0
    defaults = {o["name"]: o["default"] for o in describe_result(capsysbinary)["options"]}
    assert {k: defaults[k] for k in ("--amp-pos", "--freq", "--fade", "--motion-damp", "--settle")} == {
        "--amp-pos": 0.05, "--freq": 1.2, "--fade": 0.7, "--motion-damp": 1.0, "--settle": 0.0,
    }
    assert defaults["--cut-threshold"] == [5.0, 20.0]
    assert defaults["--dry-run"] is False and defaults["--verbose"] is False and defaults["--quiet"] is False


def _field(name, type_, minimum, exclusive_min):
    return {"name": name, "type": type_, "min": minimum, "max": None, "exclusive_min": exclusive_min}


def test_describe_compound_constraints(capsysbinary):
    assert cli.main(["--describe"]) == 0
    by_name = {o["name"]: o for o in describe_result(capsysbinary)["options"]}
    assert by_name["--cut-threshold"]["constraint"] == {
        "format": "位置,角度", "fields": [_field("位置", "float", 0, False), _field("角度", "float", 0, False)],
    }
    assert by_name["--impulse"]["constraint"] == {
        "format": "F:S:D",
        "fields": [_field("F", "int", 0, False), _field("S", "float", 0, False), _field("D", "float", 0, True)],
    }
    assert by_name["--range"]["constraint"] == {
        "format": "START:END", "fields": [_field("START", "int", 0, False), _field("END", "int", 0, False)],
    }
    assert by_name["--rot-weights"]["constraint"]["fields"] == [
        _field("P", "float", None, False), _field("Y", "float", None, False), _field("R", "float", None, False),
    ]


def test_describe_presets_shape(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    by_name = {p["name"]: p for p in r["presets"]}
    assert set(by_name) == set(presets.PRESET_NAMES)
    public = {"amp_rot", "amp_pos", "rot_weights", "freq", "motion_damp", "settle", "cut_threshold"}
    for p in r["presets"]:
        assert set(p) == {"name", "values"} and isinstance(p["values"], dict)
        assert set(p["values"]) == public


def test_describe_type_table_covers_non_meta_args():
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._D_TYPE)


def test_describe_mode_arg_error_is_error_event(capsysbinary):
    rc = cli.main(["--describe", "--seed", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--seed" and e["exit_code"] == 2


def test_machine_dry_run_emits_inspect_result(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--machine", "--dry-run", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = events[-1]
    assert r["type"] == "result" and r["mode"] == "inspect"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    assert r["output"] is None and not out.exists()
    assert r["input_kind"] == "camera"
    assert r["keys"] == 3
    assert r["frame_range"] == [0, 60]
    assert r["duration_sec"] == 60 / 30.0
    assert isinstance(r["sections"], list) and "camera" in r["sections"]
    assert isinstance(r["applied_ranges"], list)
    assert isinstance(r["max_amplitude"], float)
    assert isinstance(r["detected_cuts"], list)


def test_machine_dry_run_inspect_lists_non_camera_sections(tmp_path, capsysbinary):
    bone = [BoneKey(name_raw=b"bone".ljust(15, b"\x00"), frame=0,
                    position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                    interpolation=bytes(64))]
    inp = write_input(tmp_path / "in.vmd", bone=bone)
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--dry-run", "--no-smooth"])
    assert rc == 0
    r = machine_events(capsysbinary)[-1]
    assert r["mode"] == "inspect"
    assert set(r["sections"]) == {"camera", "bone"}


def test_machine_dry_run_inspect_counts_keys_after_merging_duplicate_frames(tmp_path, capsysbinary):
    dup = [cam(0), cam(30), cam(30, center=(9.0, 9.0, 9.0)), cam(60)]
    inp = write_input(tmp_path / "in.vmd", keys=dup)
    assert cli.main([inp, "--machine", "--dry-run"]) == 0
    assert machine_events(capsysbinary)[-1]["keys"] == 3


def test_non_machine_dry_run_writes_nothing_and_prints_no_json(tmp_path, capsys):
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()
    cap = capsys.readouterr()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")


def _raise_keyboard_interrupt(*a, **k):
    raise KeyboardInterrupt()


def test_machine_cancelled_on_keyboard_interrupt(tmp_path, capsysbinary, monkeypatch):
    monkeypatch.setattr(cli, "bake", _raise_keyboard_interrupt)
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--machine", "--no-smooth"])
    assert rc == 130
    e = machine_error(capsysbinary)
    assert e["code"] == "cancelled" and e["exit_code"] == 130 and e["field"] is None
    assert not out.exists()


def test_machine_interrupt_while_installing_break_handler_is_cancelled_event(tmp_path, capsysbinary, monkeypatch):
    monkeypatch.setattr(cli, "install_sigbreak_handler", _raise_keyboard_interrupt)
    inp = write_input(tmp_path / "in.vmd")
    assert cli.main([inp, "--machine"]) == 130
    assert machine_error(capsysbinary)["code"] == "cancelled"


def test_non_machine_cancelled_on_keyboard_interrupt(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "bake", _raise_keyboard_interrupt)
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--no-smooth"])
    assert rc == 130
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")
    assert not out.exists()


def test_machine_error_output_is_directory(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    for extra in ([], ["--overwrite"]):
        rc = cli.main([inp, "-o", str(outdir), "--machine", *extra])
        assert rc == 2
        e = machine_error(capsysbinary)
        assert e["code"] == "output_is_directory" and e["field"] == "--output"
        assert e["exit_code"] == 2 and e["path"] == str(outdir)


def test_non_machine_usage_error_is_single_error_line(capsys):
    rc = cli.main(["in.vmd", "--bogus"])
    assert rc == 2
    assert capsys.readouterr().err == "error: unrecognized arguments: --bogus\n"
