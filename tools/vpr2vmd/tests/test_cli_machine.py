import json

import pytest

from cli_options import RangeValidator
from vpr import (
    ControllerCurve,
    ControllerEvent,
    Note,
    Part,
    TempoEvent,
    Track,
    VprFormatError,
    VprProject,
    VprWarning,
)
from vpr2vmd import __version__, cli


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _note(start, dur, phonemes, *, velocity=64):
    return Note(
        start_tick=start, duration_tick=dur, pitch=60, lyric="x",
        velocity=velocity, phonemes=phonemes,
    )


def _project(notes, *, tracks=None):
    track = Track(name="Vocal", parts=[Part(name="p", start_tick=0, notes=notes)])
    return VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 120.0)],
        tracks=tracks if tracks is not None else [track],
    )


def _stub_read(monkeypatch, project, warnings=()):
    monkeypatch.setattr(cli, "read", lambda _src: (project, list(warnings)))


def _machine_events(capsysbinary):
    out = capsysbinary.readouterr().out
    assert out.endswith(b"\n") and b"\r" not in out
    return [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]


def _terminal_events(capsysbinary):
    events = _machine_events(capsysbinary)
    assert events, "stdout に少なくとも 1 イベントが要る"
    terminals = [e for e in events if e["type"] in ("result", "error")]
    assert len(terminals) == 1 and events[-1] is terminals[0]
    return events


def _machine_error(capsysbinary):
    events = _terminal_events(capsysbinary)
    assert events[-1]["type"] == "error"
    assert set(events[-1]) == {"type", "code", "exit_code", "field", "path", "message"}
    return events[-1]


_WARNING_KEYS = {
    "type", "code", "message", "section",
    "track_index", "part_index", "note_index", "related_note_index", "tick",
}
_VPR_POSITION_KEYS = ("track_index", "part_index", "note_index", "related_note_index", "tick")


def test_machine_version_stays_human(capsys):
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "vpr2vmd" in out and __version__ in out and not out.lstrip().startswith("{")


def test_machine_help_stays_human(capsys):
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


def test_help_lists_machine_flags(capsys):
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    for flag in ("--machine", "--describe", "--n-morph", "--verbose", "--version"):
        assert flag in text


def test_describe_without_input_returns_options_and_presets(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    res = _terminal_events(capsysbinary)[-1]
    assert res["type"] == "result" and res["mode"] == "describe"

    assert set(res) == {"type", "mode", "options", "presets"}

    opts = res["options"]
    for o in opts:
        assert set(o) == {"name", "type", "constraint", "default", "help"}
        assert o["help"]

    unit = {"min": 0, "max": 1, "exclusive_min": False}
    assert [(o["name"], o["type"], o["constraint"], o["default"]) for o in opts] == [
        ("input", "str", None, None),
        ("--output", "str", None, None),
        ("--overwrite", "flag", None, False),
        ("--track", "str", None, None),
        ("--model-name", "str", None, f"vpr2vmd {__version__}"),
        ("--style", "enum", {"choices": ["pop", "ballad", "powerful", "whisper", "rap"]}, "pop"),
        ("--n-morph", "flag", None, False),
        ("--open-max", "float", unit, None),
        ("--default-open", "float", unit, None),
        ("--legato-max", "float", {"min": 0, "max": None, "exclusive_min": True}, 8.0),
        ("--valley-shallow", "float", unit, None),
        ("--valley-deep", "float", unit, None),
        ("--valley-slope", "float", {"min": 0, "max": None, "exclusive_min": False}, None),
        ("--coartic-overlap", "int", {"min": 1, "max": None, "exclusive_min": False}, None),
        ("--anticipation", "int", {"min": 0, "max": None, "exclusive_min": False}, None),
        ("--ref-bpm", "float", {"min": 0, "max": None, "exclusive_min": True}, 120.0),
        ("--tempo-scale-min", "float", {"min": 0, "max": 1, "exclusive_min": True}, 0.5),
        ("--dry-run", "flag", None, False),
        ("--verbose", "flag", None, False),
    ]

    by = {o["name"]: o for o in opts}
    assert "Shift-JIS" in by["--model-name"]["help"] and "20" in by["--model-name"]["help"]
    assert "INDEX" in by["--track"]["help"]

    presets = res["presets"]
    assert [p["name"] for p in presets] == ["pop", "ballad", "powerful", "whisper", "rap"]
    for p in presets:
        assert set(p) == {"name", "values"}
        assert set(p["values"]) == {
            "open_max", "default_open", "valley_shallow", "valley_deep",
            "valley_slope", "coartic_overlap", "anticipation",
        }


def test_describe_presets_report_preset_resolved_values(capsysbinary):
    from vpr2vmd import presets

    assert cli.main(["--describe"]) == 0
    for described in _terminal_events(capsysbinary)[-1]["presets"]:
        openness_params, gen = presets.resolve(described["name"])
        assert described["values"] == {
            "open_max": openness_params.open_max,
            "default_open": openness_params.default_open,
            "valley_shallow": gen.legato_valley_shallow,
            "valley_deep": gen.legato_valley_deep,
            "valley_slope": gen.legato_valley_slope,
            "coartic_overlap": gen.coartic_overlap_max,
            "anticipation": gen.anticipation_frames,
        }


def test_numeric_constraints_are_derived_from_the_validators():
    parser = cli._build_parser()
    options = {o["name"]: o for o in cli._describe_options(parser)}
    checked = 0
    for action in parser._actions:
        if not isinstance(action.type, RangeValidator):
            continue
        name = next(s for s in action.option_strings if s.startswith("--"))
        assert options[name]["type"] == action.type.value_type
        assert options[name]["constraint"] == action.type.constraint
        checked += 1
    assert checked == 10


def test_describe_with_unknown_option_is_bad_argument(capsysbinary):
    rc = cli.main(["--describe", "--bogus"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2


def test_machine_dry_run_emits_inspect_without_writing(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--machine", "--dry-run"])
    assert rc == 0
    assert not out.exists()
    res = _terminal_events(capsysbinary)[-1]
    assert set(res) == {
        "type", "mode", "output", "input_kind", "track_index", "track_name", "style", "n_morph",
        "model_name", "params", "adopted_notes", "mouth_events", "morph_keys", "open_source",
        "open_amounts", "vowel_undetermined", "overlap_excluded", "overlap_truncated",
        "non_event_symbols",
    }
    assert res["mode"] == "inspect" and res["output"] is None and res["input_kind"] == "vpr"
    assert res["track_index"] == 0 and res["track_name"] == "Vocal"
    assert res["style"] == "pop" and res["n_morph"] is False
    assert set(res["params"]) == {
        "open_max", "default_open", "legato_max", "valley_shallow", "valley_deep",
        "valley_slope", "coartic_overlap", "anticipation", "ref_bpm", "tempo_scale_min",
        "representative_bpm",
    }
    assert res["adopted_notes"] == 1
    assert isinstance(res["mouth_events"], int) and isinstance(res["morph_keys"], int)
    assert set(res["open_amounts"]) == {"min", "max", "mean"}
    assert isinstance(res["non_event_symbols"], dict)


def test_machine_dry_run_inspect_reports_event_diagnostics(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([
        _note(0, 480, ["k", "a"]),
        _note(0, 240, ["i"]),
        _note(480, 480, ["M"]),
        _note(720, 240, ["t"]),
    ]))
    assert cli.main([src, "--machine", "--dry-run"]) == 0
    res = _terminal_events(capsysbinary)[-1]
    assert res["vowel_undetermined"] == 1
    assert res["overlap_excluded"] == 1
    assert res["overlap_truncated"] == 1
    assert res["non_event_symbols"] == {"k": 1, "t": 1}


def test_machine_dry_run_inspect_model_name_default_is_tool_and_version(
    tmp_path, capsysbinary, monkeypatch
):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--machine", "--dry-run"])
    assert rc == 0
    res = _terminal_events(capsysbinary)[-1]
    assert res["model_name"] == f"vpr2vmd {__version__}"


def test_machine_dry_run_no_adopted_warns_then_inspect(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([]))
    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 0
    events = _terminal_events(capsysbinary)
    warns = [e for e in events if e["type"] == "warning" and e["code"] == "no_adopted_notes"]
    assert len(warns) == 1
    assert set(warns[0]) == _WARNING_KEYS
    assert warns[0]["section"] is None
    assert all(warns[0][key] is None for key in _VPR_POSITION_KEYS)
    assert events[-1]["mode"] == "inspect" and events[-1]["open_amounts"] is None


def _project_with_dynamics(notes, points):
    track = Track(name="Vocal", parts=[Part(
        name="p", start_tick=0, notes=notes,
        controllers=[ControllerCurve(name="dynamics",
                                     events=[ControllerEvent(t, v) for t, v in points])],
    )])
    return VprProject(resolution=480, tempos=[TempoEvent(0, 120.0)], tracks=[track])


@pytest.mark.parametrize("project_factory, expected", [
    pytest.param(lambda: _project_with_dynamics([_note(0, 480, ["a"]), _note(480, 480, ["i"])],
                                                [(0, 30), (480, 100)]), "dynamics",
                 id="varying_dynamics"),
    pytest.param(lambda: _project_with_dynamics([_note(0, 480, ["a"]), _note(480, 480, ["i"])],
                                                [(0, 64)]), "dynamics",
                 id="flat_dynamics_is_still_dynamics"),
    pytest.param(lambda: _project([_note(0, 480, ["a"], velocity=40),
                                   _note(480, 480, ["i"], velocity=100)]), "velocity",
                 id="varying_velocity"),
    pytest.param(lambda: _project([_note(0, 480, ["a"], velocity=64),
                                   _note(480, 480, ["i"], velocity=64)]), "default",
                 id="uniform_velocity"),
    pytest.param(lambda: _project_with_dynamics([], [(0, 64)]), None,
                 id="no_adopted_notes_with_dynamics_is_none"),
])
def test_machine_dry_run_inspect_reports_open_source(
    tmp_path, capsysbinary, monkeypatch, project_factory, expected
):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, project_factory())
    assert cli.main([src, "--machine", "--dry-run"]) == 0
    assert _terminal_events(capsysbinary)[-1]["open_source"] == expected


def test_machine_dry_run_inspect_caps_open_amounts_but_keeps_resolved_default_open(
    tmp_path, capsysbinary, monkeypatch
):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"]), _note(480, 480, ["i"])]))
    assert cli.main([src, "--open-max", "0.4", "--machine", "--dry-run"]) == 0
    res = _terminal_events(capsysbinary)[-1]
    assert res["open_source"] == "default"
    assert res["params"]["default_open"] > res["params"]["open_max"]
    assert res["open_amounts"]["max"] == pytest.approx(res["params"]["open_max"])


def test_machine_convert_result_after_write(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--machine"])
    assert rc == 0
    assert out.exists()
    res = _terminal_events(capsysbinary)[-1]
    assert set(res) == {
        "type", "mode", "output", "track_index", "track_name",
        "morph_keys", "adopted_notes", "mouth_events",
    }
    assert res["type"] == "result" and res["mode"] == "convert" and res["output"] == str(out)
    assert res["track_index"] == 0 and res["track_name"] == "Vocal"
    assert res["adopted_notes"] == 1
    assert isinstance(res["mouth_events"], int)


def test_machine_convert_morph_keys_counts_keys_written_to_vmd(tmp_path, capsysbinary, monkeypatch):
    from vmd import read as vmd_read

    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 240, ["a"]), _note(480, 240, ["i"])]))
    out = tmp_path / "out.vmd"
    assert cli.main([src, "-o", str(out), "--machine"]) == 0
    doc, _ = vmd_read(str(out))
    assert _terminal_events(capsysbinary)[-1]["morph_keys"] == len(doc.morph)


def test_machine_warning_overlapping_notes_passthrough(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    w = VprWarning(
        code="overlapping_notes", message="発音区間が重なる音符", track_index=0, part_index=0,
        note_index=1, related_note_index=0, tick=240,
    )
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]), warnings=[w])
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    events = _terminal_events(capsysbinary)
    warns = [e for e in events if e["type"] == "warning"]
    assert len(warns) == 1
    wa = warns[0]
    assert set(wa) == _WARNING_KEYS
    assert wa["message"] == "発音区間が重なる音符"
    assert wa["code"] == "overlapping_notes" and wa["section"] is None
    assert (wa["track_index"], wa["part_index"], wa["note_index"]) == (0, 0, 1)
    assert wa["related_note_index"] == 0 and wa["tick"] == 240
    assert events[-1]["mode"] == "convert"


def test_machine_warning_normalize_carries_vmd_section(tmp_path, capsysbinary, monkeypatch):
    from vmd import VmdWarning

    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))
    original = cli.normalize

    def _with_warning(document, sections=None):
        doc, warnings = original(document, sections=sections)
        return doc, [*warnings, VmdWarning(
            code="normalize-duplicate", message="同一キーの重複", section="morph")]

    monkeypatch.setattr(cli, "normalize", _with_warning)
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    events = _terminal_events(capsysbinary)
    warns = [e for e in events if e["type"] == "warning"]
    assert len(warns) == 1
    wa = warns[0]
    assert set(wa) == _WARNING_KEYS
    assert wa["code"] == "normalize-duplicate" and wa["section"] == "morph"
    assert wa["message"] == "同一キーの重複"
    assert all(wa[key] is None for key in _VPR_POSITION_KEYS)
    assert events[-1]["mode"] == "convert"


@pytest.mark.parametrize("flag", ["--verbose", "--dry-run"])
def test_machine_suppresses_human_plan_and_diagnostics(tmp_path, capsysbinary, monkeypatch, flag):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine", flag])
    assert rc == 0
    events = _terminal_events(capsysbinary)
    assert all(e["type"] in ("warning", "result") for e in events)


def test_machine_error_unreadable_input(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")

    def _raise(_src):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "read", _raise)
    rc = cli.main([src, "--machine"])
    assert rc == 1
    e = _machine_error(capsysbinary)
    assert e["code"] == "not_vpr" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_unknown_option(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--machine", "--bogus"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2 and e["field"] == "--bogus"
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_missing_input(capsysbinary):
    rc = cli.main(["--machine"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "input" and e["exit_code"] == 2


def test_machine_error_bad_value_type(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--machine", "--open-max", "abc"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--open-max" and e["exit_code"] == 2


def test_machine_error_value_out_of_range(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--machine", "--open-max", "1.5"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--open-max"


def test_machine_error_output_exists(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "-o", src, "--machine"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output" and e["exit_code"] == 2


def test_machine_error_output_exists_distinct_path(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    out = _touch(tmp_path / "out.vmd")
    rc = cli.main([src, "-o", out, "--machine"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output" and e["exit_code"] == 2


def test_machine_error_valley_bounds_inverted(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--machine", "--valley-deep", "0.6", "--dry-run"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "valley_bounds_inverted" and e["field"] is None and e["exit_code"] == 2


def test_machine_error_bad_track(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))
    rc = cli.main([src, "--machine", "--track", "5"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_track" and e["field"] == "--track" and e["exit_code"] == 2


def test_machine_error_input_not_found(tmp_path, capsysbinary):
    rc = cli.main([str(tmp_path / "nope.vpr"), "--machine"])
    assert rc == 1
    e = _machine_error(capsysbinary)
    assert e["code"] == "input_not_found" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_not_vpr(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")

    def _raise(_src):
        raise VprFormatError("壊れた vpr")

    monkeypatch.setattr(cli, "read", _raise)
    rc = cli.main([src, "--machine"])
    assert rc == 1
    e = _machine_error(capsysbinary)
    assert e["code"] == "not_vpr" and e["field"] == "input" and e["exit_code"] == 1
    assert "VprFormatError" in e["message"] and "壊れた vpr" in e["message"]


def test_machine_error_no_tracks(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([], tracks=[]))
    rc = cli.main([src, "--machine"])
    assert rc == 1
    e = _machine_error(capsysbinary)
    assert e["code"] == "no_tracks" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_write_failed(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))

    def _raise(_doc, _path):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "write_file", _raise)
    out = str(tmp_path / "out.vmd")
    rc = cli.main([src, "-o", out, "--machine"])
    assert rc == 3
    e = _machine_error(capsysbinary)
    assert e["code"] == "write_failed" and e["field"] == "--output" and e["exit_code"] == 3
    assert e["path"] == out


def test_machine_error_internal_error(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))

    def _boom(*_a, **_k):
        raise RuntimeError("想定外")

    monkeypatch.setattr(cli, "generate_morph_keys", _boom)
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 1
    e = _machine_error(capsysbinary)
    assert e["code"] == "internal_error" and e["field"] is None and e["exit_code"] == 1


def test_machine_error_stdout_is_valid_json_lines_lf_only(tmp_path, capsysbinary):
    rc = cli.main([str(tmp_path / "nope.vpr"), "--machine"])
    assert rc == 1
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n") and b"\r" not in raw
    for ln in raw.decode("utf-8").split("\n"):
        if ln:
            assert "type" in json.loads(ln)


def test_describe_type_table_covers_non_meta_args():
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._DESCRIBE_TYPE_BY_DEST)


def test_machine_error_output_is_directory(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    for extra in ([], ["--overwrite"]):
        rc = cli.main([src, "-o", str(outdir), "--machine", *extra])
        assert rc == 2
        e = _machine_error(capsysbinary)
        assert e["code"] == "output_is_directory" and e["field"] == "--output"
        assert e["exit_code"] == 2 and e["path"] == str(outdir)


@pytest.mark.parametrize("output_kind, extra, expected_code", [
    pytest.param("dir", ["--overwrite"], "output_is_directory", id="output_is_directory"),
    pytest.param("file", [], "output_exists", id="output_exists"),
    pytest.param("new", ["--valley-deep", "0.6"], "valley_bounds_inverted", id="valley_bounds_inverted"),
])
def test_argument_errors_take_precedence_over_missing_input(
    tmp_path, capsysbinary, output_kind, extra, expected_code
):
    out = tmp_path / "out.vmd"
    if output_kind == "dir":
        out.mkdir()
    elif output_kind == "file":
        _touch(out)
    rc = cli.main([str(tmp_path / "nope.vpr"), "-o", str(out), "--machine", *extra])
    assert rc == 2
    assert _machine_error(capsysbinary)["code"] == expected_code


def test_main_installs_sigbreak_handler(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "install_sigbreak_handler", lambda: calls.append(True))
    cli.main([str(tmp_path / "nope.vpr")])
    assert calls == [True]


def test_interrupt_right_after_sigbreak_install_is_reported_as_machine_event(
    tmp_path, capsysbinary, monkeypatch
):
    def _interrupted():
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "install_sigbreak_handler", _interrupted)
    rc = cli.main([str(tmp_path / "nope.vpr"), "--machine"])
    assert rc == 130
    e = _machine_error(capsysbinary)
    assert e["code"] == "cancelled" and e["exit_code"] == 130
