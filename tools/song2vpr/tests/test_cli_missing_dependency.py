import json

import pytest

from song2vpr import cli

_MISSING_SOUNDFILE = ModuleNotFoundError("No module named 'soundfile'", name="soundfile")

_NAMES_BOUND_BY_EXTRA_IMPORTS = ("_pipeline", "_pitch", "RecognitionError")


@pytest.fixture
def missing_dependency(monkeypatch):
    monkeypatch.setattr(cli, "_MISSING_DEPENDENCY", _MISSING_SOUNDFILE)
    for name in _NAMES_BOUND_BY_EXTRA_IMPORTS:
        monkeypatch.delattr(cli, name)


def test_missing_dependency_is_none_in_dev_environment():
    assert cli._MISSING_DEPENDENCY is None


def test_human_message_is_single_line_without_traceback(tmp_path, capsys, missing_dependency):
    assert cli.main([str(tmp_path / "in.wav")]) == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Traceback" not in captured.err


def test_machine_mode_reports_missing_dependency(tmp_path, capsysbinary, missing_dependency):
    assert cli.main(["--machine", str(tmp_path / "in.wav")]) == 4
    text = capsysbinary.readouterr().out.decode("utf-8")
    events = [json.loads(ln) for ln in text.split("\n") if ln]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "missing_dependency"
    assert events[0]["exit_code"] == 4
    assert events[0]["field"] is None
    assert "soundfile" in events[0]["message"]


def test_lyrics_are_read_after_the_dependency_guard(tmp_path, capsysbinary, missing_dependency):
    src = tmp_path / "in.wav"
    src.write_bytes(b"")
    assert cli.main(["--machine", str(src), "--lyrics", str(tmp_path / "無い.txt")]) == 4
    text = capsysbinary.readouterr().out.decode("utf-8")
    events = [json.loads(ln) for ln in text.split("\n") if ln]
    assert events[-1]["code"] == "missing_dependency"


def test_help_succeeds_without_dependency(capsys, missing_dependency):
    assert cli.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "--separate-vocals" in captured.out
    assert captured.err == ""


def test_version_succeeds_without_dependency(capsys, missing_dependency):
    import song2vpr

    assert cli.main(["--version"]) == 0
    captured = capsys.readouterr()
    assert song2vpr.__version__ in captured.out
    assert captured.err == ""


def test_describe_succeeds_without_dependency(capsysbinary, missing_dependency):
    assert cli.main(["--describe"]) == 0
    text = capsysbinary.readouterr().out.decode("utf-8")
    events = [json.loads(ln) for ln in text.split("\n") if ln]
    assert events[-1]["mode"] == "describe"


def _argv_rejected_with_code_2_before_dependency_guard(tmp_path, case):
    src = tmp_path / "in.wav"
    src.write_bytes(b"")
    if case == "unknown_option":
        return [str(src), "--bogus"]
    if case == "missing_input":
        return []
    if case == "combination":
        return [str(src), "--recognizer-model-revision", "abc123"]
    out = tmp_path / "out.vpr"
    out.write_bytes(b"")
    return [str(src), "-o", str(out)]


@pytest.mark.parametrize("case", ["unknown_option", "missing_input", "combination",
                                  "output_exists"])
def test_argument_checks_precede_dependency_guard_without_blaming_the_dependency(
        tmp_path, capsys, missing_dependency, case):
    assert cli.main(_argv_rejected_with_code_2_before_dependency_guard(tmp_path, case)) == 2
    assert "soundfile" not in capsys.readouterr().err
