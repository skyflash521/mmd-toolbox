import json

import pytest

from song2vmd import cli

_MISSING = ModuleNotFoundError("No module named 'soundfile'", name="soundfile")

_NAMES_UNDEFINED_WITHOUT_EXTRA = (
    "AudioLoadError", "IntermediateReadError", "IntermediateWriteError", "RecognitionError",
    "SeparationError", "StageExecutionError", "_pipeline",
)


@pytest.fixture
def missing_dependency(monkeypatch):
    monkeypatch.setattr(cli, "_MISSING_DEPENDENCY", _MISSING)
    for name in _NAMES_UNDEFINED_WITHOUT_EXTRA:
        monkeypatch.delattr(cli, name)


def test_dependencies_present_in_dev_environment():
    assert cli._MISSING_DEPENDENCY is None


def test_human_message_is_single_line_without_traceback(tmp_path, capsys, missing_dependency):
    rc = cli.main([str(tmp_path / "in.wav")])
    assert rc == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Traceback" not in captured.err


def test_human_message_names_module_and_install_command(tmp_path, capsys, missing_dependency):
    cli.main([str(tmp_path / "in.wav")])
    err = capsys.readouterr().err
    assert "soundfile" in err
    assert 'pip install ".[vocal-analysis]"' in err


def test_version_succeeds_without_dependencies(capsys, missing_dependency):
    rc = cli.main(["--version"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "song2vmd" in captured.out
    assert captured.err == ""


def test_help_succeeds_without_dependencies(capsys, missing_dependency):
    rc = cli.main(["--help"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "--style" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("argv", [
    pytest.param([], id="input欠落"),
    pytest.param(["in.wav", "--forced-aligner", "sofa-forcedalign"], id="SOFA必須検証"),
    pytest.param(["in.wav", "--recognizer-model-revision", "abc"], id="組み合わせ検証"),
    pytest.param(["in.wav", "--output", "existing.vmd"], id="上書きガード"),
    pytest.param(["in.wav", "--output", "existing_dir"], id="出力先がディレクトリ"),
])
def test_argument_errors_keep_their_own_exit_code(argv, tmp_path, monkeypatch, capsys,
                                                  missing_dependency):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing.vmd").write_bytes(b"")
    (tmp_path / "existing_dir").mkdir()
    rc = cli.main(argv)
    assert rc == 2
    err = capsys.readouterr().err
    assert "soundfile" not in err


def test_machine_mode_emits_error_event(tmp_path, capsysbinary, missing_dependency):
    rc = cli.main([str(tmp_path / "in.wav"), "--machine"])
    assert rc == 4
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "error"
    assert event["code"] == "missing_dependency"
    assert event["exit_code"] == 4
    assert event["field"] is None
    assert "soundfile" in event["message"]


def test_describe_succeeds_without_dependencies(capsysbinary, missing_dependency):
    rc = cli.main(["--describe"])
    assert rc == 0
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    assert len(events) == 1
    assert events[0]["type"] == "result"
    assert events[0]["mode"] == "describe"
    assert events[0]["options"]
