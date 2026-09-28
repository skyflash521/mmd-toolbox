import json
import types

import pytest

from song2vpr import cli
from vocal_analysis.front_stage import (
    IntermediateReadError,
    IntermediateWriteError,
    StageExecutionError,
)
from vocal_analysis.io import AudioLoadError
from vocal_analysis.recognizer import RecognitionError
from vocal_analysis.separator import SeparationError

from .support import front_stage_result


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _machine_error(capsysbinary):
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    assert events[-1]["type"] == "error"
    return events[-1]


def _stub_pipeline(monkeypatch, raises=None, captured=None):
    def fake_run(input_path, **kwargs):
        if captured is not None:
            captured["input_path"] = input_path
            captured["kwargs"] = kwargs
        if raises is not None:
            raise raises
        return front_stage_result()

    monkeypatch.setattr(cli, "_pipeline", types.SimpleNamespace(run=fake_run))


def _run_machine(tmp_path, monkeypatch, exc, extra=()):
    src = _touch(tmp_path / "in.wav")
    _stub_pipeline(monkeypatch, raises=exc)
    return cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr"), *extra])


def test_machine_success_path_ends_with_exactly_one_terminal_event(tmp_path, monkeypatch,
                                                                   capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr")]) == 0
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1


def test_intermediate_read_failure_has_null_field_and_the_intermediate_path(tmp_path, monkeypatch,
                                                                           capsysbinary):
    exc = IntermediateReadError("読み直しに失敗", path=str(tmp_path / "vocal.wav"))
    assert _run_machine(tmp_path, monkeypatch, exc) == 1
    event = _machine_error(capsysbinary)
    assert event["code"] == "not_audio"
    assert event["field"] is None
    assert event["path"] == str(tmp_path / "vocal.wav")
    assert event["exit_code"] == 1


def test_input_load_failure_points_at_the_input(tmp_path, monkeypatch, capsysbinary):
    exc = AudioLoadError("音声として読めない", reason="not_audio")
    assert _run_machine(tmp_path, monkeypatch, exc) == 1
    event = _machine_error(capsysbinary)
    assert event["code"] == "not_audio"
    assert event["field"] == "input"
    assert event["exit_code"] == 1


def test_missing_decoder_exits_with_the_environment_failure_code_4(tmp_path, monkeypatch, capsysbinary):
    exc = AudioLoadError("復号器が見つからない", reason="decoder_missing")
    assert _run_machine(tmp_path, monkeypatch, exc) == 4
    event = _machine_error(capsysbinary)
    assert event["code"] == "decoder_missing"
    assert event["field"] == "input"
    assert event["exit_code"] == 4


@pytest.mark.parametrize("stage", ["separate", "recognize"])
def test_unclassified_stage_failure_names_the_stage(tmp_path, monkeypatch, capsysbinary, stage):
    exc = StageExecutionError("推論に失敗", stage=stage)
    assert _run_machine(tmp_path, monkeypatch, exc) == 4
    event = _machine_error(capsysbinary)
    assert event["code"] == "stage_failed"
    assert event["stage"] == stage
    assert event["field"] is None
    assert event["exit_code"] == 4


def test_separation_failure_names_the_separation_stage(tmp_path, monkeypatch, capsysbinary):
    assert _run_machine(tmp_path, monkeypatch, SeparationError("分離に失敗")) == 4
    event = _machine_error(capsysbinary)
    assert event["code"] == "stage_failed"
    assert event["stage"] == "separate"


def test_recognition_failure_names_the_recognition_stage(tmp_path, monkeypatch, capsysbinary):
    assert _run_machine(tmp_path, monkeypatch, RecognitionError("認識に失敗")) == 4
    event = _machine_error(capsysbinary)
    assert event["code"] == "stage_failed"
    assert event["stage"] == "recognize"


def test_intermediate_write_failure_names_keep_intermediate_and_its_directory(tmp_path, monkeypatch,
                                                                              capsysbinary):
    exc = IntermediateWriteError("書き込みに失敗")
    out = str(tmp_path / "out.vpr")
    assert _run_machine(tmp_path, monkeypatch, exc, extra=["--keep-intermediate"]) == 3
    event = _machine_error(capsysbinary)
    assert event["code"] == "write_failed"
    assert event["field"] == "--keep-intermediate"
    assert event["path"] == f"{out}.intermediate"
    assert event["exit_code"] == 3


def test_non_machine_stage_failure_is_one_line_naming_the_stage_label(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _stub_pipeline(monkeypatch, raises=StageExecutionError("推論に失敗", stage="separate"))
    assert cli.main([src, "-o", str(tmp_path / "out.vpr")]) == 4
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "ボーカル分離" in lines[0]


def test_intermediate_directory_is_the_output_path_plus_intermediate_suffix(tmp_path, monkeypatch):
    captured = {}
    src = _touch(tmp_path / "in.wav")
    out = str(tmp_path / "out.vpr")
    _stub_pipeline(monkeypatch, captured=captured)

    assert cli.main([src, "-o", out, "--keep-intermediate"]) == 0
    assert captured["kwargs"]["keep_intermediate_dir"] == f"{out}.intermediate"


def test_intermediate_is_not_saved_without_the_option(tmp_path, monkeypatch):
    captured = {}
    src = _touch(tmp_path / "in.wav")
    _stub_pipeline(monkeypatch, captured=captured)

    assert cli.main([src, "-o", str(tmp_path / "out.vpr")]) == 0
    assert captured["kwargs"]["keep_intermediate_dir"] is None


def test_intermediate_is_saved_even_in_dry_run(tmp_path, monkeypatch):
    captured = {}
    src = _touch(tmp_path / "in.wav")
    out = str(tmp_path / "out.vpr")
    _stub_pipeline(monkeypatch, captured=captured)

    assert cli.main([src, "-o", out, "--keep-intermediate", "--dry-run"]) == 0
    assert captured["kwargs"]["keep_intermediate_dir"] == f"{out}.intermediate"


def test_leftover_intermediate_directory_does_not_need_overwrite(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    out = str(tmp_path / "out.vpr")
    leftover = tmp_path / "out.vpr.intermediate"
    leftover.mkdir()
    (leftover / "vocal.wav").write_bytes(b"")
    _stub_pipeline(monkeypatch)

    assert cli.main([src, "-o", out, "--keep-intermediate"]) == 0


def test_resolved_front_stage_settings_are_passed_through(tmp_path, monkeypatch):
    captured = {}
    src = _touch(tmp_path / "in.wav")
    _stub_pipeline(monkeypatch, captured=captured)

    assert cli.main([
        src, "-o", str(tmp_path / "out.vpr"), "--separate-vocals", "never",
        "--separator", "audio-separator-htdemucs-ft", "--forced-aligner",
        "wav2vec2-ctc-forcedalign", "--max-duration", "120",
    ]) == 0
    kwargs = captured["kwargs"]
    assert captured["input_path"] == src
    assert kwargs["separate_vocals"] == "never"
    assert kwargs["separator_name"] == "audio-separator-htdemucs-ft"
    assert kwargs["forced_aligner"] == "wav2vec2-ctc-forcedalign"
    assert kwargs["chunking"].max_duration_sec == 120.0
