import json
import types

import pytest

from song2vpr import cli
from vocal_analysis.front_stage import (
    IntermediateWriteError,
    StageExecutionError,
)

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


@pytest.mark.parametrize("stage", ["separate"])
def test_unclassified_stage_failure_names_the_stage(tmp_path, monkeypatch, capsysbinary, stage):
    exc = StageExecutionError("推論に失敗", stage=stage)
    assert _run_machine(tmp_path, monkeypatch, exc) == 4
    event = _machine_error(capsysbinary)
    assert event["code"] == "stage_failed"
    assert event["stage"] == stage
    assert event["field"] is None
    assert event["exit_code"] == 4


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
