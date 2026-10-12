import sys

import pytest

from song2vmd import cli
from vmd import read as vmd_read

from .support import (
    isolate_gpu_environment,  # noqa: F401
    machine_events,
    make_result,
    stub_pipeline,
    touch,
)

pytestmark = pytest.mark.usefixtures("isolate_gpu_environment")


class _UnwritableStdout:
    class _Buffer:
        def __init__(self, error):
            self._error = error

        def write(self, _data):
            raise self._error

    def __init__(self, error):
        self.buffer = self._Buffer(error)


@pytest.mark.parametrize("error", [OSError("broken pipe"), ValueError("I/O operation on closed file")])
def test_unwritable_stdout_in_machine_mode_reports_reason_without_traceback(monkeypatch, capsys, error):
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", _UnwritableStdout(error))
        rc = cli.main(["--machine"])
    assert rc == 2
    err = capsys.readouterr().err
    lines = err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ") and "入力音声(input)が必要です" in lines[0]
    assert "Traceback" not in err


def test_unwritable_stdout_keeps_the_exit_code_of_an_output_guard_failure(tmp_path, monkeypatch, capsys):
    src = touch(tmp_path / "in.wav")
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", _UnwritableStdout(OSError("broken pipe")))
        rc = cli.main([src, "-o", src, "--machine"])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "--overwrite" in err
    assert "Traceback" not in err


def test_non_ascii_input_and_output_paths_produce_readable_vmd(tmp_path, monkeypatch):
    stub_pipeline(monkeypatch)
    src = touch(tmp_path / "ボーカル入力.wav")
    out = tmp_path / "リップモーション出力.vmd"
    rc = cli.main([src, "-o", str(out)])
    assert rc == 0
    doc, _ = vmd_read(str(out))
    assert doc.model_name_raw is not None


def test_non_ascii_input_default_output_resolves_next_to_input_and_trips_overwrite_guard(
        tmp_path, monkeypatch):
    stub_pipeline(monkeypatch)
    src = touch(tmp_path / "ボーカル.wav")
    touch(tmp_path / "ボーカル.vmd")
    assert cli.main([src]) == 2


def test_machine_stdout_keeps_non_ascii_output_path_as_utf8(tmp_path, monkeypatch, capsysbinary):
    stub_pipeline(monkeypatch)
    src = touch(tmp_path / "ボーカル入力.wav")
    out = tmp_path / "リップモーション出力.vmd"
    rc = cli.main([src, "-o", str(out), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert "リップモーション出力".encode("utf-8") in raw


def test_machine_error_event_keeps_non_ascii_path_in_utf8(tmp_path, capsysbinary):
    src = touch(tmp_path / "ボーカル入力.wav")
    out = touch(tmp_path / "既存の出力.vmd")
    rc = cli.main([src, "-o", out, "--machine"])
    assert rc == 2
    last = machine_events(capsysbinary)[-1]
    assert last["code"] == "output_exists"
    assert "既存の出力" in last["message"]


def test_non_ascii_keep_intermediate_dir_is_derived_from_output_path(tmp_path, monkeypatch):
    captured = {}

    def fake_run(input_path, **kwargs):
        captured["dir"] = kwargs["keep_intermediate_dir"]
        return make_result()

    monkeypatch.setattr(cli._pipeline, "run", fake_run)
    src = touch(tmp_path / "in.wav")
    out = tmp_path / "出力.vmd"
    assert cli.main([src, "-o", str(out), "--keep-intermediate", "--dry-run"]) == 0
    assert captured["dir"] == f"{out}.intermediate"
