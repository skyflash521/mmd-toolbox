import pytest

from song2vmd import cli

from .support import (
    isolate_gpu_environment,  # noqa: F401
    machine_events,
    stub_pipeline,
    touch,
)

pytestmark = pytest.mark.usefixtures("isolate_gpu_environment")


def test_dry_run_with_existing_output_and_no_overwrite_exits_2(tmp_path, monkeypatch):
    stub_pipeline(monkeypatch)
    src = touch(tmp_path / "in.wav")
    out = touch(tmp_path / "out.vmd")
    assert cli.main([src, "-o", out, "--dry-run"]) == 2


def test_dry_run_with_existing_default_output_and_no_overwrite_exits_2(tmp_path, monkeypatch):
    stub_pipeline(monkeypatch)
    src = touch(tmp_path / "in.wav")
    touch(tmp_path / "in.vmd")
    assert cli.main([src, "--dry-run"]) == 2


def test_dry_run_machine_reports_output_exists(tmp_path, monkeypatch, capsysbinary):
    stub_pipeline(monkeypatch)
    src = touch(tmp_path / "in.wav")
    out = touch(tmp_path / "out.vmd")
    rc = cli.main([src, "-o", out, "--dry-run", "--machine"])
    assert rc == 2
    last = machine_events(capsysbinary)[-1]
    assert last["type"] == "error"
    assert last["code"] == "output_exists" and last["field"] == "--output" and last["exit_code"] == 2


def test_dry_run_with_existing_output_does_not_run_the_pipeline(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: calls.append(1))
    src = touch(tmp_path / "in.wav")
    out = touch(tmp_path / "out.vmd")
    assert cli.main([src, "-o", out, "--dry-run"]) == 2
    assert calls == []


def test_dry_run_with_existing_output_and_overwrite_passes_the_guard(tmp_path, monkeypatch):
    stub_pipeline(monkeypatch)
    src = touch(tmp_path / "in.wav")
    out = touch(tmp_path / "out.vmd")
    assert cli.main([src, "-o", out, "--dry-run", "--overwrite"]) == 0
