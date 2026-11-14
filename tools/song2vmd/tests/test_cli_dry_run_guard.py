import pytest

from song2vmd import cli

from .support import (
    isolate_gpu_environment,  # noqa: F401
    stub_pipeline,
    touch,
)

pytestmark = pytest.mark.usefixtures("isolate_gpu_environment")


def test_dry_run_with_existing_default_output_and_no_overwrite_exits_2(tmp_path, monkeypatch):
    stub_pipeline(monkeypatch)
    src = touch(tmp_path / "in.wav")
    touch(tmp_path / "in.vmd")
    assert cli.main([src, "--dry-run"]) == 2


def test_dry_run_with_existing_output_does_not_run_the_pipeline(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: calls.append(1))
    src = touch(tmp_path / "in.wav")
    out = touch(tmp_path / "out.vmd")
    assert cli.main([src, "-o", out, "--dry-run"]) == 2
    assert calls == []
