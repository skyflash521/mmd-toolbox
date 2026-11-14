import pytest

from song2vmd import cli
from vocal_analysis.io import AudioLoadError

from .support import (
    CHAR_OUTSIDE_CP932,
    isolate_gpu_environment,  # noqa: F401
    strict_cp932_stderr,
    stub_pipeline,
    touch,
    written_text,
)

pytestmark = pytest.mark.usefixtures("isolate_gpu_environment")



def _no_traceback(text):
    return bool(text.strip()) and "UnicodeEncodeError" not in text and "Traceback" not in text


def test_usage_error_with_unencodable_char_exits_2_without_crashing(monkeypatch):
    stderr = strict_cp932_stderr(monkeypatch)
    rc = cli.main(["in.wav", "--model-name", CHAR_OUTSIDE_CP932])
    assert rc == 2
    assert _no_traceback(written_text(stderr))


def test_output_guard_failure_with_unencodable_path_exits_2_without_crashing(tmp_path, monkeypatch):
    stub_pipeline(monkeypatch)
    src = touch(tmp_path / "in.wav")
    out = touch(tmp_path / (CHAR_OUTSIDE_CP932 + ".vmd"))
    stderr = strict_cp932_stderr(monkeypatch)
    rc = cli.main([src, "-o", out])
    assert rc == 2
    text = written_text(stderr)
    assert text.startswith("error: ")
    assert _no_traceback(text)


def test_pipeline_failure_with_unencodable_message_keeps_its_exit_code_without_crashing(
        tmp_path, monkeypatch):
    def fake_run(input_path, **kwargs):
        raise AudioLoadError("読めない" + CHAR_OUTSIDE_CP932, reason="not_audio")

    monkeypatch.setattr(cli._pipeline, "run", fake_run)
    src = touch(tmp_path / "in.wav")
    stderr = strict_cp932_stderr(monkeypatch)
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    assert rc == 1
    assert _no_traceback(written_text(stderr))


def test_unexpected_exception_with_unencodable_message_exits_1_without_crashing(tmp_path, monkeypatch):
    def fake_run(input_path, **kwargs):
        raise RuntimeError("壊れた" + CHAR_OUTSIDE_CP932)

    monkeypatch.setattr(cli._pipeline, "run", fake_run)
    src = touch(tmp_path / "in.wav")
    stderr = strict_cp932_stderr(monkeypatch)
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    assert rc == 1
    text = written_text(stderr)
    assert text.startswith("error: RuntimeError: ") and "壊れた" in text
    assert _no_traceback(text)


def test_warning_with_unencodable_char_does_not_abort_the_run(tmp_path, monkeypatch):
    stub_pipeline(monkeypatch)
    monkeypatch.setattr(cli, "torch_gpu_warning", lambda device: ("test_warning", {}))
    monkeypatch.setattr(cli, "warning_texts",
                        lambda code, fields: ("msg", "警告" + CHAR_OUTSIDE_CP932))
    src = touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    stderr = strict_cp932_stderr(monkeypatch)
    rc = cli.main([src, "-o", str(out)])
    assert rc == 0
    assert out.exists()
    text = written_text(stderr)
    assert "警告" in text
    assert _no_traceback(text)
