import functools
import io
import re
import threading
import time

from mocapvmd import cli
from pmx.types import PmxFormatError

from .helpers import bone, write_vmd

_LIVE_LABEL = "[ノイズ軽減]"
_LIVE_WAIT_SEC = 5.0
_CLEARED_THEN_ERROR = re.compile(r"\r +\rerror: ")


class _TTYStringIO(io.StringIO):
    def __init__(self):
        super().__init__()
        self.live_written = threading.Event()

    def isatty(self):
        return True

    def write(self, s):
        written = super().write(s)
        if _LIVE_LABEL in s:
            self.live_written.set()
        return written


def _fast_live_progress(monkeypatch):
    monkeypatch.setattr(cli, "ProgressRouter", functools.partial(cli.ProgressRouter, interval=0.01))


def _ramp(path):
    write_vmd(path, bone=[bone("センター", f, pos=(float(f), 0.0, 0.0)) for f in range(11)])


def _tty_stderr(monkeypatch):
    stream = _TTYStringIO()
    monkeypatch.setattr("sys.stderr", stream)
    return stream


def _slow_then_raise(error):
    def fake(*args, **kwargs):
        time.sleep(0.2)
        raise error

    return fake


def _live_then_raise(error, stream):
    def fake(*args, **kwargs):
        stream.live_written.wait(timeout=_LIVE_WAIT_SEC)
        raise error

    return fake


def test_handled_failure_clears_the_live_line_before_the_error_line(tmp_path, monkeypatch):
    _fast_live_progress(monkeypatch)
    src = tmp_path / "in.vmd"
    _ramp(src)
    stderr = _tty_stderr(monkeypatch)
    monkeypatch.setattr(cli, "apply_pose_denoise", _live_then_raise(PmxFormatError("壊れたPMX"), stderr))
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose"])
    assert stderr.live_written.is_set()
    assert rc == 1
    screen = stderr.getvalue()
    assert _LIVE_LABEL in screen
    assert screen.index(_LIVE_LABEL) < screen.index("error: ")
    assert _CLEARED_THEN_ERROR.search(screen)
    assert screen.endswith("\n") and screen.count("error: ") == 1


def test_unexpected_exception_clears_the_live_line_before_the_error_line(tmp_path, monkeypatch):
    _fast_live_progress(monkeypatch)
    src = tmp_path / "in.vmd"
    _ramp(src)
    stderr = _tty_stderr(monkeypatch)
    monkeypatch.setattr(cli, "apply_pose_denoise", _live_then_raise(RuntimeError("boom"), stderr))
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose"])
    assert stderr.live_written.is_set()
    assert rc == 1
    screen = stderr.getvalue()
    assert _LIVE_LABEL in screen
    assert _CLEARED_THEN_ERROR.search(screen)
    assert screen.rstrip("\n").endswith("error: RuntimeError: boom")


def test_machine_failure_leaves_stderr_free_of_live_line_and_error_text(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "apply_pose_denoise", _slow_then_raise(PmxFormatError("壊れたPMX")))
    src = tmp_path / "in.vmd"
    _ramp(src)
    stderr = _tty_stderr(monkeypatch)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose", "--machine"])
    assert rc == 1
    assert "error: " not in stderr.getvalue()
