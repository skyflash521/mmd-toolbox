import functools
import io
import re
import threading
import time

from shakevmd import cli
from vmd import io as vmd_io
from vmd.types import CameraKey, VmdDocument

_LINEAR = bytes([20, 107, 20, 107]) * 6
_LIVE_LABEL = "[ベイク]"
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


def _write_input(path):
    keys = [CameraKey(f, -30.0, (float(f), 0.0, 0.0), (0.0, 0.0, 0.0), _LINEAR, 30, 0) for f in (0, 30, 60)]
    vmd_io.write_file(VmdDocument(camera=keys), str(path))
    return str(path)


def _fast_live_progress(monkeypatch):
    monkeypatch.setattr(cli, "ProgressRouter", functools.partial(cli.ProgressRouter, interval=0.01))


def _tty_stderr(monkeypatch):
    stream = _TTYStringIO()
    monkeypatch.setattr("sys.stderr", stream)
    return stream


def _slow_bake_then_raise(error):
    def fake_bake(*args, progress=None, **kwargs):
        progress(1, 10)
        time.sleep(0.2)
        raise error

    return fake_bake


def _live_bake_then_raise(error, stream):
    def fake_bake(*args, progress=None, **kwargs):
        progress(1, 10)
        stream.live_written.wait(timeout=_LIVE_WAIT_SEC)
        raise error

    return fake_bake


def test_handled_failure_clears_the_live_line_before_the_error_line(tmp_path, monkeypatch):
    _fast_live_progress(monkeypatch)
    inp = _write_input(tmp_path / "in.vmd")
    stderr = _tty_stderr(monkeypatch)
    monkeypatch.setattr(cli, "bake", _live_bake_then_raise(ValueError("overflow"), stderr))
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd")])
    assert stderr.live_written.is_set()
    assert rc == 2
    screen = stderr.getvalue()
    assert _LIVE_LABEL in screen
    assert screen.index(_LIVE_LABEL) < screen.index("error: ")
    assert _CLEARED_THEN_ERROR.search(screen)
    assert screen.endswith("\n") and screen.count("error: ") == 1


def test_unexpected_exception_clears_the_live_line_before_the_error_line(tmp_path, monkeypatch):
    _fast_live_progress(monkeypatch)
    inp = _write_input(tmp_path / "in.vmd")
    stderr = _tty_stderr(monkeypatch)
    monkeypatch.setattr(cli, "bake", _live_bake_then_raise(RuntimeError("boom"), stderr))
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd")])
    assert stderr.live_written.is_set()
    assert rc == 1
    screen = stderr.getvalue()
    assert _LIVE_LABEL in screen
    assert _CLEARED_THEN_ERROR.search(screen)
    assert screen.rstrip("\n").endswith("error: RuntimeError: boom")


def test_machine_failure_leaves_stderr_free_of_error_text(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "bake", _slow_bake_then_raise(ValueError("overflow")))
    inp = _write_input(tmp_path / "in.vmd")
    stderr = _tty_stderr(monkeypatch)
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 2
    assert "error: " not in stderr.getvalue()
