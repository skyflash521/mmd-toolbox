import io
import sys

from shakevmd import cli
from vmd import io as vmd_io
from vmd.types import CameraKey, VmdDocument

_LINEAR = bytes([20, 107, 20, 107]) * 6
_CHAR_OUTSIDE_CP932 = "\U0001f3a5"


def _write_input(path):
    keys = [CameraKey(f, -30.0, (float(f), 0.0, 0.0), (0.0, 0.0, 0.0), _LINEAR, 30, 0) for f in (0, 30, 60)]
    vmd_io.write_file(VmdDocument(camera=keys), str(path))
    return str(path)


def _strict_cp932_stderr(monkeypatch):
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


def _written(wrapper):
    wrapper.flush()
    return wrapper.buffer.getvalue().decode("cp932")


def _no_traceback(text):
    return bool(text.strip()) and "UnicodeEncodeError" not in text and "Traceback" not in text


def test_usage_error_with_unencodable_char_does_not_crash(monkeypatch):
    stderr = _strict_cp932_stderr(monkeypatch)
    rc = cli.main(["in.vmd", "--seed", _CHAR_OUTSIDE_CP932])
    assert rc == 2
    assert _no_traceback(_written(stderr))


def test_output_guard_failure_with_unencodable_path_does_not_crash(tmp_path, monkeypatch):
    inp = _write_input(tmp_path / (_CHAR_OUTSIDE_CP932 + ".vmd"))
    stderr = _strict_cp932_stderr(monkeypatch)
    rc = cli.main([inp, "-o", inp])
    assert rc == 2
    text = _written(stderr)
    assert text.startswith("error: ")
    assert _no_traceback(text)


def test_unexpected_exception_with_unencodable_message_exits_1_without_crashing(tmp_path, monkeypatch):
    def fake_bake(*args, **kwargs):
        raise RuntimeError("壊れた" + _CHAR_OUTSIDE_CP932)

    monkeypatch.setattr(cli, "bake", fake_bake)
    inp = _write_input(tmp_path / "in.vmd")
    stderr = _strict_cp932_stderr(monkeypatch)
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd")])
    assert rc == 1
    text = _written(stderr)
    assert text.startswith("error: RuntimeError: ") and "壊れた" in text
    assert _no_traceback(text)

