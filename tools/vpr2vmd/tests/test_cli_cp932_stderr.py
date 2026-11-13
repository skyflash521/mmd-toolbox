import io
import sys

from vpr2vmd import cli

_CHAR_OUTSIDE_CP932 = "\U0001f3a5"


def _strict_cp932_stderr(monkeypatch):
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


def _written(wrapper):
    wrapper.flush()
    return wrapper.buffer.getvalue().decode("cp932")


def _no_traceback(text):
    return bool(text.strip()) and "UnicodeEncodeError" not in text and "Traceback" not in text


def test_missing_input_with_unencodable_path_exits_1_without_crashing(tmp_path, monkeypatch):
    missing = tmp_path / (_CHAR_OUTSIDE_CP932 + ".vpr")
    stderr = _strict_cp932_stderr(monkeypatch)
    rc = cli.main([str(missing)])
    assert rc == 1
    text = _written(stderr)
    assert text.startswith("error: ")
    assert _no_traceback(text)
