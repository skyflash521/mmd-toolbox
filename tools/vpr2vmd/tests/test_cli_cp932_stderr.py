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


def test_output_guard_failure_with_unencodable_path_exits_2_without_crashing(
        tmp_path, monkeypatch):
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    out = tmp_path / (_CHAR_OUTSIDE_CP932 + ".vmd")
    out.write_bytes(b"")
    stderr = _strict_cp932_stderr(monkeypatch)
    rc = cli.main([str(src), "-o", str(out)])
    assert rc == 2
    text = _written(stderr)
    assert text.startswith("error: ")
    assert _no_traceback(text)


def test_unexpected_exception_with_unencodable_message_exits_1_without_crashing(tmp_path, monkeypatch):
    def fake_read(_src):
        raise RuntimeError("壊れた" + _CHAR_OUTSIDE_CP932)

    monkeypatch.setattr(cli, "read", fake_read)
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    stderr = _strict_cp932_stderr(monkeypatch)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd")])
    assert rc == 1
    text = _written(stderr)
    assert text.startswith("error: ")
    assert _no_traceback(text)
