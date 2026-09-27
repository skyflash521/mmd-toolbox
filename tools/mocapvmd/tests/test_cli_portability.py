import io
import sys
import types

from mocapvmd import cli
from vmd import io as vmd_io
from vmd.reduce import BONE_LINEAR_INTERP
from vmd.types import BoneKey, VmdDocument

from .helpers import bone, write_vmd

_CP932_UNENCODABLE_CHAR = "\U0001f3a5"
_EXIT_INPUT_NOT_FILE = 1


def _ramp(path):
    write_vmd(path, bone=[bone("センター", f, pos=(float(f), 0.0, 0.0)) for f in range(11)])


def test_machine_stdout_lines_are_nonempty_lf_terminated_without_cr(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _ramp(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    for line in raw.split(b"\n")[:-1]:
        assert line and not line.endswith(b"\r")


def test_machine_stdout_writes_non_ascii_output_path_as_raw_utf8(tmp_path, capsysbinary):
    src = tmp_path / "入力.vmd"
    _ramp(src)
    out = tmp_path / "出力.vmd"
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert "出力".encode("utf-8") in raw
    text = raw.decode("utf-8")
    assert "出力" in text


def test_non_ascii_input_and_output_paths_produce_readable_vmd(tmp_path):
    src = tmp_path / "モーション入力.vmd"
    _ramp(src)
    out = tmp_path / "モーション出力.vmd"
    rc = cli.main([str(src), "-o", str(out), "--no-reduce"])
    assert rc == 0
    assert out.exists()
    doc, _ = vmd_io.read(str(out))
    assert doc.bone


def _replace_stderr_with_strict_cp932(monkeypatch):
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


def test_unencodable_char_in_usage_error_on_cp932_stderr_still_exits_2(monkeypatch):
    _replace_stderr_with_strict_cp932(monkeypatch)
    rc = cli.main(["in.vmd", "--clean-strength", _CP932_UNENCODABLE_CHAR])
    assert rc == 2


def test_unencodable_char_in_warning_on_cp932_stderr_still_exits_0(tmp_path, monkeypatch):
    keys = [BoneKey(b"c".ljust(15, b"\x00"), f, (float(f), 0.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP) for f in range(4)]
    warn = types.SimpleNamespace(code="decode-error", section="bone",
                                 message="警告" + _CP932_UNENCODABLE_CHAR)

    def fake_read(path):
        return VmdDocument(bone=keys), [warn]

    monkeypatch.setattr(cli.io, "read", fake_read)
    _replace_stderr_with_strict_cp932(monkeypatch)
    existing_input_replaced_by_fake_read = tmp_path / "in.vmd"
    _ramp(existing_input_replaced_by_fake_read)
    rc = cli.main([str(existing_input_replaced_by_fake_read), "-o", str(tmp_path / "out.vmd"),
                   "--no-reduce"])
    assert rc == 0


class _UnwritableStdout:

    class _Buffer:
        def write(self, _data):
            raise OSError("broken pipe")

    def __init__(self):
        self.buffer = self._Buffer()


def test_unwritable_stdout_in_machine_mode_keeps_original_exit_and_one_original_reason_line(
        tmp_path, monkeypatch, capsys):
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", _UnwritableStdout())
        rc = cli.main([str(tmp_path / "nope.vmd"), "--machine"])
    assert rc == _EXIT_INPUT_NOT_FILE
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1
    assert err[0].startswith("error: ") and "入力が存在しないか通常ファイルでない" in err[0]
