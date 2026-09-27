import io
import json
import sys

from vmd import read as vmd_read
from vpr import Note, Part, TempoEvent, Track, VprProject, VprWarning
from vpr2vmd import cli

_NOT_IN_CP932 = "\U0001f3a5"


def _project():
    note = Note(start_tick=0, duration_tick=480, pitch=60, lyric="x", velocity=64, phonemes=["a"])
    return VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 120.0)],
        tracks=[Track(name="Vocal", parts=[Part(name="p", start_tick=0, notes=[note])])],
    )


def _stub_read(monkeypatch, warnings=()):
    monkeypatch.setattr(cli, "read", lambda _src: (_project(), list(warnings)))


def _cp932_stderr(monkeypatch):
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


def test_non_ascii_path_roundtrip_non_machine(tmp_path, monkeypatch):
    src = tmp_path / "ボーカル入力.vpr"
    src.write_bytes(b"")
    out = tmp_path / "リップモーション出力.vmd"
    _stub_read(monkeypatch)
    rc = cli.main([str(src), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    doc, _ = vmd_read(str(out))
    assert doc.morph


def test_non_ascii_path_machine(tmp_path, monkeypatch, capsysbinary):
    src = tmp_path / "ボーカル入力.vpr"
    src.write_bytes(b"")
    out = tmp_path / "リップモーション出力.vmd"
    _stub_read(monkeypatch)
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 0
    assert out.exists()
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "convert"
    assert events[-1]["output"] == str(out)


def test_stderr_safe_on_argparse_usage_error(monkeypatch):
    _cp932_stderr(monkeypatch)
    rc = cli.main(["in.vpr", "--open-max", _NOT_IN_CP932])
    assert rc == 2


def test_stderr_safe_on_warning(tmp_path, monkeypatch):
    warn = VprWarning(code="overlapping_notes", message="重なり" + _NOT_IN_CP932, track_index=0)
    _stub_read(monkeypatch, warnings=[warn])
    _cp932_stderr(monkeypatch)
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd")])
    assert rc == 0


class _UnwritableStdout:
    class _Buffer:
        def write(self, _data):
            raise OSError("broken pipe")

    def __init__(self):
        self.buffer = self._Buffer()


def test_broken_stdout_in_machine_mode_reports_reason_without_traceback(tmp_path, monkeypatch,
                                                                       capsys):
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", _UnwritableStdout())
        rc = cli.main([str(tmp_path / "nope.vpr"), "--machine"])
    assert rc == 1
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1
    assert err[0].startswith("error: ") and "入力 vpr が見つかりません" in err[0]


def test_machine_output_equals_non_machine_output(tmp_path, monkeypatch):
    _stub_read(monkeypatch)
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    out_h = tmp_path / "human.vmd"
    out_m = tmp_path / "machine.vmd"
    assert cli.main([str(src), "-o", str(out_h)]) == 0
    assert cli.main([str(src), "-o", str(out_m), "--machine"]) == 0
    assert out_h.read_bytes() == out_m.read_bytes()
