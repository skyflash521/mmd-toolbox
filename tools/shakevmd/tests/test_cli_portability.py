import io
import sys

from shakevmd import bake as bake_mod
from shakevmd import cli
from shakevmd.warn import ShakeWarning
from vmd import io as vmd_io
from vmd.types import BoneKey, CameraKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6

CHAR_NOT_IN_CP932 = "\U0001f3a5"


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, LINEAR, fov, persp)


KEYS = [
    cam(0),
    cam(30, dist=-25.0, center=(10.0, 5.0, 2.0), rot=(0.2, 0.1, 0.0)),
    cam(60, dist=-20.0, center=(20.0, 0.0, -3.0), rot=(-0.1, 0.3, 0.05)),
]


def write_input(path, keys=KEYS, **doc_kwargs):
    vmd_io.write_file(VmdDocument(camera=list(keys), **doc_kwargs), str(path))
    return str(path)


def test_machine_stdout_uses_lf_only(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    for line in raw.split(b"\n")[:-1]:
        assert line and not line.endswith(b"\r")


def test_machine_stdout_non_ascii_is_utf8(tmp_path, capsysbinary):
    bone = [BoneKey(name_raw=b"bone".ljust(15, b"\x00"), frame=0,
                    position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                    interpolation=bytes(64))]
    inp = write_input(tmp_path / "入力.vmd", bone=bone)
    out = tmp_path / "出力.vmd"
    rc = cli.main([inp, "-o", str(out), "--machine", "--no-smooth"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert "出力".encode("utf-8") in raw
    assert "カメラ以外".encode("utf-8") in raw
    text = raw.decode("utf-8")
    assert "出力" in text and "カメラ以外" in text


def test_non_ascii_path_roundtrip(tmp_path):
    inp = write_input(tmp_path / "手ぶれ入力.vmd")
    out = tmp_path / "手ぶれ出力.vmd"
    rc = cli.main([inp, "-o", str(out), "--no-smooth"])
    assert rc == 0
    assert out.exists()
    doc, _ = vmd_io.read(str(out))
    assert doc.camera


def _cp932_stderr(monkeypatch):
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


def test_stderr_safe_argparse_usage_error(monkeypatch):
    _cp932_stderr(monkeypatch)
    rc = cli.main(["in.vmd", "--seed", CHAR_NOT_IN_CP932])
    assert rc == 2


def test_stderr_safe_fail_path(tmp_path, monkeypatch):
    p = write_input(tmp_path / (CHAR_NOT_IN_CP932 + ".vmd"))
    _cp932_stderr(monkeypatch)
    rc = cli.main([p, "-o", p])
    assert rc == 2


def test_stderr_safe_warning_loop(tmp_path, monkeypatch):
    def fake_bake(camera_keys, *a, **k):
        return bake_mod.BakeResult(
            camera_keys=list(camera_keys),
            warnings=[ShakeWarning("passthrough_test", "警告" + CHAR_NOT_IN_CP932, None)],
            resolved=[(0, 60)],
        )
    monkeypatch.setattr(cli, "bake", fake_bake)
    _cp932_stderr(monkeypatch)
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--no-smooth"])
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
        rc = cli.main([str(tmp_path / "nope.vmd"), "--machine"])
    assert rc == 1
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1
    assert err[0].startswith("error: ") and "入力を VMD として読めない" in err[0]
