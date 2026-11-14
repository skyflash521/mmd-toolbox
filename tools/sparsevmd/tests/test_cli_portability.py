import io
import sys
import types

from sparsevmd import cli
from vmd import io as vmd_io
from vmd.types import BoneKey, CameraKey, VmdDocument

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()

CHAR_OUTSIDE_CP932 = "\U0001f3a5"


def cam(frame, center=(0.0, 0.0, 0.0)):
    return CameraKey(frame, -30.0, center, (0.0, 0.0, 0.0), CAM_LINEAR, 30, 0)


def bone(name, frame, pos=(0.0, 0.0, 0.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, (0.0, 0.0, 0.0, 1.0), BL)


def write_vmd(path, **sections):
    vmd_io.write_file(VmdDocument(**sections), str(path))


def linear_camera_doc():
    return [cam(f, center=(float(f), 0.0, 0.0)) for f in range(31)]


def ramp_bone_doc(name="センター"):
    return [bone(name, f, pos=(0.0, float(f), 0.0)) for f in range(31)]


def test_non_ascii_path_roundtrip_non_machine(tmp_path):
    src = tmp_path / "モーション入力.vmd"
    out = tmp_path / "モーション出力.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    assert out.exists()
    doc, _ = vmd_io.read(str(out))
    assert doc.camera


def _cp932_strict_stderr(monkeypatch):
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


def test_usage_error_with_char_outside_locale_encoding_exits_2(monkeypatch):
    _cp932_strict_stderr(monkeypatch)
    rc = cli.main(["in.vmd", "--camera-fov-tol", CHAR_OUTSIDE_CP932])
    assert rc == 2


def test_warning_with_char_outside_locale_encoding_does_not_abort(tmp_path, monkeypatch):
    cam_keys = [cam(0), cam(30, center=(5.0, 0.0, 0.0))]
    warn = types.SimpleNamespace(code="decode-error", section="bone", message="警告" + CHAR_OUTSIDE_CP932)

    def fake_read(path):
        return VmdDocument(camera=cam_keys), [warn]

    monkeypatch.setattr(cli.io, "read", fake_read)
    _cp932_strict_stderr(monkeypatch)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=cam_keys)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0


def test_machine_output_equals_non_machine_output(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    out_h = tmp_path / "human.vmd"
    out_m = tmp_path / "machine.vmd"
    assert cli.main([str(src), "-o", str(out_h), "--curve-mode", "linear"]) == 0
    assert cli.main([str(src), "-o", str(out_m), "--curve-mode", "linear", "--machine"]) == 0
    assert out_h.read_bytes() == out_m.read_bytes()


class _UnwritableStdout:
    class _Buffer:
        def write(self, _data):
            raise OSError("broken pipe")

    def __init__(self):
        self.buffer = self._Buffer()


def test_broken_stdout_in_machine_mode_reports_original_reason_in_one_line(tmp_path, monkeypatch, capsys):
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", _UnwritableStdout())
        rc = cli.main([str(tmp_path / "nope.vmd"), "--machine"])
    assert rc == 1
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1
    assert err[0].startswith("error: ") and "入力が存在しないか通常ファイルでない" in err[0]
