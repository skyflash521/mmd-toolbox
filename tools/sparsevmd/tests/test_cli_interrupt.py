import json

from sparsevmd import cli
from vmd import io
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


def cam(frame, center=(0.0, 0.0, 0.0)):
    return CameraKey(frame, -30.0, center, (0.0, 0.0, 0.0), CAM_LINEAR, 30, 0)


def bone(name, frame, pos=(0.0, 0.0, 0.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, (0.0, 0.0, 0.0, 1.0), BL)


def write_vmd(path, **sections):
    io.write_file(VmdDocument(**sections), str(path))


def linear_camera_doc():
    return [cam(f, center=(float(f), 0.0, 0.0)) for f in range(31)]


def ramp_bone_doc(name="センター"):
    return [bone(name, f, pos=(0.0, float(f), 0.0)) for f in range(31)]


def machine_error(capsysbinary):
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert events and events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    return events[-1]


def _raise_keyboard_interrupt(*a, **k):
    raise KeyboardInterrupt()


def test_machine_cancelled_on_camera_interrupt(tmp_path, capsysbinary, monkeypatch):
    monkeypatch.setattr(cli, "reduce_camera_track", _raise_keyboard_interrupt)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--machine"])
    assert rc == 130
    e = machine_error(capsysbinary)
    assert e["code"] == "cancelled" and e["exit_code"] == 130 and e["field"] is None
    assert not out.exists()


def test_machine_cancelled_on_bone_interrupt(tmp_path, capsysbinary, monkeypatch):
    monkeypatch.setattr(cli, "reduce_bone_track", _raise_keyboard_interrupt)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=ramp_bone_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "bone", "--machine"])
    assert rc == 130
    e = machine_error(capsysbinary)
    assert e["code"] == "cancelled" and e["exit_code"] == 130
    assert not out.exists()


def test_non_machine_interrupt_prints_error_line_and_leaves_no_output(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "reduce_camera_track", _raise_keyboard_interrupt)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera"])
    assert rc == 130
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")
    assert not out.exists()
