"""sparsevmd CLI の中断(Ctrl-C 等)のテスト。

削減段の KeyboardInterrupt を機械モードでは cancelled(exit 130)の終端イベント、非機械モードでは
理由を標準エラーへ 1 行(exit 130)で畳む。書き込みは全計算後に 1 回だけなので、中断で出力ファイルは
残らない。進捗のライブ表示は中断経路でも行を閉じてから終える(_Progress の finish を try/finally で保証)。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。テストは
決定論的に実行し、外部依存を使わない。
"""

import json

from vmd import io
from vmd.types import BoneKey, CameraKey, VmdDocument
from sparsevmd import cli

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
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1  # 終端はちょうど 1 つ
    return events[-1]


def _raise_keyboard_interrupt(*a, **k):
    raise KeyboardInterrupt()


def test_machine_cancelled_on_camera_interrupt(tmp_path, capsysbinary, monkeypatch):
    # カメラ削減中の KeyboardInterrupt → cancelled の error イベント・exit 130。出力は書かれない(原子性)。
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
    # ボーン削減中の KeyboardInterrupt も cancelled/130 で畳む。
    monkeypatch.setattr(cli, "reduce_bone_track", _raise_keyboard_interrupt)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=ramp_bone_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "bone", "--machine"])
    assert rc == 130
    e = machine_error(capsysbinary)
    assert e["code"] == "cancelled" and e["exit_code"] == 130
    assert not out.exists()


def test_non_machine_cancelled_on_interrupt(tmp_path, capsys, monkeypatch):
    # 非機械モードの中断は stdout に JSON を出さず理由を標準エラーへ 1 行、exit 130。出力は残らない。
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


def test_progress_line_closed_on_interrupt(tmp_path, capsys, monkeypatch):
    # ライブ表示が有効(TTY・非機械)なとき、中断でも進捗行を閉じてから終える。
    # 進捗ラベルの行に error 行が連結されない(finish の \n で閉じられている)ことで確認する。
    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli, "reduce_camera_track", _raise_keyboard_interrupt)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera"])
    assert rc == 130
    err = capsys.readouterr().err
    progress_lines = [ln for ln in err.split("\n") if "カメラ削減" in ln]
    assert progress_lines and all("error" not in ln.lower() for ln in progress_lines)


def test_progress_line_closed_on_strict_error(tmp_path, capsys, monkeypatch):
    # strict で許容を満たせず終了(exit 4)する例外経路でも、error 行を出す前に進捗行を閉じる。
    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    src = tmp_path / "in.vmd"
    # ジグザグは線形表現不能。min-segment を大きくし strict にすると分割下限まで割っても許容を満たせない。
    cam_keys = [cam(f, center=(0.0, 0.0 if f % 2 == 0 else 5.0, 0.0)) for f in range(9)]
    write_vmd(src, camera=cam_keys)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera",
                   "--curve-mode", "linear", "--strict",
                   "--min-segment-frames", "8", "--max-segment-frames", "180"])
    assert rc == 4
    err = capsys.readouterr().err
    assert "error:" in err.lower()  # 理由が 1 行出る
    progress_lines = [ln for ln in err.split("\n") if "カメラ削減" in ln]
    assert progress_lines and all("error" not in ln.lower() for ln in progress_lines)
