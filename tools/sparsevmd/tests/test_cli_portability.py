"""sparsevmd CLI 移植性・既定挙動回帰のテスト。

非ASCIIパスの受理・生成、人間向け標準エラーの符号化安全性(ロケール符号化で表せない文字でもプロセスを
落とさない)、機械モードが出力VMDを変えない(--machine の有無で出力バイト一致)ことを検証する。テスト
は決定論的に実行し、外部依存を使わない。
"""

import io
import sys
import types

from vmd import io as vmd_io
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

# cp932(Windows のロケール符号化)で表せない文字(絵文字 U+1F3A5)。ロケール符号化外の文字を
# 人間向け標準エラーへ書く経路を作り、符号化安全性を検証するために使う。
UNREP = "\U0001f3a5"


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


# --- 非ASCIIパスの受理・生成 --------------------------


def test_non_ascii_path_roundtrip_non_machine(tmp_path):
    # 日本語ファイル名の入力を受理し、日本語ファイル名の出力を生成できる(非機械)。
    src = tmp_path / "モーション入力.vmd"
    out = tmp_path / "モーション出力.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    assert out.exists()
    doc, _ = vmd_io.read(str(out))  # 生成物が妥当な VMD として読める
    assert doc.camera


# --- 人間向け標準エラーの符号化安全性 -----------------
# ロケール符号化(cp932)相当へ差し替えた標準エラーの下で、表せない文字を含む人間向け出力
# (argparse 使用法エラー・警告ループの warning 行)が UnicodeEncodeError で本体を異常終了させない
# ことを検証する。cli.py が標準エラーのエラーハンドラを backslashreplace へ緩めることで担う。


def _cp932_stderr(monkeypatch):
    """sys.stderr をロケール符号化(cp932)相当・strict へ差し替える。"""
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


def test_stderr_safe_argparse_usage_error(monkeypatch):
    # argparse 使用法エラー経路: 表せない文字を含む不正引数値。使用法エラーが符号化に失敗せず、
    # 引数エラー(2)で終える(例外を漏らさない)。
    _cp932_stderr(monkeypatch)
    rc = cli.main(["in.vmd", "--camera-fov-tol", UNREP])
    assert rc == 2


def test_stderr_safe_warning_loop(tmp_path, monkeypatch):
    # 警告ループ経路: 表せない文字を含む警告文でも本体は正常終了(0)する(符号化失敗で落とさない)。
    # io.read を差し替え、非ASCIIメッセージの警告を人間向け stderr へ流す経路を突く。
    cam_keys = [cam(0), cam(30, center=(5.0, 0.0, 0.0))]
    warn = types.SimpleNamespace(code="decode-error", section="bone", message="警告" + UNREP)

    def fake_read(path):
        return VmdDocument(camera=cam_keys), [warn]

    monkeypatch.setattr(cli.io, "read", fake_read)
    _cp932_stderr(monkeypatch)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=cam_keys)  # isfile 検査を通すため実在させる(内容は fake_read が差し替える)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0


# --- 既定挙動の回帰: 機械モードは出力VMDを変えない ---------------


def test_machine_output_equals_non_machine_output(tmp_path):
    # --machine の有無で出力VMDはバイト一致(機械モードは出力ファイル・削減結果を変えない)。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    out_h = tmp_path / "human.vmd"
    out_m = tmp_path / "machine.vmd"
    assert cli.main([str(src), "-o", str(out_h), "--curve-mode", "linear"]) == 0
    assert cli.main([str(src), "-o", str(out_m), "--curve-mode", "linear", "--machine"]) == 0
    assert out_h.read_bytes() == out_m.read_bytes()
