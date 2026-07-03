"""shakevmd CLI 移植性のテスト(shakevmd.md §2.8/§12, 規約 §10)。

機械モード標準出力の UTF-8・改行 LF 固定、非ASCIIパスの受理と生成、人間向け標準エラーの
符号化安全性(ロケール符号化で表せない文字でもプロセスを落とさない)を検証する。テスト方針は
../../../libs/vmd/vmd.md §4 に準ずる(決定論的・外部依存なし)。
"""

import io
import sys

from shakevmd import bake as bake_mod
from shakevmd import cli
from shakevmd.warn import ShakeWarning
from vmd import io as vmd_io
from vmd.types import BoneKey, CameraKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6

# cp932(Windows のロケール符号化)で表せない文字(絵文字 U+1F3A5)。ロケール符号化外の文字を
# 人間向け標準エラーへ書く経路を作り、符号化安全性を検証するために使う。
UNREP = "\U0001f3a5"


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


# --- 機械モード標準出力: UTF-8 + 改行 LF 固定(規約 §10) ---------------------


def test_machine_stdout_uses_lf_only(tmp_path, capsysbinary):
    # 機械モードの各行は LF(\n)終端で、\r を一切含まない(CRLF 変換なし。バイト列で検証)。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n")
    assert b"\r" not in raw                        # CR を混入させない(CRLF 変換なし)
    for line in raw.split(b"\n")[:-1]:
        assert line and not line.endswith(b"\r")   # 各行が非空で CR 終端でない


def test_machine_stdout_non_ascii_is_utf8(tmp_path, capsysbinary):
    # 非ASCII(日本語の出力パス・警告メッセージ)を UTF-8 のまま出す(ロケール符号化に依存しない)。
    bone = [BoneKey(name_raw=b"bone".ljust(15, b"\x00"), frame=0,
                    position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                    interpolation=bytes(64))]
    inp = write_input(tmp_path / "入力.vmd", bone=bone)
    out = tmp_path / "出力.vmd"
    rc = cli.main([inp, "-o", str(out), "--machine", "--no-smooth"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    # 出力パスの日本語が UTF-8 バイトで載る(ensure_ascii=False 相当・\uXXXX へエスケープしない)。
    assert "出力".encode("utf-8") in raw
    # 非カメラセクション透過の警告メッセージ(日本語)も UTF-8 で載る。
    assert "カメラ以外".encode("utf-8") in raw
    text = raw.decode("utf-8")                      # UTF-8 として復号できる
    assert "出力" in text and "カメラ以外" in text


# --- 非ASCIIパスの受理・生成(規約 §10) -------------------------------------


def test_non_ascii_path_roundtrip(tmp_path):
    # 日本語ファイル名の入力を受理し、日本語ファイル名の出力を生成できる。
    inp = write_input(tmp_path / "手ぶれ入力.vmd")
    out = tmp_path / "手ぶれ出力.vmd"
    rc = cli.main([inp, "-o", str(out), "--no-smooth"])
    assert rc == 0
    assert out.exists()
    doc, _ = vmd_io.read(str(out))                 # 生成物が妥当な VMD として読める
    assert doc.camera


# --- 人間向け標準エラーの符号化安全性(規約 §10) ---------------------------
# ロケール符号化(cp932)相当へ差し替えた標準エラーの下で、表せない文字を含む人間向け出力
# (argparse 使用法エラー・fail() の error 行・警告ループの warning 行)が UnicodeEncodeError で
# 本体を異常終了/internal_error へ落とさないことを検証する。この符号化安全性は cli.py が標準エラーの
# エラーハンドラを backslashreplace へ緩めることで担い、緩和が無ければ各経路は例外で失敗する。


def _cp932_stderr(monkeypatch):
    """sys.stderr をロケール符号化(cp932)相当・strict へ差し替える。"""
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


def test_stderr_safe_argparse_usage_error(monkeypatch):
    # (a) argparse 使用法エラー経路: 表せない文字を含む不正引数値。argparse が標準エラーへ書く
    # 使用法エラーが符号化に失敗せず、引数エラー(2)で終える(例外を漏らさない)。
    _cp932_stderr(monkeypatch)
    rc = cli.main(["in.vmd", "--seed", UNREP])
    assert rc == 2


def test_stderr_safe_fail_path(tmp_path, monkeypatch):
    # (b) fail() 経路: 表せない文字を含むパス。上書きガードの fail() メッセージが符号化に失敗せず、
    # 引数エラー(2)で終える(符号化失敗を internal_error(1)へ落とさない)。
    _cp932_stderr(monkeypatch)
    p = str(tmp_path / (UNREP + ".vmd"))
    rc = cli.main([p, "-o", p])   # 入力=出力・--overwrite 未指定 → output_overwrites_input(2)
    assert rc == 2


def test_stderr_safe_warning_loop(tmp_path, monkeypatch):
    # (c) 警告ループ経路: 表せない文字を含む警告文。warning 行が符号化に失敗せず本体は正常終了(0)
    # する(符号化失敗を internal_error(1)へ落とさない)。emitter 非経由の人間向け stderr を突く。
    def fake_bake(camera_keys, *a, **k):
        return bake_mod.BakeResult(
            camera_keys=list(camera_keys),
            warnings=[ShakeWarning("passthrough_test", "警告" + UNREP, None)],
            resolved=[(0, 60)],
        )
    monkeypatch.setattr(cli, "bake", fake_bake)
    _cp932_stderr(monkeypatch)
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--no-smooth"])
    assert rc == 0
