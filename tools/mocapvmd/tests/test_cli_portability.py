"""mocapvmd CLI 移植性のテスト。

機械モード標準出力の UTF-8・改行 LF 固定、非ASCIIパスの受理と生成、人間向け標準エラーの符号化安全性
(ロケール符号化で表せない文字でもプロセスを落とさない)、標準出力へ書けない場合に例外を漏らさない
ことを検証する。テストは決定論的に実行し、外部依存を使わない。
"""

import io
import sys
import types

import pytest

from mocapvmd import cli
from vmd import io as vmd_io
from vmd.reduce import BONE_LINEAR_INTERP
from vmd.types import BoneKey, VmdDocument

from .helpers import bone, write_vmd

# cp932(Windows のロケール符号化)で表せない文字(絵文字 U+1F3A5)。ロケール符号化外の文字を
# 人間向け標準エラーへ書く経路を作り、符号化安全性を検証するために使う。
UNREP = "\U0001f3a5"


def _ramp(path):
    write_vmd(path, bone=[bone("センター", f, pos=(float(f), 0.0, 0.0)) for f in range(11)])


# --- 機械モード標準出力: UTF-8 + 改行 LF 固定 -------------------------------


def test_machine_stdout_uses_lf_only(tmp_path, capsysbinary):
    # 機械モードの各行は LF(\n)終端で \r を一切含まない(CRLF 変換なし。バイト列で検証)。
    src = tmp_path / "in.vmd"
    _ramp(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    for line in raw.split(b"\n")[:-1]:
        assert line and not line.endswith(b"\r")


def test_machine_stdout_non_ascii_is_utf8(tmp_path, capsysbinary):
    # 非ASCII(日本語の出力パス)を UTF-8 のまま出す(ロケール符号化に依存しない・\uXXXX へエスケープしない)。
    src = tmp_path / "入力.vmd"
    _ramp(src)
    out = tmp_path / "出力.vmd"
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert "出力".encode("utf-8") in raw
    text = raw.decode("utf-8")  # UTF-8 として復号できる
    assert "出力" in text


# --- 非ASCIIパスの受理・生成 -------------------------------------------------


def test_non_ascii_path_roundtrip(tmp_path):
    # 日本語ファイル名の入力を受理し、日本語ファイル名の出力を生成できる。
    src = tmp_path / "モーション入力.vmd"
    _ramp(src)
    out = tmp_path / "モーション出力.vmd"
    rc = cli.main([str(src), "-o", str(out), "--no-reduce"])
    assert rc == 0
    assert out.exists()
    doc, _ = vmd_io.read(str(out))  # 生成物が妥当な VMD として読める
    assert doc.bone


# --- 人間向け標準エラーの符号化安全性 ---------------------------------------
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
    rc = cli.main(["in.vmd", "--clean-strength", UNREP])
    assert rc == 2


def test_stderr_safe_warning_loop(tmp_path, monkeypatch):
    # 警告ループ経路: 表せない文字を含む警告文でも本体は正常終了(0)する(符号化失敗で落とさない)。
    # io.read を差し替え、非ASCIIメッセージの警告を人間向け stderr へ流す経路を突く。
    keys = [BoneKey(b"c".ljust(15, b"\x00"), f, (float(f), 0.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP) for f in range(4)]
    warn = types.SimpleNamespace(code="decode-error", section="bone", message="警告" + UNREP)

    def fake_read(path):
        return VmdDocument(bone=keys), [warn]

    monkeypatch.setattr(cli.io, "read", fake_read)
    _cp932_stderr(monkeypatch)
    src = tmp_path / "in.vmd"
    _ramp(src)  # isfile 検査を通すため実在させる(内容は fake_read が差し替える)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--no-reduce"])
    assert rc == 0


# --- 標準出力へ書けない場合 -------------------------------------------------


class _UnwritableStdout:
    """buffer への書き込みが常に失敗する標準出力(呼び出し側がパイプを先に閉じた状況)。"""

    class _Buffer:
        def write(self, _data):
            raise OSError("broken pipe")

    def __init__(self):
        self.buffer = self._Buffer()


@pytest.mark.xfail(reason="impl pending: cli_events の失敗報告ヘルパ emit_failure が未実装")
def test_broken_stdout_in_machine_mode_reports_reason_without_traceback(tmp_path, monkeypatch,
                                                                       capsys):
    # 標準出力へ書けないと終端イベントを出せないが、例外をトレースバックのまま漏らさず、標準エラーへ
    # 理由1行だけを出し、その時点で確定している失敗の終了コードで終える。
    # 差し替えは CLI 呼び出しの区間だけに限り、標準エラーを読み出す前に元へ戻す。
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", _UnwritableStdout())
        rc = cli.main([str(tmp_path / "nope.vmd"), "--machine"])
    assert rc == 1  # 入力不在の終了コード(標準出力へ書けないことで変わらない)
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1  # 理由1行だけ(トレースバック等の余分な行が無い)
    # 報告する理由は元の失敗のまま(標準出力へ書けなかったこと自体を理由に差し替えない)。
    assert err[0].startswith("error: ") and "入力が存在しないか通常ファイルでない" in err[0]
