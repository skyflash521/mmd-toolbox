"""vpr2vmd CLI 移植性・既定挙動回帰のテスト。

非ASCIIパスの受理・生成、人間向け標準エラーの符号化安全性(ロケール符号化で表せない文字でもプロセスを
落とさない)、標準出力へ書けない場合に例外を漏らさないこと、機械モードが出力VMDを変えない
(--machine の有無で出力バイト一致)ことを検証する。
vpr.read は monkeypatch で差し替え、配線を決定論的に検証する。
"""

import io
import json
import sys

import pytest

from vmd import read as vmd_read
from vpr import Note, Part, TempoEvent, Track, VprProject, VprWarning
from vpr2vmd import cli

# cp932(Windows のロケール符号化)で表せない文字(絵文字 U+1F3A5)。ロケール符号化外の文字を
# 人間向け標準エラーへ書く経路を作り、符号化安全性を検証するために使う。
UNREP = "\U0001f3a5"


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
    """sys.stderr をロケール符号化(cp932)相当・strict へ差し替える(符号化安全性の検証用)。"""
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


# --- 非ASCIIパスの受理・生成 ----------------------------------------------------


def test_non_ascii_path_roundtrip_non_machine(tmp_path, monkeypatch):
    # 日本語ファイル名の入力を受理し、日本語ファイル名の出力 VMD を生成できる(非機械)。
    src = tmp_path / "ボーカル入力.vpr"
    src.write_bytes(b"")
    out = tmp_path / "リップモーション出力.vmd"
    _stub_read(monkeypatch)
    rc = cli.main([str(src), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    doc, _ = vmd_read(str(out))  # 生成物が妥当な VMD として読める
    assert doc.morph


def test_non_ascii_path_machine(tmp_path, monkeypatch, capsysbinary):
    # 日本語パスでも機械モードで convert result を出し、出力を生成する。
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


# --- 人間向け標準エラーの符号化安全性 -------------------------------------------


def test_stderr_safe_on_argparse_usage_error(monkeypatch):
    # 表せない文字を含む不正引数値でも、argparse 使用法エラーが符号化に失敗せず引数エラー(2)で終える。
    _cp932_stderr(monkeypatch)
    rc = cli.main(["in.vpr", "--open-max", UNREP])
    assert rc == 2


def test_stderr_safe_on_warning(tmp_path, monkeypatch):
    # 表せない文字を含む警告文でも本体は正常終了(0)する(符号化失敗でプロセスを落とさない)。
    warn = VprWarning(code="overlapping_notes", message="重なり" + UNREP, track_index=0)
    _stub_read(monkeypatch, warnings=[warn])
    _cp932_stderr(monkeypatch)
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd")])
    assert rc == 0


# --- 標準出力へ書けない場合 -----------------------------------------------------


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
        rc = cli.main([str(tmp_path / "nope.vpr"), "--machine"])
    assert rc == 1  # 入力不在の終了コード(標準出力へ書けないことで変わらない)
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1  # 理由1行だけ(トレースバック等の余分な行が無い)
    # 報告する理由は元の失敗のまま(標準出力へ書けなかったこと自体を理由に差し替えない)。
    assert err[0].startswith("error: ") and "入力 vpr が見つかりません" in err[0]


# --- 既定挙動の回帰: 機械モードは出力VMDを変えない ------------------------------


def test_machine_output_equals_non_machine_output(tmp_path, monkeypatch):
    # --machine の有無で出力VMDはバイト一致(機械モードは出力ファイル・変換結果を変えない)。
    _stub_read(monkeypatch)
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    out_h = tmp_path / "human.vmd"
    out_m = tmp_path / "machine.vmd"
    assert cli.main([str(src), "-o", str(out_h)]) == 0
    assert cli.main([str(src), "-o", str(out_m), "--machine"]) == 0
    assert out_h.read_bytes() == out_m.read_bytes()
