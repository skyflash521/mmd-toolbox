"""song2vmd CLI 機械モード骨格・構造化エラーのテスト。

機械モードは stdout を JSON Lines のイベント専用にし、失敗は確定 code/field/exit_code の error
イベントで終端する。非機械モードは失敗理由を標準エラーへ1行出す。既定(非機械)挙動が不変であること
(後方互換)も併せて検証する。

本モジュールは骨格(--version/--machine/--describe/--quiet フラグ・MachineArgumentParser 切替・
emitter・fail() 単一失敗経路・help= 付与・input の nargs="?" 化)と、CLI 引数解析段の構造化エラー
経路(bad_argument・output_exists)を対象にする。音声読み込み以降の成功経路のイベント
(progress/warning/result mode:"run"/"inspect")・中断は音声処理パイプラインの実装後に検証する。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。
"""

import json
import sys

import pytest

from song2vmd import cli


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def machine_events(capsysbinary):
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")  # UTF-8 固定(ロケール非依存)を前提に decode
    return [json.loads(ln) for ln in text.split("\n") if ln]


def machine_error(capsysbinary):
    events = machine_events(capsysbinary)
    assert events, "stdout に少なくとも1イベントが要る"
    assert events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1  # 終端はちょうど1つ
    return events[-1]


# --- メタ操作(--version/--help)は --machine 併用でも人間向けのまま ----------


def test_version_prints_and_exits_zero(capsys):
    from song2vmd import __version__

    rc = cli.main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "song2vmd" in out and __version__ in out


def test_machine_version_stays_human(capsys):
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "song2vmd" in out and not out.lstrip().startswith("{")


def test_machine_help_stays_human(capsys):
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


# --- 引数エラー(argparse 検出。bad_argument・exit_code 2)-------------------


def test_machine_error_bad_argument_unknown_option(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([src, "--machine", "--bogus"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2
    assert e["field"] == "--bogus"
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_bad_argument_missing_input(capsysbinary):
    rc = cli.main(["--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "input" and e["exit_code"] == 2


def test_machine_error_bad_argument_type_error_field(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([src, "--machine", "--max-duration", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--max-duration" and e["exit_code"] == 2


def test_machine_error_bad_argument_unknown_style(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([src, "--machine", "--style", "nope"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--style" and e["exit_code"] == 2


# --- 上書きガード ------------------------------------------------------------


def test_machine_error_output_exists(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([str(src), "-o", str(src), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output" and e["exit_code"] == 2


def test_machine_error_output_exists_distinct_path(tmp_path, capsysbinary):
    # 入力と別パスの既存出力も機械モードで output_exists を返すこと。
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vmd")
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output" and e["exit_code"] == 2


# --- チャネル固定(JSON Lines・LF・UTF-8)------------------------------------


def test_machine_error_stdout_is_valid_json_lines_lf_only(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([src, "-o", str(src), "--machine"])
    assert rc == 2
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n") and b"\r" not in raw
    for ln in raw.decode("utf-8").split("\n"):
        if ln:
            obj = json.loads(ln)
            assert "type" in obj


# --- 非機械モードの理由1行 --------------------------------------------------


def test_non_machine_missing_input_is_arg_error(capsys):
    rc = cli.main([])
    assert rc == 2
    assert "error:" in capsys.readouterr().err.lower()


def test_non_machine_overwrite_guard_prints_reason_to_stderr(tmp_path, capsys):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([src, "-o", src])
    assert rc == 2
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")


# --- 標準出力へ書けない場合 ---------------------------------------


class _UnwritableStdout:
    """buffer への書き込みが常に失敗する標準出力(呼び出し側がパイプを先に閉じた状況)。"""

    class _Buffer:
        def write(self, _data):
            raise OSError("broken pipe")

    def __init__(self):
        self.buffer = self._Buffer()


def test_broken_stdout_in_machine_mode_reports_reason_without_traceback(tmp_path, monkeypatch,
                                                                       capsys):
    # 標準出力へ書けないと終端イベントを出せないが、例外をトレースバックのまま漏らさず、標準エラーへ
    # 理由1行だけを出し、その時点で確定している失敗の終了コードで終える。上書きガードで失敗する経路を
    # 使い、音声処理パイプラインへ進めずに失敗報告だけを突く。
    src = _touch(tmp_path / "in.wav")
    # 差し替えは CLI 呼び出しの区間だけに限り、標準エラーを読み出す前に元へ戻す。
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", _UnwritableStdout())
        rc = cli.main([src, "-o", src, "--machine"])
    assert rc == 2  # 上書きガードの終了コード(標準出力へ書けないことで変わらない)
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1  # 理由1行だけ(トレースバック等の余分な行が無い)
    # 報告する理由は元の失敗のまま(標準出力へ書けなかったこと自体を理由に差し替えない)。
    assert err[0].startswith("error: ") and "出力先に既存ファイルがあります" in err[0]


@pytest.mark.xfail(reason="impl pending: 出力先が既存ディレクトリのときの output_is_directory が未実装")
def test_machine_error_output_is_directory(tmp_path, capsysbinary):
    # 出力先が既存ディレクトリ → output_is_directory(exit 2)。ディレクトリは --overwrite でも
    # 書けないので、併用しても同じコードで拒否する(上書きの許可を促す案内へ落とさない)。
    src = _touch(tmp_path / "in.wav")
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    for extra in ([], ["--overwrite"]):
        rc = cli.main([src, "-o", str(outdir), "--machine", *extra])
        assert rc == 2
        e = machine_error(capsysbinary)
        assert e["code"] == "output_is_directory" and e["field"] == "--output"
        assert e["exit_code"] == 2 and e["path"] == str(outdir)
