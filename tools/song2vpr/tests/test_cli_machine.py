"""song2vpr CLI 機械モード骨格・構造化エラーのテスト。

機械モードは標準出力を JSON Lines のイベント専用にし、失敗は確定 code/field/exit_code の error
イベントで終端する。非機械モードは失敗理由を標準エラーへ1行出す。

本モジュールは骨格(--version/--machine/--describe フラグ・単一の失敗経路・入力 positional の
省略可能化・終端規則)と、引数解析段の構造化エラー経路(bad_argument・output_exists・
output_is_directory)を対象にする。処理経路のイベントの中身(progress/warning/result の
mode:"run"/"inspect" のペイロード)は test_cli_output.py が見る。

機械モード標準出力は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。
"""

import json
import sys

import pytest

from song2vpr import cli


def _touch(path):
    path.write_bytes(b"")
    return str(path)


class _UnwritableStdout:
    """標準出力へ書けない状況(閉じたパイプ等)を再現する差し替え先。"""

    class _Buffer:
        def write(self, data):
            raise OSError("stdout is closed")

        def flush(self):
            raise OSError("stdout is closed")

    buffer = _Buffer()

    def write(self, text):
        raise OSError("stdout is closed")

    def flush(self):
        raise OSError("stdout is closed")


def machine_events(capsysbinary):
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")  # UTF-8 固定(ロケール非依存)を前提に decode
    return [json.loads(ln) for ln in text.split("\n") if ln]


def machine_error(capsysbinary):
    events = machine_events(capsysbinary)
    assert events, "標準出力に少なくとも1イベントが要る"
    assert events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1  # 終端はちょうど1つ
    return events[-1]


# --- メタ操作(--version/--help)は --machine 併用でも人間向けのまま ----------


def test_version_prints_and_exits_zero(capsys):
    import song2vpr

    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "song2vpr" in out and song2vpr.__version__ in out


def test_machine_version_stays_human(capsys):
    assert cli.main(["--machine", "--version"]) == 0
    out = capsys.readouterr().out
    assert "song2vpr" in out and not out.lstrip().startswith("{")


def test_machine_help_stays_human(capsys):
    assert cli.main(["--machine", "--help"]) == 0
    out = capsys.readouterr().out
    assert "song2vpr" in out and not out.lstrip().startswith("{")


# --- 正常経路の終端 ----------------------------------------------------------


def test_machine_processing_path_terminates_with_one_event(tmp_path, capsysbinary):
    """処理経路へ入った実行も、ストリームを result か error のちょうど1つで終端する。

    音声の中身を持たない入力で、共有フィクスチャが外部コマンドの照会を不在へ倒すため、前段は
    復号器の未検出で終わる。終端規則は終了コードに依らず成立するので、ここでは終了コードを見ない。
    """
    src = _touch(tmp_path / "in.wav")
    cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr"), "--dry-run"])
    events = machine_events(capsysbinary)
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1


# --- 引数解析段の構造化エラー ------------------------------------------------


def test_machine_error_bad_argument_unknown_option(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    assert cli.main(["--machine", src, "--bogus"]) == 2
    event = machine_error(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "--bogus"
    assert event["exit_code"] == 2
    assert isinstance(event["message"], str) and event["message"]


def test_machine_error_bad_argument_missing_input(capsysbinary):
    assert cli.main(["--machine"]) == 2
    event = machine_error(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "input"
    assert event["exit_code"] == 2


def test_machine_error_bad_argument_value_error_names_the_option(tmp_path, capsysbinary):
    """値の不正は、どの引数が不正かを field で示す。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main(["--machine", src, "--max-duration", "abc"]) == 2
    event = machine_error(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "--max-duration"


def test_machine_error_combination_check_names_the_option(tmp_path, capsysbinary):
    """組み合わせ検証の違反は、共有側が返す対象引数の長形式フラグ名を field に載せる。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main(["--machine", src, "--recognizer-model-revision", "abc123"]) == 2
    event = machine_error(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "--recognizer-model-revision"
    assert event["exit_code"] == 2


def test_machine_error_combination_check_names_the_missing_option(tmp_path, capsysbinary):
    """必須項目の欠落は、欠けている当の引数名を field に載せる(先に判定される項目から順に)。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main(["--machine", src, "--forced-aligner", "sofa-forcedalign"]) == 2
    event = machine_error(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "--sofa-python"
    assert event["exit_code"] == 2


def test_machine_error_output_exists(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main(["--machine", src, "-o", out, "--dry-run"]) == 2
    event = machine_error(capsysbinary)
    assert event["code"] == "output_exists"
    assert event["field"] == "--output"
    assert event["path"] is None  # この経路は対象パスを path 欄に持たない
    assert event["exit_code"] == 2


@pytest.mark.parametrize("extra", [[], ["--overwrite"]])
def test_machine_error_output_is_directory(tmp_path, capsysbinary, extra):
    """ディレクトリは --overwrite でも書けないので、上書きガードより前に専用コードで拒否する。"""
    src = _touch(tmp_path / "in.wav")
    outdir = tmp_path / "dir"
    outdir.mkdir()
    assert cli.main(["--machine", src, "-o", str(outdir), *extra, "--dry-run"]) == 2
    event = machine_error(capsysbinary)
    assert event["code"] == "output_is_directory"
    assert event["field"] == "--output"
    assert event["path"] == str(outdir)
    assert event["exit_code"] == 2


def test_machine_error_stdout_is_valid_json_lines_lf_only(tmp_path, capsysbinary):
    """機械モード標準出力は LF 区切りの JSON Lines(CR を混ぜない)。"""
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main(["--machine", src, "-o", out, "--dry-run"]) == 2
    raw = capsysbinary.readouterr().out
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    for line in raw.decode("utf-8").split("\n")[:-1]:
        json.loads(line)


def test_broken_stdout_in_machine_mode_reports_reason_without_traceback(tmp_path, monkeypatch,
                                                                       capsys):
    # 標準出力へ書けないと終端イベントを出せないが、例外をトレースバックのまま漏らさず、標準エラーへ
    # 理由1行だけを出し、その時点で確定している失敗の終了コードで終える。上書きガードで失敗する経路を
    # 使い、処理へ進めずに失敗報告だけを突く。
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    # 同じ失敗を壊れていない標準出力で走らせ、報告される理由行を控える。
    assert cli.main([src, "-o", out]) == 2
    expected = capsys.readouterr().err.splitlines()
    assert len(expected) == 1 and expected[0].startswith("error: ")

    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", _UnwritableStdout())
        rc = cli.main([src, "-o", out, "--machine"])
    assert rc == 2  # 上書きガードの終了コード(標準出力へ書けないことで変わらない)
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1  # 理由1行だけ(トレースバック等の余分な行が無い)
    # 報告する理由は元の失敗のまま(標準出力へ書けなかったこと自体を理由に差し替えない)。
    assert err[0] == expected[0]


# --- 中断と想定外の失敗 ------------------------------------------------------


def _raise(exc):
    def _f(*_args, **_kwargs):
        raise exc

    return _f


def test_interrupt_is_reported_as_cancelled(tmp_path, monkeypatch, capsysbinary):
    """中断はどの時点で届いても cancelled/130 として畳む。"""
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "_build_parser", _raise(KeyboardInterrupt()))
    assert cli.main(["--machine", src]) == 130
    event = machine_error(capsysbinary)
    assert event["code"] == "cancelled"
    assert event["field"] is None  # 中断は対象引数を持たない
    assert event["exit_code"] == 130


def test_interrupt_in_non_machine_mode_prints_single_line(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "_build_parser", _raise(KeyboardInterrupt()))
    assert cli.main([src]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""  # 非機械モードは標準出力へ JSON を出さない
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")


def test_unexpected_exception_is_reported_as_internal_error(tmp_path, monkeypatch, capsysbinary):
    """想定外の失敗はトレースバックを漏らさず internal_error/1 へ畳む。"""
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "_build_parser", _raise(RuntimeError("boom")))
    assert cli.main(["--machine", src]) == 1
    event = machine_error(capsysbinary)
    assert event["code"] == "internal_error"
    assert event["field"] is None  # 想定外の失敗は対象引数を持たない
    assert event["exit_code"] == 1


def test_unexpected_exception_in_non_machine_mode_omits_traceback(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "_build_parser", _raise(RuntimeError("boom")))
    assert cli.main([src]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert len(captured.err.splitlines()) == 1


# --- 非機械モードは同じ判定を人間向け1行で返す ------------------------------


def test_non_machine_overwrite_guard_prints_reason_to_stderr(tmp_path, capsys):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main([src, "-o", out, "--dry-run"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""  # 標準出力へ JSON を漏らさない
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")


def test_non_machine_usage_error_is_single_error_line(tmp_path, capsys):
    """argparse が検出する使用法エラーも、usage ブロックを添えず理由1行だけを出す。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--bogus"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "usage:" not in captured.err


def test_non_machine_missing_input_is_single_error_line(capsys):
    """入力の欠落は CLI 本体の検査。こちらも理由1行だけで終わる。"""
    assert cli.main([]) == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
