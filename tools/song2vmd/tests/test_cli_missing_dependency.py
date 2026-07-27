"""追加依存(vocal-analysis extra)が未導入のときの起動のテスト。

コンソールスクリプトは追加依存なしの導入でも登録されるため、依存が揃わない環境から起動されうる。
その場合にトレースバックを出さず、理由1行(または機械モードの error イベント)と終了コード4で終える
ことを検証する。判定は引数解析より前なので `--version` のようなメタ操作も同じ経路になる。

依存を実際に外して起動することはできないため、取り込み結果を保持するモジュール変数を差し替えて
未導入状態を再現する。
"""

import json

import pytest

from song2vmd import cli

_MISSING = ModuleNotFoundError("No module named 'soundfile'", name="soundfile")


@pytest.fixture
def missing_dependency(monkeypatch):
    monkeypatch.setattr(cli, "_MISSING_DEPENDENCY", _MISSING)


def test_dependencies_present_in_dev_environment():
    # 開発インストール(extra 込み)では未導入判定が立たないこと。取り込み失敗を握り潰したまま
    # 気付かない状態を防ぐ。
    assert cli._MISSING_DEPENDENCY is None


def test_human_message_is_single_line_without_traceback(tmp_path, capsys, missing_dependency):
    rc = cli.main([str(tmp_path / "in.wav")])
    assert rc == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Traceback" not in captured.err


def test_human_message_names_module_and_install_command(tmp_path, capsys, missing_dependency):
    cli.main([str(tmp_path / "in.wav")])
    err = capsys.readouterr().err
    assert "soundfile" in err
    assert 'pip install ".[vocal-analysis]"' in err


def test_meta_operations_take_the_same_path(capsys, missing_dependency):
    # --version は依存が揃っていれば終了コード0のメタ操作だが、取り込み時点で失敗しているため
    # 同じ missing_dependency 経路になる。
    rc = cli.main(["--version"])
    assert rc == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "soundfile" in captured.err


def test_machine_mode_emits_error_event(tmp_path, capsysbinary, missing_dependency):
    rc = cli.main([str(tmp_path / "in.wav"), "--machine"])
    assert rc == 4
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "error"
    assert event["code"] == "missing_dependency"
    assert event["exit_code"] == 4
    assert event["field"] is None
    assert "soundfile" in event["message"]


def test_describe_takes_the_same_path(capsysbinary, missing_dependency):
    # --describe も音声を読まないメタ操作だが、取り込み時点で失敗しているため同じ経路になる。
    rc = cli.main(["--describe"])
    assert rc == 4
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    assert [e["code"] for e in events] == ["missing_dependency"]
