"""追加依存(vocal-analysis extra)が未導入のときの起動のテスト。

コンソールスクリプトは追加依存なしの導入でも登録されるため、依存が揃わない環境から起動されうる。
その場合にトレースバックを出さず、理由1行(または機械モードの error イベント)と終了コード4で終える
ことを検証する。判定を終了へ反映するのは引数の検証をすべて終えた後なので、メタ操作(`--help`/
`--version`/`--describe`)と引数エラーは依存の有無に依らず先に成立する。

依存を実際に外して起動することはできないため、取り込み結果を保持するモジュール変数を差し替えて
未導入状態を再現する。あわせて、取り込みに失敗すれば未定義になる名前をモジュールから外す。名前が
定義されたままだと、ガードより前の経路が誤ってそれらを参照する回帰が入っても検出できない。
"""

import json

import pytest

from song2vmd import cli

_MISSING = ModuleNotFoundError("No module named 'soundfile'", name="soundfile")

# 追加依存を要する取り込みが与える名前。取り込みが失敗した環境では未定義になる。
_GUARDED_NAMES = ("AudioLoadError", "RecognitionError", "SeparationError", "_pipeline")


@pytest.fixture
def missing_dependency(monkeypatch):
    monkeypatch.setattr(cli, "_MISSING_DEPENDENCY", _MISSING)
    for name in _GUARDED_NAMES:
        monkeypatch.delattr(cli, name)


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


def test_version_succeeds_without_dependencies(capsys, missing_dependency):
    # --version は処理を起動しないメタ操作なので、追加依存が無くても人間向け1行と終了コード0で終わる。
    rc = cli.main(["--version"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "song2vmd" in captured.out
    assert captured.err == ""


def test_help_succeeds_without_dependencies(capsys, missing_dependency):
    # --help も同じく成立する。オプションの定義は追加依存を要さない取り込みだけで組み立てられる。
    rc = cli.main(["--help"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "--style" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("argv", [
    pytest.param([], id="input欠落"),
    pytest.param(["in.wav", "--forced-aligner", "sofa-forcedalign"], id="SOFA必須検証"),
    pytest.param(["in.wav", "--recognizer-model-revision", "abc"], id="組み合わせ検証"),
    pytest.param(["in.wav", "--output", "existing.vmd"], id="上書きガード"),
    pytest.param(["in.wav", "--output", "existing_dir"], id="出力先がディレクトリ"),
])
def test_argument_errors_keep_their_own_exit_code(argv, tmp_path, monkeypatch, capsys,
                                                  missing_dependency):
    # 引数の検証はすべて依存ガードより前で完結するため、引数エラーは missing_dependency へ
    # 吸われず引数エラーのまま返る(上書きガード・出力先の検査を含む)。
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing.vmd").write_bytes(b"")
    (tmp_path / "existing_dir").mkdir()
    rc = cli.main(argv)
    assert rc == 2
    err = capsys.readouterr().err
    assert "soundfile" not in err


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


def test_describe_succeeds_without_dependencies(capsysbinary, missing_dependency):
    # --describe は音声を読まない独立メタ操作なので、追加依存が無くてもオプション定義を返す
    # (機械利用側が導入前にオプションを列挙できる)。
    rc = cli.main(["--describe"])
    assert rc == 0
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    assert len(events) == 1
    assert events[0]["type"] == "result"
    assert events[0]["mode"] == "describe"
    assert events[0]["options"]
