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

from song2vpr import cli

_MISSING = ModuleNotFoundError("No module named 'soundfile'", name="soundfile")

# 追加依存を要する取り込みが与える名前。取り込みが失敗した環境では未定義になる。
_GUARDED_NAMES = ("_front_stage",)


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
    assert cli.main([str(tmp_path / "in.wav")]) == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Traceback" not in captured.err


def test_human_message_names_module_and_install_command(tmp_path, capsys, missing_dependency):
    """理由1行は不足モジュール名と、その extra を導入する pip のコマンドを示す(ツール名は入れない)。"""
    cli.main([str(tmp_path / "in.wav")])
    line = capsys.readouterr().err
    assert "soundfile" in line
    assert "pip install" in line and "vocal-analysis" in line
    assert "song2vpr" not in line


def test_machine_mode_reports_missing_dependency(tmp_path, capsysbinary, missing_dependency):
    assert cli.main(["--machine", str(tmp_path / "in.wav")]) == 4
    text = capsysbinary.readouterr().out.decode("utf-8")
    events = [json.loads(ln) for ln in text.split("\n") if ln]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "missing_dependency"
    assert events[0]["exit_code"] == 4
    assert events[0]["field"] is None
    assert "soundfile" in events[0]["message"]


def test_help_succeeds_without_dependency(capsys, missing_dependency):
    """オプション定義の組み立ては追加依存を要さない取り込みだけで済む。"""
    assert cli.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "--separate-vocals" in captured.out
    assert captured.err == ""


def test_version_succeeds_without_dependency(capsys, missing_dependency):
    import song2vpr

    assert cli.main(["--version"]) == 0
    captured = capsys.readouterr()
    assert song2vpr.__version__ in captured.out
    assert captured.err == ""


def test_describe_succeeds_without_dependency(capsysbinary, missing_dependency):
    assert cli.main(["--describe"]) == 0
    text = capsysbinary.readouterr().out.decode("utf-8")
    events = [json.loads(ln) for ln in text.split("\n") if ln]
    assert events[-1]["mode"] == "describe"


def _argv_before_guard(tmp_path, case):
    """依存ガードより前に終了コード2で終わる経路の argv。"""
    src = tmp_path / "in.wav"
    src.write_bytes(b"")
    if case == "unknown_option":
        return [str(src), "--bogus"]
    if case == "missing_input":
        return []
    if case == "value_error":
        return [str(src), "--max-duration", "abc"]
    if case == "combination":
        return [str(src), "--recognizer-model-revision", "abc123"]
    if case == "output_exists":
        out = tmp_path / "out.vpr"
        out.write_bytes(b"")
        return [str(src), "-o", str(out)]
    outdir = tmp_path / "dir"
    outdir.mkdir()
    return [str(src), "-o", str(outdir)]


@pytest.mark.parametrize("case", ["unknown_option", "missing_input", "value_error",
                                  "combination", "output_exists", "output_is_directory"])
def test_argument_checks_precede_dependency_guard(tmp_path, capsys, missing_dependency, case):
    """引数の検証と出力先のガードは処理の開始前なので、依存が無くても終了コード2で終わる。"""
    assert cli.main(_argv_before_guard(tmp_path, case)) == 2
    # 終了コード2が依存の不足を理由にしたものでないこと。
    assert "soundfile" not in capsys.readouterr().err
