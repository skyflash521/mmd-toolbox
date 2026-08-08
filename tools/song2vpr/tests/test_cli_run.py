"""song2vpr CLI の処理経路のテスト。

パイプラインが送出する失敗を仕様の `code`・`field`・`path`・終了コードへ写すこと、および中間生成物の
保存先の決め方を検証する。パイプライン自体は差し替え、CLI が持つ写像と組み立てだけを見る。
"""

import json
import types

import pytest

from song2vpr import cli
from vocal_analysis.front_stage import (
    IntermediateReadError,
    IntermediateWriteError,
    StageExecutionError,
)
from vocal_analysis.io import AudioLoadError
from vocal_analysis.recognizer import RecognitionError
from vocal_analysis.separator import SeparationError

from .support import front_stage_result


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _machine_error(capsysbinary):
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    assert events[-1]["type"] == "error"
    return events[-1]


def _stub_pipeline(monkeypatch, raises=None, captured=None):
    """パイプラインを差し替える。raises を渡すとその例外を送出し、渡さなければ引数を記録する。"""
    def fake_run(input_path, **kwargs):
        if captured is not None:
            captured["input_path"] = input_path
            captured["kwargs"] = kwargs
        if raises is not None:
            raise raises
        return front_stage_result()

    # モジュールごと差し替える(取り込みが追加依存のガードの内側にあるため、属性を差し替えるより
    # 取り込みの構造に左右されない)。
    monkeypatch.setattr(cli, "_pipeline", types.SimpleNamespace(run=fake_run))


def _run_machine(tmp_path, monkeypatch, exc, extra=()):
    src = _touch(tmp_path / "in.wav")
    _stub_pipeline(monkeypatch, raises=exc)
    return cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr"), *extra])


# --- 成功経路の終端 ----------------------------------------------------------


@pytest.mark.xfail(reason="impl pending: result を組み立てる診断の段がまだ無い", strict=True)
def test_machine_success_path_terminates_with_a_result(tmp_path, monkeypatch, capsysbinary):
    """前段を終えた実行も、ストリームを result か error のちょうど1つで終端する。

    入力不正で終わる側は別のテストが見ている。こちらはパイプラインを差し替えて前段を成功させ、
    成功して終わる実行の終端を見る。
    """
    src = _touch(tmp_path / "in.wav")
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr")]) == 0
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1


# --- 失敗の写像 --------------------------------------------------------------


def test_intermediate_read_failure_is_not_attributed_to_the_user_input(tmp_path, monkeypatch,
                                                                      capsysbinary):
    """内部生成ファイルの読み直し失敗は、利用者入力を指す field を載せず対象を path に載せる。"""
    exc = IntermediateReadError("読み直しに失敗", path=str(tmp_path / "vocal.wav"))
    assert _run_machine(tmp_path, monkeypatch, exc) == 1
    event = _machine_error(capsysbinary)
    assert event["code"] == "not_audio"
    assert event["field"] is None
    assert event["path"] == str(tmp_path / "vocal.wav")
    assert event["exit_code"] == 1


def test_input_load_failure_points_at_the_input(tmp_path, monkeypatch, capsysbinary):
    exc = AudioLoadError("音声として読めない", reason="not_audio")
    assert _run_machine(tmp_path, monkeypatch, exc) == 1
    event = _machine_error(capsysbinary)
    assert event["code"] == "not_audio"
    assert event["field"] == "input"
    assert event["exit_code"] == 1


def test_missing_decoder_is_an_environment_failure(tmp_path, monkeypatch, capsysbinary):
    """復号器の未検出は入力の不備ではなく環境の不足なので、終了コードは4。"""
    exc = AudioLoadError("復号器が見つからない", reason="decoder_missing")
    assert _run_machine(tmp_path, monkeypatch, exc) == 4
    event = _machine_error(capsysbinary)
    assert event["code"] == "decoder_missing"
    assert event["field"] == "input"
    assert event["exit_code"] == 4


@pytest.mark.parametrize("stage", ["separate", "recognize"])
def test_unclassified_stage_failure_names_the_stage(tmp_path, monkeypatch, capsysbinary, stage):
    exc = StageExecutionError("推論に失敗", stage=stage)
    assert _run_machine(tmp_path, monkeypatch, exc) == 4
    event = _machine_error(capsysbinary)
    assert event["code"] == "stage_failed"
    assert event["stage"] == stage
    assert event["field"] is None
    assert event["exit_code"] == 4


def test_separation_failure_names_the_separation_stage(tmp_path, monkeypatch, capsysbinary):
    assert _run_machine(tmp_path, monkeypatch, SeparationError("分離に失敗")) == 4
    event = _machine_error(capsysbinary)
    assert event["code"] == "stage_failed"
    assert event["stage"] == "separate"


def test_recognition_failure_names_the_recognition_stage(tmp_path, monkeypatch, capsysbinary):
    assert _run_machine(tmp_path, monkeypatch, RecognitionError("認識に失敗")) == 4
    event = _machine_error(capsysbinary)
    assert event["code"] == "stage_failed"
    assert event["stage"] == "recognize"


def test_intermediate_write_failure_points_at_the_option_that_asked_for_it(tmp_path, monkeypatch,
                                                                          capsysbinary):
    """中間生成物の書き込み失敗は、保存を要求したオプションと保存先を示す。"""
    exc = IntermediateWriteError("書き込みに失敗")
    out = str(tmp_path / "out.vpr")
    assert _run_machine(tmp_path, monkeypatch, exc, extra=["--keep-intermediate"]) == 3
    event = _machine_error(capsysbinary)
    assert event["code"] == "write_failed"
    assert event["field"] == "--keep-intermediate"
    assert event["path"] == f"{out}.intermediate"
    assert event["exit_code"] == 3


def test_stage_failure_message_names_the_stage_in_human_terms(tmp_path, monkeypatch, capsys):
    """非機械モードでも、どの工程で失敗したかを利用者向けの工程名で1行示す。"""
    src = _touch(tmp_path / "in.wav")
    _stub_pipeline(monkeypatch, raises=StageExecutionError("推論に失敗", stage="separate"))
    assert cli.main([src, "-o", str(tmp_path / "out.vpr")]) == 4
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "ボーカル分離" in lines[0]


# --- 中間生成物の保存先 ------------------------------------------------------


def test_intermediate_directory_is_adjacent_to_the_output(tmp_path, monkeypatch):
    """保存先は出力に隣接する <出力ファイル名>.intermediate。"""
    captured = {}
    src = _touch(tmp_path / "in.wav")
    out = str(tmp_path / "out.vpr")
    _stub_pipeline(monkeypatch, captured=captured)

    assert cli.main([src, "-o", out, "--keep-intermediate"]) == 0
    assert captured["kwargs"]["keep_intermediate_dir"] == f"{out}.intermediate"


def test_intermediate_is_not_saved_without_the_option(tmp_path, monkeypatch):
    captured = {}
    src = _touch(tmp_path / "in.wav")
    _stub_pipeline(monkeypatch, captured=captured)

    assert cli.main([src, "-o", str(tmp_path / "out.vpr")]) == 0
    assert captured["kwargs"]["keep_intermediate_dir"] is None


def test_intermediate_is_saved_even_in_dry_run(tmp_path, monkeypatch):
    """--dry-run が抑制するのは最終 vpr の書き出しだけで、中間生成物の保存は行う。"""
    captured = {}
    src = _touch(tmp_path / "in.wav")
    out = str(tmp_path / "out.vpr")
    _stub_pipeline(monkeypatch, captured=captured)

    assert cli.main([src, "-o", out, "--keep-intermediate", "--dry-run"]) == 0
    assert captured["kwargs"]["keep_intermediate_dir"] == f"{out}.intermediate"


# --- 前段へ渡す設定の解決 ----------------------------------------------------


def test_resolved_front_stage_settings_are_passed_through(tmp_path, monkeypatch):
    """CLI が共通引数群から解決した設定を、そのままパイプラインへ渡す。"""
    captured = {}
    src = _touch(tmp_path / "in.wav")
    _stub_pipeline(monkeypatch, captured=captured)

    assert cli.main([
        src, "-o", str(tmp_path / "out.vpr"), "--separate-vocals", "never",
        "--separator", "audio-separator-htdemucs-ft", "--forced-aligner",
        "wav2vec2-ctc-forcedalign", "--max-duration", "120",
    ]) == 0
    kwargs = captured["kwargs"]
    assert captured["input_path"] == src
    assert kwargs["separate_vocals"] == "never"
    assert kwargs["separator_name"] == "audio-separator-htdemucs-ft"
    assert kwargs["forced_aligner"] == "wav2vec2-ctc-forcedalign"
    assert kwargs["chunking"].max_duration_sec == 120.0
