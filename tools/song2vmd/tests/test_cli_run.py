"""song2vmd CLI のパイプライン配線テスト(song2vmd.md §5・§6.7・11章・12章)。

cli.py の _run() が pipeline.run() を正しい引数で呼び、その結果(PipelineResult)を
--dry-run の人間向けレポート/機械モードの result イベント、VMD 書き出し、警告発行、
エラー変換(AudioLoadError/SeparationError/RecognitionError/IntermediateWriteError)・中断(KeyboardInterrupt)へ
正しく橋渡しすることを検証する。pipeline.run 自体はモックし、実音声処理は行わない。
"""

import json

from vmd import VmdDocument
from vmd import read as vmd_read
from vocal_analysis import ContentRecognizerModel, DEFAULT_CONTENT_RECOGNIZER_MODEL
from vocal_analysis.io import AudioLoadError
from vocal_analysis.recognizer import RecognitionError
from vocal_analysis.separator import SeparationError

from song2vmd import cli
from song2vmd import events as _events
from song2vmd import pipeline as _pipeline
from song2vmd import presets as _presets
from song2vmd import report as _report


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _make_result(*, keys=3, low_dynamics=False, sample_rate=44100, channels=2):
    document = VmdDocument(model_name_raw=b"\x00" * 20, morph=[])
    diagnostics = _report.build_diagnostics(
        segments=[], mouth_events=[],
        event_diagnostics=_events.EventDiagnostics(weak_vowels=0, low_dynamics=low_dynamics, merged_morae=0),
        backends={"separator": "audio-separator-htdemucs-ft", "recognizer": "openai/whisper-medium"},
        style="pop", separated=True, duration_sec=2.5, keys=keys,
    )
    return _pipeline.PipelineResult(
        document=document, diagnostics=diagnostics, sample_rate=sample_rate, channels=channels)


def _capture_run_kwargs(monkeypatch, result=None):
    captured = {}

    def fake_run(input_path, **kwargs):
        captured["input_path"] = input_path
        captured["kwargs"] = kwargs
        return result if result is not None else _make_result()

    monkeypatch.setattr(cli._pipeline, "run", fake_run)
    return captured


def _events_of(capsysbinary):
    return [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").splitlines() if ln]


# --- pipeline.run への引数の受け渡し -------------------------------------------


def test_run_calls_pipeline_with_resolved_preset_and_default_recognizer(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "--dry-run"])
    assert rc == 0

    kwargs = captured["kwargs"]
    assert captured["input_path"] == src
    assert kwargs["content_recognizer_model"] is DEFAULT_CONTENT_RECOGNIZER_MODEL
    openness, style_gen = _presets.resolve("pop")
    assert kwargs["openness"] == openness
    assert kwargs["style_gen"] == style_gen
    assert kwargs["style_name"] == "pop"
    assert kwargs["separate_vocals"] == "auto"
    assert kwargs["separator_name"] == "audio-separator-htdemucs-ft"
    assert kwargs["max_duration_sec"] == 300.0
    assert kwargs["use_n_morph"] is True
    # --vowel-gain の既定 1:1:1:1:1 は presets.resolve での乗算後もプリセット値のまま
    # (song2vmd.md 8.2)。pipeline.run へは style_gen.vowel_scale として渡る。
    assert kwargs["style_gen"].vowel_scale == _presets.resolve("pop")[1].vowel_scale
    assert kwargs["intensity_curve"] == 0.6
    assert kwargs["silence_on"] == 0.06
    assert kwargs["model_name"] == ""
    assert kwargs["progress"] is not None
    assert kwargs["keep_intermediate_dir"] is None
    assert kwargs["forced_aligner"] == "wav2vec2-ctc-forcedalign"
    assert kwargs["sofa_aligner"] is None


def test_run_builds_sofa_aligner_config_from_cli_options(tmp_path, monkeypatch):
    """--forced-aligner sofa-forcedalign選択時、--sofa-*からSofaAlignerConfigが組み立てられる。"""
    from pathlib import Path

    from vocal_analysis import SofaAlignerConfig

    src = _touch(tmp_path / "in.wav")
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-root", "/sofa",
        "--sofa-checkpoint", "/ckpt.ckpt", "--sofa-timeout", "120", "--dry-run",
    ])
    assert rc == 0
    kwargs = captured["kwargs"]
    assert kwargs["forced_aligner"] == "sofa-forcedalign"
    assert kwargs["sofa_aligner"] == SofaAlignerConfig(
        sofa_python=Path("/venv/python"), sofa_root=Path("/sofa"),
        checkpoint_path=Path("/ckpt.ckpt"), timeout_sec=120.0)


def test_keep_intermediate_resolves_to_output_path_plus_suffix(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "-o", str(out), "--keep-intermediate", "--dry-run"])
    assert rc == 0
    assert captured["kwargs"]["keep_intermediate_dir"] == f"{out}.intermediate"


def test_run_builds_custom_content_recognizer_model_from_cli_options(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([
        src, "--recognizer-model-id", "org/model", "--recognizer-model-revision", "rev1", "--dry-run",
    ])
    assert rc == 0
    model = captured["kwargs"]["content_recognizer_model"]
    assert model == ContentRecognizerModel(model_id="org/model", model_revision="rev1")


def test_run_passes_default_retry_enabled(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert captured["kwargs"]["retry"] is True


def test_no_recognizer_retry_passes_false(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "--no-recognizer-retry", "--dry-run"])
    assert rc == 0
    assert captured["kwargs"]["retry"] is False


def test_run_passes_preset_overrides_through(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "--style", "powerful", "--open-max", "0.5", "--coarticulation", "9", "--dry-run"])
    assert rc == 0
    openness, style_gen = _presets.resolve("powerful", open_max=0.5, coarticulation=9)
    assert captured["kwargs"]["openness"] == openness
    assert captured["kwargs"]["style_gen"] == style_gen


# --- --dry-run: 人間向けレポート / 機械モード inspect --------------------------


def test_dry_run_non_machine_prints_report_text_to_stdout(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(keys=5))

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "keys: 5" in out


def test_dry_run_machine_emits_inspect_result_with_input_metadata(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(keys=7, sample_rate=48000, channels=1))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 0
    results = [e for e in _events_of(capsysbinary) if e["type"] == "result"]
    assert len(results) == 1
    r = results[0]
    assert r["mode"] == "inspect"
    assert r["output"] is None
    assert r["keys"] == 7
    assert r["input_kind"] == "audio"
    assert r["sample_rate"] == 48000
    assert r["channels"] == 1


def test_dry_run_does_not_write_vmd(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "-o", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()


# --- 通常実行: VMD書き出し・機械モード result ---------------------------------


def test_normal_run_writes_vmd_and_returns_zero(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result(keys=2))

    rc = cli.main([src, "-o", str(out)])
    assert rc == 0
    assert out.exists()
    document, _warnings = vmd_read(str(out))
    assert document.morph == []


def test_normal_run_machine_emits_run_result(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result(keys=4))

    rc = cli.main([src, "-o", str(out), "--machine"])
    assert rc == 0
    results = [e for e in _events_of(capsysbinary) if e["type"] == "result"]
    assert len(results) == 1
    r = results[0]
    assert r["mode"] == "run"
    assert r["output"] == str(out)
    assert r["keys"] == 4


def test_normal_run_emits_write_progress_stage_before_writing(tmp_path, monkeypatch, capsysbinary):
    # VMD書き出し(song2vmd.md 12.1の段id "write")はcli.py自身の責務なので、pipeline.run()の
    # 内部でなくcli.py側でprogress.stage("write")を発行する必要がある。
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result())

    rc = cli.main([src, "-o", str(out), "--machine"])
    assert rc == 0
    stages = [e["stage"] for e in _events_of(capsysbinary) if e["type"] == "progress"]
    assert "write" in stages


# --- low_dynamics_suppressed 警告 ----------------------------------------------


def test_low_dynamics_suppressed_warning_emitted_in_machine_mode(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(low_dynamics=True))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 0
    warnings = [e for e in _events_of(capsysbinary) if e["type"] == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["code"] == "low_dynamics_suppressed"


def test_no_low_dynamics_warning_when_not_suppressed(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(low_dynamics=False))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 0
    assert not [e for e in _events_of(capsysbinary) if e["type"] == "warning"]


def test_low_dynamics_suppressed_warning_printed_to_stderr_non_machine(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(low_dynamics=True))

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "low_dynamics_suppressed" in err or "ダイナミックレンジ" in err


def test_low_dynamics_suppressed_warning_survives_quiet(tmp_path, monkeypatch, capsys):
    # --quiet は進捗表示だけを抑制し、警告は抑制しない(song2vmd.md 5.2)。--quiet指定時も
    # 警告そのものは実際に残ることを検証する。
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(low_dynamics=True))

    rc = cli.main([src, "--dry-run", "--quiet"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "low_dynamics_suppressed" in err or "ダイナミックレンジ" in err


# --- エラー変換(音声前段の失敗。12.3) -----------------------------------------


def test_audio_load_error_maps_to_decoder_missing(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(AudioLoadError("no ffmpeg")))

    rc = cli.main([src, "--dry-run"])
    assert rc == 4


def test_audio_load_error_machine_mode_emits_decoder_missing_error(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(AudioLoadError("no ffmpeg")))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 4
    events = _events_of(capsysbinary)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "decoder_missing"
    assert events[-1]["field"] == "input"
    assert events[-1]["exit_code"] == 4


def test_separation_error_maps_to_stage_failed_separate(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(
        cli._pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(SeparationError("missing")))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 4
    events = _events_of(capsysbinary)
    assert events[-1]["code"] == "stage_failed"
    assert events[-1]["stage"] == "separate"


def test_recognition_error_maps_to_stage_failed_recognize(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(
        cli._pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(RecognitionError("failed")))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 4
    events = _events_of(capsysbinary)
    assert events[-1]["code"] == "stage_failed"
    assert events[-1]["stage"] == "recognize"


def test_intermediate_write_error_maps_to_write_failed(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(
        cli._pipeline, "run",
        lambda *a, **k: (_ for _ in ()).throw(_pipeline.IntermediateWriteError("disk full")))

    rc = cli.main([src, "--keep-intermediate", "--machine", "--dry-run"])
    assert rc == 3
    events = _events_of(capsysbinary)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "write_failed"
    assert events[-1]["field"] == "--keep-intermediate"
    assert events[-1]["path"] == f"{tmp_path / 'in.vmd'}.intermediate"
    assert events[-1]["exit_code"] == 3


def test_recognizer_model_revision_without_id_machine_mode_emits_bad_argument(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "--recognizer-model-revision", "abc123", "--machine", "--dry-run"])
    assert rc == 2
    events = _events_of(capsysbinary)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "bad_argument"
    assert events[0]["field"] == "--recognizer-model-revision"
    assert events[0]["exit_code"] == 2


def test_forced_aligner_sofa_all_missing_machine_mode_reports_only_first_field(
    tmp_path, monkeypatch, capsysbinary
):
    """sofa-python・sofa-root・sofa-checkpointが全て欠落していても、走査順で最初の1件だけ報告する
    (song2vmd.md 12.3。複数欠落を1つのエラーへまとめて返す設計は採らない)。"""
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "--forced-aligner", "sofa-forcedalign", "--machine", "--dry-run"])
    assert rc == 2
    events = _events_of(capsysbinary)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "bad_argument"
    assert events[0]["field"] == "--sofa-python"
    assert events[0]["exit_code"] == 2


# --- 中断(KeyboardInterrupt。12.4) ---------------------------------------------


def test_keyboard_interrupt_during_pipeline_is_cancelled_130_non_machine(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))

    rc = cli.main([src, "--dry-run"])
    assert rc == 130
    assert capsys.readouterr().out == ""


def test_keyboard_interrupt_during_pipeline_is_cancelled_130_machine(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 130
    events = _events_of(capsysbinary)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "cancelled"
    assert events[0]["exit_code"] == 130


# --- 書き込み失敗(write_failed。12.3) -----------------------------------------


def test_write_failure_maps_to_write_failed(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "missing_parent" / "out.vmd"  # 親ディレクトリが無い→書き込み失敗
    _capture_run_kwargs(monkeypatch, result=_make_result())

    rc = cli.main([src, "-o", str(out)])
    assert rc == 3


def test_write_failure_machine_mode_emits_write_failed_error(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "missing_parent" / "out.vmd"  # 親ディレクトリが無い→書き込み失敗
    _capture_run_kwargs(monkeypatch, result=_make_result())

    rc = cli.main([src, "-o", str(out), "--machine"])
    assert rc == 3
    events = _events_of(capsysbinary)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "write_failed"
    assert events[-1]["field"] == "--output"
    assert events[-1]["path"] == str(out)
    assert events[-1]["exit_code"] == 3
