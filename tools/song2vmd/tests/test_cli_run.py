import builtins
import json
import os
import sys

import pytest

from cli_progress_router import ProgressEmitError
from cli_resource_watch import torch_config as _torch_config
from cli_resource_watch import watch as _watch_module
from song2vmd import __version__, cli
from song2vmd import events as _events
from song2vmd import pipeline as _pipeline
from song2vmd import presets as _presets
from song2vmd import report as _report
from song2vmd.resource_watch import warning_texts
from vmd import VmdDocument
from vmd import read as vmd_read
from vocal_analysis import (
    DEFAULT_CONTENT_RECOGNIZER_MODEL,
    ChunkingPolicy,
    ContentRecognizerModel,
)
from vocal_analysis.front_stage import (
    IntermediateReadError,
    IntermediateWriteError,
    StageExecutionError,
)
from vocal_analysis.io import AudioLoadError
from vocal_analysis.recognizer import RecognitionError
from vocal_analysis.separator import SeparationError


@pytest.fixture(autouse=True)
def _isolate_gpu_environment(monkeypatch):
    """CUDA_VISIBLE_DEVICES はこのフィクスチャが退避・復元する。値を要するテストは os.environ へ
    直接入れること(monkeypatch の取り消しはすべてのフィクスチャの解除より後に走るので、
    monkeypatch.setenv の復元はこのフィクスチャの復元を上書きする)。"""
    monkeypatch.setattr(_torch_config.shutil, "which", lambda name: None)
    saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    yield
    if saved is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _make_result(*, keys=3, low_dynamics=False, forced_split=False, sample_rate=44100, channels=2):
    document = VmdDocument(model_name_raw=b"\x00" * 20, morph=[])
    diagnostics = _report.build_diagnostics(
        segments=[], mouth_events=[], mora_event_group_sizes=[],
        event_diagnostics=_events.EventDiagnostics(weak_vowels=0, low_dynamics=low_dynamics, merged_morae=0),
        backends={"separator": "audio-separator-htdemucs-ft", "recognizer": "openai/whisper-medium"},
        style="pop", separated=True, duration_sec=2.5, keys=keys, forced_split=forced_split,
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
    assert kwargs["separate_vocals"] == "always"
    assert kwargs["separator_name"] == "audio-separator-htdemucs-ft"
    assert kwargs["chunking"] == ChunkingPolicy(max_duration_sec=300.0)
    assert kwargs["use_n_morph"] is False
    assert kwargs["style_gen"].vowel_scale == _presets.resolve("pop")[1].vowel_scale
    assert kwargs["intensity_curve"] == 0.6
    assert kwargs["silence_on"] == 0.06
    assert kwargs["model_name"] == f"song2vmd {__version__}"
    assert kwargs["progress"] is not None
    assert kwargs["keep_intermediate_dir"] is None
    assert kwargs["forced_aligner"] == "wav2vec2-ctc-forcedalign"
    assert kwargs["sofa_aligner"] is None


def test_silence_threshold_off_side_does_not_reach_pipeline(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    runs = []
    for text in ("0.06:0.10", "0.06:0.90"):
        captured = _capture_run_kwargs(monkeypatch)
        assert cli.main([src, "--silence-threshold", text, "--dry-run"]) == 0
        runs.append({k: v for k, v in captured["kwargs"].items() if k != "progress"})
    assert runs[0] == runs[1]


def test_dry_run_report_shows_both_silence_threshold_sides(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "--silence-threshold", "0.05:0.12", "--dry-run"])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert "silence_threshold_on: 0.05" in lines
    assert "silence_threshold_off: 0.12" in lines


def test_run_builds_sofa_aligner_config_from_cli_options(tmp_path, monkeypatch):
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


def test_run_passes_default_english_katakana_method(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert captured["kwargs"]["english_katakana_method"] == "arpakana"


def test_english_katakana_method_option_selects_tinyllama(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([
        src, "--english-katakana-method", "tinyllama-katakana-converter", "--dry-run",
    ])
    assert rc == 0
    assert captured["kwargs"]["english_katakana_method"] == "tinyllama-katakana-converter"


def _capture_cvd_at_pipeline_start(monkeypatch):
    seen = {}

    def fake_run(input_path, **kwargs):
        seen["cvd"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return _make_result()

    monkeypatch.setattr(cli._pipeline, "run", fake_run)
    return seen


def test_device_cpu_hides_cuda_before_pipeline_runs(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    seen = _capture_cvd_at_pipeline_start(monkeypatch)

    rc = cli.main([src, "--device", "cpu", "--dry-run"])
    assert rc == 0
    assert seen["cvd"] == "-1"


def test_device_auto_default_keeps_user_cuda_visible_devices(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seen = _capture_cvd_at_pipeline_start(monkeypatch)

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert seen["cvd"] == "0"


def test_device_unknown_value_is_arg_error_without_starting_pipeline(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    seen = _capture_cvd_at_pipeline_start(monkeypatch)

    rc = cli.main([src, "--device", "gpu", "--dry-run"])
    assert rc == 2
    assert "cvd" not in seen


def _force_gpu_oversubscription_on_second_check(monkeypatch):
    mib = 2**20
    seq = iter([(4000 * mib, 8192 * mib, 0), (4000 * mib, 8192 * mib, 5600 * mib)])
    monkeypatch.setattr(_watch_module, "_default_gpu_probe",
                        lambda: next(seq, (4000 * mib, 8192 * mib, 5600 * mib)))
    monkeypatch.setattr(_watch_module, "_default_ram_probe", lambda: (0, 0, 0))


def _fake_run_reporting_load_then_separate(monkeypatch):

    def fake_run(input_path, **kwargs):
        kwargs["progress"].stage("load")
        kwargs["progress"].stage("separate")
        return _make_result()

    monkeypatch.setattr(cli._pipeline, "run", fake_run)


def test_resource_warning_emitted_as_machine_event_without_changing_result(
        tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription_on_second_check(monkeypatch)
    _fake_run_reporting_load_then_separate(monkeypatch)

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 0
    events = _events_of(capsysbinary)
    warnings = [e for e in events if e["type"] == "warning" and e["code"] == "gpu_memory_oversubscribed"]
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning["message"] == "GPUメモリの要求量が空き容量を超過しました"
    assert warning["stage"] == "separate"
    assert warning["reserved_mib"] == 5600 and warning["free_at_start_mib"] == 4000
    assert warning["total_mib"] == 8192
    assert events[-1]["type"] == "result"


def test_resource_warning_printed_to_stderr_in_human_mode(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription_on_second_check(monkeypatch)
    _fake_run_reporting_load_then_separate(monkeypatch)

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "warning: gpu_memory_oversubscribed:" in err
    assert "5600MiB" in err and "--device cpu" in err


def test_resource_warning_closes_live_line_before_stderr_write(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription_on_second_check(monkeypatch)
    _fake_run_reporting_load_then_separate(monkeypatch)
    order = []

    class _OrderReporter:
        def stage(self, stage_id, **kwargs):
            pass

        def close(self):
            order.append("close")

        def summary(self, message):
            pass

    class _OrderStderr:
        def write(self, text):
            if text.startswith("warning:"):
                order.append("warning")

        def flush(self):
            pass

    monkeypatch.setattr(cli._progress, "build_router", lambda **kwargs: _OrderReporter())
    monkeypatch.setattr(cli.sys, "stderr", _OrderStderr())

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert order[:2] == ["close", "warning"]


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


def test_verbose_normal_run_writes_vmd_and_prints_report(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result(keys=6))

    rc = cli.main([src, "-o", str(out), "--verbose"])
    assert rc == 0
    assert out.exists()
    out_text = capsys.readouterr().out
    assert "keys: 6" in out_text


def test_normal_run_without_verbose_prints_no_report(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result(keys=6))

    rc = cli.main([src, "-o", str(out)])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert out_text == ""


def test_verbose_machine_run_does_not_print_report_text(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result(keys=6))

    rc = cli.main([src, "-o", str(out), "--machine", "--verbose"])
    assert rc == 0
    results = [e for e in _events_of(capsysbinary) if e["type"] == "result"]
    assert len(results) == 1
    assert results[0]["mode"] == "run"


def test_normal_run_emits_write_progress_stage(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result())

    rc = cli.main([src, "-o", str(out), "--machine"])
    assert rc == 0
    stages = [e["stage"] for e in _events_of(capsysbinary) if e["type"] == "progress"]
    assert "write" in stages


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
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(low_dynamics=True))

    rc = cli.main([src, "--dry-run", "--quiet"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "low_dynamics_suppressed" in err or "ダイナミックレンジ" in err


def test_low_dynamics_human_warning_is_one_code_prefixed_stderr_line(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result(low_dynamics=True))

    rc = cli.main([src, "-o", str(out)])
    assert rc == 0
    stdout, err = capsys.readouterr()
    assert stdout == ""
    lines = err.splitlines()
    assert len(lines) == 1
    prefix = "warning: low_dynamics_suppressed: "
    assert lines[0].startswith(prefix)
    body = lines[0][len(prefix):]
    assert body.strip()
    assert body.lstrip() == body


def test_forced_split_warning_emitted_in_machine_mode_without_stage_key(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(forced_split=True))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 0
    warnings = [e for e in _events_of(capsysbinary) if e["type"] == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["code"] == "forced_split"
    assert "stage" not in warnings[0]


def test_no_forced_split_warning_when_not_forced(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(forced_split=False))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 0
    assert not [e for e in _events_of(capsysbinary) if e["type"] == "warning"]


def test_forced_split_warning_printed_to_stderr_non_machine(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(forced_split=True))

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "forced_split" in err


def test_forced_split_warning_survives_quiet(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(forced_split=True))

    rc = cli.main([src, "--dry-run", "--quiet"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "forced_split" in err


def test_forced_split_human_warning_is_one_code_prefixed_stderr_line(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result(forced_split=True))

    rc = cli.main([src, "-o", str(out)])
    assert rc == 0
    stdout, err = capsys.readouterr()
    assert stdout == ""
    lines = err.splitlines()
    assert len(lines) == 1
    prefix = "warning: forced_split: "
    assert lines[0].startswith(prefix)
    body = lines[0][len(prefix):]
    assert body.strip()
    assert body.lstrip() == body


def test_audio_load_error_maps_to_decoder_missing(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(
        cli._pipeline, "run",
        lambda *a, **k: (_ for _ in ()).throw(AudioLoadError("no ffmpeg", reason="decoder_missing")),
    )

    rc = cli.main([src, "--dry-run"])
    assert rc == 4


def test_audio_load_error_machine_mode_emits_decoder_missing_error(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(
        cli._pipeline, "run",
        lambda *a, **k: (_ for _ in ()).throw(AudioLoadError("no ffmpeg", reason="decoder_missing")),
    )

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 4
    events = _events_of(capsysbinary)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "decoder_missing"
    assert events[-1]["field"] == "input"
    assert events[-1]["exit_code"] == 4


def test_broken_input_audio_maps_to_not_audio_with_exit_code_one(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(
        cli._pipeline, "run",
        lambda *a, **k: (_ for _ in ()).throw(AudioLoadError("broken input", reason="not_audio")))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 1
    events = _events_of(capsysbinary)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "not_audio"
    assert events[-1]["field"] == "input"
    assert events[-1]["exit_code"] == 1


def test_intermediate_read_error_maps_to_not_audio_with_path_and_no_field(
        tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    vocal = tmp_path / "vocal.wav"
    monkeypatch.setattr(
        cli._pipeline, "run",
        lambda *a, **k: (_ for _ in ()).throw(
            IntermediateReadError("broken vocal wav", path=vocal)))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 1
    events = _events_of(capsysbinary)
    assert events[-1]["code"] == "not_audio"
    assert events[-1]["field"] is None
    assert events[-1]["path"] == str(vocal)
    assert events[-1]["exit_code"] == 1


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
        lambda *a, **k: (_ for _ in ()).throw(IntermediateWriteError("disk full")))

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


def test_write_to_missing_parent_directory_maps_to_write_failed(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "missing_parent" / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result())

    rc = cli.main([src, "-o", str(out)])
    assert rc == 3


def test_write_to_missing_parent_directory_machine_mode_emits_write_failed_error(
        tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "missing_parent" / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result())

    rc = cli.main([src, "-o", str(out), "--machine"])
    assert rc == 3
    events = _events_of(capsysbinary)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "write_failed"
    assert events[-1]["field"] == "--output"
    assert events[-1]["path"] == str(out)
    assert events[-1]["exit_code"] == 3


class _SpyProgressRouter:
    calls = None

    def __init__(self, **kwargs):
        pass

    def stage(self, *args, **kwargs):
        pass

    def close(self):
        _SpyProgressRouter.calls.append("close")

    def summary(self, message):
        _SpyProgressRouter.calls.append(("summary", message))


@pytest.fixture
def spy_progress(monkeypatch):
    """close・summary・標準エラーへの print・レポート生成を、呼ばれた順に1本のリストへ記録して返す。"""
    calls = []
    _SpyProgressRouter.calls = calls
    monkeypatch.setattr(cli._progress, "build_router", lambda **kwargs: _SpyProgressRouter())

    real_print = print

    def spy_print(*args, **kwargs):
        if kwargs.get("file") is sys.stderr and args:
            calls.append(("stderr_print", args[0]))
        real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", spy_print)

    real_render = cli._report.render_report_text

    def spy_render(*args, **kwargs):
        calls.append("render_report")
        return real_render(*args, **kwargs)

    monkeypatch.setattr(cli._report, "render_report_text", spy_render)
    return calls


def test_normal_run_closes_progress_then_shows_completion(tmp_path, monkeypatch, spy_progress):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result())

    rc = cli.main([src, "-o", str(out)])
    assert rc == 0
    assert spy_progress[:2] == ["close", ("summary", f"完了 {out}")]
    assert all(call == "close" for call in spy_progress[2:])


def test_dry_run_closes_progress_before_report_without_completion_line(tmp_path, monkeypatch, spy_progress):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result())

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert spy_progress[0] == "close"
    assert "render_report" in spy_progress
    assert spy_progress.index("close") < spy_progress.index("render_report")
    assert not any(isinstance(c, tuple) and c[0] == "summary" for c in spy_progress)


def test_low_dynamics_warning_closes_progress_before_stderr_print(tmp_path, monkeypatch, spy_progress):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(low_dynamics=True))

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    close_positions = [i for i, c in enumerate(spy_progress) if c == "close"]
    print_positions = [i for i, c in enumerate(spy_progress)
                       if isinstance(c, tuple) and c[0] == "stderr_print"]
    assert close_positions and print_positions
    assert close_positions[0] < print_positions[0]


@pytest.mark.parametrize("make_exc", [
    lambda: AudioLoadError("no ffmpeg", reason="decoder_missing"),
    lambda: SeparationError("sep failed"),
    lambda: RecognitionError("rec failed"),
    lambda: StageExecutionError("RuntimeError: 分離の失敗", stage="separate"),
])
def test_pipeline_failure_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress, make_exc):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(make_exc()))

    rc = cli.main([src, "--dry-run"])
    assert rc == 4
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))


def test_intermediate_write_error_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(
        cli._pipeline, "run",
        lambda *a, **k: (_ for _ in ()).throw(IntermediateWriteError("disk full")))

    rc = cli.main([src, "--keep-intermediate", "--dry-run"])
    assert rc == 3
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))


def test_write_failure_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "missing_parent" / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_result())

    rc = cli.main([src, "-o", str(out)])
    assert rc == 3
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))


def test_keyboard_interrupt_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))

    rc = cli.main([src, "--dry-run"])
    assert rc == 130
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))


def test_cpu_only_torch_warning_precedes_first_progress_event(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(
        cli, "torch_gpu_warning",
        lambda device: ("cpu_only_torch", {"torch_version": "2.13.0"}))

    def fake_run(input_path, **kwargs):
        kwargs["progress"].stage("load")
        return _make_result()

    monkeypatch.setattr(cli._pipeline, "run", fake_run)

    rc = cli.main([src, "--dry-run", "--machine"])

    assert rc == 0
    events = _events_of(capsysbinary)
    types = [(e.get("type"), e.get("code")) for e in events]
    assert ("warning", "cpu_only_torch") in types
    assert types.index(("warning", "cpu_only_torch")) < next(
        i for i, (kind, _) in enumerate(types) if kind == "progress")
    warning = next(e for e in events if e.get("type") == "warning")
    assert warning["message"] == warning_texts("cpu_only_torch", {"torch_version": "2.13.0"})[0]
    assert warning["torch_version"] == "2.13.0"


def _raise_from_pipeline(monkeypatch, error):
    monkeypatch.setattr(
        cli._pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(error))


@pytest.mark.parametrize("stage", ["separate", "recognize"])
def test_stage_execution_error_maps_to_stage_failed_with_only_stage_as_extra_key(
        tmp_path, monkeypatch, capsysbinary, stage):
    src = _touch(tmp_path / "in.wav")
    _raise_from_pipeline(
        monkeypatch,
        StageExecutionError("RuntimeError: CUDA out of memory", stage=stage))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 4
    events = _events_of(capsysbinary)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "stage_failed"
    assert events[-1]["stage"] == stage
    assert events[-1]["field"] is None
    assert events[-1]["path"] is None
    assert events[-1]["exit_code"] == 4
    assert "CUDA out of memory" in events[-1]["message"]
    assert set(events[-1]) == {"type", "code", "exit_code", "field", "path", "message", "stage"}


@pytest.mark.parametrize("stage, stage_label", [("separate", "ボーカル分離"), ("recognize", "音素認識")])
def test_stage_execution_error_names_the_stage_label_in_one_human_line(
        tmp_path, monkeypatch, capsys, stage, stage_label):
    src = _touch(tmp_path / "in.wav")
    _raise_from_pipeline(
        monkeypatch, StageExecutionError("RuntimeError: CUDA out of memory", stage=stage))

    rc = cli.main([src, "--dry-run"])
    assert rc == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert stage_label in lines[0]
    assert "CUDA out of memory" in lines[0]
    assert "Traceback" not in captured.err


def test_progress_emit_error_is_reported_as_internal_error_not_stage_failed(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _raise_from_pipeline(monkeypatch, ProgressEmitError("stdout is closed"))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 1
    events = _events_of(capsysbinary)
    assert events[-1]["code"] == "internal_error"


def test_unexpected_exception_reports_internal_error_without_traceback(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _raise_from_pipeline(monkeypatch, RuntimeError("unexpected"))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 1
    captured = capsysbinary.readouterr()
    events = [json.loads(ln) for ln in captured.out.decode("utf-8").splitlines() if ln]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "internal_error"
    assert events[-1]["field"] is None
    assert events[-1]["path"] is None
    assert events[-1]["exit_code"] == 1
    assert "RuntimeError" in events[-1]["message"]
    assert "unexpected" in events[-1]["message"]
    assert "Traceback" not in captured.err.decode("utf-8")


def test_unexpected_exception_reports_single_line_without_machine(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _raise_from_pipeline(monkeypatch, RuntimeError("unexpected"))

    rc = cli.main([src, "--dry-run"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "RuntimeError" in lines[0]
    assert "unexpected" in lines[0]
    assert "Traceback" not in captured.err
