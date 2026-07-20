"""song2vmd CLI のパイプライン配線テスト。

cli.py の _run() が pipeline.run() を正しい引数で呼び、その結果(PipelineResult)を
--dry-run の人間向けレポート/機械モードの result イベント、VMD 書き出し、警告発行、
エラー変換(AudioLoadError/SeparationError/RecognitionError/IntermediateWriteError)・中断(KeyboardInterrupt)へ
正しく橋渡しすることを検証する。pipeline.run 自体はモックし、実音声処理は行わない。
"""

import json
import os
import sys

import pytest

from vmd import VmdDocument
from vmd import read as vmd_read
from vocal_analysis import ContentRecognizerModel, DEFAULT_CONTENT_RECOGNIZER_MODEL
from vocal_analysis.io import AudioLoadError
from vocal_analysis.recognizer import RecognitionError
from vocal_analysis.separator import SeparationError

from song2vmd import __version__
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
        segments=[], mouth_events=[], mora_event_group_sizes=[],
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
    assert kwargs["separate_vocals"] == "always"
    assert kwargs["separator_name"] == "audio-separator-htdemucs-ft"
    assert kwargs["max_duration_sec"] == 300.0
    assert kwargs["use_n_morph"] is False
    # --vowel-gain の既定 1:1:1:1:1 は presets.resolve での乗算後もプリセット値のまま。
    # pipeline.run へは style_gen.vowel_scale として渡る。
    assert kwargs["style_gen"].vowel_scale == _presets.resolve("pop")[1].vowel_scale
    assert kwargs["intensity_curve"] == 0.6
    assert kwargs["silence_on"] == 0.06
    assert kwargs["model_name"] == f"song2vmd {__version__}"
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


def test_run_passes_default_english_oov_katakana_method(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert captured["kwargs"]["english_oov_katakana_method"] == "arpakana"


def test_english_oov_katakana_method_option_selects_tinyllama(tmp_path, monkeypatch):
    """--english-oov-katakana-method tinyllama-katakana-converter を指定すると、
    pipeline.run へその方式が渡る(recognize()を経て実際に選択できることの配線検証)。"""
    src = _touch(tmp_path / "in.wav")
    captured = _capture_run_kwargs(monkeypatch)

    rc = cli.main([
        src, "--english-oov-katakana-method", "tinyllama-katakana-converter", "--dry-run",
    ])
    assert rc == 0
    assert captured["kwargs"]["english_oov_katakana_method"] == "tinyllama-katakana-converter"


def _capture_cvd_at_run(monkeypatch):
    """pipeline.run 呼び出し時点の CUDA_VISIBLE_DEVICES を捕捉する。

    --device の設定はパイプライン起動前(最初のCUDA照会前)に済んでいなければ効かないため、
    設定の有無だけでなく「pipeline.run より前」というタイミングを呼び出し時点の観測で検証する。
    """
    seen = {}

    def fake_run(input_path, **kwargs):
        seen["cvd"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return _make_result()

    monkeypatch.setattr(cli._pipeline, "run", fake_run)
    return seen


def test_device_cpu_hides_cuda_before_pipeline_runs(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    seen = _capture_cvd_at_run(monkeypatch)

    rc = cli.main([src, "--device", "cpu", "--dry-run"])
    assert rc == 0
    assert seen["cvd"] == "-1"


def test_device_auto_default_leaves_environment_untouched(tmp_path, monkeypatch):
    """既定(auto)は環境に触れない。利用者が自分で設定した CUDA_VISIBLE_DEVICES も壊さない。"""
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    seen = _capture_cvd_at_run(monkeypatch)

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert seen["cvd"] == "0"


def test_device_rejects_unknown_value(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    seen = _capture_cvd_at_run(monkeypatch)

    rc = cli.main([src, "--device", "gpu", "--dry-run"])
    assert rc == 2
    assert "cvd" not in seen  # 引数エラーで pipeline は起動しない


def _force_gpu_oversubscription(monkeypatch):
    """資源逼迫のGPU probe を、2回目の判定で超過が成立する系列(バイト値)に差し替える。"""
    mib = 2**20
    seq = iter([(4000 * mib, 8192 * mib, 0), (4000 * mib, 8192 * mib, 5600 * mib)])
    monkeypatch.setattr(cli._resource_watch, "_default_gpu_probe",
                        lambda: next(seq, (4000 * mib, 8192 * mib, 5600 * mib)))
    monkeypatch.setattr(cli._resource_watch, "_default_ram_probe", lambda: (0, 0, 0))


def _fake_run_with_stages(monkeypatch):
    """pipeline.run を、progress へ2段(load→separate)を報告するフェイクへ差し替える。"""

    def fake_run(input_path, **kwargs):
        kwargs["progress"].stage("load")
        kwargs["progress"].stage("separate")
        return _make_result()

    monkeypatch.setattr(cli._pipeline, "run", fake_run)


def test_resource_warning_emitted_as_machine_event(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription(monkeypatch)
    _fake_run_with_stages(monkeypatch)

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
    # 警告は診断・終了を変えない: result で正常終端する。
    assert events[-1]["type"] == "result"


def test_resource_warning_printed_to_stderr_in_human_mode(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription(monkeypatch)
    _fake_run_with_stages(monkeypatch)

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "warning: gpu_memory_oversubscribed:" in err
    assert "5600MiB" in err and "--device cpu" in err


def test_resource_warning_closes_live_line_before_stderr_write(tmp_path, monkeypatch):
    """人間向け警告は、ライブ進捗行の消去(close)→標準エラーへの書き込みの順で出る
    (ライブ行と警告行の混線防止の順序保証)。"""
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription(monkeypatch)
    _fake_run_with_stages(monkeypatch)
    order = []

    class _OrderReporter:
        def __init__(self, **kwargs):
            self.enabled = False

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

    monkeypatch.setattr(cli._progress, "ProgressReporter", _OrderReporter)
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


# --- --verbose: 通常実行での診断レポート追加表示 --------------------------------


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


def test_normal_run_emits_write_progress_stage_before_writing(tmp_path, monkeypatch, capsysbinary):
    # VMD書き出し(段id "write")はcli.py自身の責務なので、pipeline.run()の
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
    # --quiet は進捗表示だけを抑制し、警告は抑制しない。--quiet指定時も
    # 警告そのものは実際に残ることを検証する。
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result(low_dynamics=True))

    rc = cli.main([src, "--dry-run", "--quiet"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "low_dynamics_suppressed" in err or "ダイナミックレンジ" in err


def test_human_warning_line_uses_common_format(tmp_path, monkeypatch, capsys):
    # 警告行は共通コードのラベルで1行にまとめて標準エラーへ出す(安定コードは機械モードの
    # warning イベントと同じ値)。通常実行(--dry-run なし)で標準出力には何も漏らさない。
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
    assert body.lstrip() == body  # 接頭辞直後に空白文字(タブ・全角空白等)が無い


# --- forced_split 警告 ----------------------------------------------------------


def _make_forced_split_result(*, forced_split):
    # _make_resultはforced_splitに未対応のため、build_diagnosticsを直接呼ぶ専用ヘルパー。
    document = VmdDocument(model_name_raw=b"\x00" * 20, morph=[])
    diagnostics = _report.build_diagnostics(
        segments=[], mouth_events=[], mora_event_group_sizes=[],
        event_diagnostics=_events.EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=0),
        backends={"separator": "audio-separator-htdemucs-ft", "recognizer": "openai/whisper-medium"},
        style="pop", separated=True, duration_sec=2.5, keys=3, forced_split=forced_split)
    return _pipeline.PipelineResult(document=document, diagnostics=diagnostics, sample_rate=44100, channels=2)


@pytest.mark.xfail(reason="impl pending: 項目1 forced_split警告", strict=True)
def test_forced_split_warning_emitted_in_machine_mode(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_forced_split_result(forced_split=True))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 0
    warnings = [e for e in _events_of(capsysbinary) if e["type"] == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["code"] == "forced_split"
    assert "stage" not in warnings[0]  # low_dynamics_suppressedと同じ形でstageキーは持たない


@pytest.mark.xfail(reason="impl pending: 項目1 forced_split警告", strict=True)
def test_no_forced_split_warning_when_not_forced(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_forced_split_result(forced_split=False))

    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 0
    assert not [e for e in _events_of(capsysbinary) if e["type"] == "warning"]


@pytest.mark.xfail(reason="impl pending: 項目1 forced_split警告", strict=True)
def test_forced_split_warning_printed_to_stderr_non_machine(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_forced_split_result(forced_split=True))

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "forced_split" in err


@pytest.mark.xfail(reason="impl pending: 項目1 forced_split警告", strict=True)
def test_forced_split_warning_survives_quiet(tmp_path, monkeypatch, capsys):
    # --quiet は進捗表示だけを抑制し、警告は抑制しない(low_dynamics_suppressedと同じ扱い)。
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_forced_split_result(forced_split=True))

    rc = cli.main([src, "--dry-run", "--quiet"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "forced_split" in err


@pytest.mark.xfail(reason="impl pending: 項目1 forced_split警告", strict=True)
def test_forced_split_human_warning_line_uses_common_format_and_appears_once(tmp_path, monkeypatch, capsys):
    # low_dynamics_suppressedのtest_human_warning_line_uses_common_formatと対になる検証:
    # 警告行は共通コードのラベルで実行につき1行だけ標準エラーへ出す。
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    _capture_run_kwargs(monkeypatch, result=_make_forced_split_result(forced_split=True))

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
    assert body.lstrip() == body  # 接頭辞直後に空白文字(タブ・全角空白等)が無い


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
    (複数欠落を1つのエラーへまとめて返す設計は採らない)。"""
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


# --- 進捗ライブ表示の終端処理(全終了経路で close、正常終了時のみ完了行) -------------------


class _SpyProgressReporter:
    """ProgressReporter の差し替え。close/summary の呼び出しを、共有の calls リストへ記録する
    (下記 spy_progress フィクスチャが標準エラー出力の print 呼び出しも同じリストへ記録するため、
    close とエラー行表示の相対順序を1本のタイムラインで検証できる)。"""

    calls = None  # クラス変数: monkeypatch 先のコンストラクタから書けるよう、テストごとにリセットする

    def __init__(self, **kwargs):
        pass

    def stage(self, *args, **kwargs):
        pass

    def close(self):
        _SpyProgressReporter.calls.append("close")

    def summary(self, message):
        _SpyProgressReporter.calls.append(("summary", message))


@pytest.fixture
def spy_progress(monkeypatch):
    calls = []
    _SpyProgressReporter.calls = calls
    monkeypatch.setattr(cli._progress, "ProgressReporter", _SpyProgressReporter)

    real_print = print

    def spy_print(*args, **kwargs):
        if kwargs.get("file") is sys.stderr and args:
            calls.append(("stderr_print", args[0]))
        real_print(*args, **kwargs)

    monkeypatch.setattr(cli, "print", spy_print, raising=False)

    # --dry-run の人間向けレポート生成呼び出しも同じタイムラインへ記録し、close との
    # 相対順序(ライブ行を消してからレポートを書く)を検証できるようにする。
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
    # close は冪等なので、正常終了の明示的な close/summary の後に、全終了経路を保証する
    # 保険としての再呼び出しが続いてもよい(先頭2件の順序だけを固定する)。
    assert spy_progress[:2] == ["close", ("summary", f"完了 {out}")]
    assert all(call == "close" for call in spy_progress[2:])


def test_dry_run_closes_progress_without_completion_line(tmp_path, monkeypatch, spy_progress):
    # --dry-run は VMD を書かないため完了行を出さない(close はライブ行の終端として必ず呼ぶ)。
    # close はレポート生成(標準出力とライブ行が同じ端末で連結しうる)より前でなければならない。
    src = _touch(tmp_path / "in.wav")
    _capture_run_kwargs(monkeypatch, result=_make_result())

    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert spy_progress[0] == "close"
    assert "render_report" in spy_progress
    assert spy_progress.index("close") < spy_progress.index("render_report")
    assert not any(isinstance(c, tuple) and c[0] == "summary" for c in spy_progress)


def test_low_dynamics_warning_closes_progress_before_stderr_print(tmp_path, monkeypatch, spy_progress):
    # low_dynamics 警告(非機械モード)は、標準エラーがライブ行と同じ端末につながるため、
    # close でライブ行を消してから出す。
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
    lambda: AudioLoadError("no ffmpeg"),
    lambda: SeparationError("sep failed"),
    lambda: RecognitionError("rec failed"),
])
def test_pipeline_failure_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress, make_exc):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(make_exc()))

    rc = cli.main([src, "--dry-run"])
    assert rc == 4
    # close がエラー行の標準エラー出力より前に呼ばれる(ライブ行を消してからエラーを表示する)。
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))


def test_intermediate_write_error_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(
        cli._pipeline, "run",
        lambda *a, **k: (_ for _ in ()).throw(_pipeline.IntermediateWriteError("disk full")))

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
