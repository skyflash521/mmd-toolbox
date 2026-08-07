"""song2vpr パイプライン統合のテスト。

音声前段の実行そのものは共有側が持つので、ここでは song2vpr 側の結線——前段へ渡す設定の組み立てと、
返った共有出力を後段が使える形で取り出すこと——だけを検証する。前段の実行エンジンはモックし、
呼び出しが受け取った引数と、返した共有出力の行き先を見る。
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

from song2vpr import pipeline
from vocal_analysis import (
    AnalysisResult,
    AudioPcm,
    ChunkingPolicy,
    ContentRecognizerModel,
    RmsEnvelope,
    Segment,
    front_stage,
)
from vocal_analysis.front_stage import FrontStageResult

_TEST_MODEL = ContentRecognizerModel(model_id="test-content-recognizer")


def seg(type_, start, end, phoneme=None, confidence=None):
    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=phoneme, confidence=confidence)


def _common_kwargs(**overrides):
    kw = dict(
        separate_vocals="always", separator_name="audio-separator-htdemucs-ft",
        content_recognizer_model=_TEST_MODEL, retry=True,
        chunking=ChunkingPolicy(max_duration_sec=300.0),
        forced_aligner="wav2vec2-ctc-forcedalign", sofa_aligner=None,
        english_katakana_method="arpakana",
    )
    kw.update(overrides)
    return kw


class _RecordingProgress:
    def __init__(self):
        self.calls = []

    def stage(self, stage, *, done=0, total=None, note="", elapsed=0.0):
        self.calls.append(stage)


def _pcm(seconds=1.0, sample_rate=8000, channels=2):
    n = int(seconds * sample_rate)
    return AudioPcm(samples=np.zeros((n, channels), dtype=np.float32), sample_rate=sample_rate)


def _stub_front_stage(monkeypatch, captured, *, segments=None, vocal_pcm=None, duration_sec=1.0,
                      forced_split=False):
    """前段実行エンジンを差し替え、受け取った引数を記録して最小の共有出力を返す。"""
    given = segments if segments is not None else [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]
    vocal = vocal_pcm if vocal_pcm is not None else _pcm()

    def fake_run_front_stage(input_path, **kwargs):
        captured["input_path"] = input_path
        captured["kwargs"] = kwargs
        if kwargs.get("on_progress") is not None:
            kwargs["on_progress"]("load", done=0, total=None, note="", elapsed=0.0)
        rms = RmsEnvelope(
            times_sec=np.array([0.0, 0.5]), values=np.array([0.5, 0.5]), dynamic_range_db=20.0)
        return FrontStageResult(
            analysis=AnalysisResult(vocal_wav=Path("vocal.wav"), segments=given, rms=rms),
            pcm=_pcm(), vocal_pcm=vocal, duration_sec=duration_sec, forced_split=forced_split)

    monkeypatch.setattr(pipeline, "run_front_stage", fake_run_front_stage)


# --- 前段へ渡す設定 ----------------------------------------------------------


def test_front_stage_receives_the_settings_and_the_intermediate_directory(tmp_path, monkeypatch):
    """CLI が解決した音声前段の設定は、解釈されずそのまま前段へ渡る。"""
    captured = {}
    _stub_front_stage(monkeypatch, captured)

    pipeline.run("in.wav", keep_intermediate_dir=str(tmp_path / "keep"), **_common_kwargs(
        separator_name="sep-x", chunking=ChunkingPolicy(max_duration_sec=42.0)))

    kwargs = captured["kwargs"]
    # 実エンジンの公開シグネチャへ束縛できることを見る(必須の欠落・名前の取り違えに加えて、
    # 受け付けない余分なキーワードを渡す実装も落とす)。
    inspect.signature(front_stage.run_front_stage).bind(captured["input_path"], **kwargs)
    assert captured["input_path"] == "in.wav"
    assert kwargs["separate_vocals"] == "always"
    assert kwargs["separator"] == "sep-x"
    assert kwargs["content_recognizer_model"] is _TEST_MODEL
    assert kwargs["retry"] is True
    assert kwargs["chunking"] == ChunkingPolicy(max_duration_sec=42.0)
    assert kwargs["keep_intermediate_dir"] == str(tmp_path / "keep")
    assert kwargs["forced_aligner"] == "wav2vec2-ctc-forcedalign"
    assert kwargs["sofa_aligner"] is None
    assert kwargs["english_katakana_method"] == "arpakana"


def test_intermediate_directory_is_not_passed_when_unset(monkeypatch):
    """保存先を渡さない実行では、前段へも None が渡る(前段は何も書かない)。"""
    captured = {}
    _stub_front_stage(monkeypatch, captured)

    pipeline.run("in.wav", **_common_kwargs())

    assert captured["kwargs"]["keep_intermediate_dir"] is None


def test_sofa_aligner_is_relayed_as_given(monkeypatch):
    """強制アライメント段の設定も解釈せずそのまま渡す。"""
    captured = {}
    _stub_front_stage(monkeypatch, captured)
    sofa = object()

    pipeline.run("in.wav", **_common_kwargs(
        forced_aligner="sofa-forcedalign", sofa_aligner=sofa))

    assert captured["kwargs"]["forced_aligner"] == "sofa-forcedalign"
    assert captured["kwargs"]["sofa_aligner"] is sofa


# --- 前段の共有出力の取り出し ------------------------------------------------


def test_front_stage_outputs_are_available_to_the_later_stages(monkeypatch):
    """後段が使う共有出力(ボーカルPCM・音素セグメント列・RMS・入力PCM・尺)を保持する。

    ボーカルWAVは前段が既に読んだ PCM を受け取るので、song2vpr 側で読み直さない。
    """
    captured = {}
    segments = [seg("vowel", 0.0, 0.4, phoneme="a", confidence=0.9),
                seg("gap", 0.4, 1.0)]
    vocal = _pcm(seconds=2.0, sample_rate=16000, channels=1)
    _stub_front_stage(monkeypatch, captured, segments=segments, vocal_pcm=vocal, duration_sec=2.0)

    result = pipeline.run("in.wav", **_common_kwargs())

    assert result.segments == segments
    assert result.vocal_wav == Path("vocal.wav")
    assert result.vocal_pcm is vocal
    assert result.rms.dynamic_range_db == 20.0
    assert result.duration_sec == 2.0
    # テンポ・拍子の推定は分離前の入力を使うので、波形そのものを保持する。
    assert result.pcm.sample_rate == 8000
    assert result.pcm.samples.shape[1] == 2


@pytest.mark.parametrize("forced_split", [True, False])
def test_forced_split_is_carried_from_the_front_stage(monkeypatch, forced_split):
    """強制分割が起きたかは前段から受け取った値をそのまま運ぶ(固定値を返さない)。"""
    captured = {}
    _stub_front_stage(monkeypatch, captured, forced_split=forced_split)

    assert pipeline.run("in.wav", **_common_kwargs()).forced_split is forced_split


# --- 進捗の中継 --------------------------------------------------------------


def test_progress_is_relayed_to_the_front_stage(monkeypatch):
    """前段の段は共有側が報告する。song2vpr は中継先を渡すだけで自分では報告し直さない。"""
    captured = {}
    _stub_front_stage(monkeypatch, captured)
    progress = _RecordingProgress()

    pipeline.run("in.wav", progress=progress, **_common_kwargs())

    assert captured["kwargs"]["on_progress"] is not None
    assert progress.calls == ["load"]


def test_run_works_without_progress_reporter(monkeypatch):
    """進捗の中継先が無い実行では、前段へも None を渡す。"""
    captured = {}
    _stub_front_stage(monkeypatch, captured)

    pipeline.run("in.wav", **_common_kwargs())

    assert captured["kwargs"]["on_progress"] is None
