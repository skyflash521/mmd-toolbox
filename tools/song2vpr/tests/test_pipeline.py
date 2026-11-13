import inspect
from pathlib import Path

import numpy as np

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


def seg(type_, start, end, phoneme=None):
    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=phoneme)


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
    given = segments if segments is not None else [seg("vowel", 0.0, 1.0, phoneme="a")]
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


def test_front_stage_receives_the_settings_unchanged_and_bindable_to_its_signature(tmp_path, monkeypatch):
    captured = {}
    _stub_front_stage(monkeypatch, captured)

    pipeline.run("in.wav", keep_intermediate_dir=str(tmp_path / "keep"), **_common_kwargs(
        separator_name="sep-x", chunking=ChunkingPolicy(max_duration_sec=42.0)))

    kwargs = captured["kwargs"]
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


def test_progress_is_relayed_through_the_front_stage_callback_without_re_reporting(monkeypatch):
    captured = {}
    _stub_front_stage(monkeypatch, captured)
    progress = _RecordingProgress()

    pipeline.run("in.wav", progress=progress, **_common_kwargs())

    assert captured["kwargs"]["on_progress"] is not None
    assert progress.calls == ["load"]


def test_on_progress_is_none_without_progress_reporter(monkeypatch):
    captured = {}
    _stub_front_stage(monkeypatch, captured)

    pipeline.run("in.wav", **_common_kwargs())

    assert captured["kwargs"]["on_progress"] is None


def test_backends_carry_the_settings_that_were_passed(monkeypatch):
    _stub_front_stage(monkeypatch, {})
    result = pipeline.run("in.wav", **_common_kwargs())
    assert result.backends == {
        "separator": "audio-separator-htdemucs-ft",
        "recognizer": "test-content-recognizer",
        "recognizer_revision": None,
        "forced_aligner": "wav2vec2-ctc-forcedalign",
        "english_katakana_method": "arpakana",
    }
