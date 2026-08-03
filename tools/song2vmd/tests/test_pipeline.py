"""song2vmd パイプライン統合のテスト。

音声前段の実行そのものは共有側が持つので、ここでは song2vmd 側の結線——前段の共有出力を口形イベント
確定(events)・モーフ生成(morphs)・診断(report)へ渡すこと——だけを検証する。重い外部アダプタ
(S1分離・S2認識)はモックし、軽量な純粋数値処理(S0読込・S3 RMS)は短い合成WAVフィクスチャで実関数を
そのまま通す。
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lipsync import MouthEvent, MouthShape
from song2vmd import pipeline, presets
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


def write_wav(path, seconds, sample_rate=8000, channels=2, amplitude=0.5):
    n = int(seconds * sample_rate)
    mono = (amplitude * np.sin(2 * np.pi * 220 * np.arange(n) / sample_rate)).astype(np.float32)
    samples = np.stack([mono] * channels, axis=1)
    sf.write(path, samples, sample_rate)
    return samples, sample_rate


def seg(type_, start, end, phoneme=None, confidence=None):
    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=phoneme, confidence=confidence)


def _common_kwargs(**overrides):
    openness, style_gen = presets.resolve("pop")
    kw = dict(
        separate_vocals="always", separator_name="audio-separator-htdemucs-ft",
        content_recognizer_model=_TEST_MODEL, retry=True,
        chunking=ChunkingPolicy(max_duration_sec=300.0), use_n_morph=True,
        intensity_curve=0.6, silence_on=0.06,
        openness=openness, style_gen=style_gen, style_name="pop", model_name="",
        forced_aligner="wav2vec2-ctc-forcedalign", sofa_aligner=None,
    )
    kw.update(overrides)
    return kw


def test_single_run_diagnostics_forced_split_is_always_false(tmp_path, monkeypatch):
    # 非分割経路(_run_single)は forced_split=False で固定(強制分割は長尺分割時のみ起こりうる)。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        front_stage._recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    result = pipeline.run(input_path, **_common_kwargs())
    assert result.diagnostics.forced_split is False


def test_single_run_forwards_mora_event_group_sizes_to_build_diagnostics(tmp_path, monkeypatch):
    # events.confirm_mouth_events の3件目の戻り値(母音的口形ユニットごとの分割数列)が、
    # report.build_diagnostics へ mora_event_group_sizes として正しく中継されることを検証する。
    # confirm_mouth_events自体を完全にモックし、pipeline.pyの配線だけをevents.pyの分割ロジック
    # から独立に検証する(実際の分割数計算が正しいかどうかに依存させない)。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        front_stage._recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    fake_group_sizes = [3]  # 1モーラが3件のMouthEventへ分割されたことを模擬

    def fake_confirm(segments, rms, **kwargs):
        fake_events = [
            MouthEvent(shape=MouthShape.A, start=0.0, end=10.0, open_amount=0.5),
            MouthEvent(shape=MouthShape.A, start=10.0, end=20.0, open_amount=0.6),
            MouthEvent(shape=MouthShape.A, start=20.0, end=30.0, open_amount=0.7),
        ]
        fake_diag = pipeline.events.EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=0)
        return fake_events, fake_diag, fake_group_sizes

    monkeypatch.setattr(pipeline.events, "confirm_mouth_events", fake_confirm)

    real_build_diagnostics = pipeline.report.build_diagnostics
    captured = {}

    def spy_build_diagnostics(**kwargs):
        captured["kwargs"] = kwargs
        return real_build_diagnostics(**kwargs)

    monkeypatch.setattr(pipeline.report, "build_diagnostics", spy_build_diagnostics)

    result = pipeline.run(input_path, progress=None, **_common_kwargs())

    assert captured["kwargs"]["mora_event_group_sizes"] == fake_group_sizes
    assert result.diagnostics.morae == 1  # 3件のMouthEventが1モーラとして集計される


def test_single_run_diagnostics_reflect_backends_style_and_separated(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        front_stage._recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    result = pipeline.run(input_path, **_common_kwargs(
        separator_name="sep-x", content_recognizer_model=ContentRecognizerModel(model_id="rec-y"),
        style_name="ballad", separate_vocals="always"))

    assert result.diagnostics.backends == {
        "separator": "sep-x", "recognizer": "rec-y", "forced_aligner": "wav2vec2-ctc-forcedalign",
        "english_katakana_method": "arpakana"}
    assert result.diagnostics.style == "ballad"
    assert result.diagnostics.separated is True
    assert result.diagnostics.phonemes == 1
    assert result.diagnostics.duration_sec == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("mode,expected", [("always", True), ("never", False)])
def test_separated_flag_matches_separate_vocals_mode(tmp_path, monkeypatch, mode, expected):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, m, **kwargs: vocal_path)
    monkeypatch.setattr(
        front_stage._recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    result = pipeline.run(input_path, **_common_kwargs(separate_vocals=mode))
    assert result.diagnostics.separated is expected


def test_generation_params_are_built_from_openness_and_style_gen(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        front_stage._recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    captured = {}
    real_build = pipeline.morphs.build_vmd_document

    def spy_build_vmd_document(events_, params, model_name):
        captured["params"] = params
        return real_build(events_, params, model_name)

    monkeypatch.setattr(pipeline.morphs, "build_vmd_document", spy_build_vmd_document)

    # vowel_gain は presets.resolve がプリセットの母音別倍率へ乗算して style_gen.vowel_scale に
    # 織り込み済み。pipeline は style_gen の値をそのまま lipsync へ渡す。
    openness, style_gen = presets.resolve("powerful", vowel_gain=(1.1, 0.9, 1.0, 1.0, 1.2))
    pipeline.run(input_path, **_common_kwargs(
        openness=openness, style_gen=style_gen, style_name="powerful"))

    params = captured["params"]
    assert params.open_cap == pytest.approx(openness.open_max)
    assert params.vowel_scale == style_gen.vowel_scale
    assert params.vowel_scale == (1.1, 0.9, 1.0, 1.0, 1.2, 1.0)  # powerfulのプリセット倍率は全て1.0
    assert params.attack_frames == style_gen.attack_frames
    assert params.release_frames == style_gen.release_frames
    assert params.min_hold_frames == style_gen.min_hold_frames
    assert params.triangle_min_frames == style_gen.triangle_min_frames
    assert params.coartic_overlap_max == style_gen.coartic_overlap_max
    assert params.anticipation_frames == style_gen.anticipation_frames
    assert params.legato_valley_shallow == pytest.approx(style_gen.legato_valley_shallow)
    assert params.legato_valley_deep == pytest.approx(style_gen.legato_valley_deep)
    assert params.legato_valley_slope == pytest.approx(style_gen.legato_valley_slope)
    assert params.exaggeration == pytest.approx(style_gen.exaggeration)
    assert params.vibrato_threshold == style_gen.vibrato_threshold
    assert params.vibrato_amp == pytest.approx(style_gen.vibrato_amp)
    assert params.vibrato_period == style_gen.vibrato_period


@pytest.mark.parametrize("boundary_pairs, expected_forced_split", [
    ([(3.0, False), (6.0, False)], False),  # 全境界が無音採用
    ([(3.0, True), (6.0, False)], True),  # 一部が強制分割
    ([(3.0, False), (6.0, True)], True),  # 一部が強制分割(順序が逆でも同じ判定)
])
def test_chunked_run_diagnostics_forced_split_reflects_any_boundary(
        tmp_path, monkeypatch, boundary_pairs, expected_forced_split):
    # find_chunk_boundariesが返すタプル列のうち、いずれか1つでもforced=Trueならforced_split=True
    # (単純な一括判定ではなく境界ごとの論理和)。merge_chunk_segments が受け取る boundaries_sec は
    # forced フラグを含まない座標だけの list[float]。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: boundary_pairs)

    def fake_merge(chunk_segments_list, chunk_offsets_sec, boundaries_sec):
        assert boundaries_sec == [3.0, 6.0]  # 座標だけの list[float]
        return [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(front_stage._chunking, "merge_chunk_segments", fake_merge)
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        front_stage._recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    result = pipeline.run(input_path, **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))
    assert result.diagnostics.forced_split is expected_forced_split


def test_chunked_run_renormalizes_openness_over_whole_song_not_per_chunk(tmp_path, monkeypatch):
    # 長尺分割時、開き量決定の声量レンジ再正規化はチャンク単位でなく結合後の全曲モーラ集合に
    # 対して1回だけ適用される(confirm_mouth_eventsが結合後のセグメント・RMSで1回だけ
    # 呼ばれる配線のため追加のチャンク対応は不要)。最終的にmorphs.build_vmd_documentへ
    # 渡されるmouth_eventsのopen_amountで検証する。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=6.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=6.0, amplitude=0.5)

    # 境界(3.0秒)の前後で異なる母音にし、_merge_adjacentで1件へ統合されないようにする。
    combined_segments = [
        seg("vowel", 0.0, 1.0, phoneme="ɯ", confidence=0.9),
        seg("vowel", 1.0, 2.0, phoneme="e̞", confidence=0.9),
        seg("vowel", 2.0, 3.0, phoneme="a", confidence=0.9),  # 境界直前(第1チャンク)
        seg("vowel", 3.0, 4.0, phoneme="i", confidence=0.9),  # 境界直後(第2チャンク)
        seg("vowel", 4.0, 5.0, phoneme="o̞", confidence=0.9),
        seg("vowel", 5.0, 6.0, phoneme="ɯ", confidence=0.9),
    ]
    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False)])
    monkeypatch.setattr(front_stage._chunking, "merge_chunk_segments", lambda *a, **k: combined_segments)
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        front_stage._recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    raw_values = [0.10, 0.30, 0.50, 0.60, 0.80, 0.99]  # モーラ(0〜5番目)ごとにRMSを変える
    hop = 0.010
    n = round(6.0 / hop) + 1
    times = np.array([0.0125 + i * hop for i in range(n)])
    values = np.array([raw_values[min(int(t // 1.0), 5)] for t in times])
    compute_rms_call_count = [0]
    real_compute_rms = front_stage._rms.compute_rms

    def fake_compute_rms(pcm):
        compute_rms_call_count[0] += 1
        if compute_rms_call_count[0] == 1:
            return real_compute_rms(pcm)  # 境界決定用(生音声側)はそのまま実関数を通す
        return RmsEnvelope(times_sec=times, values=values, dynamic_range_db=20.0)

    monkeypatch.setattr(front_stage._rms, "compute_rms", fake_compute_rms)

    captured = {}
    real_build_vmd_document = pipeline.morphs.build_vmd_document

    def spy_build_vmd_document(mouth_events, gen_params, model_name):
        captured["mouth_events"] = mouth_events
        return real_build_vmd_document(mouth_events, gen_params, model_name)

    monkeypatch.setattr(pipeline.morphs, "build_vmd_document", spy_build_vmd_document)

    openness = presets.OpennessParams(open_lo=0.0, open_hi=1.0, open_max=1.0)
    pipeline.run(input_path, **_common_kwargs(
        chunking=ChunkingPolicy(max_duration_sec=3.0), openness=openness, intensity_curve=1.0))

    mouth_events = captured["mouth_events"]
    boundary_next_event = next(e for e in mouth_events if e.shape == MouthShape.I)

    whole_song_arr = np.array(raw_values)
    p10 = np.percentile(whole_song_arr, 10, method="linear")
    p90 = np.percentile(whole_song_arr, 90, method="linear")
    expected = np.clip((0.60 - p10) / (p90 - p10), 0.0, 1.0)

    chunk2_arr = np.array([0.60, 0.80, 0.99])  # 第2チャンク単体のモーラ集合
    p10_chunk = np.percentile(chunk2_arr, 10, method="linear")
    p90_chunk = np.percentile(chunk2_arr, 90, method="linear")
    per_chunk_only = np.clip((0.60 - p10_chunk) / (p90_chunk - p10_chunk), 0.0, 1.0)

    assert boundary_next_event.open_amount == pytest.approx(float(expected), abs=1e-6)
    assert boundary_next_event.open_amount != pytest.approx(float(per_chunk_only), abs=1e-3)


def test_run_works_without_progress_reporter(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        front_stage._recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    result = pipeline.run(input_path, **_common_kwargs())
    assert result.document is not None


# --- 前段への受け渡しと固有段の報告 ----------------------------------------------


class _RecordingProgress:
    def __init__(self):
        self.calls = []

    def stage(self, stage, *, done=0, total=None, note="", elapsed=0.0):
        self.calls.append(stage)


def _stub_front_stage(monkeypatch, captured, *, segments=None):
    """前段実行エンジンを差し替え、受け取った引数を記録して最小の共有出力を返す。"""
    given = segments if segments is not None else [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    def fake_run_front_stage(input_path, **kwargs):
        captured["input_path"] = input_path
        captured["kwargs"] = kwargs
        if kwargs.get("on_progress") is not None:
            kwargs["on_progress"]("load", done=0, total=None, note="", elapsed=0.0)
        pcm = AudioPcm(samples=np.zeros((8000, 2), dtype=np.float32), sample_rate=8000)
        rms = RmsEnvelope(
            times_sec=np.array([0.0, 0.5]), values=np.array([0.5, 0.5]), dynamic_range_db=20.0)
        return FrontStageResult(
            analysis=AnalysisResult(vocal_wav=Path("vocal.wav"), segments=given, rms=rms),
            pcm=pcm, vocal_pcm=pcm, duration_sec=1.0, forced_split=False)

    monkeypatch.setattr(pipeline, "run_front_stage", fake_run_front_stage)


def test_front_stage_receives_the_settings_and_the_intermediate_directory(tmp_path, monkeypatch):
    captured = {}
    _stub_front_stage(monkeypatch, captured)

    pipeline.run("in.wav", keep_intermediate_dir=str(tmp_path / "keep"), **_common_kwargs(
        separator_name="sep-x", chunking=ChunkingPolicy(max_duration_sec=42.0)))

    kwargs = captured["kwargs"]
    assert captured["input_path"] == "in.wav"
    assert kwargs["separator"] == "sep-x"
    assert kwargs["chunking"] == ChunkingPolicy(max_duration_sec=42.0)
    assert kwargs["keep_intermediate_dir"] == str(tmp_path / "keep")
    assert kwargs["content_recognizer_model"] is _TEST_MODEL
    assert kwargs["forced_aligner"] == "wav2vec2-ctc-forcedalign"
    assert kwargs["sofa_aligner"] is None
    assert kwargs["english_katakana_method"] == "arpakana"


def test_progress_is_relayed_to_the_front_stage_and_own_stages_follow_it(monkeypatch):
    # 前段の段は共有側が報告し、song2vmd 固有の2段はその後に自分で報告する。
    captured = {}
    _stub_front_stage(monkeypatch, captured)
    progress = _RecordingProgress()

    pipeline.run("in.wav", progress=progress, **_common_kwargs())

    assert progress.calls == ["load", "events", "generate"]


def test_front_stage_receives_no_progress_when_omitted(monkeypatch):
    captured = {}
    _stub_front_stage(monkeypatch, captured)

    pipeline.run("in.wav", **_common_kwargs())

    assert captured["kwargs"]["on_progress"] is None
