"""song2vmd パイプライン統合のテスト。

vocal_analysis(S0読込・S1分離・S2認識・S3 RMS)から song2vmd 自身の口形イベント確定(events)・
長尺分割(chunking)・モーフ生成(morphs)までの呼び出し順序・受け渡しを検証する。重い外部アダプタ
(S1分離・S2認識)はモックし、軽量な純粋数値処理(S0読込・S3 RMS)は短い合成WAVフィクスチャで
実関数をそのまま通す(長尺分割の境界計算・結合そのものは tests/test_chunking.py が担うため、
ここではモックしてパイプライン側の呼び出し・受け渡しだけを検証する)。
"""

import json

import numpy as np
import pytest
import soundfile as sf

from cli_progress_router import ProgressEmitError
from lipsync import MouthEvent, MouthShape
from song2vmd import pipeline, presets
from vocal_analysis import ContentRecognizerModel, RmsEnvelope, Segment
from vocal_analysis.io import AudioLoadError
from vocal_analysis.recognizer import RecognitionError
from vocal_analysis.separator import SeparationError
from vocal_analysis.types import AudioPcm

_TEST_MODEL = ContentRecognizerModel(model_id="test-content-recognizer")


def write_wav(path, seconds, sample_rate=8000, channels=2, amplitude=0.5):
    n = int(seconds * sample_rate)
    mono = (amplitude * np.sin(2 * np.pi * 220 * np.arange(n) / sample_rate)).astype(np.float32)
    samples = np.stack([mono] * channels, axis=1)
    sf.write(path, samples, sample_rate)
    return samples, sample_rate


def seg(type_, start, end, phoneme=None, confidence=None):
    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=phoneme, confidence=confidence)


class _RecordingProgress:
    def __init__(self):
        self.calls = []

    def stage(self, stage, *, done=0, total=None, note="", elapsed=0.0):
        self.calls.append({"stage": stage, "done": done, "total": total, "note": note})
        assert elapsed >= 0.0


def _common_kwargs(**overrides):
    openness, style_gen = presets.resolve("pop")
    kw = dict(
        separate_vocals="always", separator_name="audio-separator-htdemucs-ft",
        content_recognizer_model=_TEST_MODEL, retry=True,
        max_duration_sec=300.0, use_n_morph=True,
        intensity_curve=0.6, silence_on=0.06,
        openness=openness, style_gen=style_gen, style_name="pop", model_name="",
        forced_aligner="wav2vec2-ctc-forcedalign", sofa_aligner=None,
    )
    kw.update(overrides)
    return kw


# --- 単一実行(--max-duration 未超過) ------------------------------------------


def test_single_run_calls_stages_in_order(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    progress = _RecordingProgress()
    result = pipeline.run(input_path, progress=progress, **_common_kwargs())

    assert [c["stage"] for c in progress.calls] == [
        "load", "separate", "recognize", "rms", "events", "generate",
    ]
    assert result.document is not None
    assert len(result.document.morph) == result.diagnostics.keys
    assert result.sample_rate == 8000
    assert result.channels == 2


def test_single_run_passes_separator_name_to_separate(tmp_path, monkeypatch):
    # --separator で選んだ安定 id は診断表示だけでなく、実際の分離呼び出しへ渡る。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    captured = {}

    def spy_separate(pcm, mode, **kwargs):
        captured["separator"] = kwargs.get("separator")
        return vocal_path

    monkeypatch.setattr(pipeline._va_separator, "separate", spy_separate)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    pipeline.run(input_path, **_common_kwargs(separator_name="audio-separator-htdemucs-ft"))

    assert captured["separator"] == "audio-separator-htdemucs-ft"


def test_single_run_diagnostics_forced_split_is_always_false(tmp_path, monkeypatch):
    # 非分割経路(_run_single)は forced_split=False で固定(強制分割は長尺分割時のみ起こりうる)。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    result = pipeline.run(input_path, **_common_kwargs())
    assert result.diagnostics.forced_split is False


def test_single_run_calls_underlying_functions_in_order_with_correct_data_flow(tmp_path, monkeypatch):
    # progress.stage()の順序だけでなく、実際の下位関数の呼び出し順序とデータの受け渡し
    # (separateへ渡すpcm・recognizeへ渡すvocal_path・events.confirm_mouth_eventsへ渡す
    # segments/rms/各パラメータ)を検証する。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)
    given_segments = [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    call_order = []
    real_load_audio = pipeline._va_io.load_audio
    real_compute_rms = pipeline._va_rms.compute_rms
    real_confirm = pipeline.events.confirm_mouth_events
    real_build = pipeline.morphs.build_vmd_document
    captured = {}

    def spy_load_audio(path):
        call_order.append("load_audio")
        result = real_load_audio(path)
        captured.setdefault("load_audio_results", []).append(result)
        return result

    def spy_separate(pcm, mode, **kwargs):
        call_order.append("separate")
        captured["separate_pcm"] = pcm
        return vocal_path

    def spy_recognize(path, **kwargs):
        call_order.append("recognize")
        captured["recognize_path"] = path
        return given_segments

    def spy_compute_rms(pcm):
        call_order.append("compute_rms")
        result = real_compute_rms(pcm)
        captured["compute_rms_result"] = result
        return result

    def spy_confirm(segments, rms, **kwargs):
        call_order.append("confirm_mouth_events")
        captured["confirm_segments"] = segments
        captured["confirm_rms"] = rms
        captured["confirm_kwargs"] = kwargs
        return real_confirm(segments, rms, **kwargs)

    def spy_build(events_, params, model_name):
        call_order.append("build_vmd_document")
        return real_build(events_, params, model_name)

    class _OrderRecordingProgress:
        def stage(self, stage, *, done=0, total=None, note="", elapsed=0.0):
            call_order.append(f"progress:{stage}")

    monkeypatch.setattr(pipeline._va_io, "load_audio", spy_load_audio)
    monkeypatch.setattr(pipeline._va_separator, "separate", spy_separate)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", spy_recognize)
    monkeypatch.setattr(pipeline._va_rms, "compute_rms", spy_compute_rms)
    monkeypatch.setattr(pipeline.events, "confirm_mouth_events", spy_confirm)
    monkeypatch.setattr(pipeline.morphs, "build_vmd_document", spy_build)

    openness, style_gen = presets.resolve("pop")
    pipeline.run(input_path, progress=_OrderRecordingProgress(), **_common_kwargs(
        openness=openness, intensity_curve=0.7, silence_on=0.05, use_n_morph=False))

    # load_audioは入力読み込みと(分離後の)ボーカル読み込みの2回呼ばれる。各段のprogress発行
    # ("progress:X")は、その段の実処理("X")より必ず前に来る(「各段は開始時に
    # 最低1本のprogressを出す」挙動の後退を防ぐ回帰テスト)。
    assert call_order == [
        "progress:load", "load_audio", "progress:separate", "separate", "progress:recognize",
        "recognize", "progress:rms", "load_audio", "compute_rms", "progress:events",
        "confirm_mouth_events", "progress:generate", "build_vmd_document",
    ]
    assert str(captured["recognize_path"]) == str(vocal_path)
    # 1回目のload_audioの戻り値がそのままseparateへ渡ること、compute_rmsの戻り値がそのまま
    # confirm_mouth_eventsのrms引数へ渡ることを、同一性(取り違え・別経路生成が無いこと)で確認する。
    assert captured["separate_pcm"] is captured["load_audio_results"][0]
    assert captured["confirm_rms"] is captured["compute_rms_result"]
    assert captured["confirm_segments"] == given_segments
    assert captured["confirm_kwargs"] == {
        "open_lo": openness.open_lo, "open_hi": openness.open_hi, "open_max": openness.open_max,
        "intensity_curve": 0.7, "silence_on": 0.05, "use_n_morph": False,
    }


def test_single_run_forwards_mora_event_group_sizes_to_build_diagnostics(tmp_path, monkeypatch):
    # events.confirm_mouth_events の3件目の戻り値(母音的口形ユニットごとの分割数列)が、
    # report.build_diagnostics へ mora_event_group_sizes として正しく中継されることを検証する。
    # confirm_mouth_events自体を完全にモックし、pipeline.pyの配線だけをevents.pyの分割ロジック
    # から独立に検証する(実際の分割数計算が正しいかどうかに依存させない)。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
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

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
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

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, m, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    result = pipeline.run(input_path, **_common_kwargs(separate_vocals=mode))
    assert result.diagnostics.separated is expected


def test_recognizer_receives_selected_content_recognizer_model(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    received = {}

    def fake_recognize(path, content_recognizer_model, retry, forced_aligner, sofa_aligner,
                       english_katakana_method, on_progress=None):
        received["content_recognizer_model"] = content_recognizer_model
        received["forced_aligner"] = forced_aligner
        received["sofa_aligner"] = sofa_aligner
        return [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", fake_recognize)

    given_model = ContentRecognizerModel(model_id="org/custom-recognizer", model_revision="rev1")
    given_sofa_config = object()
    pipeline.run(input_path, **_common_kwargs(
        content_recognizer_model=given_model, forced_aligner="sofa-forcedalign",
        sofa_aligner=given_sofa_config))
    assert received["content_recognizer_model"] is given_model
    assert received["forced_aligner"] == "sofa-forcedalign"
    assert received["sofa_aligner"] is given_sofa_config


def test_recognizer_receives_selected_english_katakana_method(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    received = {}

    def fake_recognize(path, content_recognizer_model, retry, forced_aligner, sofa_aligner,
                       english_katakana_method, on_progress=None):
        received["english_katakana_method"] = english_katakana_method
        return [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", fake_recognize)

    pipeline.run(input_path, **_common_kwargs(
        english_katakana_method="tinyllama-katakana-converter"))
    assert received["english_katakana_method"] == "tinyllama-katakana-converter"


def test_generation_params_are_built_from_openness_and_style_gen(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
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


# --- 長尺分割(--max-duration 超過) ---------------------------------------------


@pytest.mark.parametrize("boundary_pairs, expected_forced_split", [
    ([(3.0, False), (6.0, False)], False),  # 全境界が無音採用
    ([(3.0, True), (6.0, False)], True),  # 一部が強制分割
    ([(3.0, False), (6.0, True)], True),  # 一部が強制分割(順序が逆でも同じ判定)
])
def test_chunked_run_diagnostics_forced_split_reflects_any_boundary(
        tmp_path, monkeypatch, boundary_pairs, expected_forced_split):
    # find_chunk_boundariesが返すタプル列のうち、いずれか1つでもforced=Trueならforced_split=True
    # (単純な一括判定ではなく境界ごとの論理和)。_run_chunkedはboundaries_secへ座標だけを渡す
    # ので、merge_chunk_segmentsが受け取るboundaries_secは従来どおりlist[float]のまま。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: boundary_pairs)

    def fake_merge(chunk_segments_list, chunk_offsets_sec, boundaries_sec):
        assert boundaries_sec == [3.0, 6.0]  # 座標だけのlist[float]で従来どおり
        return [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline.chunking, "merge_chunk_segments", fake_merge)
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    result = pipeline.run(input_path, **_common_kwargs(max_duration_sec=3.0))
    assert result.diagnostics.forced_split is expected_forced_split


def test_chunked_run_calls_separate_and_recognize_once_per_chunk(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False), (6.0, False)])

    def fake_merge(chunk_segments_list, chunk_offsets_sec, boundaries_sec):
        assert len(chunk_segments_list) == 3
        # 境界[3.0, 6.0]・オーバーラップ1.0秒・全長10.0秒のとき、各チャンクの範囲は
        # [0,4]・[2,7]・[5,10](先頭・末尾は片側のみオーバーラップ)なので、
        # チャンクローカル時刻0に対応するグローバル時刻(オフセット)は [0.0, 2.0, 5.0] になる。
        assert chunk_offsets_sec == pytest.approx([0.0, 2.0, 5.0])
        assert boundaries_sec == [3.0, 6.0]
        return [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline.chunking, "merge_chunk_segments", fake_merge)

    separate_calls = []
    separate_kwargs = []

    def fake_separate(pcm, mode, **kwargs):
        separate_calls.append(len(pcm.samples) / pcm.sample_rate)
        separate_kwargs.append(kwargs)
        return vocal_path

    recognize_calls = []

    def fake_recognize(path, **kwargs):
        recognize_calls.append((path, kwargs))
        return [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline._va_separator, "separate", fake_separate)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", fake_recognize)

    sofa_config = object()
    progress = _RecordingProgress()
    result = pipeline.run(input_path, progress=progress, **_common_kwargs(
        max_duration_sec=3.0, forced_aligner="sofa-forcedalign", sofa_aligner=sofa_config,
        english_katakana_method="tinyllama-katakana-converter"))

    assert len(separate_calls) == 3
    assert len(recognize_calls) == 3
    # forced_aligner・sofa_aligner・english_katakana_methodは長尺分割の全チャンクへ
    # 同一の値で伝播する。
    for _, kwargs in recognize_calls:
        assert kwargs["forced_aligner"] == "sofa-forcedalign"
        assert kwargs["sofa_aligner"] is sofa_config
        assert kwargs["english_katakana_method"] == "tinyllama-katakana-converter"
    # S1の分離アダプタ選択も同様に全チャンクへ伝播する。
    for kwargs in separate_kwargs:
        assert kwargs["separator"] == "audio-separator-htdemucs-ft"
    # 各チャンクは前後1.0秒のオーバーラップを持つ(先頭・末尾は片側のみ)。
    assert separate_calls[0] == pytest.approx(4.0, abs=0.05)  # [0, 3+1]
    assert separate_calls[1] == pytest.approx(5.0, abs=0.05)  # [3-1, 6+1]
    assert separate_calls[2] == pytest.approx(5.0, abs=0.05)  # [6-1, 10]
    assert result.diagnostics.duration_sec == pytest.approx(10.0, abs=0.05)
    # doneは「このチャンクを始める時点までに完了したチャンク数」(0始まり)。
    separate_done_totals = [(c["done"], c["total"]) for c in progress.calls if c["stage"] == "separate"]
    assert separate_done_totals == [(0, 3), (1, 3), (2, 3)]


def test_chunked_run_uses_raw_audio_rms_for_boundaries_and_whole_vocal_rms_for_events(tmp_path, monkeypatch):
    # 境界決定(find_chunk_boundaries)には分離前の生音声のRMSを使い、口形イベント確定
    # (events.confirm_mouth_events)にはチャンクの核区間を連結した曲全体のボーカルRMSを
    # 1回だけ渡す(チャンクごとに個別正規化しない)ことを検証する。
    # rms.compute_rmsは実関数をそのまま通し、どちらの呼び出しがどの下流(境界決定/events)へ
    # 渡ったかは戻り値の同一性(is)で識別する(io.load_audioがピーク正規化するため、振幅の
    # 大小では入力を識別できない)。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0, amplitude=0.5)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    boundary_calls = []

    def fake_find_boundaries(duration_sec, rms_times_sec, rms_values, **kwargs):
        boundary_calls.append((duration_sec, rms_times_sec, rms_values))
        return [(3.0, False), (6.0, False)]

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", fake_find_boundaries)
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)])
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    compute_rms_results = []
    real_compute_rms = pipeline._va_rms.compute_rms

    def recording_compute_rms(pcm):
        result = real_compute_rms(pcm)
        compute_rms_results.append(result)
        return result

    monkeypatch.setattr(pipeline._va_rms, "compute_rms", recording_compute_rms)

    confirm_rms_args = []
    real_confirm = pipeline.events.confirm_mouth_events

    def spy_confirm(segments, rms, **kwargs):
        confirm_rms_args.append(rms)
        return real_confirm(segments, rms, **kwargs)

    monkeypatch.setattr(pipeline.events, "confirm_mouth_events", spy_confirm)

    pipeline.run(input_path, **_common_kwargs(max_duration_sec=3.0))

    assert len(boundary_calls) == 1
    assert boundary_calls[0][0] == pytest.approx(10.0, abs=0.05)
    # compute_rmsはちょうど2回: (1)境界決定用の生音声、(2)events用の曲全体ボーカル。
    assert len(compute_rms_results) == 2
    # 1回目のcompute_rmsの戻り値がそのままfind_chunk_boundariesへ、2回目の戻り値がそのまま
    # confirm_mouth_eventsへ渡ること(取り違えが無いこと)を同一性で確認する。
    assert boundary_calls[0][1] is compute_rms_results[0].times_sec
    assert boundary_calls[0][2] is compute_rms_results[0].values
    assert len(confirm_rms_args) == 1
    assert confirm_rms_args[0] is compute_rms_results[1]


def test_chunked_run_preserves_relative_loudness_across_chunks(tmp_path, monkeypatch):
    # 各チャンクの分離済みボーカル音声の振幅が異なるとき(曲の強弱)、チャンクごとの読み込みで
    # vocal_analysis.io.load_audioのピーク正規化(目標値固定)を経由すると、静かなチャンクも
    # 大きいチャンクも独立に同じ目標振幅へ引き伸ばされ、チャンク間の相対的な強弱(曲全体基準の
    # RMS)が壊れる。核区間を連結した曲全体のボーカル音声が、チャンクごとの
    # 元の振幅差を保持していることを検証する。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=6.0)
    quiet_vocal = tmp_path / "vocal_quiet.wav"
    write_wav(quiet_vocal, seconds=6.0, amplitude=0.1)
    loud_vocal = tmp_path / "vocal_loud.wav"
    write_wav(loud_vocal, seconds=6.0, amplitude=0.9)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False)])
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 6.0, phoneme="a", confidence=0.9)])

    separate_call_count = [0]

    def fake_separate(pcm, mode, **kwargs):
        separate_call_count[0] += 1
        return quiet_vocal if separate_call_count[0] == 1 else loud_vocal

    monkeypatch.setattr(pipeline._va_separator, "separate", fake_separate)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    compute_rms_pcms = []
    real_compute_rms = pipeline._va_rms.compute_rms

    def recording_compute_rms(pcm):
        compute_rms_pcms.append(pcm)
        return real_compute_rms(pcm)

    monkeypatch.setattr(pipeline._va_rms, "compute_rms", recording_compute_rms)

    pipeline.run(input_path, **_common_kwargs(max_duration_sec=3.0))

    # compute_rmsは2回呼ばれる: (1)境界決定用の生音声、(2)events用の曲全体ボーカル。
    whole_vocal_samples = compute_rms_pcms[1].samples
    half = len(whole_vocal_samples) // 2
    first_half_peak = float(np.max(np.abs(whole_vocal_samples[:half])))
    second_half_peak = float(np.max(np.abs(whole_vocal_samples[half:])))
    # ピーク正規化がチャンクごとにかかっていれば両半分とも同じ目標振幅になり見分けがつかない。
    # 個別正規化を経ていなければ、静かな前半(0.1)と大きい後半(0.9)の振幅差が保たれる。
    assert first_half_peak < second_half_peak * 0.5


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
    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False)])
    monkeypatch.setattr(pipeline.chunking, "merge_chunk_segments", lambda *a, **k: combined_segments)
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    raw_values = [0.10, 0.30, 0.50, 0.60, 0.80, 0.99]  # モーラ(0〜5番目)ごとにRMSを変える
    hop = 0.010
    n = round(6.0 / hop) + 1
    times = np.array([0.0125 + i * hop for i in range(n)])
    values = np.array([raw_values[min(int(t // 1.0), 5)] for t in times])
    compute_rms_call_count = [0]
    real_compute_rms = pipeline._va_rms.compute_rms

    def fake_compute_rms(pcm):
        compute_rms_call_count[0] += 1
        if compute_rms_call_count[0] == 1:
            return real_compute_rms(pcm)  # 境界決定用(生音声側)はそのまま実関数を通す
        return RmsEnvelope(times_sec=times, values=values, dynamic_range_db=20.0)

    monkeypatch.setattr(pipeline._va_rms, "compute_rms", fake_compute_rms)

    captured = {}
    real_build_vmd_document = pipeline.morphs.build_vmd_document

    def spy_build_vmd_document(mouth_events, gen_params, model_name):
        captured["mouth_events"] = mouth_events
        return real_build_vmd_document(mouth_events, gen_params, model_name)

    monkeypatch.setattr(pipeline.morphs, "build_vmd_document", spy_build_vmd_document)

    openness = presets.OpennessParams(open_lo=0.0, open_hi=1.0, open_max=1.0)
    pipeline.run(input_path, **_common_kwargs(
        max_duration_sec=3.0, openness=openness, intensity_curve=1.0))

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


def test_chunked_run_reports_recognize_progress_with_chunk_totals(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [(5.0, False)])
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)])
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    progress = _RecordingProgress()
    pipeline.run(input_path, progress=progress, **_common_kwargs(max_duration_sec=3.0))

    # doneは「このチャンクを始める時点までに完了したチャンク数」(0始まり)。
    recognize_done_totals = [(c["done"], c["total"]) for c in progress.calls if c["stage"] == "recognize"]
    assert recognize_done_totals == [(0, 2), (1, 2)]


def test_non_chunked_progress_reports_done_zero_total_none_for_separate_and_recognize(tmp_path, monkeypatch):
    # 分割しない場合や内訳の無い段は done=0, total=None。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    progress = _RecordingProgress()
    pipeline.run(input_path, progress=progress, **_common_kwargs())

    by_stage = {c["stage"]: c for c in progress.calls}
    assert by_stage["separate"]["done"] == 0
    assert by_stage["separate"]["total"] is None
    assert by_stage["recognize"]["done"] == 0
    assert by_stage["recognize"]["total"] is None


# --- 音素認識のモデルダウンロード進捗の中継 --------------------------------------
#
# vocal_analysis.recognizer.recognize() の on_progress 引数(モデル初回取得が実際にダウンロードを
# 要した区間だけ進捗文言を渡すコールバック)へ、進捗の報告先を橋渡しする。
# 橋渡しの要点: (1) recognize() へ on_progress を渡す、(2) on_progress が呼ばれたら
# progress.stage("recognize", note=<文言>) へ反映する、(3) その際の done/total は呼び出し時点の
# 進捗(分割時はチャンク進捗)をそのまま保つ(0/None へ巻き戻さない)。


def test_non_chunked_recognize_on_progress_forwards_download_note(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    captured = {}

    def fake_recognize(path, on_progress=None, **kwargs):
        captured["on_progress"] = on_progress
        if on_progress is not None:
            on_progress("ダウンロード中: dummy-model 42%")
        return [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", fake_recognize)

    progress = _RecordingProgress()
    pipeline.run(input_path, progress=progress, **_common_kwargs())

    assert captured["on_progress"] is not None
    recognize_calls = [c for c in progress.calls if c["stage"] == "recognize"]
    assert any(c["note"] == "ダウンロード中: dummy-model 42%" for c in recognize_calls)
    # 分割しない実行の既定(done=0, total=None)を、ダウンロード通知後も保ったままにする。
    assert all(c["done"] == 0 and c["total"] is None for c in recognize_calls)


def test_chunked_recognize_on_progress_preserves_chunk_done_total(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [(5.0, False)])
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)])
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    def fake_recognize(path, on_progress=None, **kwargs):
        if on_progress is not None:
            on_progress("ダウンロード中: dummy-model 10%")
        return [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline._va_recognizer, "recognize", fake_recognize)

    progress = _RecordingProgress()
    pipeline.run(input_path, progress=progress, **_common_kwargs(max_duration_sec=3.0))

    recognize_calls = [c for c in progress.calls if c["stage"] == "recognize"]
    download_notes = [c for c in recognize_calls if c["note"] == "ダウンロード中: dummy-model 10%"]
    # 2チャンクとも、そのチャンクの done/total(チャンク進捗)を保ったままダウンロード通知が出る。
    assert [(c["done"], c["total"]) for c in download_notes] == [(0, 2), (1, 2)]


def test_non_chunked_separate_on_progress_forwards_download_note(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    captured = {}

    def fake_separate(pcm, mode, on_progress=None, **kwargs):
        captured["on_progress"] = on_progress
        if on_progress is not None:
            on_progress("ダウンロード中: 42%")
        return vocal_path

    monkeypatch.setattr(pipeline._va_separator, "separate", fake_separate)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", lambda path, **kwargs: [
        seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    progress = _RecordingProgress()
    pipeline.run(input_path, progress=progress, **_common_kwargs())

    assert captured["on_progress"] is not None
    separate_calls = [c for c in progress.calls if c["stage"] == "separate"]
    assert any(c["note"] == "ダウンロード中: 42%" for c in separate_calls)
    # 分割しない実行の既定(done=0, total=None)を、ダウンロード通知後も保ったままにする。
    assert all(c["done"] == 0 and c["total"] is None for c in separate_calls)


def test_chunked_separate_on_progress_preserves_chunk_done_total(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [(5.0, False)])
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)])

    def fake_separate(pcm, mode, on_progress=None, **kwargs):
        if on_progress is not None:
            on_progress("ダウンロード中: 10%")
        return vocal_path

    monkeypatch.setattr(pipeline._va_separator, "separate", fake_separate)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", lambda path, **kwargs: [
        seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    progress = _RecordingProgress()
    pipeline.run(input_path, progress=progress, **_common_kwargs(max_duration_sec=3.0))

    separate_calls = [c for c in progress.calls if c["stage"] == "separate"]
    download_notes = [c for c in separate_calls if c["note"] == "ダウンロード中: 10%"]
    # 2チャンクとも、そのチャンクの done/total(チャンク進捗)を保ったままダウンロード通知が出る。
    assert [(c["done"], c["total"]) for c in download_notes] == [(0, 2), (1, 2)]


def test_run_without_progress_reporter_passes_on_progress_none_to_separate(tmp_path, monkeypatch):
    # progress 省略時は separate() へも on_progress=None を渡す(recognize と同じ契約)。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    captured = {}

    def fake_separate(pcm, mode, *, on_progress, **kwargs):
        captured["on_progress"] = on_progress
        return vocal_path

    monkeypatch.setattr(pipeline._va_separator, "separate", fake_separate)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", lambda path, **kwargs: [
        seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    pipeline.run(input_path, **_common_kwargs())

    assert captured["on_progress"] is None


def test_run_without_progress_reporter_passes_on_progress_none_to_recognize(tmp_path, monkeypatch):
    # progress 省略時は recognize() へ on_progress=None を渡す(存在しない進捗表示へ橋渡しする
    # 無意味なコールバックを作らない)。on_progress を必須キーワード引数にして、現行の
    # (on_progress を渡さない)実装では TypeError で確実に落ちるようにする(既定値だと
    # 未実装のままでも偶然 None のまま通ってしまい印の意味が無くなるため)。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    captured = {}

    def fake_recognize(path, *, on_progress, **kwargs):
        captured["on_progress"] = on_progress
        return [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", fake_recognize)

    pipeline.run(input_path, **_common_kwargs())

    assert captured["on_progress"] is None


def test_chunked_run_without_progress_reporter_passes_on_progress_none_to_recognize(tmp_path, monkeypatch):
    # 上と同じ契約(progress省略→on_progress=None)を、_run_chunked 側の recognize 呼び出し箇所
    # (_run_single とは別のコード経路)でも独立に固定する。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [(5.0, False)])
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)])
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    captured = []

    def fake_recognize(path, *, on_progress, **kwargs):
        captured.append(on_progress)
        return [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline._va_recognizer, "recognize", fake_recognize)

    pipeline.run(input_path, **_common_kwargs(max_duration_sec=3.0))

    assert captured == [None, None]


# --- 進捗レポータ省略時 --------------------------------------------------------


def test_run_works_without_progress_reporter(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    result = pipeline.run(input_path, **_common_kwargs())
    assert result.document is not None


# --- --keep-intermediate(中間生成物の保存) ------------------------------------


def test_keep_intermediate_dir_none_creates_nothing(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    keep_dir = tmp_path / "out.vmd.intermediate"
    pipeline.run(input_path, **_common_kwargs())  # keep_intermediate_dir省略(既定None)
    assert not keep_dir.exists()


def test_keep_intermediate_saves_normalized_input_vocal_and_segments(tmp_path, monkeypatch):
    # input と vocal を異なるサンプルレートにし、保存先が入れ替わる退行を検出できるようにする。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0, sample_rate=8000)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, sample_rate=11025, amplitude=0.8)
    given_segments = [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize", lambda path, **kwargs: given_segments)

    keep_dir = tmp_path / "out.vmd.intermediate"
    pipeline.run(input_path, keep_intermediate_dir=keep_dir, **_common_kwargs())

    assert (keep_dir / "input_normalized.wav").exists()
    assert (keep_dir / "vocal.wav").exists()
    _, saved_input_sr = sf.read(str(keep_dir / "input_normalized.wav"))
    _, saved_vocal_sr = sf.read(str(keep_dir / "vocal.wav"))
    assert saved_input_sr == 8000
    assert saved_vocal_sr == 11025
    segments_path = keep_dir / "segments.json"
    assert segments_path.exists()
    saved = json.loads(segments_path.read_text(encoding="utf-8"))
    assert saved == [
        {"type": "vowel", "start_sec": 0.0, "end_sec": 1.0, "phoneme": "a", "confidence": 0.9},
    ]


def test_keep_intermediate_write_failure_raises_intermediate_write_error(tmp_path, monkeypatch):
    # sf.write はlibsndfileが開くため、失敗を OSError でなく sf.SoundFileError 系で送出する。
    # モックで OSError を偽装せず、書き込み先の名前をディレクトリで塞いで実際に失敗させる。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    keep_dir = tmp_path / "out.vmd.intermediate"
    keep_dir.mkdir()
    (keep_dir / "input_normalized.wav").mkdir()  # 同名ディレクトリで sf.write の書き込み先を塞ぐ

    with pytest.raises(pipeline.IntermediateWriteError):
        pipeline.run(input_path, keep_intermediate_dir=keep_dir, **_common_kwargs())


def test_keep_intermediate_chunked_saves_concatenated_vocal(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=6.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=6.0, amplitude=0.8)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False)])
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 6.0, phoneme="a", confidence=0.9)])
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    keep_dir = tmp_path / "out.vmd.intermediate"
    pipeline.run(input_path, keep_intermediate_dir=keep_dir, **_common_kwargs(max_duration_sec=3.0))

    saved_samples, saved_sr = sf.read(str(keep_dir / "vocal.wav"), dtype="float32", always_2d=True)
    # 曲全体(6秒)分の核区間連結ボーカルが保存される(チャンクごとの重複区間を含まない)。
    assert saved_samples.shape[0] == pytest.approx(6.0 * saved_sr, abs=saved_sr * 0.01)


# --- PCMスライス/連結ヘルパ(純粋ロジック) -------------------------------------


def test_slice_pcm_extracts_the_requested_time_range():
    samples = np.arange(100, dtype=np.float32).reshape(-1, 1)
    pcm = AudioPcm(samples=samples, sample_rate=10)
    sliced = pipeline._slice_pcm(pcm, 2.0, 5.0)
    assert sliced.sample_rate == 10
    np.testing.assert_array_equal(sliced.samples[:, 0], samples[20:50, 0])


def test_concat_pcm_joins_slices_in_order():
    pcm_a = AudioPcm(samples=np.array([[1.0], [2.0]], dtype=np.float32), sample_rate=10)
    pcm_b = AudioPcm(samples=np.array([[3.0], [4.0]], dtype=np.float32), sample_rate=10)
    joined = pipeline._concat_pcm([pcm_a, pcm_b])
    np.testing.assert_array_equal(joined.samples[:, 0], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert joined.sample_rate == 10


# --- 内部生成ファイルの読み直し失敗(利用者入力の読み込み失敗と区別する)---


def test_vocal_reread_failure_raises_dedicated_error_with_path(tmp_path, monkeypatch):
    # 音量解析段の分離後ボーカルの読み直し失敗は、利用者入力の読み込み失敗と混ざらないよう
    # 専用例外へ包み、対象パスを保持する。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    real_load_audio = pipeline._va_io.load_audio

    def fail_on_vocal(path):
        if str(path) == str(vocal_path):
            raise AudioLoadError("broken vocal wav", reason="not_audio")
        return real_load_audio(path)

    monkeypatch.setattr(pipeline._va_io, "load_audio", fail_on_vocal)
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kw: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kw: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    openness, style_gen = presets.resolve("pop")
    with pytest.raises(pipeline.IntermediateReadError) as exc:
        pipeline.run(input_path, progress=_RecordingProgress(), **_common_kwargs(
            openness=openness, intensity_curve=0.7, silence_on=0.05, use_n_morph=False))
    assert str(exc.value.path) == str(vocal_path)


def test_chunked_vocal_reread_failure_raises_dedicated_error_with_path(tmp_path, monkeypatch):
    # 長尺分割の経路は各チャンクのボーカルWAVを別実装で読むが、内部生成ファイルの読み直しである点は
    # 同じなので同じ専用例外へ包む。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(
        pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False), (6.0, False)])
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)])
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode, **kw: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kw: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    # soundfile は入力の読み込みでも使うので、失敗させるのは分離後ボーカルの読み直しだけに限る。
    real_read = pipeline.sf.read

    def fail_on_vocal(path, *a, **kw):
        if str(path) == str(vocal_path):
            # soundfile が読み込み失敗で実際に送出する例外型に合わせる。
            raise sf.LibsndfileError(1, prefix=str(vocal_path))
        return real_read(path, *a, **kw)

    monkeypatch.setattr(pipeline.sf, "read", fail_on_vocal)

    with pytest.raises(pipeline.IntermediateReadError) as exc:
        pipeline.run(input_path, **_common_kwargs(max_duration_sec=3.0))
    assert str(exc.value.path) == str(vocal_path)


def test_input_read_failure_is_not_wrapped_in_dedicated_error(tmp_path, monkeypatch):
    # 利用者入力の読み込み失敗は音声読み込みの例外のまま送出する(専用例外へ包まない)。
    # 派生関係で通り抜けないよう、型そのものを固定する。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)

    def always_fail(path):
        raise AudioLoadError("broken input", reason="not_audio")

    monkeypatch.setattr(pipeline._va_io, "load_audio", always_fail)

    openness, style_gen = presets.resolve("pop")
    with pytest.raises(AudioLoadError) as exc:
        pipeline.run(input_path, progress=_RecordingProgress(), **_common_kwargs(
            openness=openness, intensity_curve=0.7, silence_on=0.05, use_n_morph=False))
    assert type(exc.value) is AudioLoadError


# --- 外部推論から漏れた例外のステージ写像 --------------------------------------


def _stage_failure_kwargs(tmp_path, monkeypatch, *, failing_stage, error, chunked=False):
    """分離または認識だけを失敗させたパイプライン実行の材料を組み立てる。"""
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0 if chunked else 1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0 if chunked else 1.0, amplitude=0.8)

    def separate(pcm, mode, **kw):
        if failing_stage == "separate":
            raise error
        return vocal_path

    def recognize(path, **kw):
        if failing_stage == "recognize":
            raise error
        return [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline._va_separator, "separate", separate)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", recognize)
    if chunked:
        monkeypatch.setattr(
            pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False), (6.0, False)])
        monkeypatch.setattr(
            pipeline.chunking, "merge_chunk_segments",
            lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)])
    return input_path, _common_kwargs(max_duration_sec=3.0 if chunked else 300.0)


class _UnlistedError(Exception):
    """除外一覧に無い、推論器が独自に定義しうる例外を模した型。"""


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failing_stage, stage_label", [
    ("separate", "ボーカル分離"), ("recognize", "音素認識")])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, MemoryError, OSError, _UnlistedError])
def test_bare_exception_from_inference_carries_failing_stage(
        tmp_path, monkeypatch, failing_stage, stage_label, error_type, chunked):
    # 除外対象以外の例外は、型を問わず、どの工程で失敗したかを持つ例外へ写像する。
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, failing_stage=failing_stage,
        error=error_type("推論の失敗"), chunked=chunked)

    with pytest.raises(pipeline.StageExecutionError) as exc:
        pipeline.run(input_path, **kwargs)
    assert exc.value.stage == failing_stage
    # 原因を追えるよう、元の例外の型名と文言を message に残す。stage キーを持たない非機械モードでも
    # どの工程が失敗したか分かるよう、工程名も message に入れる。
    assert f"{stage_label}に失敗しました" in str(exc.value)
    assert error_type.__name__ in str(exc.value)
    assert "推論の失敗" in str(exc.value)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failing_stage", ["separate", "recognize"])
@pytest.mark.parametrize("error_factory", [
    lambda: SeparationError("separator failed"),
    lambda: RecognitionError("recognizer failed"),
    lambda: AudioLoadError("broken wav", reason="not_audio"),
    lambda: pipeline.IntermediateReadError("broken vocal wav", path="vocal.wav"),
    lambda: pipeline.IntermediateWriteError("disk full"),
])
def test_classified_exceptions_from_inference_pass_through_unchanged(
        tmp_path, monkeypatch, failing_stage, error_factory, chunked):
    # 既に分類が定まっている例外は写像せず、型そのままで送出する(終了コードの分岐を保つため)。
    error = error_factory()
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, failing_stage=failing_stage, error=error, chunked=chunked)

    with pytest.raises(type(error)) as exc:
        pipeline.run(input_path, **kwargs)
    assert exc.value is error


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failing_stage", ["separate", "recognize"])
def test_keyboard_interrupt_from_inference_passes_through_unchanged(
        tmp_path, monkeypatch, failing_stage, chunked):
    # 中断は工程の失敗ではないので、中断の経路へそのまま届ける。
    error = KeyboardInterrupt()
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, failing_stage=failing_stage, error=error, chunked=chunked)

    with pytest.raises(KeyboardInterrupt) as exc:
        pipeline.run(input_path, **kwargs)
    assert exc.value is error


@pytest.mark.parametrize("chunked", [False, True])
def test_bare_exception_is_not_mapped_when_separation_is_skipped(tmp_path, monkeypatch, chunked):
    # 分離しない指定では外部推論を呼ばないので、この呼び出しの失敗を工程失敗として報告しない。
    error = RuntimeError("一時ファイルを書けません")
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, failing_stage="separate", error=error, chunked=chunked)
    kwargs["separate_vocals"] = "never"

    with pytest.raises(RuntimeError) as exc:
        pipeline.run(input_path, **kwargs)
    assert exc.value is error


@pytest.mark.parametrize("chunked", [False, True])
def test_bare_exception_outside_inference_is_not_mapped_to_stage(tmp_path, monkeypatch, chunked):
    # 写像するのは外部推論の呼び出しだけで、その外側(音量解析等)の失敗はそのまま通す。
    error = RuntimeError("rms failed")
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, failing_stage=None, error=error, chunked=chunked)
    monkeypatch.setattr(
        pipeline._va_rms, "compute_rms", lambda *a, **k: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError) as exc:
        pipeline.run(input_path, **kwargs)
    assert exc.value is error


@pytest.mark.parametrize("chunked", [False, True])
def test_bare_exception_right_after_inference_is_not_mapped_to_stage(
        tmp_path, monkeypatch, chunked):
    # 推論呼び出しの直後に続く処理(内部生成ファイルの読み直し)まで写像範囲へ巻き込まないことを固定する。
    # 長尺分割では推論と読み直しが同じ繰り返しの中にあるため、範囲が広いと工程失敗へ化ける。
    error = RuntimeError("read failed")
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, failing_stage=None, error=error, chunked=chunked)
    monkeypatch.setattr(
        pipeline, "_read_intermediate", lambda *a, **k: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError) as exc:
        pipeline.run(input_path, **kwargs)
    assert exc.value is error


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failing_stage", ["separate", "recognize"])
def test_progress_emit_failure_from_inference_passes_through_unchanged(
        tmp_path, monkeypatch, failing_stage, chunked):
    # 推論中の進捗送出の失敗は工程の失敗ではないので写像せず、想定外例外の経路へ落とす。
    error = ProgressEmitError("stdout is closed")
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, failing_stage=failing_stage, error=error, chunked=chunked)

    with pytest.raises(ProgressEmitError) as exc:
        pipeline.run(input_path, **kwargs)
    assert exc.value is error
