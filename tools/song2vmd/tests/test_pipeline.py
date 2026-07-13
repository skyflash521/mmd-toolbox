"""song2vmd パイプライン統合のテスト(song2vmd.md §4章・§6章)。

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

from vocal_analysis import ContentRecognizerModel, Segment
from vocal_analysis.types import AudioPcm

from song2vmd import pipeline, presets

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
        separate_vocals="auto", separator_name="audio-separator-htdemucs-ft",
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

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
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

    def spy_separate(pcm, mode):
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
    # ("progress:X")は、その段の実処理("X")より必ず前に来る(song2vmd.md 12.1「各段は開始時に
    # 最低1本のprogressを出す」の後退を防ぐ回帰テスト)。
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


def test_single_run_diagnostics_reflect_backends_style_and_separated(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    result = pipeline.run(input_path, **_common_kwargs(
        separator_name="sep-x", content_recognizer_model=ContentRecognizerModel(model_id="rec-y"),
        style_name="ballad", separate_vocals="always"))

    assert result.diagnostics.backends == {
        "separator": "sep-x", "recognizer": "rec-y", "forced_aligner": "wav2vec2-ctc-forcedalign"}
    assert result.diagnostics.style == "ballad"
    assert result.diagnostics.separated is True
    assert result.diagnostics.phonemes == 1
    assert result.diagnostics.duration_sec == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("mode,expected", [("auto", True), ("always", True), ("never", False)])
def test_separated_flag_matches_separate_vocals_mode(tmp_path, monkeypatch, mode, expected):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, m: vocal_path)
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

    def fake_recognize(path, content_recognizer_model, retry, forced_aligner, sofa_aligner):
        received["content_recognizer_model"] = content_recognizer_model
        received["forced_aligner"] = forced_aligner
        received["sofa_aligner"] = sofa_aligner
        return [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
    monkeypatch.setattr(pipeline._va_recognizer, "recognize", fake_recognize)

    given_model = ContentRecognizerModel(model_id="org/custom-recognizer", model_revision="rev1")
    given_sofa_config = object()
    pipeline.run(input_path, **_common_kwargs(
        content_recognizer_model=given_model, forced_aligner="sofa-forcedalign",
        sofa_aligner=given_sofa_config))
    assert received["content_recognizer_model"] is given_model
    assert received["forced_aligner"] == "sofa-forcedalign"
    assert received["sofa_aligner"] is given_sofa_config


def test_generation_params_are_built_from_openness_and_style_gen(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
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
    # 織り込み済み(song2vmd.md 8.2)。pipeline は style_gen の値をそのまま lipsync へ渡す。
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


def test_chunked_run_calls_separate_and_recognize_once_per_chunk(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [3.0, 6.0])

    def fake_merge(chunk_segments_list, chunk_offsets_sec, boundaries_sec):
        assert len(chunk_segments_list) == 3
        # 境界[3.0, 6.0]・オーバーラップ1.0秒・全長10.0秒のとき、各チャンクの範囲は
        # [0,4]・[2,7]・[5,10](先頭・末尾は片側のみオーバーラップ。song2vmd.md 6.6)なので、
        # チャンクローカル時刻0に対応するグローバル時刻(オフセット)は [0.0, 2.0, 5.0] になる。
        assert chunk_offsets_sec == pytest.approx([0.0, 2.0, 5.0])
        assert boundaries_sec == [3.0, 6.0]
        return [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)]

    monkeypatch.setattr(pipeline.chunking, "merge_chunk_segments", fake_merge)

    separate_calls = []

    def fake_separate(pcm, mode):
        separate_calls.append(len(pcm.samples) / pcm.sample_rate)
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
        max_duration_sec=3.0, forced_aligner="sofa-forcedalign", sofa_aligner=sofa_config))

    assert len(separate_calls) == 3
    assert len(recognize_calls) == 3
    # forced_aligner・sofa_alignerは長尺分割の全チャンクへ同一の値で伝播する(song2vmd.md 6.6)。
    for _, kwargs in recognize_calls:
        assert kwargs["forced_aligner"] == "sofa-forcedalign"
        assert kwargs["sofa_aligner"] is sofa_config
    # 各チャンクは前後1.0秒のオーバーラップを持つ(先頭・末尾は片側のみ。song2vmd.md 6.6)。
    assert separate_calls[0] == pytest.approx(4.0, abs=0.05)  # [0, 3+1]
    assert separate_calls[1] == pytest.approx(5.0, abs=0.05)  # [3-1, 6+1]
    assert separate_calls[2] == pytest.approx(5.0, abs=0.05)  # [6-1, 10]
    assert result.diagnostics.duration_sec == pytest.approx(10.0, abs=0.05)
    # doneは「このチャンクを始める時点までに完了したチャンク数」(0始まり。song2vmd.md 12.1)。
    separate_done_totals = [(c["done"], c["total"]) for c in progress.calls if c["stage"] == "separate"]
    assert separate_done_totals == [(0, 3), (1, 3), (2, 3)]


def test_chunked_run_uses_raw_audio_rms_for_boundaries_and_whole_vocal_rms_for_events(tmp_path, monkeypatch):
    # 境界決定(find_chunk_boundaries)には分離前の生音声のRMSを使い、口形イベント確定
    # (events.confirm_mouth_events)にはチャンクの核区間を連結した曲全体のボーカルRMSを
    # 1回だけ渡す(チャンクごとに個別正規化しない。song2vmd.md 6.4・6.6)ことを検証する。
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
        return [3.0, 6.0]

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", fake_find_boundaries)
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)])
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
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
    # RMS。song2vmd.md 6.4・6.6)が壊れる。核区間を連結した曲全体のボーカル音声が、チャンクごとの
    # 元の振幅差を保持していることを検証する。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=6.0)
    quiet_vocal = tmp_path / "vocal_quiet.wav"
    write_wav(quiet_vocal, seconds=6.0, amplitude=0.1)
    loud_vocal = tmp_path / "vocal_loud.wav"
    write_wav(loud_vocal, seconds=6.0, amplitude=0.9)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [3.0])
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 6.0, phoneme="a", confidence=0.9)])

    separate_call_count = [0]

    def fake_separate(pcm, mode):
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


def test_chunked_run_reports_recognize_progress_with_chunk_totals(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [5.0])
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a", confidence=0.9)])
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
    monkeypatch.setattr(
        pipeline._va_recognizer, "recognize",
        lambda path, **kwargs: [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)])

    progress = _RecordingProgress()
    pipeline.run(input_path, progress=progress, **_common_kwargs(max_duration_sec=3.0))

    # doneは「このチャンクを始める時点までに完了したチャンク数」(0始まり。song2vmd.md 12.1)。
    recognize_done_totals = [(c["done"], c["total"]) for c in progress.calls if c["stage"] == "recognize"]
    assert recognize_done_totals == [(0, 2), (1, 2)]


def test_non_chunked_progress_reports_done_zero_total_none_for_separate_and_recognize(tmp_path, monkeypatch):
    # 分割しない場合や内訳の無い段は done=0, total=None(song2vmd.md 12.1)。
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
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


# --- 進捗レポータ省略時 --------------------------------------------------------


def test_run_works_without_progress_reporter(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
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

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
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

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
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

    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
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

    monkeypatch.setattr(pipeline.chunking, "find_chunk_boundaries", lambda *a, **k: [3.0])
    monkeypatch.setattr(
        pipeline.chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 6.0, phoneme="a", confidence=0.9)])
    monkeypatch.setattr(pipeline._va_separator, "separate", lambda pcm, mode: vocal_path)
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
