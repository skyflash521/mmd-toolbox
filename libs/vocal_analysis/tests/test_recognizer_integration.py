from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def test_downmix_to_mono_averages_channels():
    from vocal_analysis.recognizer import _downmix_to_mono

    samples = np.array([[0.2, 0.6], [-0.4, 0.0]], dtype=np.float32)

    mono = _downmix_to_mono(samples)

    assert np.allclose(mono, [0.4, -0.2])


def test_downmix_to_mono_passthrough_for_already_mono():
    from vocal_analysis.recognizer import _downmix_to_mono

    samples = np.array([[0.3], [-0.1]], dtype=np.float32)

    mono = _downmix_to_mono(samples)

    assert np.allclose(mono, [0.3, -0.1])


def test_resample_to_target_same_rate_is_passthrough():
    from vocal_analysis.recognizer import _resample_to_target

    mono = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)

    result = _resample_to_target(mono, sample_rate=16000, target_sample_rate=16000)

    assert np.array_equal(result, mono)


def test_resample_to_target_downsamples_to_expected_length():
    from vocal_analysis.recognizer import _resample_to_target

    mono = np.linspace(-0.5, 0.5, 48000, dtype=np.float32)

    result = _resample_to_target(mono, sample_rate=48000, target_sample_rate=16000)

    assert len(result) == 16000


class _FakeTokenizer:
    def __init__(self, decoder, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self._decoder = decoder

    def get_vocab(self):
        return {v: k for k, v in self._decoder.items()}


class _FakeProcessor:
    def __init__(self, decoder, pad_token_id=0):
        self.tokenizer = _FakeTokenizer(decoder, pad_token_id)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    sf.write(path, samples, sample_rate)
    return path


def _loud_samples(num_samples: int, sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(num_samples) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32).reshape(-1, 1)


_SIX_FRAME_PAU_A_PAU_LOG_PROBS = np.array(
    [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
)


def test_recognize_builds_segments_from_mocked_pipeline(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    samples_for_six_frames = 1920
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(samples_for_six_frames), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 3
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.02)
    assert segments[1].type == "vowel"
    assert segments[1].phoneme == "a"
    assert segments[1].start_sec == pytest.approx(0.02)
    assert segments[1].end_sec == pytest.approx(0.10)
    assert segments[2].type == "gap"
    assert segments[2].phoneme is None
    assert segments[2].start_sec == pytest.approx(0.10)
    assert segments[2].end_sec == pytest.approx(0.12)


def test_recognize_releases_content_pipeline_before_phoneme_model_load(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)
    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS

    calls = []
    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_release_content_recognizer_pipeline", lambda: calls.append("release"))

    def fake_load_model(on_progress=None):
        calls.append("load_phoneme")
        return _FakeProcessor(decoder), object()

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fake_load_model)
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    assert calls == ["release", "load_phoneme"]
    assert any(seg.type == "vowel" for seg in segments)


def test_recognize_releases_content_pipeline_without_loading_phoneme_model_when_no_alignment_targets(
        tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    calls = []
    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("", None))
    monkeypatch.setattr(
        recognizer_module, "_release_content_recognizer_pipeline", lambda: calls.append("release"))

    def fake_load_model(on_progress=None):
        calls.append("load_phoneme")
        raise AssertionError("アライメント対象0件では音素モデルをロードしない")

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fake_load_model)

    segments = recognizer_module.recognize(wav_path)

    assert calls == ["release"]
    assert all(seg.type == "gap" for seg in segments)


def test_recognize_skips_content_recognition_for_silent_segment(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    loud = _loud_samples(32000)
    silence = np.zeros((48000, 1), dtype=np.float32)
    wav_path = _write_wav(tmp_path / "vocal.wav", np.concatenate([loud, silence], axis=0), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS
    call_count = {"transcribe": 0}

    def fake_transcribe(pipeline, samples):
        call_count["transcribe"] += 1
        return "あ", None

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    assert call_count["transcribe"] == 1
    assert segments[-1].type == "gap"
    assert segments[-1].phoneme is None
    assert segments[-1].end_sec == pytest.approx(5.0)


def test_recognize_loads_content_recognizer_pipeline_once_for_multiple_segments(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    loud1 = _loud_samples(32000)
    silence = np.zeros((16000, 1), dtype=np.float32)
    loud2 = _loud_samples(32000)
    wav_path = _write_wav(
        tmp_path / "vocal.wav", np.concatenate([loud1, silence, loud2], axis=0), 16000
    )

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS
    build_calls = {"count": 0}

    def fake_transformers_pipeline(*args, **kwargs):
        build_calls["count"] += 1
        return object()

    def fake_transcribe(pipeline, samples):
        return "あ", None

    monkeypatch.setattr(recognizer_module, "_content_recognizer_pipeline_cache", None)
    monkeypatch.setattr(recognizer_module, "_transformers_pipeline", fake_transformers_pipeline)
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    recognizer_module.recognize(wav_path)

    assert build_calls["count"] == 1


def test_recognize_forwards_on_progress_to_model_loaders(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)
    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS
    received = {}

    def fake_load_pipeline(content_recognizer_model, on_progress=None):
        received["pipeline_on_progress"] = on_progress
        return object()

    def fake_load_model_and_processor(on_progress=None):
        received["model_on_progress"] = on_progress
        return _FakeProcessor(decoder), object()

    monkeypatch.setattr(recognizer_module, "_load_content_recognizer_pipeline", fake_load_pipeline)
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fake_load_model_and_processor)
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    def sentinel_on_progress(note):
        pass

    recognizer_module.recognize(wav_path, on_progress=sentinel_on_progress)

    assert received["pipeline_on_progress"] is sentinel_on_progress
    assert received["model_on_progress"] is sentinel_on_progress


def test_recognize_reannounces_segment_note_after_loading_note(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)
    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS
    notes = []

    def fake_load_pipeline(content_recognizer_model, on_progress=None):
        if on_progress is not None:
            on_progress("内容認識モデル読み込み中: dummy/model")
        return object()

    monkeypatch.setattr(recognizer_module, "_load_content_recognizer_pipeline", fake_load_pipeline)
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object()),
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    recognizer_module.recognize(wav_path, on_progress=notes.append)

    segment_notes = [n for n in notes if n.startswith("歌詞書き起こし中")]
    load_index = notes.index("内容認識モデル読み込み中: dummy/model")
    assert len(segment_notes) >= 2
    assert notes[load_index + 1] == segment_notes[0]


def test_recognize_reports_transcription_and_alignment_notes_with_time_range_and_counts(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)
    decoder = {0: "<pad>", 1: "a"}
    notes = []

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object()),
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs",
        lambda processor, model, samples: _SIX_FRAME_PAU_A_PAU_LOG_PROBS,
    )

    recognizer_module.recognize(wav_path, on_progress=notes.append)

    assert "歌詞書き起こし中: 0:00-0:00(1/1)" in notes
    assert "音素アライメント中: 0:00-0:00(1/1)" in notes


def test_format_time_range_uses_minutes_and_zero_padded_seconds():
    from vocal_analysis.recognizer import _format_time_range

    assert _format_time_range(65.4, 125.0) == "1:05-2:05"


def test_recognize_without_on_progress_still_works(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)
    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object()),
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path, on_progress=None)

    assert len(segments) == 3


def test_recognize_trims_leading_silence_and_offsets_segments(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    silence = np.zeros((24000, 1), dtype=np.float32)
    loud = _loud_samples(32000)
    wav_path = _write_wav(tmp_path / "vocal.wav", np.concatenate([silence, loud], axis=0), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    voiced_start_sec = 1.5
    trim_margin_sec = 0.1
    trimmed_start_sec = voiced_start_sec - trim_margin_sec
    assert len(segments) == 3
    assert segments[0].type == "gap"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(trimmed_start_sec + 0.02)
    assert segments[1].type == "vowel"
    assert segments[1].phoneme == "a"
    assert segments[1].start_sec == pytest.approx(trimmed_start_sec + 0.02)
    assert segments[1].end_sec == pytest.approx(trimmed_start_sec + 0.10)
    assert segments[2].type == "gap"
    assert segments[2].start_sec == pytest.approx(trimmed_start_sec + 0.10)
    assert segments[2].end_sec == pytest.approx(3.5)


def test_recognize_treats_high_phoneme_density_chunk_as_gap_without_loading_phoneme_model(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment",
        lambda pipeline, samples: ("あいうえおかきくけこさしすせそ", None))
    monkeypatch.setattr(
        recognizer_module, "_transcribe_text_only",
        lambda samples, content_recognizer_model, on_progress=None: "あいうえおかきくけこさしすせそ")
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: ["a"] * 100)

    def fail_if_called():
        raise AssertionError("音素密度が高い区間で音素モデルをロードしてはならない")

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fail_if_called)

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(2.0)


def test_recognize_empty_transcription_confirms_gap_without_g2p_or_model(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment",
        lambda pipeline, samples: ("  ", None))

    def fail_g2p(text, method=None, **kwargs):
        raise AssertionError("空文字列の区間でG2Pを呼んではならない")

    def fail_load_model():
        raise AssertionError("空文字列の区間で音素モデルをロードしてはならない")

    monkeypatch.setattr(recognizer_module, "_g2p", fail_g2p)
    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fail_load_model)

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(2.0)


def test_recognize_computes_and_passes_word_windows_to_windowed_alignment(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS
    captured = {}

    def fake_transcribe(pipeline, samples):
        return "あ", [("あ", 0.02, 0.06)]

    def fake_forced_align_windowed(log_probs_arg, token_ids, windows_sec):
        captured["windows_sec"] = windows_sec
        return recognizer_module._forced_align(log_probs_arg, token_ids)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )
    monkeypatch.setattr(recognizer_module, "_forced_align_windowed", fake_forced_align_windowed)

    segments = recognizer_module.recognize(wav_path)

    assert "windows_sec" in captured
    margin = recognizer_module._WORD_WINDOW_MARGIN_SEC
    windows_sec = captured["windows_sec"]
    leading_pau, *word_substates, trailing_pau = windows_sec
    assert len(word_substates) == 2
    assert leading_pau == pytest.approx((0.0, 0.02 + margin))
    assert all(w == pytest.approx((0.02 - margin, 0.06 + margin)) for w in word_substates)
    assert trailing_pau == pytest.approx((0.06 - margin, 0.12))
    assert segments[1].phoneme == "a"


def test_recognize_falls_back_to_band_alignment_when_word_window_infeasible(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS

    def fake_transcribe(pipeline, samples):
        return "あ", [("あ", 0.02, 0.06)]

    def fail_windowed(log_probs_arg, token_ids, windows_sec):
        raise recognizer_module.RecognitionError("単語窓制約下で強制アライメントが末尾トークンへ到達できませんでした")

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )
    monkeypatch.setattr(recognizer_module, "_forced_align_windowed", fail_windowed)

    segments = recognizer_module.recognize(wav_path)

    assert segments[1].type == "vowel"
    assert segments[1].phoneme == "a"


def test_recognize_falls_back_to_global_min_stay_when_windowed_alignment_fails(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS
    calls = {"local": 0, "global": 0}

    def fake_transcribe(pipeline, samples):
        return "あ", [("あ", 0.02, 0.06)]

    def fail_windowed(log_probs_arg, token_ids, windows_sec):
        raise recognizer_module.RecognitionError("単語窓制約下で強制アライメントが末尾トークンへ到達できませんでした")

    original_local = recognizer_module._expand_min_stay_local
    original_global = recognizer_module._expand_min_stay

    def spy_local(*args, **kwargs):
        calls["local"] += 1
        return original_local(*args, **kwargs)

    def spy_global(*args, **kwargs):
        calls["global"] += 1
        return original_global(*args, **kwargs)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )
    monkeypatch.setattr(recognizer_module, "_forced_align_windowed", fail_windowed)
    monkeypatch.setattr(recognizer_module, "_expand_min_stay_local", spy_local)
    monkeypatch.setattr(recognizer_module, "_expand_min_stay", spy_global)

    segments = recognizer_module.recognize(wav_path)

    assert calls["local"] == 1
    assert calls["global"] == 1
    assert segments[1].type == "vowel"
    assert segments[1].phoneme == "a"


def test_recognize_places_multiple_words_in_order_via_per_word_g2p(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    samples_for_seven_frames = 2240
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(samples_for_seven_frames), 16000)

    decoder = {0: "<pad>", 1: "a", 2: "i"}
    pau = [5.0, -5.0, -5.0]
    a = [-5.0, 5.0, -5.0]
    i = [-5.0, -5.0, 5.0]
    log_probs = np.array([pau, a, a, pau, i, i, pau])

    def fake_transcribe(pipeline, samples):
        return "あ い", [("あ", 0.02, 0.04), ("い", 0.06, 0.08)]

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(
        recognizer_module, "_g2p",
        lambda text, method=None, **kwargs: {"あ": ["a"], "い": ["i"], "あ い": ["a", "i"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    vowels = [(s.phoneme, s.start_sec, s.end_sec) for s in segments if s.type == "vowel"]
    assert [p for p, _, _ in vowels] == ["a", "i"]
    assert vowels[0][2] <= vowels[1][1]


def test_recognize_hallucination_density_sums_across_words(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(16000), 16000)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment",
        lambda pipeline, samples: ("あ い", [("あ", 0.1, 0.4), ("い", 0.5, 0.9)]))
    phonemes_per_word_below_threshold_alone = 15
    monkeypatch.setattr(
        recognizer_module, "_g2p",
        lambda text, method=None, **kwargs: ["a"] * phonemes_per_word_below_threshold_alone)

    def fail_if_called():
        raise AssertionError("音素密度が高い区間で音素モデルをロードしてはならない")

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fail_if_called)

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "gap"


def test_recognize_empty_word_list_with_nonempty_text_falls_back_like_no_timestamps(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS

    def fail_windowed(*args, **kwargs):
        raise AssertionError("単語タイムスタンプ非取得扱いでは単語窓制約経路を呼んではならない")

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", []))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )
    monkeypatch.setattr(recognizer_module, "_forced_align_windowed", fail_windowed)

    segments = recognizer_module.recognize(wav_path)

    assert segments[1].phoneme == "a"


def test_recognize_fully_silent_input_never_loads_phoneme_model(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", np.zeros((32000, 1), dtype=np.float32), 16000)

    def fail_if_called():
        raise AssertionError("無音のみの入力で音素モデルをロードしてはならない")

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fail_if_called)

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(2.0)


def test_recognize_last_local_segment_extends_exactly_to_segment_boundary_despite_fewer_model_frames(
        tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    assert segments[-1].type == "gap"
    assert segments[-1].end_sec == pytest.approx(2.0)


def test_recognize_content_recognizer_missing_library_raises_error_naming_transformers_and_torch(
        tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    original_error = ImportError("no transformers")
    decoder = {0: "<pad>"}
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )

    def fake_transcribe(pipeline, samples):
        raise original_error

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)

    with pytest.raises(recognizer_module.RecognitionError, match="transformers") as excinfo:
        recognizer_module.recognize(wav_path)

    assert "torch" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_error


def test_recognize_content_recognizer_model_fetch_failure_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    original_error = OSError("model not found in cache and offline")
    decoder = {0: "<pad>"}
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )

    def fake_transcribe(pipeline, samples):
        raise original_error

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)

    with pytest.raises(recognizer_module.RecognitionError, match="内容認識モデル") as excinfo:
        recognizer_module.recognize(wav_path)

    assert excinfo.value.__cause__ is original_error


def test_recognize_g2p_missing_library_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    original_error = ImportError("no pyopenjtalk")
    decoder = {0: "<pad>"}
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))

    def fake_g2p(text, method=None, **kwargs):
        raise original_error

    monkeypatch.setattr(recognizer_module, "_g2p", fake_g2p)

    with pytest.raises(recognizer_module.RecognitionError, match="pyopenjtalk-plus") as excinfo:
        recognizer_module.recognize(wav_path)

    assert excinfo.value.__cause__ is original_error


def test_recognize_phoneme_model_missing_library_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    original_error = ImportError("no transformers")
    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: ["a"])

    def fake_load(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fake_load)

    with pytest.raises(recognizer_module.RecognitionError, match="transformers") as excinfo:
        recognizer_module.recognize(wav_path)

    assert "torch" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_error


def test_recognize_phoneme_model_fetch_failure_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    original_error = OSError("model not found in cache and offline")
    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: ["a"])

    def fake_load(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fake_load)

    with pytest.raises(recognizer_module.RecognitionError, match="認識モデル") as excinfo:
        recognizer_module.recognize(wav_path)

    assert excinfo.value.__cause__ is original_error


def test_load_model_and_processor_passes_pinned_config_and_disables_phonemizer(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from vocal_analysis import RECOGNIZER_CONFIG
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.recognizer import _load_model_and_processor

    captured = {}

    class _FakeModel:
        def to(self, device):
            captured["model_to_device"] = device
            return self

        def eval(self):
            return self

    def fake_processor_from_pretrained(model_id, revision=None, do_phonemize=None, **kwargs):
        captured["processor_model_id"] = model_id
        captured["processor_revision"] = revision
        captured["do_phonemize"] = do_phonemize
        return _FakeProcessor({0: "<pad>"})

    def fake_model_from_pretrained(model_id, revision=None, dtype=None, **kwargs):
        captured["model_model_id"] = model_id
        captured["model_revision"] = revision
        captured["model_dtype"] = dtype
        return _FakeModel()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_processor_from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCTC, "from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr(recognizer_module, "_select_device", lambda: "fake-device")

    _load_model_and_processor()

    assert captured["do_phonemize"] is False
    assert captured["processor_model_id"] == RECOGNIZER_CONFIG.model_id
    assert captured["processor_revision"] == RECOGNIZER_CONFIG.model_revision
    assert captured["model_model_id"] == RECOGNIZER_CONFIG.model_id
    assert captured["model_revision"] == RECOGNIZER_CONFIG.model_revision
    assert captured["model_dtype"] == getattr(torch, RECOGNIZER_CONFIG.dtype)
    assert captured["model_to_device"] == "fake-device"


def test_load_content_recognizer_pipeline_uses_cpu_and_float32_when_gpu_unavailable(monkeypatch):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.recognizer import _load_content_recognizer_pipeline

    captured = {}

    def fake_pipeline(task, model=None, revision=None, device=None, dtype=None, **kwargs):
        captured["task"] = task
        captured["model"] = model
        captured["revision"] = revision
        captured["device"] = device
        captured["dtype"] = dtype
        return object()

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(recognizer_module, "_content_recognizer_pipeline_cache", None)

    _load_content_recognizer_pipeline(DEFAULT_CONTENT_RECOGNIZER_MODEL)

    assert captured["task"] == "automatic-speech-recognition"
    assert captured["model"] == DEFAULT_CONTENT_RECOGNIZER_MODEL.model_id
    assert captured["revision"] == DEFAULT_CONTENT_RECOGNIZER_MODEL.model_revision
    assert captured["device"] == "cpu"
    assert captured["dtype"] == torch.float32


def test_load_content_recognizer_pipeline_passes_through_omitted_revision(monkeypatch):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.recognizer import _load_content_recognizer_pipeline

    custom_model = ContentRecognizerModel(model_id="openai/whisper-large-v3")
    captured = {}

    def fake_pipeline(task, model=None, revision=None, device=None, dtype=None, **kwargs):
        captured["revision"] = revision
        return object()

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(recognizer_module, "_content_recognizer_pipeline_cache", None)

    _load_content_recognizer_pipeline(custom_model)

    assert captured["revision"] is None


def test_load_content_recognizer_pipeline_uses_gpu_and_float16_when_available(monkeypatch):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.recognizer import _load_content_recognizer_pipeline

    captured = {}

    def fake_pipeline(task, model=None, revision=None, device=None, dtype=None, **kwargs):
        captured["device"] = device
        captured["dtype"] = dtype
        return object()

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(recognizer_module, "_content_recognizer_pipeline_cache", None)

    _load_content_recognizer_pipeline(DEFAULT_CONTENT_RECOGNIZER_MODEL)

    assert captured["device"] == "cuda"
    assert captured["dtype"] == torch.float16


def test_load_content_recognizer_pipeline_evicts_previous_model_before_loading_next(monkeypatch):
    transformers = pytest.importorskip("transformers")
    import gc
    import weakref

    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.recognizer import _load_content_recognizer_pipeline

    model_a = ContentRecognizerModel(model_id="org/model-a")
    model_b = ContentRecognizerModel(model_id="org/model-b")
    observed = {}

    # object() のインスタンスには弱参照を張れない。
    class _Pipeline:
        pass

    def fake_pipeline(task, model=None, revision=None, device=None, **kwargs):
        if model == model_b.model_id:
            gc.collect()
            observed["model_a_pipeline_alive"] = observed["model_a_pipeline_ref"]() is not None
        return _Pipeline()

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr(recognizer_module, "_select_device", lambda: "cpu")
    monkeypatch.setattr(recognizer_module, "_content_recognizer_pipeline_cache", None)

    model_a_pipeline = _load_content_recognizer_pipeline(model_a)
    observed["model_a_pipeline_ref"] = weakref.ref(model_a_pipeline)
    del model_a_pipeline

    _load_content_recognizer_pipeline(model_b)

    assert observed["model_a_pipeline_alive"] is False


def _fake_whisper_pipeline_class(result, captured=None):
    class _FakePromptIds:
        def to(self, device):
            return "PROMPT_IDS"

    class _FakePipeline:
        device = "cpu"

        class tokenizer:
            @staticmethod
            def get_prompt_ids(prompt, return_tensors):
                return _FakePromptIds()

        def __call__(self, samples, return_timestamps, generate_kwargs):
            if captured is not None:
                captured["generate_kwargs"] = generate_kwargs
            return result

    return _FakePipeline


def test_transcribe_segment_extracts_word_timestamps_from_chunks(monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    pipeline_class = _fake_whisper_pipeline_class({
        "text": "あ い",
        "chunks": [
            {"text": "あ", "timestamp": (0.0, 0.5)},
            {"text": "い", "timestamp": (0.5, 1.0)},
        ],
    })

    samples = np.zeros(16000, dtype=np.float32)
    text, words = recognizer_module._transcribe_segment(pipeline_class(), samples)

    assert text == "あ い"
    assert words == [("あ", 0.0, 0.5), ("い", 0.5, 1.0)]


def test_transcribe_segment_returns_none_words_when_pipeline_has_no_chunks(monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    pipeline_class = _fake_whisper_pipeline_class({"text": "あ"})

    text, words = recognizer_module._transcribe_segment(
        pipeline_class(), np.zeros(16000, dtype=np.float32)
    )

    assert text == "あ"
    assert words is None


def test_transcribe_segment_bounds_generation_length(monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    captured = {}
    pipeline_class = _fake_whisper_pipeline_class({"text": "あ"}, captured)

    recognizer_module._transcribe_segment(pipeline_class(), np.zeros(16000, dtype=np.float32))

    assert captured["generate_kwargs"]["max_new_tokens"] == 380


def test_transcribe_text_only_bounds_generation_length(monkeypatch):
    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module

    captured = {}

    class _FakePipeline:
        def __call__(self, samples, generate_kwargs):
            captured["generate_kwargs"] = generate_kwargs
            return {"text": "あ"}

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda model, on_progress=None: _FakePipeline()
    )

    recognizer_module._transcribe_text_only(
        np.zeros(16000, dtype=np.float32), ContentRecognizerModel(model_id="org/model")
    )

    assert captured["generate_kwargs"]["max_new_tokens"] == 420


def test_transcribe_text_only_forwards_on_progress_to_pipeline_loader(monkeypatch):
    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module

    captured = {}

    class _FakePipeline:
        def __call__(self, samples, generate_kwargs):
            return {"text": "あ"}

    def fake_loader(model, on_progress=None):
        captured["on_progress"] = on_progress
        return _FakePipeline()

    monkeypatch.setattr(recognizer_module, "_load_content_recognizer_pipeline", fake_loader)

    sentinel = lambda note: None  # noqa: E731
    recognizer_module._transcribe_text_only(
        np.zeros(16000, dtype=np.float32), ContentRecognizerModel(model_id="org/model"),
        on_progress=sentinel,
    )

    assert captured["on_progress"] is sentinel


def test_recognize_default_content_recognizer_model_loads_pinned_pipeline_with_kana_prompt(tmp_path, monkeypatch):
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL, KANA_PROMPT
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS
    calls = {}

    class _FakePromptIds:
        def to(self, device):
            calls["prompt_ids_device"] = device
            return "PROMPT_IDS"

    class _FakePipeline:
        device = "cpu"

        class tokenizer:
            @staticmethod
            def get_prompt_ids(prompt, return_tensors):
                calls["prompt_text"] = prompt
                return _FakePromptIds()

        def __call__(self, samples, return_timestamps, generate_kwargs):
            calls["generate_kwargs"] = generate_kwargs
            calls["return_timestamps"] = return_timestamps
            return {"text": "あ"}

    def fake_load_pipeline(content_recognizer_model, on_progress=None):
        calls["loaded_model"] = content_recognizer_model
        return _FakePipeline()

    monkeypatch.setattr(recognizer_module, "_load_content_recognizer_pipeline", fake_load_pipeline)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    assert calls["loaded_model"] == DEFAULT_CONTENT_RECOGNIZER_MODEL
    assert calls["prompt_text"] == KANA_PROMPT
    assert calls["generate_kwargs"]["prompt_ids"] == "PROMPT_IDS"
    assert calls["return_timestamps"] == "word"
    assert segments[1].phoneme == "a"


def test_recognize_custom_content_recognizer_model_is_passed_through(tmp_path, monkeypatch):
    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module

    custom_model = ContentRecognizerModel(model_id="sbintuitions/kana-whisper")

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS
    calls = {}

    pipeline_class = _fake_whisper_pipeline_class({"text": "あ"})

    def fake_load_pipeline(content_recognizer_model, on_progress=None):
        calls["loaded_model"] = content_recognizer_model
        return pipeline_class()

    monkeypatch.setattr(recognizer_module, "_load_content_recognizer_pipeline", fake_load_pipeline)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    recognizer_module.recognize(wav_path, content_recognizer_model=custom_model)

    assert calls["loaded_model"] == custom_model


def test_recognize_english_katakana_method_is_passed_through_to_every_g2p_call(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = _SIX_FRAME_PAU_A_PAU_LOG_PROBS
    g2p_calls = []

    def fake_g2p(text, method=None, **kwargs):
        g2p_calls.append(method)
        return {"あ": ["a"]}[text]

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", fake_g2p)
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    recognizer_module.recognize(
        wav_path, english_katakana_method="tinyllama-katakana-converter")

    assert len(g2p_calls) >= 1
    assert all(m == "tinyllama-katakana-converter" for m in g2p_calls)


def test_recognize_english_katakana_method_reaches_per_word_g2p(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(2240), 16000)

    decoder = {0: "<pad>", 1: "a", 2: "i"}
    pau = [5.0, -5.0, -5.0]
    a = [-5.0, 5.0, -5.0]
    i = [-5.0, -5.0, 5.0]
    log_probs = np.array([pau, a, a, pau, i, i, pau])
    g2p_calls = []

    def fake_g2p(text, method=None, **kwargs):
        g2p_calls.append((text, method))
        return {"あ": ["a"], "い": ["i"], "あ い": ["a", "i"]}[text]

    def fake_transcribe(pipeline, samples):
        return "あ い", [("あ", 0.02, 0.04), ("い", 0.06, 0.08)]

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", fake_g2p)
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor",
        lambda on_progress=None: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    recognizer_module.recognize(
        wav_path, english_katakana_method="tinyllama-katakana-converter")

    per_word_calls = [c for c in g2p_calls if c[0] in ("あ", "い")]
    assert len(per_word_calls) == 2
    assert all(method == "tinyllama-katakana-converter" for _, method in per_word_calls)


def test_recognize_content_recognizer_model_fetch_failure_names_that_model(tmp_path, monkeypatch):
    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)
    custom_model = ContentRecognizerModel(model_id="openai/whisper-large-v3", model_revision="deadbeef")

    original_error = OSError("model not found in cache and offline")

    def fail_load_pipeline(content_recognizer_model, on_progress=None):
        raise original_error

    monkeypatch.setattr(recognizer_module, "_load_content_recognizer_pipeline", fail_load_pipeline)

    with pytest.raises(recognizer_module.RecognitionError, match="openai/whisper-large-v3") as excinfo:
        recognizer_module.recognize(wav_path, content_recognizer_model=custom_model)

    assert "deadbeef" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_error


def _make_sofa_config(tmp_path):
    from vocal_analysis import SofaAlignerConfig

    return SofaAlignerConfig(
        sofa_python=tmp_path / "sofa-venv" / "python",
        sofa_root=tmp_path / "SOFA",
        checkpoint_path=tmp_path / "checkpoint.ckpt",
    )


def test_recognize_sofa_without_config_raises_recognition_error(tmp_path):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(16000), 16000)

    with pytest.raises(recognizer_module.RecognitionError):
        recognizer_module.recognize(wav_path, forced_aligner="sofa-forcedalign", sofa_aligner=None)


def test_recognize_unknown_forced_aligner_raises_recognition_error_before_reading_audio(tmp_path):
    from vocal_analysis import recognizer as recognizer_module

    with pytest.raises(recognizer_module.RecognitionError):
        recognizer_module.recognize(tmp_path / "does_not_exist.wav", forced_aligner="unknown-aligner")


def test_recognize_default_forced_aligner_does_not_call_sofa(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.config import DEFAULT_FORCED_ALIGNER

    assert DEFAULT_FORCED_ALIGNER == "wav2vec2-ctc-forcedalign"

    wav_path = _write_wav(tmp_path / "vocal.wav", np.zeros((16000, 1), dtype=np.float32), 16000)

    def fail_if_called(targets, config):
        raise AssertionError("既定のforced_alignerでSOFAを呼び出してはならない")

    monkeypatch.setattr(recognizer_module.sofa_align, "_align_batch", fail_if_called)

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "gap"


def test_recognize_sofa_path_splits_words_batches_once_and_reassembles_segments(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(6400), 16000)
    config = _make_sofa_config(tmp_path)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment",
        lambda pipeline, samples: ("かき", [("か", 0.1, 0.2), ("き", 0.25, 0.35)]),
    )
    monkeypatch.setattr(
        recognizer_module, "_g2p",
        lambda text, method=None, **kwargs: {"か": ["k", "a"], "き": ["k", "i"], "かき": ["k", "a", "k", "i"]}[text]
    )

    align_batch_calls = []

    def fake_align_batch(targets, sofa_aligner_config):
        align_batch_calls.append((targets, sofa_aligner_config))
        return {
            "segment_0000": [(0.0, 0.05, "k"), (0.05, 0.1, "a")],
            "segment_0001": [(0.0, 0.04, "k"), (0.04, 0.1, "i")],
        }

    monkeypatch.setattr(recognizer_module.sofa_align, "_align_batch", fake_align_batch)

    segments = recognizer_module.recognize(
        wav_path, forced_aligner="sofa-forcedalign", sofa_aligner=config
    )

    assert len(align_batch_calls) == 1
    targets, passed_config = align_batch_calls[0]
    assert passed_config is config
    assert len(targets) == 2
    assert targets[0][2] == ["k", "a"]
    assert targets[1][2] == ["k", "i"]

    expected = [
        ("gap", None, 0.0, 0.1),
        ("consonant", "k", 0.1, 0.15),
        ("vowel", "a", 0.15, 0.2),
        ("gap", None, 0.2, 0.25),
        ("consonant", "k", 0.25, 0.29),
        ("vowel", "i", 0.29, 0.35),
        ("gap", None, 0.35, 0.4),
    ]
    assert len(segments) == len(expected)
    for seg, (exp_type, exp_phoneme, exp_start, exp_end) in zip(segments, expected, strict=True):
        assert seg.type == exp_type
        assert seg.phoneme == exp_phoneme
        assert seg.start_sec == pytest.approx(exp_start)
        assert seg.end_sec == pytest.approx(exp_end)


def test_recognize_sofa_path_uses_whole_region_when_no_word_timestamps(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(3200), 16000)
    config = _make_sofa_config(tmp_path)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])

    align_batch_calls = []

    def fake_align_batch(targets, sofa_aligner_config):
        align_batch_calls.append(targets)
        return {"segment_0000": [(0.0, 0.2, "a")]}

    monkeypatch.setattr(recognizer_module.sofa_align, "_align_batch", fake_align_batch)

    segments = recognizer_module.recognize(
        wav_path, forced_aligner="sofa-forcedalign", sofa_aligner=config
    )

    assert len(align_batch_calls) == 1
    assert len(align_batch_calls[0]) == 1
    assert align_batch_calls[0][0][2] == ["a"]

    assert len(segments) == 1
    assert segments[0].type == "vowel"
    assert segments[0].phoneme == "a"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.2)


def test_recognize_sofa_path_never_loads_wav2vec2_model(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(3200), 16000)
    config = _make_sofa_config(tmp_path)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None, **kwargs: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module.sofa_align, "_align_batch",
        lambda targets, cfg: {"segment_0000": [(0.0, 0.2, "a")]},
    )

    def fail_if_called():
        raise AssertionError("SOFA経路でwav2vec2の音素モデルをロードしてはならない")

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fail_if_called)

    recognizer_module.recognize(wav_path, forced_aligner="sofa-forcedalign", sofa_aligner=config)


def test_recognize_sofa_path_silent_input_calls_align_batch_with_no_targets(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", np.zeros((16000, 1), dtype=np.float32), 16000)
    config = _make_sofa_config(tmp_path)

    calls = []

    def fake_align_batch(targets, cfg):
        calls.append(targets)
        return {}

    monkeypatch.setattr(recognizer_module.sofa_align, "_align_batch", fake_align_batch)

    segments = recognizer_module.recognize(wav_path, forced_aligner="sofa-forcedalign", sofa_aligner=config)

    assert len(calls) == 1
    assert calls[0] == []
    assert len(segments) == 1
    assert segments[0].type == "gap"


def test_recognize_widens_cramped_word_before_assigning_windows(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.recognizer import _WORD_WINDOW_MARGIN_SEC

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(6400), 16000)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model, on_progress=None: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment",
        lambda pipeline, samples: ("な", [("な", 0.30, 0.35)]))
    monkeypatch.setattr(recognizer_module, "_g2p",
                        lambda text, method=None, **kwargs: {"な": ["n", "a"]}[text])
    decoder = {0: "<pad>", 1: "n", 2: "a"}
    monkeypatch.setattr(recognizer_module, "_load_model_and_processor",
                        lambda on_progress=None: (_FakeProcessor(decoder), object()))
    captured = {}

    def fake_align(processor, model, vocab, blank_token_id, threshold, job):
        captured["words"] = job["words_phonemes"]
        captured["windows"] = job["windows_sec"]
        return []

    monkeypatch.setattr(recognizer_module, "_align_wav2vec2_job", fake_align)

    recognizer_module.recognize(wav_path)

    phonemes, start, end = captured["words"][0]
    widened_start = 0.35 - 2 * 0.04
    assert phonemes == ["n", "a"]
    assert start == pytest.approx(widened_start)
    assert end == pytest.approx(0.35)
    first_phoneme_window = captured["windows"][1]
    assert first_phoneme_window[0] == pytest.approx(widened_start - _WORD_WINDOW_MARGIN_SEC)
    assert first_phoneme_window[1] == pytest.approx(0.35 + _WORD_WINDOW_MARGIN_SEC)
