import gc
import importlib
import re
import weakref

import numpy as np
import pytest
import soundfile as sf

_SIX_FRAME_LOG_PROBS = np.array(
    [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
)


def _mod(name):
    return importlib.import_module(f"vocal_analysis.{name}")


def _loud(num_samples, sample_rate=16000):
    t = np.arange(num_samples) / sample_rate
    return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32).reshape(-1, 1)


def _write_loud_wav(directory, name="vocal.wav", num_samples=1920):
    path = directory / name
    sf.write(path, _loud(num_samples), 16000)
    return path


def _write_silent_wav(directory, name="silent.wav"):
    path = directory / name
    sf.write(path, np.zeros((16000, 1), dtype=np.float32), 16000)
    return path


class _Pipeline:
    pass


class _PhonemeModel:
    pass


class _FakeTokenizer:
    pad_token_id = 0

    def get_vocab(self):
        return {"<pad>": 0, "a": 1}


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class _Fakes:
    def __init__(self, monkeypatch):
        self.events = []
        self.pipeline_refs = []
        self.model_refs = []
        self.load_pipeline_progress = []
        self.load_phoneme_progress = []
        self.text = "あ"
        self.texts = []
        self.words = None
        self.g2p_calls = []
        self.g2p_phonemes = None
        self.retry_texts = []
        self.katakana_refs = []
        self.katakana_alive_at_phoneme_load = None
        monkeypatch.setattr(_mod("transcription"), "load_pipeline", self._load_pipeline)
        monkeypatch.setattr(_mod("transcription"), "transcribe_segment", self._transcribe_segment)
        monkeypatch.setattr(_mod("transcription"), "transcribe_text_only", self._transcribe_text_only)
        monkeypatch.setattr(_mod("transcription"), "g2p", self._g2p)
        monkeypatch.setattr(_mod("alignment"), "load_phoneme_model", self._load_phoneme_model)
        monkeypatch.setattr(_mod("alignment"), "compute_log_probs", self._compute_log_probs)
        monkeypatch.setattr(_mod("model_store"), "release_gpu_memory", lambda: self.events.append("release"))

    def _load_pipeline(self, content_recognizer_model, on_progress=None):
        self.events.append("load_pipeline")
        self.load_pipeline_progress.append(on_progress)
        pipeline = _Pipeline()
        self.pipeline_refs.append(weakref.ref(pipeline))
        return pipeline

    def _transcribe_segment(self, pipeline, samples):
        self.events.append("transcribe")
        text = self.texts.pop(0) if self.texts else self.text
        return text, self.words

    def _transcribe_text_only(self, pipeline, samples):
        self.events.append("retranscribe")
        assert "release" not in self.events
        return self.retry_texts.pop(0)

    def _g2p(self, text, method=None, **kwargs):
        self.g2p_calls.append((len(self.events), text, method))
        if self.g2p_phonemes is not None:
            return self.g2p_phonemes(text, method)
        return ["a"]

    def _load_phoneme_model(self, on_progress=None):
        self.events.append("load_phoneme")
        gc.collect()
        self.katakana_alive_at_phoneme_load = any(ref() is not None for ref in self.katakana_refs)
        self.load_phoneme_progress.append(on_progress)
        model = _PhonemeModel()
        self.model_refs.append(weakref.ref(model))
        return _FakeProcessor(), model

    def _compute_log_probs(self, processor, model, samples):
        self.events.append("align")
        return _SIX_FRAME_LOG_PROBS

    def count(self, name):
        return self.events.count(name)

    def g2p_positions(self, text):
        return [position for position, called_text, _ in self.g2p_calls if called_text == text]


@pytest.fixture
def fakes(monkeypatch):
    return _Fakes(monkeypatch)


_TINYLLAMA = "tinyllama-katakana-converter"


class _TokenizedPrompt(dict):
    def to(self, device):
        return self


class _KatakanaTokenizer:
    def __init__(self, events):
        self._events = events

    def __call__(self, prompt, return_tensors=None):
        import torch

        word = re.findall(r"英語: (\S+)\n単語音素", prompt)[-1]
        self._events.append(("generate", word))
        return _TokenizedPrompt(input_ids=torch.zeros((1, 3), dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return "スカイ"


class _KatakanaModel:
    device = "cpu"

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **inputs):
        import torch

        return torch.zeros((1, 5), dtype=torch.long)


@pytest.fixture
def katakana(fakes, monkeypatch):
    pytest.importorskip("torch")
    import transformers

    english_katakana = _mod("english_katakana")
    english_katakana._conversion_cache.clear()

    def from_pretrained_tokenizer(*args, **kwargs):
        return _KatakanaTokenizer(fakes.events)

    def from_pretrained_model(*args, **kwargs):
        fakes.events.append("load_katakana")
        model = _KatakanaModel()
        fakes.katakana_refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr(english_katakana, "_katakana_model_cache", None)
    monkeypatch.setattr(
        english_katakana, "_find_target_words", lambda text: re.findall(r"[A-Za-z]+", text))
    monkeypatch.setattr(english_katakana, "_lookup_cmudict_phonemes", lambda word: "HH AH0")
    monkeypatch.setattr(_mod("model_store"), "select_device", lambda: "cpu")
    monkeypatch.setattr(
        transformers, "AutoTokenizer",
        type("_Tokenizers", (), {"from_pretrained": staticmethod(from_pretrained_tokenizer)}))
    monkeypatch.setattr(
        transformers, "AutoModelForCausalLM",
        type("_Models", (), {"from_pretrained": staticmethod(from_pretrained_model)}))
    yield fakes
    english_katakana._conversion_cache.clear()


def _session(**kwargs):
    from vocal_analysis.recognizer import RecognizerSession

    return RecognizerSession(**kwargs)


def test_session_loads_each_model_once_across_chunks(tmp_path, fakes):
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(3)]

    with _session() as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        results = [session.align(t) for t in transcriptions]

    assert fakes.count("load_pipeline") == 1
    assert fakes.count("load_phoneme") == 1
    assert fakes.count("transcribe") == 3
    assert fakes.count("align") == 3
    for segments in results:
        assert [s.type for s in segments] == ["gap", "vowel", "gap"]


def test_session_finishes_all_transcriptions_before_first_alignment(tmp_path, fakes):
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(3)]

    with _session() as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        for t in transcriptions:
            session.align(t)

    assert fakes.events == [
        "load_pipeline", "transcribe", "transcribe", "transcribe",
        "release", "load_phoneme", "align", "align", "align",
        "release",
    ]


def test_session_transcribe_alone_neither_loads_phoneme_model_nor_releases_pipeline(tmp_path, fakes):
    wav = _write_loud_wav(tmp_path)

    with _session() as session:
        session.transcribe(wav)
        assert fakes.events == ["load_pipeline", "transcribe"]


def test_session_drops_content_pipeline_at_first_alignment(tmp_path, fakes):
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(2)]

    with _session() as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        gc.collect()
        assert fakes.pipeline_refs[0]() is not None
        session.align(transcriptions[0])
        gc.collect()
        assert fakes.pipeline_refs[0]() is None
        session.align(transcriptions[1])


def test_session_close_drops_phoneme_model_and_releases_gpu_memory(tmp_path, fakes):
    wav = _write_loud_wav(tmp_path)
    session = _session()
    session.align(session.transcribe(wav))
    gc.collect()
    assert fakes.model_refs[0]() is not None
    assert fakes.events[-1] == "align"

    session.close()

    gc.collect()
    assert fakes.model_refs[0]() is None
    assert fakes.events[-1] == "release"


def test_session_close_is_idempotent(tmp_path, fakes):
    wav = _write_loud_wav(tmp_path)
    session = _session()
    session.align(session.transcribe(wav))
    session.close()
    events_after_first_close = list(fakes.events)

    session.close()

    assert fakes.events == events_after_first_close


def test_session_context_exit_releases_resources_when_body_raises(tmp_path, fakes):
    wav = _write_loud_wav(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        with _session() as session:
            session.transcribe(wav)
            raise RuntimeError("boom")

    gc.collect()
    assert fakes.pipeline_refs[0]() is None
    assert fakes.events[-1] == "release"


def test_session_close_before_any_model_load_releases_nothing(fakes):
    session = _session()

    session.close()

    assert fakes.events == []


def test_session_rejects_use_after_close(tmp_path, fakes):
    wav = _write_loud_wav(tmp_path)
    session = _session()
    transcription = session.transcribe(wav)
    session.close()

    with pytest.raises(RuntimeError):
        session.transcribe(wav)
    with pytest.raises(RuntimeError):
        session.align(transcription)


def test_session_rejects_transcribe_after_alignment_started(tmp_path, fakes):
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(2)]

    with _session() as session:
        first = session.transcribe(wavs[0])
        session.align(first)

        with pytest.raises(RuntimeError):
            session.transcribe(wavs[1])


def test_session_without_alignment_targets_releases_pipeline_but_never_loads_phoneme_model(tmp_path, fakes):
    fakes.text = ""
    wav = _write_loud_wav(tmp_path)

    with _session() as session:
        segments = session.align(session.transcribe(wav))

    assert fakes.events == ["load_pipeline", "transcribe", "release"]
    assert all(s.type == "gap" for s in segments)


def test_session_with_only_silent_chunks_loads_no_model_and_releases_nothing(tmp_path, fakes):
    wav = _write_silent_wav(tmp_path)

    with _session() as session:
        segments = session.align(session.transcribe(wav))

    assert fakes.events == []
    assert len(segments) == 1
    assert segments[0].type == "gap"


def test_session_skips_phoneme_model_when_every_chunk_is_hallucinated(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(_mod("transcription"), "g2p", lambda text, method=None, **kwargs: ["a"] * 100)
    wav = _write_loud_wav(tmp_path)

    with _session(retry=False) as session:
        segments = session.align(session.transcribe(wav))

    assert fakes.count("load_phoneme") == 0
    assert all(s.type == "gap" for s in segments)


def test_session_forwards_each_calls_on_progress_to_the_model_loaders(tmp_path, fakes):
    wav = _write_loud_wav(tmp_path)

    def transcribe_progress(note):
        pass

    def align_progress(note):
        pass

    with _session() as session:
        transcription = session.transcribe(wav, on_progress=transcribe_progress)
        session.align(transcription, on_progress=align_progress)

    assert fakes.load_pipeline_progress == [transcribe_progress]
    assert fakes.load_phoneme_progress == [align_progress]


def test_session_reports_transcription_notes_from_transcribe_and_alignment_notes_from_align(tmp_path, fakes):
    wav = _write_loud_wav(tmp_path)
    transcribe_notes = []
    align_notes = []

    with _session() as session:
        transcription = session.transcribe(wav, on_progress=transcribe_notes.append)
        assert align_notes == []
        session.align(transcription, on_progress=align_notes.append)

    assert "歌詞書き起こし中: 0:00-0:00(1/1)" in transcribe_notes
    assert not any(n.startswith("音素アライメント中") for n in transcribe_notes)
    assert "音素アライメント中: 0:00-0:00(1/1)" in align_notes
    assert not any(n.startswith("歌詞書き起こし中") for n in align_notes)


def test_session_times_of_each_chunk_are_relative_to_that_chunks_own_audio(tmp_path, fakes):
    first = _write_loud_wav(tmp_path, "a.wav", 1920)
    second = _write_loud_wav(tmp_path, "b.wav", 3200)

    with _session() as session:
        t1 = session.transcribe(first)
        t2 = session.transcribe(second)
        s1 = session.align(t1)
        s2 = session.align(t2)

    assert s1[0].start_sec == pytest.approx(0.0)
    assert s1[-1].end_sec == pytest.approx(0.12)
    assert s2[0].start_sec == pytest.approx(0.0)
    assert s2[-1].end_sec == pytest.approx(0.2)


def test_session_two_phase_result_equals_single_call_recognize(tmp_path, fakes):
    from vocal_analysis.recognizer import recognize

    wav = _write_loud_wav(tmp_path)

    with _session() as session:
        two_phase = session.align(session.transcribe(wav))
    single = recognize(wav)

    assert two_phase == single


def test_recognize_uses_one_session_and_follows_the_two_phase_order(tmp_path, fakes):
    from vocal_analysis.recognizer import recognize

    wav = _write_loud_wav(tmp_path)

    recognize(wav)

    assert fakes.events == [
        "load_pipeline", "transcribe", "release", "load_phoneme", "align", "release",
    ]
    assert any(
        fakes.events.index("release") < position <= fakes.events.index("load_phoneme")
        for position in fakes.g2p_positions("あ"))


def test_session_passes_constructor_model_to_pipeline_loader(tmp_path, monkeypatch, fakes):
    from vocal_analysis import ContentRecognizerModel

    model = ContentRecognizerModel(model_id="org/custom", model_revision="rev9")
    received = []

    def load_pipeline(content_recognizer_model, on_progress=None):
        received.append(content_recognizer_model)
        return _Pipeline()

    monkeypatch.setattr(_mod("transcription"), "load_pipeline", load_pipeline)
    wav = _write_loud_wav(tmp_path)

    with _session(content_recognizer_model=model) as session:
        session.transcribe(wav)

    assert received == [model]


def test_session_defaults_to_the_default_content_recognizer_model(tmp_path, monkeypatch, fakes):
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL

    received = []

    def load_pipeline(content_recognizer_model, on_progress=None):
        received.append(content_recognizer_model)
        return _Pipeline()

    monkeypatch.setattr(_mod("transcription"), "load_pipeline", load_pipeline)

    with _session() as session:
        session.transcribe(_write_loud_wav(tmp_path))

    assert received == [DEFAULT_CONTENT_RECOGNIZER_MODEL]


def test_session_constructor_rejects_unknown_forced_aligner():
    from vocal_analysis.recognizer import RecognitionError

    with pytest.raises(RecognitionError, match="forced_aligner"):
        _session(forced_aligner="nonexistent")


def test_session_constructor_requires_sofa_config_for_sofa_aligner():
    from vocal_analysis.recognizer import RecognitionError

    with pytest.raises(RecognitionError, match="sofa_aligner"):
        _session(forced_aligner="sofa-forcedalign")


def test_session_constructor_failure_loads_nothing(fakes):
    from vocal_analysis.recognizer import RecognitionError

    with pytest.raises(RecognitionError):
        _session(forced_aligner="nonexistent")

    assert fakes.events == []


def test_session_transcribe_converts_missing_library_to_recognition_error(tmp_path, fakes, monkeypatch):
    from vocal_analysis.recognizer import RecognitionError

    original = ImportError("no transformers")

    def broken_transcribe(pipeline, samples):
        raise original

    monkeypatch.setattr(_mod("transcription"), "transcribe_segment", broken_transcribe)

    with _session() as session:
        with pytest.raises(RecognitionError, match="transformers") as excinfo:
            session.transcribe(_write_loud_wav(tmp_path))

    assert excinfo.value.__cause__ is original


def test_session_align_converts_phoneme_model_fetch_failure_to_recognition_error(tmp_path, fakes, monkeypatch):
    from vocal_analysis.recognizer import RecognitionError

    original = OSError("offline")

    def broken_load(on_progress=None):
        raise original

    monkeypatch.setattr(_mod("alignment"), "load_phoneme_model", broken_load)

    with _session() as session:
        transcription = session.transcribe(_write_loud_wav(tmp_path))
        with pytest.raises(RecognitionError, match="認識モデル") as excinfo:
            session.align(transcription)

    assert excinfo.value.__cause__ is original


def _sofa_config(tmp_path):
    from vocal_analysis import SofaAlignerConfig

    return SofaAlignerConfig(
        sofa_python=tmp_path / "sofa-venv" / "python",
        sofa_root=tmp_path / "SOFA",
        checkpoint_path=tmp_path / "checkpoint.ckpt",
    )


def test_session_sofa_path_aligns_after_all_transcriptions_and_never_loads_phoneme_model(
        tmp_path, fakes, monkeypatch):
    fakes.text = "かき"
    fakes.words = [("か", 0.1, 0.2), ("き", 0.25, 0.35)]
    monkeypatch.setattr(
        _mod("transcription"), "g2p",
        lambda text, method=None, **kwargs: {"か": ["k", "a"], "き": ["k", "i"], "かき": ["k", "a", "k", "i"]}[text])

    def fake_align_batch(targets, config):
        fakes.events.append("sofa_batch")
        return {
            "segment_0000": [(0.0, 0.05, "k"), (0.05, 0.1, "a")],
            "segment_0001": [(0.0, 0.04, "k"), (0.04, 0.1, "i")],
        }

    monkeypatch.setattr(_mod("sofa_align"), "_align_batch", fake_align_batch)
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav", 6400) for i in range(2)]

    with _session(forced_aligner="sofa-forcedalign", sofa_aligner=_sofa_config(tmp_path)) as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        for t in transcriptions:
            session.align(t)

    assert fakes.events == [
        "load_pipeline", "transcribe", "transcribe", "release", "sofa_batch", "sofa_batch",
    ]


def test_session_tinyllama_loads_each_model_once_and_converts_all_chunks_words_in_one_pass(
        tmp_path, katakana):
    fakes = katakana
    fakes.texts = ["あhello", "いworld"]
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(2)]

    with _session(english_katakana_method=_TINYLLAMA) as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        assert fakes.events == ["load_pipeline", "transcribe", "transcribe"]
        for t in transcriptions:
            session.align(t)

    assert fakes.events == [
        "load_pipeline", "transcribe", "transcribe",
        "release", "load_katakana", ("generate", "hello"), ("generate", "world"), "release",
        "load_phoneme", "align", "align",
        "release",
    ]
    assert fakes.count("load_pipeline") == 1
    assert fakes.count("load_katakana") == 1
    assert fakes.count("load_phoneme") == 1
    assert fakes.katakana_alive_at_phoneme_load is False
    conversion_cache = _mod("english_katakana")._conversion_cache
    assert conversion_cache[(_TINYLLAMA, "hello")] == "スカイ"
    assert conversion_cache[(_TINYLLAMA, "world")] == "スカイ"


def test_session_tinyllama_final_g2p_runs_with_the_converter_only_after_the_converter_is_released(
        tmp_path, katakana):
    fakes = katakana
    fakes.texts = ["あhello", "いworld"]
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(2)]

    with _session(english_katakana_method=_TINYLLAMA) as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        for t in transcriptions:
            session.align(t)

    last_generate = fakes.events.index(("generate", "world"))
    load_phoneme = fakes.events.index("load_phoneme")
    final_g2p = [(position, text) for position, text, method in fakes.g2p_calls if method == _TINYLLAMA]
    assert final_g2p
    assert all(last_generate + 1 < position <= load_phoneme for position, _ in final_g2p)
    assert {text for _, text in final_g2p} == {"あhello", "いworld"}


def test_session_tinyllama_density_g2p_runs_in_transcribe_phase_without_the_converter(tmp_path, katakana):
    fakes = katakana
    fakes.texts = ["あhello", "いworld"]
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(2)]

    with _session(english_katakana_method=_TINYLLAMA) as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        density_calls = list(fakes.g2p_calls)
        assert fakes.events == ["load_pipeline", "transcribe", "transcribe"]
        for t in transcriptions:
            session.align(t)

    assert {text for _, text, _ in density_calls} == {"あhello", "いworld"}
    assert all(method == "arpakana" for _, _, method in density_calls)
    release = fakes.events.index("release")
    assert all(position < release for position, _, _ in density_calls)
    final_methods = [method for _, _, method in fakes.g2p_calls[len(density_calls):]]
    assert final_methods
    assert set(final_methods) == {_TINYLLAMA}


def test_session_tinyllama_retranscribes_dense_english_chunk_within_transcribe_phase_and_adopts_the_result(
        tmp_path, katakana):
    fakes = katakana
    fakes.texts = ["あhello"]
    fakes.retry_texts = ["いworld"]
    fakes.g2p_phonemes = lambda text, method: ["a"] * 100 if (text, method) == ("あhello", "arpakana") else ["a"]
    wav = _write_loud_wav(tmp_path)

    with _session(english_katakana_method=_TINYLLAMA) as session:
        transcription = session.transcribe(wav)
        assert fakes.events == ["load_pipeline", "transcribe", "retranscribe"]
        segments = session.align(transcription)

    assert fakes.events == [
        "load_pipeline", "transcribe", "retranscribe",
        "release", "load_katakana", ("generate", "world"), "release",
        "load_phoneme", "align", "release",
    ]
    assert fakes.count("load_pipeline") == 1
    assert [s.type for s in segments] == ["gap", "vowel", "gap"]
    assert [text for _, text, _ in fakes.g2p_calls] == ["あhello", "いworld", "いworld"]
    assert [method for _, _, method in fakes.g2p_calls] == ["arpakana", "arpakana", _TINYLLAMA]
    conversion_cache = _mod("english_katakana")._conversion_cache
    assert (_TINYLLAMA, "hello") not in conversion_cache
    assert conversion_cache[(_TINYLLAMA, "world")] == "スカイ"


def test_session_tinyllama_dense_english_chunk_without_retry_keeps_the_original_text(tmp_path, katakana):
    fakes = katakana
    fakes.texts = ["あhello"]
    fakes.g2p_phonemes = lambda text, method: ["a"] * 100 if method == "arpakana" else ["a"]
    wav = _write_loud_wav(tmp_path)

    with _session(english_katakana_method=_TINYLLAMA, retry=False) as session:
        transcription = session.transcribe(wav)
        assert "retranscribe" not in fakes.events
        session.align(transcription)

    assert fakes.count("retranscribe") == 0
    assert fakes.count("load_pipeline") == 1
    assert ("generate", "hello") in fakes.events


def test_session_tinyllama_converts_a_word_shared_by_chunks_once(tmp_path, katakana):
    fakes = katakana
    fakes.texts = ["hello", "hello world"]
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(2)]

    with _session(english_katakana_method=_TINYLLAMA) as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        for t in transcriptions:
            session.align(t)

    generated = [e for e in fakes.events if isinstance(e, tuple)]
    assert generated == [("generate", "hello"), ("generate", "world")]
    assert fakes.count("load_katakana") == 1


def test_session_tinyllama_does_not_load_converter_when_every_word_is_already_converted(tmp_path, katakana):
    fakes = katakana
    fakes.texts = ["あhello", "いworld"]
    conversion_cache = _mod("english_katakana")._conversion_cache
    conversion_cache[(_TINYLLAMA, "hello")] = "ハロー"
    conversion_cache[(_TINYLLAMA, "world")] = "ワールド"
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(2)]

    with _session(english_katakana_method=_TINYLLAMA) as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        for t in transcriptions:
            session.align(t)

    assert fakes.events == [
        "load_pipeline", "transcribe", "transcribe", "release",
        "load_phoneme", "align", "align", "release",
    ]


def test_session_tinyllama_does_not_load_converter_when_no_chunk_has_english_words(tmp_path, katakana):
    fakes = katakana
    fakes.texts = ["あ", "い"]
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(2)]

    with _session(english_katakana_method=_TINYLLAMA) as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        for t in transcriptions:
            session.align(t)

    assert fakes.count("load_katakana") == 0
    assert fakes.count("release") == 2


def test_session_arpakana_never_loads_converter_or_inspects_uncached_words(tmp_path, katakana, monkeypatch):
    fakes = katakana
    fakes.texts = ["あhello", "いworld"]
    monkeypatch.setattr(
        _mod("english_katakana"), "uncached_target_words",
        lambda text, method: pytest.fail("arpakana では対象語検出を行わない"))
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(2)]

    with _session(english_katakana_method="arpakana") as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        assert fakes.events == ["load_pipeline", "transcribe", "transcribe"]
        for t in transcriptions:
            session.align(t)

    assert fakes.events == [
        "load_pipeline", "transcribe", "transcribe", "release",
        "load_phoneme", "align", "align", "release",
    ]


def test_session_tinyllama_reports_converter_loading_note_from_align_not_transcribe(tmp_path, katakana, monkeypatch):
    from vocal_analysis.config import ENGLISH_KATAKANA_MODEL

    fakes = katakana
    fakes.texts = ["あhello"]
    monkeypatch.setattr(_mod("model_store"), "prefetch_with_progress", lambda repo_id, revision, on_progress: False)
    transcribe_notes = []
    align_notes = []
    loading_note = f"カタカナ生成モデル読み込み中: {ENGLISH_KATAKANA_MODEL.model_id}"

    with _session(english_katakana_method=_TINYLLAMA) as session:
        transcription = session.transcribe(_write_loud_wav(tmp_path), on_progress=transcribe_notes.append)
        session.align(transcription, on_progress=align_notes.append)

    assert loading_note not in transcribe_notes
    assert loading_note in align_notes


def test_session_tinyllama_converter_load_failure_is_recognition_error_and_loads_no_phoneme_model(
        tmp_path, katakana, monkeypatch):
    import transformers

    from vocal_analysis.recognizer import RecognitionError

    fakes = katakana
    fakes.texts = ["あhello"]
    original = OSError("offline")

    def broken_load(*args, **kwargs):
        raise original

    monkeypatch.setattr(
        transformers, "AutoModelForCausalLM",
        type("_Models", (), {"from_pretrained": staticmethod(broken_load)}))

    with _session(english_katakana_method=_TINYLLAMA) as session:
        transcription = session.transcribe(_write_loud_wav(tmp_path))
        with pytest.raises(RecognitionError, match="カタカナ生成モデル") as excinfo:
            session.align(transcription)

    assert excinfo.value.__cause__ is original
    assert fakes.count("load_phoneme") == 0


def test_session_runs_g2p_for_every_chunk_after_all_transcriptions_and_before_phoneme_model_load(tmp_path, fakes):
    fakes.texts = ["あ", "い", "う"]
    wavs = [_write_loud_wav(tmp_path, f"c{i}.wav") for i in range(3)]

    with _session() as session:
        transcriptions = [session.transcribe(w) for w in wavs]
        for t in transcriptions:
            session.align(t)

    pipeline_release = fakes.events.index("release")
    load_phoneme = fakes.events.index("load_phoneme")
    for text in ("あ", "い", "う"):
        assert any(pipeline_release < position <= load_phoneme for position in fakes.g2p_positions(text))


def test_recognizer_facade_still_exports_public_names():
    from vocal_analysis import recognizer

    for name in ("RecognizerSession", "Transcription", "RecognitionError", "recognize", "convert_with_g2p"):
        assert hasattr(recognizer, name)
