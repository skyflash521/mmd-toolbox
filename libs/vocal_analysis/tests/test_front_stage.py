import json

import numpy as np
import pytest
import soundfile as sf

from vocal_analysis import ChunkingPolicy, ContentRecognizerModel, Segment, front_stage
from vocal_analysis.front_stage import run_front_stage
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


def seg(type_, start, end, phoneme=None):
    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=phoneme)


class _RecordingProgress:
    def __init__(self):
        self.calls = []

    def stage(self, stage, *, done=0, total=None, note="", elapsed=0.0):
        self.calls.append({"stage": stage, "done": done, "total": total, "note": note})
        assert elapsed >= 0.0


def _common_kwargs(**overrides):
    kw = dict(
        separate_vocals="always", separator="audio-separator-htdemucs-ft",
        content_recognizer_model=_TEST_MODEL, retry=True,
        chunking=ChunkingPolicy(max_duration_sec=300.0),
        forced_aligner="wav2vec2-ctc-forcedalign", sofa_aligner=None,
        english_katakana_method="arpakana",
    )
    kw.update(overrides)
    return kw


_PHASES = ("prepare_dictionary", "transcribe", "align")


class _FakeTranscription:
    def __init__(self, index, path):
        self.index = index
        self.path = path


class _FakeSession:
    def __init__(self, fakes, content_recognizer_model, retry, forced_aligner, sofa_aligner,
                 english_katakana_method):
        self.fakes = fakes
        self.settings = {
            "content_recognizer_model": content_recognizer_model, "retry": retry,
            "forced_aligner": forced_aligner, "sofa_aligner": sofa_aligner,
            "english_katakana_method": english_katakana_method,
        }
        self.entered = False
        self.closed = False
        self.transcribed = []
        self.aligned = []

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        self.closed = True
        self.fakes.events.append("close")

    def _run_phase(self, name, on_progress):
        assert not self.closed
        self.fakes.events.append(name)
        self.fakes.on_progress_args[name].append(on_progress)
        note = self.fakes.notes.get(name)
        if note is not None and on_progress is not None:
            on_progress(note)
        if self.fakes.failing_phase == name:
            raise self.fakes.error

    def prepare_dictionary(self, *, on_progress):
        self._run_phase("prepare_dictionary", on_progress)

    def transcribe(self, vocal_wav_path, *, on_progress):
        self._run_phase("transcribe", on_progress)
        transcribed = _FakeTranscription(len(self.transcribed), vocal_wav_path)
        self.transcribed.append(transcribed)
        return transcribed

    def align(self, transcribed, *, on_progress):
        self._run_phase("align", on_progress)
        self.aligned.append(transcribed)
        return self.fakes.align_result(transcribed.index)


class _SessionFakes:
    def __init__(self, monkeypatch):
        self.events = []
        self.sessions = []
        self.notes = {}
        self.on_progress_args = {name: [] for name in _PHASES}
        self.failing_phase = None
        self.error = None
        self.create_error = None
        self.align_result = lambda index: [seg("vowel", 0.0, 1.0, phoneme="a")]
        monkeypatch.setattr(front_stage._recognizer, "RecognizerSession", self._create)
        monkeypatch.setattr(front_stage._recognizer, "recognize", self._forbidden_recognize)

    def _create(self, content_recognizer_model, *, retry, forced_aligner, sofa_aligner,
                english_katakana_method):
        if self.create_error is not None:
            raise self.create_error
        session = _FakeSession(
            self, content_recognizer_model, retry, forced_aligner, sofa_aligner,
            english_katakana_method)
        self.sessions.append(session)
        return session

    @staticmethod
    def _forbidden_recognize(*args, **kwargs):
        raise AssertionError("front_stage は recognize() を呼ばずセッションを使う")

    def fail_in(self, phase, error):
        self.failing_phase = phase
        self.error = error

    def progress_to_events(self):
        def relay(stage, *, done=0, total=None, note="", elapsed=0.0):
            self.events.append(("progress", stage, done, total, note))
        return relay

    def count(self, name):
        return self.events.count(name)


@pytest.fixture(autouse=True)
def session_fakes(monkeypatch):
    return _SessionFakes(monkeypatch)


def test_single_run_calls_stages_in_order(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    progress = _RecordingProgress()
    result = run_front_stage(input_path, on_progress=progress.stage, **_common_kwargs())

    assert [c["stage"] for c in progress.calls] == ["load", "separate", "recognize", "rms"]
    assert result.analysis.vocal_wav == vocal_path
    assert result.analysis.segments == [seg("vowel", 0.0, 1.0, phoneme="a")]
    assert result.pcm.sample_rate == 8000
    assert result.pcm.samples.shape[1] == 2
    assert result.duration_sec == pytest.approx(1.0)
    assert result.forced_split is False


def test_single_run_passes_separator_name_to_separate(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    captured = {}

    def spy_separate(pcm, mode, **kwargs):
        captured["separator"] = kwargs.get("separator")
        return vocal_path

    monkeypatch.setattr(front_stage._separator, "separate", spy_separate)

    run_front_stage(input_path, **_common_kwargs(separator="audio-separator-htdemucs-ft"))

    assert captured["separator"] == "audio-separator-htdemucs-ft"


def test_single_run_reports_each_stage_before_running_it_and_passes_data_through(
        tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)
    given_segments = [seg("vowel", 0.0, 1.0, phoneme="a")]
    session_fakes.align_result = lambda index: given_segments

    call_order = []
    session_fakes.events = call_order
    real_load_audio = front_stage._io.load_audio
    real_compute_rms = front_stage._rms.compute_rms
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

    def spy_compute_rms(pcm):
        call_order.append("compute_rms")
        result = real_compute_rms(pcm)
        captured["compute_rms_result"] = result
        return result

    class _OrderRecordingProgress:
        def stage(self, stage, *, done=0, total=None, note="", elapsed=0.0):
            call_order.append(f"progress:{stage}")

    monkeypatch.setattr(front_stage._io, "load_audio", spy_load_audio)
    monkeypatch.setattr(front_stage._separator, "separate", spy_separate)
    monkeypatch.setattr(front_stage._rms, "compute_rms", spy_compute_rms)

    result = run_front_stage(
        input_path, on_progress=_OrderRecordingProgress().stage, **_common_kwargs())

    assert [c for c in call_order if c not in ("prepare_dictionary", "close")] == [
        "progress:load", "load_audio", "progress:separate", "separate", "progress:recognize",
        "transcribe", "align", "progress:rms", "load_audio", "compute_rms",
    ]
    (session,) = session_fakes.sessions
    (transcribed,) = session.transcribed
    assert str(transcribed.path) == str(vocal_path)
    assert session.aligned == [transcribed]
    assert captured["separate_pcm"] is captured["load_audio_results"][0]
    assert result.analysis.rms is captured["compute_rms_result"]
    assert result.analysis.segments == given_segments
    assert result.vocal_pcm is captured["load_audio_results"][1]


def test_session_receives_selected_content_recognizer_model(tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    given_model = ContentRecognizerModel(model_id="org/custom-recognizer", model_revision="rev1")
    given_sofa_config = object()
    run_front_stage(input_path, **_common_kwargs(
        content_recognizer_model=given_model, forced_aligner="sofa-forcedalign",
        sofa_aligner=given_sofa_config))
    (session,) = session_fakes.sessions
    assert session.settings["content_recognizer_model"] is given_model
    assert session.settings["forced_aligner"] == "sofa-forcedalign"
    assert session.settings["sofa_aligner"] is given_sofa_config


def test_session_receives_selected_english_katakana_method(tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    run_front_stage(input_path, **_common_kwargs(
        english_katakana_method="tinyllama-katakana-converter"))
    (session,) = session_fakes.sessions
    assert session.settings["english_katakana_method"] == "tinyllama-katakana-converter"


@pytest.mark.parametrize("retry", [True, False])
def test_session_receives_retry_setting(tmp_path, monkeypatch, session_fakes, retry):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    run_front_stage(input_path, **_common_kwargs(retry=retry))
    (session,) = session_fakes.sessions
    assert session.settings["retry"] is retry


def test_input_exactly_max_duration_is_not_chunked(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=3.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=3.0, amplitude=0.8)

    def fail_find_boundaries(*a, **k):
        raise AssertionError("尺が目標長ちょうどの入力で分割境界を探してはならない")

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", fail_find_boundaries)
    separate_calls = []
    monkeypatch.setattr(
        front_stage._separator, "separate",
        lambda pcm, mode, **kwargs: separate_calls.append(pcm) or vocal_path)

    progress = _RecordingProgress()
    result = run_front_stage(input_path, on_progress=progress.stage,
                             **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    assert len(separate_calls) == 1
    assert result.forced_split is False
    assert all(c["done"] == 0 and c["total"] is None for c in progress.calls)


def test_chunked_run_calls_separate_transcribe_and_align_once_per_chunk(
        tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False), (6.0, False)])

    def fake_merge(chunk_segments_list, chunk_offsets_sec, boundaries_sec):
        assert len(chunk_segments_list) == 3
        overlap = 1.0
        assert chunk_offsets_sec == pytest.approx([0.0, 3.0 - overlap, 6.0 - overlap])
        assert boundaries_sec == [3.0, 6.0]
        return [seg("vowel", 0.0, 10.0, phoneme="a")]

    monkeypatch.setattr(front_stage._chunking, "merge_chunk_segments", fake_merge)

    separate_calls = []
    separate_kwargs = []

    def fake_separate(pcm, mode, **kwargs):
        separate_calls.append(len(pcm.samples) / pcm.sample_rate)
        separate_kwargs.append(kwargs)
        return vocal_path

    monkeypatch.setattr(front_stage._separator, "separate", fake_separate)

    sofa_config = object()
    progress = _RecordingProgress()
    result = run_front_stage(input_path, on_progress=progress.stage, **_common_kwargs(
        chunking=ChunkingPolicy(max_duration_sec=3.0), forced_aligner="sofa-forcedalign", sofa_aligner=sofa_config,
        english_katakana_method="tinyllama-katakana-converter"))

    assert len(separate_calls) == 3
    assert session_fakes.count("prepare_dictionary") == 1
    assert session_fakes.count("transcribe") == 3
    assert session_fakes.count("align") == 3
    (session,) = session_fakes.sessions
    assert session.settings["forced_aligner"] == "sofa-forcedalign"
    assert session.settings["sofa_aligner"] is sofa_config
    assert session.settings["english_katakana_method"] == "tinyllama-katakana-converter"
    for kwargs in separate_kwargs:
        assert kwargs["separator"] == "audio-separator-htdemucs-ft"
    assert separate_calls[0] == pytest.approx(3.0 + 1.0, abs=0.05)
    assert separate_calls[1] == pytest.approx((6.0 + 1.0) - (3.0 - 1.0), abs=0.05)
    assert separate_calls[2] == pytest.approx(10.0 - (6.0 - 1.0), abs=0.05)
    assert result.duration_sec == pytest.approx(10.0, abs=0.05)
    separate_done_totals = [(c["done"], c["total"]) for c in progress.calls if c["stage"] == "separate"]
    assert separate_done_totals == [(0, 3), (1, 3), (2, 3)]
    rms_done_totals = [(c["done"], c["total"]) for c in progress.calls if c["stage"] == "rms"]
    assert rms_done_totals == [(0, None)]


def _stub_three_chunks(monkeypatch, vocal_path, session_fakes):
    monkeypatch.setattr(
        front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False), (6.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])
    monkeypatch.setattr(
        front_stage._separator, "separate",
        lambda pcm, mode, **kwargs: session_fakes.events.append("separate") or vocal_path)


def _calls_only(events):
    return [e for e in events if isinstance(e, str) and e != "close"]


def test_chunked_run_separates_all_chunks_then_prepares_dictionary_then_transcribes_then_aligns(
        tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)
    _stub_three_chunks(monkeypatch, vocal_path, session_fakes)

    run_front_stage(
        input_path, on_progress=session_fakes.progress_to_events(),
        **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    assert _calls_only(session_fakes.events) == [
        "separate", "separate", "separate", "prepare_dictionary",
        "transcribe", "transcribe", "transcribe", "align", "align", "align",
    ]


def test_single_run_separates_then_prepares_dictionary_then_transcribes_then_aligns(
        tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)
    monkeypatch.setattr(
        front_stage._separator, "separate",
        lambda pcm, mode, **kwargs: session_fakes.events.append("separate") or vocal_path)

    run_front_stage(input_path, on_progress=session_fakes.progress_to_events(), **_common_kwargs())

    assert _calls_only(session_fakes.events) == ["separate", "prepare_dictionary", "transcribe", "align"]


def test_chunked_run_transcribes_each_chunks_vocal_and_aligns_each_transcription_in_order(
        tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_paths = []
    for i in range(3):
        path = tmp_path / f"vocal{i}.wav"
        write_wav(path, seconds=10.0, amplitude=0.8)
        vocal_paths.append(path)
    monkeypatch.setattr(
        front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False), (6.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])
    queued = iter(vocal_paths)
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: next(queued))

    run_front_stage(input_path, **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    (session,) = session_fakes.sessions
    assert [str(t.path) for t in session.transcribed] == [str(p) for p in vocal_paths]
    assert session.aligned == session.transcribed


def test_chunked_run_merges_each_chunks_aligned_segments(tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)
    _stub_three_chunks(monkeypatch, vocal_path, session_fakes)
    per_chunk = [[seg("vowel", 0.0, 1.0, phoneme=p)] for p in ("a", "i", "u")]
    session_fakes.align_result = lambda index: per_chunk[index]
    merged = [seg("vowel", 0.0, 10.0, phoneme="e")]
    received = {}

    def fake_merge(chunk_segments_list, chunk_offsets_sec, boundaries_sec):
        received["chunk_segments_list"] = chunk_segments_list
        return merged

    monkeypatch.setattr(front_stage._chunking, "merge_chunk_segments", fake_merge)

    result = run_front_stage(input_path, **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    assert received["chunk_segments_list"] == per_chunk
    assert result.analysis.segments == merged


def test_single_run_uses_one_session_and_closes_it(tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    run_front_stage(input_path, **_common_kwargs())

    (session,) = session_fakes.sessions
    assert session.entered
    assert session.closed
    assert session_fakes.count("close") == 1
    assert len(session.transcribed) == 1


def test_chunked_run_uses_one_session_for_all_chunks_and_closes_it(tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)
    _stub_three_chunks(monkeypatch, vocal_path, session_fakes)

    run_front_stage(input_path, **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    (session,) = session_fakes.sessions
    assert session.entered
    assert session.closed
    assert session_fakes.count("close") == 1
    assert len(session.transcribed) == 3
    assert len(session.aligned) == 3


def test_chunked_run_reports_forced_split(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=6.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=6.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, True)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 6.0, phoneme="a")])
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    result = run_front_stage(input_path, **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    assert result.forced_split is True


def test_chunked_run_uses_unseparated_audio_rms_for_boundaries_and_whole_vocal_rms_for_output(
        tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0, amplitude=0.5)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    boundary_calls = []

    def fake_find_boundaries(duration_sec, rms_times_sec, rms_values, **kwargs):
        boundary_calls.append((duration_sec, rms_times_sec, rms_values))
        return [(3.0, False), (6.0, False)]

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", fake_find_boundaries)
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    compute_rms_results = []
    real_compute_rms = front_stage._rms.compute_rms

    def recording_compute_rms(pcm):
        result = real_compute_rms(pcm)
        compute_rms_results.append(result)
        return result

    monkeypatch.setattr(front_stage._rms, "compute_rms", recording_compute_rms)

    result = run_front_stage(
        input_path, **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    assert len(boundary_calls) == 1
    assert boundary_calls[0][0] == pytest.approx(10.0, abs=0.05)
    assert len(compute_rms_results) == 2
    unseparated_rms, whole_vocal_rms = compute_rms_results
    assert boundary_calls[0][1] is unseparated_rms.times_sec
    assert boundary_calls[0][2] is unseparated_rms.values
    assert result.analysis.rms is whole_vocal_rms


def test_chunked_run_preserves_relative_loudness_across_chunks(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=6.0)
    quiet_vocal = tmp_path / "vocal_quiet.wav"
    write_wav(quiet_vocal, seconds=6.0, amplitude=0.1)
    loud_vocal = tmp_path / "vocal_loud.wav"
    write_wav(loud_vocal, seconds=6.0, amplitude=0.9)

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 6.0, phoneme="a")])

    separate_call_count = [0]

    def fake_separate(pcm, mode, **kwargs):
        separate_call_count[0] += 1
        return quiet_vocal if separate_call_count[0] == 1 else loud_vocal

    monkeypatch.setattr(front_stage._separator, "separate", fake_separate)

    compute_rms_pcms = []
    real_compute_rms = front_stage._rms.compute_rms

    def recording_compute_rms(pcm):
        compute_rms_pcms.append(pcm)
        return real_compute_rms(pcm)

    monkeypatch.setattr(front_stage._rms, "compute_rms", recording_compute_rms)

    run_front_stage(input_path, **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    _, whole_vocal_pcm = compute_rms_pcms
    whole_vocal_samples = whole_vocal_pcm.samples
    half = len(whole_vocal_samples) // 2
    first_half_peak = float(np.max(np.abs(whole_vocal_samples[:half])))
    second_half_peak = float(np.max(np.abs(whole_vocal_samples[half:])))
    assert first_half_peak < second_half_peak * 0.5


def test_chunked_run_reports_recognize_progress_over_transcription_then_alignment_of_every_chunk(
        tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(5.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    progress = _RecordingProgress()
    run_front_stage(input_path, on_progress=progress.stage,
                    **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    recognize_done_totals = [
        (c["done"], c["total"]) for c in progress.calls if c["stage"] == "recognize" and c["note"] == ""]
    assert recognize_done_totals == [(0, 4), (1, 4), (2, 4), (3, 4)]


def test_chunked_run_reports_each_recognize_start_right_before_its_transcription_or_alignment(
        tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)
    _stub_three_chunks(monkeypatch, vocal_path, session_fakes)

    run_front_stage(
        input_path, on_progress=session_fakes.progress_to_events(),
        **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    finished_calls = 0
    phases_after_start = []
    for event in session_fakes.events:
        if event in ("transcribe", "align"):
            finished_calls += 1
        elif isinstance(event, tuple) and event[1] == "recognize" and event[4] == "":
            _, _, done, total, _ = event
            assert (done, total) == (finished_calls, 6)
            phases_after_start.append(done)
    assert phases_after_start == [0, 1, 2, 3, 4, 5]


def test_non_chunked_progress_reports_done_zero_total_none_for_every_stage(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    progress = _RecordingProgress()
    run_front_stage(input_path, on_progress=progress.stage, **_common_kwargs())

    by_stage = {c["stage"]: c for c in progress.calls}
    for stage in ("load", "separate", "recognize", "rms"):
        assert by_stage[stage]["done"] == 0
        assert by_stage[stage]["total"] is None
    assert [c["stage"] for c in progress.calls].count("recognize") == 1


@pytest.mark.parametrize("phase", _PHASES)
def test_non_chunked_recognize_on_progress_forwards_download_note(
        tmp_path, monkeypatch, session_fakes, phase):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)
    session_fakes.notes[phase] = "ダウンロード中: dummy-model 42%"

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    progress = _RecordingProgress()
    run_front_stage(input_path, on_progress=progress.stage, **_common_kwargs())

    assert all(on_progress is not None for on_progress in session_fakes.on_progress_args[phase])
    recognize_calls = [c for c in progress.calls if c["stage"] == "recognize"]
    assert any(c["note"] == "ダウンロード中: dummy-model 42%" for c in recognize_calls)
    assert all(c["done"] == 0 and c["total"] is None for c in recognize_calls)


@pytest.mark.parametrize("phase", ["transcribe", "align"])
def test_chunked_recognize_on_progress_preserves_chunk_done_total(
        tmp_path, monkeypatch, session_fakes, phase):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)
    session_fakes.notes[phase] = "ダウンロード中: dummy-model 10%"

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(5.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    progress = _RecordingProgress()
    run_front_stage(input_path, on_progress=progress.stage,
                    **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    recognize_calls = [c for c in progress.calls if c["stage"] == "recognize"]
    download_notes = [c for c in recognize_calls if c["note"] == "ダウンロード中: dummy-model 10%"]
    expected = [(0, 4), (1, 4)] if phase == "transcribe" else [(2, 4), (3, 4)]
    assert [(c["done"], c["total"]) for c in download_notes] == expected


def test_non_chunked_dictionary_note_reaches_progress_as_recognize_stage_between_separation_and_transcription(
        tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)
    session_fakes.notes["prepare_dictionary"] = "辞書取得中"
    monkeypatch.setattr(
        front_stage._separator, "separate",
        lambda pcm, mode, **kwargs: session_fakes.events.append("separate") or vocal_path)

    run_front_stage(input_path, on_progress=session_fakes.progress_to_events(), **_common_kwargs())

    events = session_fakes.events
    note_index = events.index(("progress", "recognize", 0, None, "辞書取得中"))
    assert events.index("separate") < events.index("prepare_dictionary") < note_index
    assert note_index < events.index("transcribe")


def test_chunked_dictionary_note_reaches_progress_as_recognize_stage_after_all_separations_and_before_transcription(
        tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)
    session_fakes.notes["prepare_dictionary"] = "辞書取得中"
    _stub_three_chunks(monkeypatch, vocal_path, session_fakes)

    run_front_stage(
        input_path, on_progress=session_fakes.progress_to_events(),
        **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    events = session_fakes.events
    note_indices = [
        i for i, e in enumerate(events) if isinstance(e, tuple) and e[1] == "recognize" and e[4] == "辞書取得中"]
    assert len(note_indices) == 1
    last_separate = max(i for i, e in enumerate(events) if e == "separate")
    assert last_separate < note_indices[0] < events.index("transcribe")


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

    monkeypatch.setattr(front_stage._separator, "separate", fake_separate)

    progress = _RecordingProgress()
    run_front_stage(input_path, on_progress=progress.stage, **_common_kwargs())

    assert captured["on_progress"] is not None
    separate_calls = [c for c in progress.calls if c["stage"] == "separate"]
    assert any(c["note"] == "ダウンロード中: 42%" for c in separate_calls)
    assert all(c["done"] == 0 and c["total"] is None for c in separate_calls)


def test_chunked_separate_on_progress_preserves_chunk_done_total(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(5.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])

    def fake_separate(pcm, mode, on_progress=None, **kwargs):
        if on_progress is not None:
            on_progress("ダウンロード中: 10%")
        return vocal_path

    monkeypatch.setattr(front_stage._separator, "separate", fake_separate)

    progress = _RecordingProgress()
    run_front_stage(input_path, on_progress=progress.stage,
                    **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    separate_calls = [c for c in progress.calls if c["stage"] == "separate"]
    download_notes = [c for c in separate_calls if c["note"] == "ダウンロード中: 10%"]
    assert [(c["done"], c["total"]) for c in download_notes] == [(0, 2), (1, 2)]


def test_run_without_progress_reporter_passes_on_progress_none_to_separate(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    captured = {}

    def fake_separate(pcm, mode, *, on_progress, **kwargs):
        captured["on_progress"] = on_progress
        return vocal_path

    monkeypatch.setattr(front_stage._separator, "separate", fake_separate)

    run_front_stage(input_path, **_common_kwargs())

    assert captured["on_progress"] is None


@pytest.mark.parametrize("phase", _PHASES)
def test_run_without_progress_reporter_passes_on_progress_none_to_session(
        tmp_path, monkeypatch, session_fakes, phase):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    run_front_stage(input_path, **_common_kwargs())

    assert session_fakes.on_progress_args[phase] == [None]


@pytest.mark.parametrize("phase", _PHASES)
def test_chunked_run_without_progress_reporter_passes_on_progress_none_to_session(
        tmp_path, monkeypatch, session_fakes, phase):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(5.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    run_front_stage(input_path, **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    expected_calls = 1 if phase == "prepare_dictionary" else 2
    assert session_fakes.on_progress_args[phase] == [None] * expected_calls


def test_keep_intermediate_dir_none_creates_nothing(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    keep_dir = tmp_path / "out.vmd.intermediate"
    run_front_stage(input_path, **_common_kwargs())
    assert not keep_dir.exists()


def test_keep_intermediate_saves_normalized_input_vocal_and_segments(tmp_path, monkeypatch, session_fakes):
    input_path = tmp_path / "in.wav"
    input_sample_rate = 8000
    write_wav(input_path, seconds=1.0, sample_rate=input_sample_rate)
    vocal_path = tmp_path / "vocal.wav"
    vocal_sample_rate = 11025
    write_wav(vocal_path, seconds=1.0, sample_rate=vocal_sample_rate, amplitude=0.8)
    given_segments = [seg("vowel", 0.0, 1.0, phoneme="ɯ")]

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)
    session_fakes.align_result = lambda index: given_segments

    keep_dir = tmp_path / "out.vmd.intermediate"
    run_front_stage(input_path, keep_intermediate_dir=keep_dir, **_common_kwargs())

    assert (keep_dir / "input_normalized.wav").exists()
    assert (keep_dir / "vocal.wav").exists()
    _, saved_input_sr = sf.read(str(keep_dir / "input_normalized.wav"))
    _, saved_vocal_sr = sf.read(str(keep_dir / "vocal.wav"))
    assert saved_input_sr == input_sample_rate
    assert saved_vocal_sr == vocal_sample_rate
    segments_path = keep_dir / "segments.json"
    assert segments_path.exists()
    segments_text = segments_path.read_text(encoding="utf-8")
    assert "ɯ" in segments_text
    assert json.loads(segments_text) == [
        {"type": "vowel", "start_sec": 0.0, "end_sec": 1.0, "phoneme": "ɯ"},
    ]


def test_keep_intermediate_write_failure_raises_intermediate_write_error(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    keep_dir = tmp_path / "out.vmd.intermediate"
    keep_dir.mkdir()
    (keep_dir / "input_normalized.wav").mkdir()

    with pytest.raises(front_stage.IntermediateWriteError):
        run_front_stage(input_path, keep_intermediate_dir=keep_dir, **_common_kwargs())


def test_chunked_run_returns_and_saves_concatenated_whole_vocal(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=6.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=6.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 6.0, phoneme="a")])
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    keep_dir = tmp_path / "out.vmd.intermediate"
    result = run_front_stage(input_path, keep_intermediate_dir=keep_dir,
                             **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    for path in (keep_dir / "vocal.wav", result.analysis.vocal_wav):
        saved_samples, saved_sr = sf.read(str(path), dtype="float32", always_2d=True)
        assert saved_samples.shape[0] == pytest.approx(6.0 * saved_sr, abs=saved_sr * 0.01)


def test_chunked_run_private_whole_vocal_write_failure_propagates_without_classification(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=6.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=6.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 6.0, phoneme="a")])
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    error = OSError("disk full")

    def fail_write(path, *a, **k):
        raise error

    monkeypatch.setattr(front_stage.sf, "write", fail_write)

    with pytest.raises(OSError) as exc:
        run_front_stage(input_path, **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))
    assert exc.value is error


def test_chunked_run_reports_elapsed_from_each_call_start_without_accumulating_across_chunks(
        tmp_path, monkeypatch, session_fakes):
    import itertools

    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(5.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kwargs: vocal_path)

    session_fakes.notes["transcribe"] = "認識中"
    tick = 10.0
    clock = itertools.count(0.0, tick)
    monkeypatch.setattr(front_stage.time, "monotonic", lambda: next(clock))

    elapsed_of_notes = []

    def on_progress(stage, *, done, total, note, elapsed):
        if note == "認識中":
            elapsed_of_notes.append(elapsed)

    run_front_stage(input_path, on_progress=on_progress,
                    **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))

    assert elapsed_of_notes == [tick, tick]


def test_slice_pcm_extracts_the_requested_time_range():
    samples = np.arange(100, dtype=np.float32).reshape(-1, 1)
    pcm = AudioPcm(samples=samples, sample_rate=10)
    sliced = front_stage._slice_pcm(pcm, 2.0, 5.0)
    assert sliced.sample_rate == 10
    np.testing.assert_array_equal(sliced.samples[:, 0], samples[20:50, 0])


def test_concat_pcm_joins_slices_in_order():
    pcm_a = AudioPcm(samples=np.array([[1.0], [2.0]], dtype=np.float32), sample_rate=10)
    pcm_b = AudioPcm(samples=np.array([[3.0], [4.0]], dtype=np.float32), sample_rate=10)
    joined = front_stage._concat_pcm([pcm_a, pcm_b])
    np.testing.assert_array_equal(joined.samples[:, 0], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert joined.sample_rate == 10


def test_vocal_reread_failure_raises_dedicated_error_with_path(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=1.0, amplitude=0.8)

    real_load_audio = front_stage._io.load_audio

    def fail_on_vocal(path):
        if str(path) == str(vocal_path):
            raise AudioLoadError("broken vocal wav", reason="not_audio")
        return real_load_audio(path)

    monkeypatch.setattr(front_stage._io, "load_audio", fail_on_vocal)
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kw: vocal_path)

    with pytest.raises(front_stage.IntermediateReadError) as exc:
        run_front_stage(input_path, on_progress=_RecordingProgress().stage, **_common_kwargs())
    assert str(exc.value.path) == str(vocal_path)


def test_chunked_vocal_reread_failure_raises_dedicated_error_with_path(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0, amplitude=0.8)

    monkeypatch.setattr(
        front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False), (6.0, False)])
    monkeypatch.setattr(
        front_stage._chunking, "merge_chunk_segments",
        lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])
    monkeypatch.setattr(front_stage._separator, "separate", lambda pcm, mode, **kw: vocal_path)

    real_read = front_stage.sf.read

    def fail_on_vocal(path, *a, **kw):
        if str(path) == str(vocal_path):
            raise sf.LibsndfileError(1, prefix=str(vocal_path))
        return real_read(path, *a, **kw)

    monkeypatch.setattr(front_stage.sf, "read", fail_on_vocal)

    with pytest.raises(front_stage.IntermediateReadError) as exc:
        run_front_stage(input_path, **_common_kwargs(chunking=ChunkingPolicy(max_duration_sec=3.0)))
    assert str(exc.value.path) == str(vocal_path)


def test_input_read_failure_is_not_wrapped_in_dedicated_error(tmp_path, monkeypatch):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=1.0)

    def always_fail(path):
        raise AudioLoadError("broken input", reason="not_audio")

    monkeypatch.setattr(front_stage._io, "load_audio", always_fail)

    with pytest.raises(AudioLoadError) as exc:
        run_front_stage(input_path, on_progress=_RecordingProgress().stage, **_common_kwargs())
    assert type(exc.value) is AudioLoadError


_FAILING_STAGES = ["separate", *_PHASES]


def _reported_stage(failing_stage):
    return "separate" if failing_stage == "separate" else "recognize"


def _stage_failure_kwargs(tmp_path, monkeypatch, session_fakes, *, failing_stage, error, chunked=False):
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0 if chunked else 1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0 if chunked else 1.0, amplitude=0.8)

    def separate(pcm, mode, **kw):
        if failing_stage == "separate":
            raise error
        return vocal_path

    if failing_stage in _PHASES:
        session_fakes.fail_in(failing_stage, error)
    monkeypatch.setattr(front_stage._separator, "separate", separate)
    if chunked:
        monkeypatch.setattr(
            front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False), (6.0, False)])
        monkeypatch.setattr(
            front_stage._chunking, "merge_chunk_segments",
            lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])
    return input_path, _common_kwargs(
        chunking=ChunkingPolicy(max_duration_sec=3.0 if chunked else 300.0))


class _UnlistedError(Exception):
    pass


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failing_stage", _FAILING_STAGES)
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, MemoryError, OSError, _UnlistedError])
def test_unclassified_exception_from_inference_becomes_stage_error_keeping_type_message_and_cause(
        tmp_path, monkeypatch, session_fakes, failing_stage, error_type, chunked):
    error = error_type("推論の失敗")
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, session_fakes, failing_stage=failing_stage, error=error, chunked=chunked)

    with pytest.raises(front_stage.StageExecutionError) as exc:
        run_front_stage(input_path, **kwargs)
    assert exc.value.stage == _reported_stage(failing_stage)
    assert error_type.__name__ in str(exc.value)
    assert "推論の失敗" in str(exc.value)
    assert exc.value.__cause__ is error


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failing_stage", _FAILING_STAGES)
@pytest.mark.parametrize("error_factory", [
    lambda: SeparationError("separator failed"),
    lambda: RecognitionError("recognizer failed"),
    lambda: AudioLoadError("broken wav", reason="not_audio"),
    lambda: front_stage.IntermediateReadError("broken vocal wav", path="vocal.wav"),
    lambda: front_stage.IntermediateWriteError("disk full"),
])
def test_classified_exceptions_from_inference_pass_through_unchanged(
        tmp_path, monkeypatch, session_fakes, failing_stage, error_factory, chunked):
    error = error_factory()
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, session_fakes, failing_stage=failing_stage, error=error, chunked=chunked)

    with pytest.raises(type(error)) as exc:
        run_front_stage(input_path, **kwargs)
    assert exc.value is error


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failing_stage", _FAILING_STAGES)
def test_keyboard_interrupt_from_inference_passes_through_unchanged(
        tmp_path, monkeypatch, session_fakes, failing_stage, chunked):
    error = KeyboardInterrupt()
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, session_fakes, failing_stage=failing_stage, error=error, chunked=chunked)

    with pytest.raises(KeyboardInterrupt) as exc:
        run_front_stage(input_path, **kwargs)
    assert exc.value is error


@pytest.mark.parametrize("chunked", [False, True])
def test_bare_exception_is_not_mapped_when_separation_is_skipped(
        tmp_path, monkeypatch, session_fakes, chunked):
    error = RuntimeError("一時ファイルを書けません")
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, session_fakes, failing_stage="separate", error=error, chunked=chunked)
    kwargs["separate_vocals"] = "never"

    with pytest.raises(RuntimeError) as exc:
        run_front_stage(input_path, **kwargs)
    assert exc.value is error


@pytest.mark.parametrize("chunked", [False, True])
def test_bare_exception_outside_inference_is_not_mapped_to_stage(
        tmp_path, monkeypatch, session_fakes, chunked):
    error = RuntimeError("rms failed")
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, session_fakes, failing_stage=None, error=error, chunked=chunked)
    monkeypatch.setattr(
        front_stage._rms, "compute_rms", lambda *a, **k: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError) as exc:
        run_front_stage(input_path, **kwargs)
    assert exc.value is error


@pytest.mark.parametrize("chunked", [False, True])
def test_bare_exception_from_vocal_reread_right_after_inference_is_not_mapped_to_stage(
        tmp_path, monkeypatch, session_fakes, chunked):
    error = RuntimeError("read failed")
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, session_fakes, failing_stage=None, error=error, chunked=chunked)
    monkeypatch.setattr(
        front_stage, "_read_intermediate", lambda *a, **k: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError) as exc:
        run_front_stage(input_path, **kwargs)
    assert exc.value is error


class _CallbackFailure(Exception):
    pass


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("stage", _FAILING_STAGES)
def test_progress_callback_failure_during_inference_passes_through_unchanged(
        tmp_path, monkeypatch, session_fakes, stage, chunked):
    error = _CallbackFailure("stdout is closed")
    input_path = tmp_path / "in.wav"
    write_wav(input_path, seconds=10.0 if chunked else 1.0)
    vocal_path = tmp_path / "vocal.wav"
    write_wav(vocal_path, seconds=10.0 if chunked else 1.0, amplitude=0.8)

    def separate(pcm, mode, on_progress=None, **kw):
        if stage == "separate" and on_progress is not None:
            on_progress("モデル取得中")
        return vocal_path

    if stage in _PHASES:
        session_fakes.notes[stage] = "モデル取得中"
    monkeypatch.setattr(front_stage._separator, "separate", separate)
    if chunked:
        monkeypatch.setattr(
            front_stage._chunking, "find_chunk_boundaries", lambda *a, **k: [(3.0, False), (6.0, False)])
        monkeypatch.setattr(
            front_stage._chunking, "merge_chunk_segments",
            lambda *a, **k: [seg("vowel", 0.0, 10.0, phoneme="a")])

    def failing_progress(stage_id, **kwargs):
        if kwargs.get("note"):
            raise error

    kwargs = _common_kwargs(
        chunking=ChunkingPolicy(max_duration_sec=3.0 if chunked else 300.0))
    with pytest.raises(_CallbackFailure) as exc:
        run_front_stage(input_path, on_progress=failing_progress, **kwargs)
    assert exc.value is error


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failing_stage", _PHASES)
def test_session_is_closed_when_a_recognize_phase_fails(
        tmp_path, monkeypatch, session_fakes, failing_stage, chunked):
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, session_fakes, failing_stage=failing_stage,
        error=RuntimeError("推論の失敗"), chunked=chunked)

    with pytest.raises(front_stage.StageExecutionError):
        run_front_stage(input_path, **kwargs)

    (session,) = session_fakes.sessions
    assert session.closed


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failing_stage", ["prepare_dictionary", "transcribe"])
def test_alignment_does_not_start_after_an_earlier_recognize_phase_fails(
        tmp_path, monkeypatch, session_fakes, failing_stage, chunked):
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, session_fakes, failing_stage=failing_stage,
        error=RuntimeError("推論の失敗"), chunked=chunked)

    with pytest.raises(front_stage.StageExecutionError):
        run_front_stage(input_path, **kwargs)

    assert session_fakes.count("align") == 0
    if failing_stage == "prepare_dictionary":
        assert session_fakes.count("transcribe") == 0


@pytest.mark.parametrize("chunked", [False, True])
def test_recognition_error_from_session_creation_passes_through_unchanged(
        tmp_path, monkeypatch, session_fakes, chunked):
    error = RecognitionError("forced_aligner を指定してください")
    session_fakes.create_error = error
    input_path, kwargs = _stage_failure_kwargs(
        tmp_path, monkeypatch, session_fakes, failing_stage=None, error=error, chunked=chunked)

    with pytest.raises(RecognitionError) as exc:
        run_front_stage(input_path, **kwargs)
    assert exc.value is error
    assert session_fakes.sessions == []
