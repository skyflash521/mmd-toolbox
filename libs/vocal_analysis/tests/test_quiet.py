import os
import subprocess
import sys
import threading
import types

import numpy as np
import pytest
import soundfile as sf

from vocal_analysis import quiet

try:
    import huggingface_hub.utils as hf_utils
    import huggingface_hub.utils.logging as hf_logging
    import torch
    import tqdm.std as tqdm_std
    import transformers

    PRISTINE_TQDM_INIT = tqdm_std.tqdm.__dict__["__init__"]
    THIRD_PARTY_MISSING = False
except ImportError:
    THIRD_PARTY_MISSING = True

_SIX_FRAME_LOGITS = [[[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]]


@pytest.fixture
def third_party_state():
    if THIRD_PARTY_MISSING:
        pytest.skip("tqdm・huggingface_hub・transformers・torch が必要です")
    saved_progress_bars_disabled = hf_utils.are_progress_bars_disabled()
    saved_hf_verbosity = hf_logging.get_verbosity()
    saved_transformers_verbosity = transformers.utils.logging.get_verbosity()
    tqdm_std.tqdm.__init__ = PRISTINE_TQDM_INIT
    hf_utils.enable_progress_bars()
    hf_logging.set_verbosity(hf_logging.WARNING)
    transformers.utils.logging.set_verbosity(transformers.logging.WARNING)
    yield
    tqdm_std.tqdm.__init__ = PRISTINE_TQDM_INIT
    hf_logging.set_verbosity(saved_hf_verbosity)
    transformers.utils.logging.set_verbosity(saved_transformers_verbosity)
    if saved_progress_bars_disabled:
        hf_utils.disable_progress_bars()
    else:
        hf_utils.enable_progress_bars()


def _observe():
    bar = tqdm_std.tqdm(total=1)
    try:
        tqdm_disabled = bar.disable
    finally:
        bar.close()
    return {
        "tqdm_disabled": tqdm_disabled,
        "hf_disabled": hf_utils.are_progress_bars_disabled(),
        "hf_verbosity": hf_logging.get_verbosity(),
        "transformers_verbosity": transformers.utils.logging.get_verbosity(),
    }


def _assert_silenced(observed):
    assert observed == {
        "tqdm_disabled": True,
        "hf_disabled": True,
        "hf_verbosity": hf_logging.CRITICAL,
        "transformers_verbosity": transformers.logging.CRITICAL,
    }


def _assert_restored():
    assert tqdm_std.tqdm.__dict__["__init__"] is PRISTINE_TQDM_INIT
    assert _observe() == {
        "tqdm_disabled": False,
        "hf_disabled": False,
        "hf_verbosity": hf_logging.WARNING,
        "transformers_verbosity": transformers.logging.WARNING,
    }


class _Recorder:
    def __init__(self):
        self.observed = []

    def record(self):
        self.observed.append(_observe())

    def assert_silenced_in_every_call(self, expected_calls=1):
        assert len(self.observed) == expected_calls
        for observed in self.observed:
            _assert_silenced(observed)


class _FakeProcessor:
    class tokenizer:
        pad_token_id = 0

        @staticmethod
        def get_vocab():
            return {"<pad>": 0, "a": 1}

    def __call__(self, samples, sampling_rate=None, return_tensors=None):
        return types.SimpleNamespace(input_values=torch.zeros(1, 4))


class _FakeCtcModel:
    def __init__(self, recorder=None):
        self._recorder = recorder

    def __call__(self, input_values):
        if self._recorder is not None:
            self._recorder.record()
        return types.SimpleNamespace(logits=torch.tensor(_SIX_FRAME_LOGITS))

    def to(self, device):
        return self

    def eval(self):
        return self


class _FakePipeline:
    device = "cpu"

    def __init__(self, recorder=None):
        self._recorder = recorder
        self.tokenizer = types.SimpleNamespace(
            get_prompt_ids=lambda prompt, return_tensors=None: types.SimpleNamespace(to=lambda device: None))

    def __call__(self, samples, **kwargs):
        if self._recorder is not None:
            self._recorder.record()
        return {"text": "あ"}


def _install_fake_transformers(monkeypatch, recorder):
    # 遅延ロードの transformers は属性アクセスで sys.modules の実体を入れ替えるので、属性の差し替えは効かない。
    real = sys.modules["transformers"]
    fake = types.ModuleType("transformers")
    fake.__dict__["__getattr__"] = lambda name: getattr(real, name)
    fake.pipeline = lambda *args, **kwargs: _FakePipeline(recorder)
    fake.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeProcessor())
    fake.AutoModelForCTC = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _FakeCtcModel(recorder))
    monkeypatch.setitem(sys.modules, "transformers", fake)


def test_silenced_third_party_output_disables_third_party_output_inside(third_party_state):
    with quiet.silenced_third_party_output():
        observed = _observe()

    _assert_silenced(observed)


def test_silenced_third_party_output_restores_state_on_exit(third_party_state):
    with quiet.silenced_third_party_output():
        pass

    _assert_restored()


def test_silenced_third_party_output_restores_state_after_exception(third_party_state):
    with pytest.raises(ValueError):
        with quiet.silenced_third_party_output():
            raise ValueError("boom")

    _assert_restored()


def test_silenced_third_party_output_restores_non_default_original_state(third_party_state):
    hf_utils.disable_progress_bars()
    hf_logging.set_verbosity(hf_logging.INFO)
    transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

    with quiet.silenced_third_party_output():
        _assert_silenced(_observe())

    assert tqdm_std.tqdm.__dict__["__init__"] is PRISTINE_TQDM_INIT
    assert hf_utils.are_progress_bars_disabled() is True
    assert hf_logging.get_verbosity() == hf_logging.INFO
    assert transformers.utils.logging.get_verbosity() == transformers.logging.ERROR


def test_silenced_third_party_output_stays_silenced_until_outermost_exit(third_party_state):
    with quiet.silenced_third_party_output():
        with quiet.silenced_third_party_output():
            _assert_silenced(_observe())
        _assert_silenced(_observe())

    _assert_restored()


def test_silenced_third_party_output_can_be_entered_again_after_exit(third_party_state):
    with quiet.silenced_third_party_output():
        pass
    with quiet.silenced_third_party_output():
        _assert_silenced(_observe())

    _assert_restored()


def test_silenced_third_party_output_restores_per_group_progress_bar_settings(third_party_state):
    import importlib

    hf_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")

    hf_utils.enable_progress_bars()
    hf_utils.disable_progress_bars("group_a")
    hf_utils.enable_progress_bars("group_b")
    saved_states = dict(hf_tqdm.progress_bar_states)

    with quiet.silenced_third_party_output():
        assert hf_utils.are_progress_bars_disabled() is True

    assert dict(hf_tqdm.progress_bar_states) == saved_states
    assert hf_utils.are_progress_bars_disabled() is False
    assert hf_utils.are_progress_bars_disabled("group_a") is True
    assert hf_utils.are_progress_bars_disabled("group_b") is False


def test_silenced_third_party_output_undoes_partial_entry_when_a_step_fails(
        third_party_state, monkeypatch):
    def failing():
        raise RuntimeError("boom")

    monkeypatch.setattr(quiet, "_silence_huggingface_hub", failing)

    with pytest.raises(RuntimeError):
        with quiet.silenced_third_party_output():
            pytest.fail("入場が失敗したのに本体へ入った")

    _assert_restored()
    monkeypatch.undo()
    with quiet.silenced_third_party_output():
        _assert_silenced(_observe())
    _assert_restored()


@pytest.mark.parametrize("module_name", [
    "model_store", "separator", "transcription", "alignment", "recognizer", "front_stage",
    "english_katakana",
])
def test_importing_library_module_leaves_tqdm_init_unchanged(third_party_state, module_name):
    code = (
        "import importlib, tqdm.std\n"
        "original = tqdm.std.tqdm.__dict__['__init__']\n"
        f"importlib.import_module('vocal_analysis.{module_name}')\n"
        "assert tqdm.std.tqdm.__dict__['__init__'] is original\n"
        "assert tqdm.std.tqdm(total=1).disable is False\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )

    assert result.returncode == 0, result.stderr


def test_load_with_progress_silences_third_party_output_only_while_building(third_party_state):
    from vocal_analysis import model_store

    recorder = _Recorder()

    def build():
        recorder.record()
        return "model"

    result = model_store.load_with_progress(
        "repo", "rev", on_progress=None, loading_note="loading", build=build)

    assert result == "model"
    recorder.assert_silenced_in_every_call()
    _assert_restored()


def test_load_with_progress_restores_state_when_build_fails(third_party_state):
    from vocal_analysis import model_store

    def build():
        raise OSError("offline")

    with pytest.raises(OSError):
        model_store.load_with_progress(
            "repo", "rev", on_progress=None, loading_note="loading", build=build)

    _assert_restored()


def test_separate_silences_third_party_output_only_while_audio_separator_runs(
        third_party_state, monkeypatch):
    from vocal_analysis import model_store, separator
    from vocal_analysis.types import AudioPcm

    recorder = _Recorder()

    class _FakeSeparator:
        def __init__(self, **kwargs):
            pass

        def load_model(self, model_filename=None):
            pass

        def separate(self, input_path):
            recorder.record()
            return ["vocals.wav"]

    fake_module = types.ModuleType("audio_separator.separator")
    fake_module.Separator = _FakeSeparator
    monkeypatch.setitem(sys.modules, "audio_separator.separator", fake_module)
    monkeypatch.setattr(model_store, "release_gpu_memory", lambda: None)

    pcm = AudioPcm(samples=np.zeros((1600, 1), dtype=np.float32), sample_rate=16000)
    separator.separate(pcm, "always", on_progress=None)

    recorder.assert_silenced_in_every_call()
    _assert_restored()


def test_transcribe_segment_silences_third_party_output_only_while_pipeline_runs(third_party_state):
    from vocal_analysis import transcription

    recorder = _Recorder()

    text, words = transcription.transcribe_segment(
        _FakePipeline(recorder), np.zeros(1600, dtype=np.float32))

    assert (text, words) == ("あ", None)
    recorder.assert_silenced_in_every_call()
    _assert_restored()


def test_transcribe_text_only_silences_third_party_output_only_while_pipeline_runs(third_party_state):
    from vocal_analysis import transcription

    recorder = _Recorder()

    text = transcription.transcribe_text_only(_FakePipeline(recorder), np.zeros(1600, dtype=np.float32))

    assert text == "あ"
    recorder.assert_silenced_in_every_call()
    _assert_restored()


def test_compute_log_probs_silences_third_party_output_only_while_phoneme_model_runs(
        third_party_state, monkeypatch):
    from vocal_analysis import alignment, model_store

    monkeypatch.setattr(model_store, "select_device", lambda: "cpu")
    recorder = _Recorder()

    log_probs = alignment.compute_log_probs(
        _FakeProcessor(), _FakeCtcModel(recorder), np.zeros(1600, dtype=np.float32))

    assert log_probs.shape == (6, 2)
    recorder.assert_silenced_in_every_call()
    _assert_restored()


def test_recognizer_session_restores_third_party_state_after_transcribe_and_align(
        third_party_state, monkeypatch, tmp_path):
    from vocal_analysis import model_store, transcription
    from vocal_analysis.recognizer import RecognizerSession

    recorder = _Recorder()
    monkeypatch.setattr(model_store, "select_device", lambda: "cpu")
    monkeypatch.setattr(model_store, "release_gpu_memory", lambda: None)
    _install_fake_transformers(monkeypatch, recorder)
    monkeypatch.setattr(transcription, "g2p", lambda text, method=None, **kwargs: ["a"])
    t = np.arange(1920) / 16000
    wav_path = tmp_path / "vocal.wav"
    sf.write(wav_path, (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), 16000)

    with RecognizerSession() as session:
        transcribed = session.transcribe(wav_path)
        session.align(transcribed)

    recorder.assert_silenced_in_every_call(expected_calls=2)
    _assert_restored()


def test_suppress_native_stderr_hides_direct_fd_writes(capfd):
    os.write(2, b"before\n")
    with quiet.suppress_native_stderr():
        os.write(2, b"hidden\n")
    os.write(2, b"after\n")

    captured = capfd.readouterr()
    assert "before" in captured.err
    assert "hidden" not in captured.err
    assert "after" in captured.err


def test_suppress_native_stderr_restores_fd_after_exception(capfd):
    with pytest.raises(ValueError):
        with quiet.suppress_native_stderr():
            raise ValueError("boom")
    os.write(2, b"after-exception\n")

    captured = capfd.readouterr()
    assert "after-exception" in captured.err


def test_suppress_native_stderr_restores_fd_even_if_flush_fails(monkeypatch, capfd):
    class _BrokenStderr:
        def flush(self):
            raise OSError("flush failed")

    monkeypatch.setattr(quiet.sys, "stderr", _BrokenStderr())

    with quiet.suppress_native_stderr():
        pass
    os.write(2, b"after-flush-failure\n")

    captured = capfd.readouterr()
    assert "after-flush-failure" in captured.err


def _try_acquire_stderr_write_lock_from_other_thread():
    result = {}

    def _try_acquire():
        result["acquired"] = quiet.STDERR_WRITE_LOCK.acquire(blocking=False)
        if result["acquired"]:
            quiet.STDERR_WRITE_LOCK.release()

    # RLock は保有スレッド自身なら常に再取得できる。
    t = threading.Thread(target=_try_acquire)
    t.start()
    t.join(2.0)
    return result.get("acquired", False)


def test_suppress_native_stderr_holds_stderr_write_lock_for_other_threads():
    with quiet.suppress_native_stderr():
        acquired = _try_acquire_stderr_write_lock_from_other_thread()

    assert acquired is False


def test_suppress_native_stderr_releases_stderr_write_lock_after_exit():
    with quiet.suppress_native_stderr():
        pass

    assert _try_acquire_stderr_write_lock_from_other_thread() is True
