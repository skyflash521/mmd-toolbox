import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _make_pcm(sample_rate=8000, duration_sec=0.2, channels=2):
    from vocal_analysis.types import AudioPcm

    frames = int(sample_rate * duration_sec)
    rng = np.random.default_rng(0)
    samples = ((rng.random((frames, channels)) - 0.5) * 0.5).astype(np.float32)
    return AudioPcm(samples=samples, sample_rate=sample_rate)


def test_separate_never_mode_returns_input_unseparated():
    from vocal_analysis.separator import separate

    pcm = _make_pcm()

    result = separate(pcm, mode="never")

    assert isinstance(result, Path)
    assert result.exists()
    samples, sr = sf.read(result, dtype="float32", always_2d=True)
    assert sr == pcm.sample_rate
    assert np.allclose(samples, pcm.samples, atol=1e-4)


def test_separate_never_mode_does_not_call_separator_factory(monkeypatch):
    from vocal_analysis import separator as separator_module

    def _fail(output_dir):
        raise AssertionError("never モードで分離器ファクトリを呼んではならない")

    monkeypatch.setattr(separator_module, "_build_separator", _fail)

    result = separator_module.separate(_make_pcm(), mode="never")

    assert result.exists()


def test_separate_invalid_mode_raises_value_error():
    from vocal_analysis.separator import separate

    with pytest.raises(ValueError):
        separate(_make_pcm(), mode="bogus")


def test_separate_unknown_separator_id_raises_separation_error():
    from vocal_analysis.separator import SeparationError, separate

    with pytest.raises(SeparationError):
        separate(_make_pcm(), mode="always", separator="bogus-separator")


def test_separate_unknown_separator_id_raises_even_when_not_separating():
    from vocal_analysis.separator import SeparationError, separate

    with pytest.raises(SeparationError):
        separate(_make_pcm(), mode="never", separator="bogus-separator")


def test_published_separator_ids_all_have_implementations():
    from vocal_analysis import SEPARATOR_IDS
    from vocal_analysis import separator as separator_module

    assert set(SEPARATOR_IDS) == set(separator_module._SEPARATOR_IMPLS)


class _FakeSeparator:
    def __init__(self, calls):
        self._calls = calls

    def load_model(self, model_filename):
        self._calls["model_filename"] = model_filename

    def separate(self, audio_file_path):
        self._calls["audio_file_path"] = audio_file_path
        out = Path(audio_file_path).parent / "vocals_output.wav"
        out.write_bytes(b"")
        # audio-separator は output_dir 相対のファイル名だけを返す。
        return [out.name]


def test_separate_always_mode_returns_existing_vocal_wav_path(monkeypatch):
    from vocal_analysis import separator as separator_module

    calls = {}

    def fake_build_separator(output_dir):
        calls["output_dir"] = output_dir
        return _FakeSeparator(calls)

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)

    result = separator_module.separate(_make_pcm(), mode="always")

    assert result == Path(calls["audio_file_path"]).parent / "vocals_output.wav"
    assert result.exists()
    assert calls["model_filename"] == "htdemucs_ft.yaml"


def test_separate_dispatches_registered_separator_id(monkeypatch):
    from vocal_analysis import DEFAULT_SEPARATOR
    from vocal_analysis import separator as separator_module

    calls = {}

    monkeypatch.setattr(
        separator_module, "_build_separator", lambda output_dir: _FakeSeparator(calls))

    result = separator_module.separate(_make_pcm(), mode="always", separator=DEFAULT_SEPARATOR)

    assert result.exists()
    assert calls["model_filename"] == "htdemucs_ft.yaml"


def test_separate_uses_pinned_separator_config(monkeypatch):
    from vocal_analysis import SEPARATOR_CONFIG
    from vocal_analysis import separator as separator_module

    calls = {}

    def fake_build_separator(output_dir):
        return _FakeSeparator(calls)

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)

    separator_module.separate(_make_pcm(), mode="always")

    assert calls["model_filename"] == SEPARATOR_CONFIG.model_filename


def test_separate_registers_work_dir_cleanup_that_removes_it(monkeypatch):
    import shutil

    from vocal_analysis import separator as separator_module

    registered = []
    monkeypatch.setattr(
        separator_module.atexit, "register",
        lambda fn, *args, **kwargs: registered.append((fn, args, kwargs)))

    result = separator_module.separate(_make_pcm(), mode="never")
    work_dir = result.parent
    assert work_dir.exists()

    assert len(registered) == 1
    fn, args, kwargs = registered[0]
    assert fn is shutil.rmtree
    assert args == (work_dir,)
    assert kwargs == {"ignore_errors": True}
    fn(*args, **kwargs)

    assert not work_dir.exists()


def test_separate_missing_library_raises_clear_error(monkeypatch):
    from vocal_analysis import separator as separator_module

    def fake_build_separator(output_dir):
        raise ImportError("no audio_separator")

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)

    with pytest.raises(separator_module.SeparationError):
        separator_module.separate(_make_pcm(), mode="always")


def _fake_download_file_if_not_exists(self, url, output_path):
    import audio_separator.separator.separator as as_mod

    bar = as_mod.tqdm(total=100, unit="iB", unit_scale=True)
    bar.update(40)
    bar.update(60)
    bar.close()


def test_separate_relays_download_progress_and_clears_it_before_separating(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.separator as as_mod

    from vocal_analysis import separator as separator_module

    calls = {}
    notes = []

    class _FakeSeparatorWithDownload(_FakeSeparator):
        def load_model(self, model_filename):
            as_mod.Separator.download_file_if_not_exists(
                self, "https://example.invalid/htdemucs_ft.yaml", "/models/htdemucs_ft.yaml")
            super().load_model(model_filename)

        def separate(self, audio_file_path):
            notes.append("SEPARATE_STARTED")
            return super().separate(audio_file_path)

    def fake_build_separator(output_dir):
        return _FakeSeparatorWithDownload(calls)

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)
    monkeypatch.setattr(as_mod.Separator, "download_file_if_not_exists", _fake_download_file_if_not_exists)

    separator_module.separate(_make_pcm(), mode="always", on_progress=notes.append)

    assert any("40" in n for n in notes)
    assert any("100" in n for n in notes)
    assert any("htdemucs_ft.yaml" in n for n in notes)
    assert notes.index("") < notes.index("SEPARATE_STARTED")


def test_separate_download_progress_labels_multi_file_models_by_real_filename(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.separator as as_mod

    from vocal_analysis import separator as separator_module

    calls = {}
    notes = []
    filenames = ["04573f0d-f3cf25b2.th", "htdemucs_ft.yaml"]

    class _FakeSeparatorWithMultiFileDownload(_FakeSeparator):
        def load_model(self, model_filename):
            for name in filenames:
                as_mod.Separator.download_file_if_not_exists(
                    self, f"https://example.invalid/{name}", f"/models/{name}")
            super().load_model(model_filename)

    def fake_build_separator(output_dir):
        return _FakeSeparatorWithMultiFileDownload(calls)

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)
    monkeypatch.setattr(as_mod.Separator, "download_file_if_not_exists", _fake_download_file_if_not_exists)

    separator_module.separate(_make_pcm(), mode="always", on_progress=notes.append)

    file1_notes = [n for n in notes if filenames[0] in n]
    file2_notes = [n for n in notes if filenames[1] in n]
    assert any("100" in n for n in file1_notes)
    assert any("100" in n for n in file2_notes)
    assert not any(filenames[1] in n for n in file1_notes)


def test_separate_shows_loading_note_but_no_download_note_when_no_download_happens(monkeypatch):
    pytest.importorskip("audio_separator")
    from vocal_analysis import SEPARATOR_CONFIG
    from vocal_analysis import separator as separator_module

    calls = {}
    notes = []

    def fake_build_separator(output_dir):
        return _FakeSeparator(calls)

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)

    separator_module.separate(_make_pcm(), mode="always", on_progress=notes.append)

    assert notes == [f"モデル読み込み中: {SEPARATOR_CONFIG.model_filename}"]


def test_load_model_with_progress_restores_original_tqdm_and_download_method(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.separator as as_mod

    from vocal_analysis.separator import _load_model_with_progress

    original_tqdm = as_mod.tqdm
    original_download = as_mod.Separator.download_file_if_not_exists

    class _FakeSep:
        def load_model(self, model_filename):
            assert as_mod.tqdm is not original_tqdm
            assert as_mod.Separator.download_file_if_not_exists is not original_download

    _load_model_with_progress(_FakeSep(), on_progress=lambda note: None)

    assert as_mod.tqdm is original_tqdm
    assert as_mod.Separator.download_file_if_not_exists is original_download


def test_load_model_with_progress_skips_relay_when_on_progress_is_none():
    pytest.importorskip("audio_separator")
    import audio_separator.separator.separator as as_mod

    from vocal_analysis import SEPARATOR_CONFIG
    from vocal_analysis.separator import _load_model_with_progress

    original = as_mod.tqdm
    calls = {}

    class _FakeSep:
        def load_model(self, model_filename):
            calls["model_filename"] = model_filename
            assert as_mod.tqdm is original

    downloaded = _load_model_with_progress(_FakeSep(), on_progress=None)

    assert downloaded is False
    assert calls["model_filename"] == SEPARATOR_CONFIG.model_filename


def test_load_model_with_progress_loads_anyway_and_shows_loading_note_when_patch_target_is_gone(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.separator as as_mod

    from vocal_analysis import SEPARATOR_CONFIG
    from vocal_analysis.separator import _load_model_with_progress

    monkeypatch.delattr(as_mod, "tqdm")

    calls = {}

    class _FakeSep:
        def load_model(self, model_filename):
            calls["model_filename"] = model_filename

    notes = []
    downloaded = _load_model_with_progress(_FakeSep(), on_progress=notes.append)

    assert downloaded is False
    assert calls["model_filename"] == SEPARATOR_CONFIG.model_filename
    assert notes == [f"モデル読み込み中: {SEPARATOR_CONFIG.model_filename}"]


def test_separate_with_progress_relays_inference_progress_scaled_to_percent(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.architectures.demucs_separator as demucs_mod

    from vocal_analysis.separator import _separate_with_progress

    def fake_apply_model(*args, set_progress_bar=None, **kwargs):
        set_progress_bar(0.1, 0.4)
        set_progress_bar(0.1, 0.8)
        return "dummy_source"

    monkeypatch.setattr(demucs_mod, "apply_model", fake_apply_model)

    class _FakeSep:
        def separate(self, audio_file_path):
            demucs_mod.apply_model(set_progress_bar=None)
            return ["vocals_output.wav"]

    notes = []
    _separate_with_progress(_FakeSep(), Path("input.wav"), on_progress=notes.append)

    assert notes == ["分離中: 50%", "分離中: 100%"]


def test_separate_with_progress_restores_original_apply_model(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.architectures.demucs_separator as demucs_mod

    from vocal_analysis.separator import _separate_with_progress

    original_apply_model = demucs_mod.apply_model

    class _FakeSep:
        def separate(self, audio_file_path):
            assert demucs_mod.apply_model is not original_apply_model
            return ["vocals_output.wav"]

    _separate_with_progress(_FakeSep(), Path("input.wav"), on_progress=lambda note: None)

    assert demucs_mod.apply_model is original_apply_model


def test_separate_with_progress_skips_relay_when_on_progress_is_none(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.architectures.demucs_separator as demucs_mod

    from vocal_analysis.separator import _separate_with_progress

    original_apply_model = demucs_mod.apply_model

    class _FakeSep:
        def separate(self, audio_file_path):
            assert demucs_mod.apply_model is original_apply_model
            return ["vocals_output.wav"]

    _separate_with_progress(_FakeSep(), Path("input.wav"), on_progress=None)


def test_separate_with_progress_separates_anyway_and_clears_note_when_hook_argument_is_gone(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.architectures.demucs_separator as demucs_mod

    from vocal_analysis.separator import _separate_with_progress

    def apply_model_without_hook(*args, **kwargs):
        raise AssertionError("set_progress_bar を受け取れない apply_model を差し替えてはならない")

    monkeypatch.setattr(demucs_mod, "apply_model", apply_model_without_hook)

    calls = {}

    class _FakeSep:
        def separate(self, audio_file_path):
            calls["audio_file_path"] = audio_file_path
            assert demucs_mod.apply_model is apply_model_without_hook
            return ["vocals_output.wav"]

    notes = []
    result = _separate_with_progress(_FakeSep(), Path("input.wav"), on_progress=notes.append)

    assert result == ["vocals_output.wav"]
    assert calls["audio_file_path"] == "input.wav"
    assert notes == [""]


def test_separate_with_progress_separates_anyway_when_patch_target_is_gone(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.architectures.demucs_separator as demucs_mod

    from vocal_analysis.separator import _separate_with_progress

    monkeypatch.delattr(demucs_mod, "apply_model")

    class _FakeSep:
        def separate(self, audio_file_path):
            return ["vocals_output.wav"]

    notes = []
    result = _separate_with_progress(_FakeSep(), Path("input.wav"), on_progress=notes.append)

    assert result == ["vocals_output.wav"]
    assert notes == [""]


def test_separate_with_progress_ignores_unexpected_callback_arity(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.architectures.demucs_separator as demucs_mod

    from vocal_analysis.separator import _separate_with_progress

    def fake_apply_model(*args, set_progress_bar=None, **kwargs):
        set_progress_bar(0.1)
        set_progress_bar(0.1, 0.4)
        return "dummy_source"

    monkeypatch.setattr(demucs_mod, "apply_model", fake_apply_model)

    class _FakeSep:
        def separate(self, audio_file_path):
            demucs_mod.apply_model(set_progress_bar=None)
            return ["vocals_output.wav"]

    notes = []
    result = _separate_with_progress(_FakeSep(), Path("input.wav"), on_progress=notes.append)

    assert result == ["vocals_output.wav"]
    assert notes == ["分離中: 50%"]


def test_load_model_with_progress_downloads_anyway_when_call_form_changed(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.separator as as_mod

    from vocal_analysis.separator import _load_model_with_progress

    def _renamed_argument_download(self, url, destination):
        bar = as_mod.tqdm(total=100, unit="iB", unit_scale=True)
        bar.update(100)
        bar.close()

    monkeypatch.setattr(as_mod.Separator, "download_file_if_not_exists", _renamed_argument_download)

    class _FakeSep:
        def load_model(self, model_filename):
            as_mod.Separator.download_file_if_not_exists(
                self, url="https://example.invalid/x.yaml", destination="/models/x.yaml")

    notes = []
    downloaded = _load_model_with_progress(_FakeSep(), on_progress=notes.append)

    assert downloaded is True
    assert any("100%" in note for note in notes)


def test_build_separator_configures_real_separator_with_pinned_values():
    pytest.importorskip("audio_separator")
    from vocal_analysis import SEPARATOR_CONFIG
    from vocal_analysis.separator import _build_separator

    with tempfile.TemporaryDirectory() as tmp_dir:
        sep = _build_separator(Path(tmp_dir))

    assert sep.output_single_stem == SEPARATOR_CONFIG.output_single_stem
    # Separator は demucs_params を arch_specific_params["Demucs"] へ格納する。
    assert sep.arch_specific_params["Demucs"]["shifts"] == SEPARATOR_CONFIG.shifts
    assert sep.logger.level == logging.CRITICAL
