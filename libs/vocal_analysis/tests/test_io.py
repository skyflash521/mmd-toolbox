import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FIXTURES = Path(__file__).parent / "fixtures"


def test_reads_wav_without_ffmpeg(monkeypatch):
    from vocal_analysis.io import load_audio
    from vocal_analysis.types import AudioPcm

    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: None)
    pcm = load_audio(FIXTURES / "beep.wav")

    assert isinstance(pcm, AudioPcm)
    assert pcm.sample_rate == 48000
    assert pcm.samples.ndim == 2
    assert pcm.samples.shape[1] == 1


def test_reads_flac_without_ffmpeg(monkeypatch):
    from vocal_analysis.io import TARGET_PEAK, load_audio

    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: None)
    pcm = load_audio(FIXTURES / "beep.flac")

    assert pcm.sample_rate == 48000
    assert pcm.samples.dtype == np.float32
    assert np.max(np.abs(pcm.samples)) == pytest.approx(TARGET_PEAK, abs=1e-4)


def test_reads_ogg_without_ffmpeg(monkeypatch):
    from vocal_analysis.io import TARGET_PEAK, load_audio

    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: None)
    pcm = load_audio(FIXTURES / "beep.ogg")

    assert pcm.sample_rate == 48000
    assert pcm.samples.dtype == np.float32
    assert np.max(np.abs(pcm.samples)) == pytest.approx(TARGET_PEAK, abs=1e-4)


def test_reads_mp3_without_ffmpeg(monkeypatch):
    from vocal_analysis.io import TARGET_PEAK, load_audio

    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: None)
    pcm = load_audio(FIXTURES / "beep.mp3")

    assert pcm.sample_rate == 48000
    assert pcm.samples.dtype == np.float32
    assert np.max(np.abs(pcm.samples)) == pytest.approx(TARGET_PEAK, abs=1e-4)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not found in PATH")
def test_reads_m4a_via_ffmpeg_fallback():
    from vocal_analysis.io import TARGET_PEAK, load_audio

    pcm = load_audio(FIXTURES / "beep.m4a")

    assert pcm.sample_rate == 48000
    assert pcm.samples.shape[1] == 1
    assert pcm.samples.dtype == np.float32
    assert np.max(np.abs(pcm.samples)) == pytest.approx(TARGET_PEAK, abs=1e-4)


def test_missing_input_file_raises_clear_error_without_attempting_ffmpeg(monkeypatch, tmp_path):
    from vocal_analysis.io import AudioLoadError, load_audio

    called = {"ffmpeg": False}
    monkeypatch.setattr(
        "vocal_analysis.io.subprocess.run",
        lambda *a, **k: called.__setitem__("ffmpeg", True))

    missing_path = tmp_path / "does_not_exist.wav"
    with pytest.raises(AudioLoadError) as exc_info:
        load_audio(missing_path)

    assert "見つかりません" in str(exc_info.value)
    assert "ffmpeg" not in str(exc_info.value)
    assert exc_info.value.reason == "not_audio"
    assert called["ffmpeg"] is False


def test_directory_input_raises_clear_error_without_attempting_ffmpeg(monkeypatch, tmp_path):
    from vocal_analysis.io import AudioLoadError, load_audio

    called = {"ffmpeg": False}
    monkeypatch.setattr(
        "vocal_analysis.io.subprocess.run",
        lambda *a, **k: called.__setitem__("ffmpeg", True))

    with pytest.raises(AudioLoadError) as exc_info:
        load_audio(tmp_path)

    assert "見つかりません" in str(exc_info.value)
    assert exc_info.value.reason == "not_audio"
    assert called["ffmpeg"] is False


def test_format_unreadable_by_soundfile_without_ffmpeg_raises_error_naming_ffmpeg(monkeypatch):
    from vocal_analysis.io import AudioLoadError, load_audio

    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: None)

    with pytest.raises(AudioLoadError, match="ffmpeg"):
        load_audio(FIXTURES / "beep.m4a")


def test_missing_ffmpeg_error_has_decoder_missing_reason(monkeypatch):
    from vocal_analysis.io import AudioLoadError, load_audio

    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: None)

    with pytest.raises(AudioLoadError) as exc_info:
        load_audio(FIXTURES / "beep.m4a")
    assert exc_info.value.reason == "decoder_missing"


def test_ffmpeg_conversion_failure_raises_audio_load_error_with_not_audio_reason(monkeypatch):
    from vocal_analysis.io import AudioLoadError, load_audio

    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "vocal_analysis.io.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "ffmpeg")))

    with pytest.raises(AudioLoadError) as exc_info:
        load_audio(FIXTURES / "beep.m4a")
    assert exc_info.value.reason == "not_audio"


def test_ffmpeg_reread_failure_raises_audio_load_error_with_not_audio_reason(monkeypatch, tmp_path):
    from vocal_analysis.io import AudioLoadError, load_audio

    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr("vocal_analysis.io.subprocess.run", lambda *a, **k: None)
    monkeypatch.setattr(
        "vocal_analysis.io.sf.read",
        lambda *a, **k: (_ for _ in ()).throw(sf.LibsndfileError(1, "decode error")))

    with pytest.raises(AudioLoadError) as exc_info:
        load_audio(tmp_path / "broken.m4a")
    assert exc_info.value.reason == "not_audio"


def test_channels_and_sample_rate_preserved_for_stereo(tmp_path):
    from vocal_analysis.io import load_audio

    sr = 44100
    stereo = np.stack(
        [
            np.linspace(-0.5, 0.5, sr, dtype=np.float32),
            np.linspace(0.5, -0.5, sr, dtype=np.float32),
        ],
        axis=1,
    )
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, sr)

    pcm = load_audio(path)

    assert pcm.sample_rate == sr
    assert pcm.samples.shape[1] == 2


def test_target_peak_is_fixed_at_0_95():
    from vocal_analysis.io import TARGET_PEAK

    assert TARGET_PEAK == 0.95


def test_peak_normalization_reaches_target_peak():
    from vocal_analysis.io import TARGET_PEAK, _normalize_peak

    samples = np.array([[0.2], [-0.4], [0.1]], dtype=np.float32)

    normalized = _normalize_peak(samples)

    assert np.max(np.abs(normalized)) == pytest.approx(TARGET_PEAK)


def test_peak_normalization_uses_global_max_across_channels():
    from vocal_analysis.io import TARGET_PEAK, _normalize_peak

    samples = np.array([[0.1, 0.8], [-0.2, -0.4], [0.05, 0.1]], dtype=np.float32)

    normalized = _normalize_peak(samples)

    scale = TARGET_PEAK / 0.8
    assert np.allclose(normalized, samples * scale, atol=1e-6)


def test_peak_normalization_is_invariant_to_uniform_gain():
    from vocal_analysis.io import _normalize_peak

    base = np.array([[0.3], [-0.6], [0.15]], dtype=np.float32)
    gained = base * 2.5

    assert np.allclose(_normalize_peak(base), _normalize_peak(gained), atol=1e-6)


def test_peak_normalization_leaves_silence_unchanged():
    from vocal_analysis.io import _normalize_peak

    silence = np.zeros((10, 1), dtype=np.float32)

    assert np.array_equal(_normalize_peak(silence), silence)


def test_load_audio_normalizes_peak_to_target_as_float32():
    from vocal_analysis.io import TARGET_PEAK, load_audio

    pcm = load_audio(FIXTURES / "beep.wav")

    assert np.max(np.abs(pcm.samples)) == pytest.approx(TARGET_PEAK, abs=1e-4)
    assert pcm.samples.dtype == np.float32
