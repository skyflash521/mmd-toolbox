"""S0 入力読み込みのテスト。

soundfile 経由で読める形式(WAV/FLAC/OGG/mp3)は、短い合成音の形式別フィクスチャで読み込みを
確認する。これら最低保証の形式は ffmpeg 未検出(_find_ffmpeg を差し替え)でも読めることを直接
確認し、ffmpeg フォールバック経由の形式(m4a)は実環境の ffmpeg 未検出時のみ skip する(ffmpeg
はオプション)。レベル正規化は合成配列でファイル入出力に依存せず決定論的に検証する。
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FIXTURES = Path(__file__).parent / "fixtures"
_XFAIL_REASON_ATTR = pytest.mark.xfail(reason="impl pending: AudioLoadErrorのreason属性", strict=True)


def test_reads_wav_without_ffmpeg(monkeypatch):
    from vocal_analysis.io import load_audio
    from vocal_analysis.types import AudioPcm

    # soundfile(同梱 libsndfile)が読める最低保証形式は ffmpeg 無しでも読める。
    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: None)
    pcm = load_audio(FIXTURES / "beep.wav")

    assert isinstance(pcm, AudioPcm)
    assert pcm.sample_rate == 48000
    assert pcm.samples.ndim == 2
    assert pcm.samples.shape[1] == 1  # beep はモノラル


def test_reads_flac_without_ffmpeg(monkeypatch):
    from vocal_analysis.io import TARGET_PEAK, load_audio

    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: None)
    pcm = load_audio(FIXTURES / "beep.flac")

    assert pcm.sample_rate == 48000
    # 公開型契約(float32)とピーク正規化(target_peak 0.95)を形式ごとに確認し、特定形式だけ dtype/正規化が
    # 抜け落ちる誤実装を検出する(beep.flac は非無音で実ピーク約0.20)。
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

    # m4a/aac は soundfile(同梱 libsndfile)で読めないため ffmpeg フォールバック経路を通る。
    # 実環境の ffmpeg 検出に依存する結合テストなので、ここだけは実際の PATH を使う。
    pcm = load_audio(FIXTURES / "beep.m4a")

    assert pcm.sample_rate == 48000
    assert pcm.samples.shape[1] == 1
    # ffmpeg 経路も soundfile 経路と同じ公開型契約(float32)とピーク正規化を満たす
    # (beep.m4a は非無音で実ピーク約0.31。経路ごとに dtype/正規化が抜け落ちる誤実装を検出する)。
    assert pcm.samples.dtype == np.float32
    assert np.max(np.abs(pcm.samples)) == pytest.approx(TARGET_PEAK, abs=1e-4)


def test_missing_ffmpeg_raises_clear_error(monkeypatch):
    from vocal_analysis.io import AudioLoadError, load_audio

    # soundfile で読めない形式で ffmpeg も見つからない場合、黙って失敗させず分かるエラーにする。
    # ffmpeg 検出点(_find_ffmpeg)そのものを差し替えて、実環境の ffmpeg 有無に依存させない。
    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: None)

    with pytest.raises(AudioLoadError, match="ffmpeg"):
        load_audio(FIXTURES / "beep.m4a")


@_XFAIL_REASON_ATTR
def test_missing_ffmpeg_error_has_decoder_missing_reason(monkeypatch):
    from vocal_analysis.io import AudioLoadError, load_audio

    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: None)

    with pytest.raises(AudioLoadError) as exc_info:
        load_audio(FIXTURES / "beep.m4a")
    assert exc_info.value.reason == "decoder_missing"


@_XFAIL_REASON_ATTR
def test_ffmpeg_conversion_failure_raises_audio_load_error_with_not_audio_reason(monkeypatch):
    from vocal_analysis.io import AudioLoadError, load_audio

    # soundfile で読めない形式で ffmpeg は見つかるが、変換自体が失敗する場合(壊れた入力ファイル等)。
    # 復号器が無いのではなく入力が壊れているケースなので、decoder_missingでなくnot_audioになる。
    monkeypatch.setattr("vocal_analysis.io._find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "vocal_analysis.io.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "ffmpeg")))

    with pytest.raises(AudioLoadError) as exc_info:
        load_audio(FIXTURES / "beep.m4a")
    assert exc_info.value.reason == "not_audio"


@_XFAIL_REASON_ATTR
def test_ffmpeg_reread_failure_raises_audio_load_error_with_not_audio_reason(monkeypatch, tmp_path):
    from vocal_analysis.io import AudioLoadError, load_audio

    # ffmpeg変換自体は成功するが、変換後ファイルの再読み込みが失敗する場合(壊れた入力ファイル等)。
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

    # S1 はステレオ原音の方が分離品質が高いため、ここで mono化・低レート化しない。
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

    # target_peak は 0.95 に確定し、実装が独自に変えない。
    assert TARGET_PEAK == 0.95


def test_peak_normalization_reaches_target_peak():
    from vocal_analysis.io import TARGET_PEAK, _normalize_peak

    samples = np.array([[0.2], [-0.4], [0.1]], dtype=np.float32)

    normalized = _normalize_peak(samples)

    assert np.max(np.abs(normalized)) == pytest.approx(TARGET_PEAK)


def test_peak_normalization_uses_global_max_across_channels():
    from vocal_analysis.io import TARGET_PEAK, _normalize_peak

    # peak は全チャンネル・全フレームの絶対値の最大(チャンネルごとに別々に正規化しない)。
    # 右chの方が大きいピーク(0.8)を持ち、左ch(最大0.2)はそれに引きずられて正規化される。
    samples = np.array([[0.1, 0.8], [-0.2, -0.4], [0.05, 0.1]], dtype=np.float32)

    normalized = _normalize_peak(samples)

    scale = TARGET_PEAK / 0.8
    assert np.allclose(normalized, samples * scale, atol=1e-6)


def test_peak_normalization_is_homogeneous_to_gain():
    from vocal_analysis.io import _normalize_peak

    # 正の一様ゲイン g に対し正規化後PCMが不変(斉次)。
    base = np.array([[0.3], [-0.6], [0.15]], dtype=np.float32)
    gained = base * 2.5

    assert np.allclose(_normalize_peak(base), _normalize_peak(gained), atol=1e-6)


def test_peak_normalization_leaves_silence_unchanged():
    from vocal_analysis.io import _normalize_peak

    # peak == 0(完全無音)は0除算を避けて入力をそのまま返す。
    silence = np.zeros((10, 1), dtype=np.float32)

    assert np.array_equal(_normalize_peak(silence), silence)


def test_load_audio_normalizes_peak_to_target():
    from vocal_analysis.io import TARGET_PEAK, load_audio

    # beep フィクスチャは非無音(実ピーク約0.2)なので、正規化後は target_peak に達するはず
    # (無音回避や過小ゲインなど正規化が事実上効かない誤実装をここで検出する)。
    pcm = load_audio(FIXTURES / "beep.wav")

    assert np.max(np.abs(pcm.samples)) == pytest.approx(TARGET_PEAK, abs=1e-4)
    # AudioPcm.samples は float32。soundfile の既定 float64 のまま返す誤実装を検出する。
    assert pcm.samples.dtype == np.float32
