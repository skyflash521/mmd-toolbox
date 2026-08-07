"""song2vpr のピッチ推定のテスト。

分離後ボーカルの PCM から、時刻ごとの基本周波数(F0)・有声/無声・音高を得ることを検証する。
音符へ切る規則は後段が持つので、ここでは時刻ごとの値だけを見る。
"""

import numpy as np
import pytest

from vocal_analysis import AudioPcm

pitch = pytest.importorskip("song2vpr.pitch", reason="ピッチ推定のモジュールがまだ無い")

SR = 22050


def _tone(freq_hz, seconds=1.0, sample_rate=SR, channels=1, amplitude=0.5):
    """倍音を持つ合成音(正弦波だけだと推定器が基本波を取り違えやすい)。"""
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    wave = sum(amplitude / (k + 1) * np.sin(2 * np.pi * freq_hz * (k + 1) * t) for k in range(4))
    samples = np.stack([wave.astype(np.float32)] * channels, axis=1)
    return AudioPcm(samples=samples, sample_rate=sample_rate)


def _silence(seconds=1.0, sample_rate=SR, channels=1):
    n = int(seconds * sample_rate)
    return AudioPcm(samples=np.zeros((n, channels), dtype=np.float32), sample_rate=sample_rate)


def _concat(*pcms):
    return AudioPcm(samples=np.concatenate([p.samples for p in pcms], axis=0),
                    sample_rate=pcms[0].sample_rate)


# --- 時刻ごとの値の形 --------------------------------------------------------


def test_track_arrays_share_one_length():
    track = pitch.estimate(_tone(440.0))
    assert len(track.times_sec) == len(track.f0_hz) == len(track.voiced) == len(track.midi)
    assert len(track.times_sec) > 0


def test_times_are_increasing_and_start_at_zero():
    track = pitch.estimate(_tone(440.0))
    assert track.times_sec[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(track.times_sec) > 0)


def test_track_covers_the_input_duration():
    """時間格子は入力の尺をほぼ覆う(末尾を大きく取りこぼさない)。"""
    track = pitch.estimate(_tone(440.0, seconds=2.0))
    assert track.times_sec[-1] == pytest.approx(2.0, abs=0.1)


# --- 音高 --------------------------------------------------------------------


@pytest.mark.parametrize("freq_hz,expected_midi", [
    (220.0, 57),  # A3
    (440.0, 69),  # A4
    (880.0, 81),  # A5
])
def test_steady_tone_maps_to_its_midi_note(freq_hz, expected_midi):
    """定常音の音高は、その周波数の MIDI ノート番号へ写る。"""
    track = pitch.estimate(_tone(freq_hz))
    voiced_midi = track.midi[track.voiced]
    assert voiced_midi.size > 0
    assert np.median(np.rint(voiced_midi)) == expected_midi


def test_midi_is_not_rounded():
    """音高は丸めずに持つ(丸めは音符へ切る段の責務)。"""
    # A4 と A#4 の中間(半音の 1/2 上)。丸めた値を持つ実装ではこの差が消える。
    track = pitch.estimate(_tone(440.0 * 2 ** (0.5 / 12)))
    voiced_midi = track.midi[track.voiced]
    assert voiced_midi.size > 0
    assert np.median(voiced_midi) == pytest.approx(69.5, abs=0.2)


def test_pitch_follows_the_time_axis():
    """音高は時刻ごとに求める(有声フレームを代表値1つへ潰さない)。"""
    track = pitch.estimate(_concat(_tone(440.0, seconds=0.5), _silence(0.2),
                                   _tone(330.0, seconds=0.5)))
    early = track.voiced & (track.times_sec < 0.45)
    late = track.voiced & (track.times_sec > 0.75)
    assert early.any() and late.any()
    assert np.median(np.rint(track.midi[early])) == 69  # A4
    assert np.median(np.rint(track.midi[late])) == 64  # E4(330Hz)


def test_midi_and_f0_agree():
    """音高は F0 から導く(別々に持たない)。"""
    track = pitch.estimate(_tone(440.0))
    voiced = track.voiced
    expected = 69.0 + 12.0 * np.log2(track.f0_hz[voiced] / 440.0)
    assert np.allclose(track.midi[voiced], expected, atol=1e-6)


def test_unvoiced_frames_carry_no_value():
    """無声フレームは F0 も音高も持たない(0 のような値で埋めない)。"""
    track = pitch.estimate(_concat(_silence(0.5), _tone(440.0, seconds=0.5)))
    assert np.all(np.isnan(track.f0_hz[~track.voiced]))
    assert np.all(np.isnan(track.midi[~track.voiced]))


# --- 有声/無声 ---------------------------------------------------------------


def test_silence_is_unvoiced_and_tone_is_voiced():
    track = pitch.estimate(_concat(_silence(0.5), _tone(440.0, seconds=1.0)))
    early = track.times_sec < 0.4
    late = track.times_sec > 0.7
    assert not track.voiced[early].any()
    assert track.voiced[late].mean() > 0.8


def test_all_silence_has_no_voiced_frame():
    track = pitch.estimate(_silence(1.0))
    assert not track.voiced.any()


# --- 入力の受け取り方 --------------------------------------------------------


def test_stereo_input_is_accepted():
    """分離後ボーカルは多チャネルでも来る。混ぜて1本として扱う。"""
    mono = pitch.estimate(_tone(440.0, channels=1))
    stereo = pitch.estimate(_tone(440.0, channels=2))
    assert np.median(np.rint(stereo.midi[stereo.voiced])) == np.median(np.rint(mono.midi[mono.voiced]))


@pytest.mark.parametrize("sample_rate", [16000, 22050, 44100])
def test_time_grid_does_not_depend_on_the_sample_rate(sample_rate):
    """時間格子は秒で決まる(入力の標本化周波数で刻みが変わらない)。"""
    track = pitch.estimate(_tone(440.0, seconds=1.0, sample_rate=sample_rate))
    steps = np.diff(track.times_sec)
    assert np.allclose(steps, steps[0], atol=1e-9)
    assert steps[0] == pytest.approx(pitch.FRAME_SEC, abs=1e-9)


@pytest.mark.parametrize("sample_rate", [16000, 44100])
def test_pitch_does_not_depend_on_the_sample_rate(sample_rate):
    track = pitch.estimate(_tone(440.0, sample_rate=sample_rate))
    assert np.median(np.rint(track.midi[track.voiced])) == 69


def test_gain_does_not_change_the_pitch():
    """入力ゲインで音高は変わらない。"""
    quiet = pitch.estimate(_tone(440.0, amplitude=0.05))
    loud = pitch.estimate(_tone(440.0, amplitude=0.5))
    assert quiet.voiced.any() and loud.voiced.any()  # 小音量でも有声として拾える
    assert np.median(np.rint(quiet.midi[quiet.voiced])) == np.median(np.rint(loud.midi[loud.voiced]))


# --- 決定論 ------------------------------------------------------------------


def test_same_input_gives_the_same_output():
    """同一入力に対し決定論的(採用手法に課した条件そのもの)。"""
    pcm = _concat(_tone(440.0, seconds=0.5), _silence(0.2), _tone(330.0, seconds=0.5))
    first = pitch.estimate(pcm)
    second = pitch.estimate(pcm)
    assert np.array_equal(first.voiced, second.voiced)
    assert np.array_equal(np.nan_to_num(first.f0_hz, nan=-1.0),
                          np.nan_to_num(second.f0_hz, nan=-1.0))
    assert np.array_equal(np.nan_to_num(first.midi, nan=-1.0),
                          np.nan_to_num(second.midi, nan=-1.0))
