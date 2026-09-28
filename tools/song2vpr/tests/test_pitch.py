import numpy as np
import pytest

from song2vpr import pitch
from vocal_analysis import AudioPcm

SR = 22050


def _harmonic_tone(freq_hz, seconds=1.0, sample_rate=SR, channels=1, amplitude=0.5):
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


def test_track_arrays_share_one_length():
    track = pitch.estimate(_harmonic_tone(440.0))
    assert len(track.times_sec) == len(track.f0_hz) == len(track.voiced) == len(track.midi)
    assert len(track.times_sec) > 0


def test_times_are_increasing_and_start_at_zero():
    track = pitch.estimate(_harmonic_tone(440.0))
    assert track.times_sec[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(track.times_sec) > 0)


def test_track_covers_the_input_duration():
    track = pitch.estimate(_harmonic_tone(440.0, seconds=2.0))
    assert track.times_sec[-1] == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize("freq_hz,expected_midi", [
    pytest.param(220.0, 57, id="A3"),
    pytest.param(440.0, 69, id="A4"),
    pytest.param(880.0, 81, id="A5"),
])
def test_steady_tone_maps_to_its_midi_note(freq_hz, expected_midi):
    track = pitch.estimate(_harmonic_tone(freq_hz))
    voiced_midi = track.midi[track.voiced]
    assert voiced_midi.size > 0
    assert np.median(np.rint(voiced_midi)) == expected_midi


def test_midi_halfway_between_semitones_is_not_rounded():
    track = pitch.estimate(_harmonic_tone(440.0 * 2 ** (0.5 / 12)))
    voiced_midi = track.midi[track.voiced]
    assert voiced_midi.size > 0
    assert np.median(voiced_midi) == pytest.approx(69.5, abs=0.2)


def test_pitch_is_estimated_per_frame_not_collapsed_to_one_value():
    track = pitch.estimate(_concat(_harmonic_tone(440.0, seconds=0.5), _silence(0.2),
                                   _harmonic_tone(330.0, seconds=0.5)))
    early = track.voiced & (track.times_sec < 0.45)
    late = track.voiced & (track.times_sec > 0.75)
    assert early.any() and late.any()
    assert np.median(np.rint(track.midi[early])) == 69
    assert np.median(np.rint(track.midi[late])) == 64


def test_midi_is_derived_from_f0():
    track = pitch.estimate(_harmonic_tone(440.0))
    voiced = track.voiced
    expected = 69.0 + 12.0 * np.log2(track.f0_hz[voiced] / 440.0)
    assert np.allclose(track.midi[voiced], expected, atol=1e-6)


def test_unvoiced_frames_carry_nan_for_f0_and_midi():
    track = pitch.estimate(_concat(_silence(0.5), _harmonic_tone(440.0, seconds=0.5)))
    assert np.all(np.isnan(track.f0_hz[~track.voiced]))
    assert np.all(np.isnan(track.midi[~track.voiced]))


def test_silence_is_unvoiced_and_tone_is_voiced():
    track = pitch.estimate(_concat(_silence(0.5), _harmonic_tone(440.0, seconds=1.0)))
    early = track.times_sec < 0.4
    late = track.times_sec > 0.7
    assert not track.voiced[early].any()
    assert track.voiced[late].mean() > 0.8


def test_all_silence_has_no_voiced_frame():
    track = pitch.estimate(_silence(1.0))
    assert not track.voiced.any()


def test_stereo_input_gives_the_same_pitch_as_mono():
    mono = pitch.estimate(_harmonic_tone(440.0, channels=1))
    stereo = pitch.estimate(_harmonic_tone(440.0, channels=2))
    assert np.median(np.rint(stereo.midi[stereo.voiced])) == np.median(np.rint(mono.midi[mono.voiced]))


@pytest.mark.parametrize("sample_rate", [16000, 22050, 44100])
def test_time_grid_does_not_depend_on_the_sample_rate(sample_rate):
    track = pitch.estimate(_harmonic_tone(440.0, seconds=1.0, sample_rate=sample_rate))
    steps = np.diff(track.times_sec)
    assert np.allclose(steps, steps[0], atol=1e-9)
    assert steps[0] == pytest.approx(pitch.FRAME_SEC, abs=1e-9)


@pytest.mark.parametrize("sample_rate", [16000, 44100])
def test_pitch_does_not_depend_on_the_sample_rate(sample_rate):
    track = pitch.estimate(_harmonic_tone(440.0, sample_rate=sample_rate))
    assert np.median(np.rint(track.midi[track.voiced])) == 69


def test_quiet_input_is_still_voiced_and_gain_does_not_change_the_pitch():
    quiet = pitch.estimate(_harmonic_tone(440.0, amplitude=0.05))
    loud = pitch.estimate(_harmonic_tone(440.0, amplitude=0.5))
    assert quiet.voiced.any() and loud.voiced.any()
    assert np.median(np.rint(quiet.midi[quiet.voiced])) == np.median(np.rint(loud.midi[loud.voiced]))


def test_same_input_gives_the_same_output():
    pcm = _concat(_harmonic_tone(440.0, seconds=0.5), _silence(0.2), _harmonic_tone(330.0, seconds=0.5))
    first = pitch.estimate(pcm)
    second = pitch.estimate(pcm)
    assert np.array_equal(first.voiced, second.voiced)
    assert np.array_equal(np.nan_to_num(first.f0_hz, nan=-1.0),
                          np.nan_to_num(second.f0_hz, nan=-1.0))
    assert np.array_equal(np.nan_to_num(first.midi, nan=-1.0),
                          np.nan_to_num(second.midi, nan=-1.0))
