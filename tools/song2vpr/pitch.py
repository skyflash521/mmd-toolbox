from dataclasses import dataclass

import librosa
import numpy as np

FRAME_SEC = 0.01

_SINGING_FMIN_HZ = 65.0
_SINGING_FMAX_HZ = 1200.0

_PYIN_FRAME_LENGTH_SAMPLES = 2048


@dataclass(frozen=True)
class PitchTrack:
    """4つの配列は同じ長さで、同じ添字が同じ時刻を指す。無声フレームの f0_hz・midi は NaN。"""

    times_sec: np.ndarray
    f0_hz: np.ndarray
    voiced: np.ndarray
    midi: np.ndarray


def _nearest_on_grid(src_times, values, grid_times):
    if len(src_times) == 0:
        return np.full(len(grid_times), np.nan)
    idx = np.searchsorted(src_times, grid_times)
    idx = np.clip(idx, 0, len(src_times) - 1)
    left = np.clip(idx - 1, 0, len(src_times) - 1)
    take_left = np.abs(src_times[left] - grid_times) <= np.abs(src_times[idx] - grid_times)
    return values[np.where(take_left, left, idx)]


def estimate(pcm) -> PitchTrack:
    samples = np.asarray(pcm.samples, dtype=np.float32)
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    sample_rate = pcm.sample_rate

    hop = max(1, int(round(sample_rate * FRAME_SEC)))
    f0, voiced, _probability = librosa.pyin(
        mono, fmin=_SINGING_FMIN_HZ, fmax=_SINGING_FMAX_HZ, sr=sample_rate,
        frame_length=_PYIN_FRAME_LENGTH_SAMPLES, hop_length=hop)
    pyin_times = librosa.times_like(f0, sr=sample_rate, hop_length=hop)

    grid_times = np.arange(int(len(mono) / sample_rate / FRAME_SEC) + 1) * FRAME_SEC
    f0_hz = _nearest_on_grid(pyin_times, f0, grid_times)
    is_voiced = _nearest_on_grid(pyin_times, voiced.astype(float), grid_times) > 0.5
    is_voiced &= ~np.isnan(f0_hz)
    f0_hz = np.where(is_voiced, f0_hz, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        midi = 69.0 + 12.0 * np.log2(f0_hz / 440.0)

    return PitchTrack(times_sec=grid_times, f0_hz=f0_hz, voiced=is_voiced, midi=midi)
