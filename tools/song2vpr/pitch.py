"""song2vpr のピッチ推定。

分離後ボーカルの PCM から、時刻ごとの基本周波数(F0)・有声/無声・音高を求める。音符へ切る規則
(丸め・無声の扱い・音節の切れ目)は後段が持つので、ここは時刻ごとの値を出すことに徹する。

推定には確率的 YIN を使う。手法は候補3系統の実測比較で選び、半音丸め後の音高一致率が最も高く、
同一入力に対して同じ出力を返すことを条件に確定した(学習モデル系は精度では上回るが、同じ入力でも
実行ごとに出力が変わるため採れない)。
"""

from dataclasses import dataclass

import librosa
import numpy as np

# 時間格子の刻み(秒)。入力の標本化周波数に依らず一定にして、後段が時刻で扱えるようにする。
FRAME_SEC = 0.01

# 歌唱の想定音域(Hz)。C2 から D6 相当。この外は推定の対象にしない。
_FMIN_HZ = 65.0
_FMAX_HZ = 1200.0

# 解析窓長(標本数)。最も低い想定音の2周期を含む必要があり、想定する標本化周波数(16k〜48k)で
# それを満たす最小の2の冪。
_FRAME_LENGTH = 2048


@dataclass(frozen=True)
class PitchTrack:
    """時刻ごとの推定結果。4つの配列は同じ長さで、同じ添字が同じ時刻を指す。

    無声フレームの `f0_hz`・`midi` は NaN。0 のような値で埋めると、後段が音高として読んでしまう。
    """

    times_sec: np.ndarray
    f0_hz: np.ndarray
    voiced: np.ndarray
    midi: np.ndarray


def _to_grid(src_times, values, times):
    """推定器の時間格子上の値を、共通の時間格子へ最近傍で写す。"""
    if len(src_times) == 0:
        return np.full(len(times), np.nan)
    idx = np.searchsorted(src_times, times)
    idx = np.clip(idx, 0, len(src_times) - 1)
    left = np.clip(idx - 1, 0, len(src_times) - 1)
    take_left = np.abs(src_times[left] - times) <= np.abs(src_times[idx] - times)
    return values[np.where(take_left, left, idx)]


def estimate(pcm) -> PitchTrack:
    """分離後ボーカルの PCM から時刻ごとの F0・有声/無声・音高を求める。"""
    samples = np.asarray(pcm.samples, dtype=np.float32)
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    sample_rate = pcm.sample_rate

    hop = max(1, int(round(sample_rate * FRAME_SEC)))
    f0, voiced, _probability = librosa.pyin(
        mono, fmin=_FMIN_HZ, fmax=_FMAX_HZ, sr=sample_rate,
        frame_length=_FRAME_LENGTH, hop_length=hop)
    src_times = librosa.times_like(f0, sr=sample_rate, hop_length=hop)

    # 推定器の刻みは hop を標本数へ丸めた分だけ FRAME_SEC からずれるので、共通格子へ写し直す。
    times = np.arange(int(len(mono) / sample_rate / FRAME_SEC) + 1) * FRAME_SEC
    f0_hz = _to_grid(src_times, f0, times)
    is_voiced = _to_grid(src_times, voiced.astype(float), times) > 0.5
    # 有声と判定されても F0 が得られないフレームは無声として扱う(音高を持てないため)。
    is_voiced &= ~np.isnan(f0_hz)
    f0_hz = np.where(is_voiced, f0_hz, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        midi = 69.0 + 12.0 * np.log2(f0_hz / 440.0)

    return PitchTrack(times_sec=times, f0_hz=f0_hz, voiced=is_voiced, midi=midi)
