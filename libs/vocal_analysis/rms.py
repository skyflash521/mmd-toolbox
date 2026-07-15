"""S3 強弱エンベロープ。

ボーカルWAVのRMSエンベロープを numpy/scipy で算出する。外部ツールに依存しない。フレーム化とパーセンタイル
相対正規化・dynamic_range_db 算出はいずれも入力ゲインに不変(S0 のピーク正規化と同じく、正の一様ゲインに
対して結果が変わらない)。
"""

import numpy as np

from .types import AudioPcm, RmsEnvelope

FRAME_SEC = 0.025
HOP_SEC = 0.010


def compute_rms(pcm: AudioPcm) -> RmsEnvelope:
    """ボーカルの相対正規化RMSエンベロープを算出する。"""
    times_sec, raw_rms = _frame_rms(pcm.samples, pcm.sample_rate)
    values = _normalize_rms(raw_rms)
    dynamic_range_db = _dynamic_range_db(raw_rms)
    return RmsEnvelope(times_sec=times_sec, values=values, dynamic_range_db=dynamic_range_db)


def _frame_rms(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """フレーム化した生RMS(パーセンタイル正規化前)と各フレーム中心時刻を返す。

    フレーム長・ホップは対象サンプルレートで round() してサンプル数へ換算する。フレームは時刻0から
    信号長を超えない範囲でホップ幅ずつずらして生成し、末尾の不完全フレームは切り捨てる(ゼロ詰めしない)。
    複数チャンネルはフレーム内の全チャンネル・全サンプルをまとめてRMSを取る(チャンネルごとの独立計算・
    平均はしない)。
    """
    frame_samples = round(FRAME_SEC * sample_rate)
    hop_samples = round(HOP_SEC * sample_rate)
    length = samples.shape[0]

    starts = np.arange(0, length - frame_samples + 1, hop_samples)
    if starts.size == 0:
        return np.array([]), np.array([])

    times_sec = (starts + frame_samples / 2) / sample_rate
    raw_rms = np.array(
        [np.sqrt(np.mean(np.square(samples[start : start + frame_samples]))) for start in starts]
    )
    return times_sec, raw_rms


def _normalize_rms(raw_rms: np.ndarray) -> np.ndarray:
    """曲全体のパーセンタイル(p10/p90)で相対正規化する。"""
    if raw_rms.size == 0:
        return raw_rms
    p10 = np.percentile(raw_rms, 10, method="linear")
    p90 = np.percentile(raw_rms, 90, method="linear")
    if p90 == p10:
        return np.zeros_like(raw_rms)
    return np.clip((raw_rms - p10) / (p90 - p10), 0.0, 1.0)


def _dynamic_range_db(raw_rms: np.ndarray) -> float:
    """曲全体の95/5パーセンタイル RMS の dB 差を算出する。"""
    if raw_rms.size == 0:
        return 0.0
    p5 = np.percentile(raw_rms, 5, method="linear")
    p95 = np.percentile(raw_rms, 95, method="linear")
    if p95 == 0:
        return 0.0
    if p5 == 0:
        p5 = p95 * 1e-6
    return float(20 * np.log10(p95 / p5))
