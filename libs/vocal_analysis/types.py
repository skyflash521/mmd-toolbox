from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# np.ndarray の == は要素ごとの配列を返し、真偽値として評価すると ValueError になる。
@dataclass(eq=False)
class AudioPcm:
    """samples の形状は (フレーム数, チャンネル数)、dtype は float32、値域は [-1, 1]。"""

    samples: np.ndarray
    sample_rate: int


@dataclass
class Segment:
    """phoneme は母音・子音なら IPA、gap なら None。confidence は 0〜1。"""

    type: Literal["vowel", "consonant", "gap"]
    start_sec: float
    end_sec: float
    phoneme: str | None
    confidence: float | None


# np.ndarray の == は要素ごとの配列を返し、真偽値として評価すると ValueError になる。
@dataclass(eq=False)
class RmsEnvelope:
    """times_sec は各フレームの中心時刻。values は 0〜1 へ相対正規化した RMS。dynamic_range_db は
    曲全体の RMS の 95 パーセンタイルと 5 パーセンタイルの dB 差。"""

    times_sec: np.ndarray
    values: np.ndarray
    dynamic_range_db: float


@dataclass(eq=False)
class AnalysisResult:
    vocal_wav: Path
    segments: list[Segment]
    rms: RmsEnvelope
