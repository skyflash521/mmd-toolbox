"""正規化中間形式の公開データ型(vocal_analysis.md §2.1)。

各ステージ(S0/S2/S3)をつなぐ中間形式と共有出力コンテナを、特定の外部ツールに依存しない
正規化モデルとして公開する。時刻は秒(float)、numpy 配列は np.ndarray。S1 出力(ボーカル WAV)は
専用 dataclass を設けず pathlib.Path で表す(出力形式は分離器に従う)。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# eq=False: np.ndarray フィールドを持つ型は既定の == が配列の要素比較になり真偽が曖昧
# (ValueError)になるため、識別子比較(object 既定)にして事故を防ぐ。値の一致は利用先が
# np.allclose 等で明示的に確かめる。
@dataclass(eq=False)
class AudioPcm:
    """S0 出力(入力レベル正規化済み PCM)。

    samples は形状 (フレーム数, チャンネル数) の float32・値域 [-1, 1]。チャンネル数は
    samples.shape[1](ch/SR を保持し mono 化・再サンプリングしない。§3)。
    """

    samples: np.ndarray
    sample_rate: int


@dataclass
class Segment:
    """S2 出力の要素(時刻付き音素セグメント)。

    type は母音/子音/gap の3種。phoneme は IPA(母音/子音のみ。gap は None)。confidence は
    任意(0〜1)。セグメント列は時間順・隙間なく連続・非重複で全時間軸を被覆する。
    """

    type: Literal["vowel", "consonant", "gap"]
    start_sec: float
    end_sec: float
    phoneme: str | None
    confidence: float | None


# eq=False: AudioPcm と同じく np.ndarray フィールドを持つため(理由は AudioPcm 参照)。
@dataclass(eq=False)
class RmsEnvelope:
    """S3 出力(相対正規化した強弱エンベロープ。§6.1)。

    times_sec は各フレーム中心時刻(秒)、values は相対正規化済み RMS。dynamic_range_db は曲全体の
    ダイナミックレンジ(95/5 パーセンタイル RMS の dB 差。ゲイン不変)で、正規化済み values からは
    復元できないため別フィールドで持つ。
    """

    times_sec: np.ndarray
    values: np.ndarray
    dynamic_range_db: float


# eq=False: RmsEnvelope(eq=False)を内包し、既定の == が意味を持たないため。
@dataclass(eq=False)
class AnalysisResult:
    """共有出力コンテナ(§2.1)。利用先は必要な部分集合だけを使う。"""

    vocal_wav: Path
    segments: list[Segment]
    rms: RmsEnvelope
