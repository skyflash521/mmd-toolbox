"""song2vpr のテストが共有する、音声前段の共有出力の代役。

パイプラインを差し替えるテストは、後段が読む形の値を返す必要がある(CLI は戻り値を使う)。
テスト側から相対取り込みで使う。
"""

from pathlib import Path

import numpy as np

from song2vpr.pipeline import PipelineResult
from vocal_analysis import AudioPcm
from vocal_analysis.rms import compute_rms

RATE = 22050


def front_stage_result(duration_sec=0.3):
    """短い無音を持つ共有出力。後段は音符を作らずに通る。"""
    pcm = AudioPcm(samples=np.zeros((int(RATE * duration_sec), 1), dtype=np.float32),
                   sample_rate=RATE)
    return PipelineResult(vocal_wav=Path("vocal.wav"), vocal_pcm=pcm, segments=[],
                          rms=compute_rms(pcm), pcm=pcm, duration_sec=duration_sec,
                          forced_split=False)
