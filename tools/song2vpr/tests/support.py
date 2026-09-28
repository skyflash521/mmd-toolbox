from pathlib import Path

import numpy as np

from song2vpr.pipeline import PipelineResult
from vocal_analysis import AudioPcm
from vocal_analysis.rms import compute_rms

RATE = 22050


def front_stage_result(duration_sec=0.3):
    pcm = AudioPcm(samples=np.zeros((int(RATE * duration_sec), 1), dtype=np.float32),
                   sample_rate=RATE)
    return PipelineResult(vocal_wav=Path("vocal.wav"), vocal_pcm=pcm, segments=[],
                          rms=compute_rms(pcm), pcm=pcm, duration_sec=duration_sec,
                          forced_split=False, backends={})
