from dataclasses import dataclass
from pathlib import Path

from vocal_analysis import AnalysisResult, AudioPcm, RmsEnvelope, Segment
from vocal_analysis.front_stage import run_front_stage


@dataclass(frozen=True)
class PipelineResult:
    vocal_wav: Path
    vocal_pcm: AudioPcm
    segments: list[Segment]
    rms: RmsEnvelope
    pcm: AudioPcm
    duration_sec: float
    forced_split: bool
    backends: dict


def _from_front_stage(front, requested_backends) -> PipelineResult:
    analysis: AnalysisResult = front.analysis
    return PipelineResult(
        vocal_wav=analysis.vocal_wav, vocal_pcm=front.vocal_pcm, segments=analysis.segments,
        rms=analysis.rms, pcm=front.pcm, duration_sec=front.duration_sec,
        forced_split=front.forced_split, backends=requested_backends)


def run(input_path, *, separate_vocals, separator_name, content_recognizer_model, retry,
        forced_aligner, sofa_aligner, english_katakana_method, chunking,
        keep_intermediate_dir=None, progress=None) -> PipelineResult:
    front = run_front_stage(
        input_path, separate_vocals=separate_vocals, separator=separator_name,
        content_recognizer_model=content_recognizer_model, retry=retry,
        forced_aligner=forced_aligner, sofa_aligner=sofa_aligner,
        english_katakana_method=english_katakana_method, chunking=chunking,
        keep_intermediate_dir=keep_intermediate_dir,
        on_progress=progress.stage if progress is not None else None)

    return _from_front_stage(front, {
        "separator": separator_name,
        "recognizer": content_recognizer_model.model_id,
        "recognizer_revision": content_recognizer_model.model_revision,
        "forced_aligner": forced_aligner,
        "english_katakana_method": english_katakana_method,
    })
