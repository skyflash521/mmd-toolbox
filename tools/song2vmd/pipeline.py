from dataclasses import dataclass

from lipsync import GenerationParams
from vmd import VmdDocument
from vocal_analysis import DEFAULT_ENGLISH_KATAKANA_METHOD as _DEFAULT_ENGLISH_KATAKANA_METHOD
from vocal_analysis.front_stage import run_front_stage

from . import events, morphs, report


@dataclass(frozen=True)
class PipelineResult:
    document: VmdDocument
    diagnostics: report.Diagnostics
    sample_rate: int
    channels: int


def _build_generation_params(openness, style_gen):
    return GenerationParams(
        open_cap=openness.open_max,
        vowel_scale=style_gen.vowel_scale,
        attack_frames=style_gen.attack_frames,
        release_frames=style_gen.release_frames,
        min_hold_frames=style_gen.min_hold_frames,
        triangle_min_frames=style_gen.triangle_min_frames,
        coartic_overlap_max=style_gen.coartic_overlap_max,
        anticipation_frames=style_gen.anticipation_frames,
        legato_valley_shallow=style_gen.legato_valley_shallow,
        legato_valley_deep=style_gen.legato_valley_deep,
        legato_valley_slope=style_gen.legato_valley_slope,
        exaggeration=style_gen.exaggeration,
        vibrato_threshold=style_gen.vibrato_threshold,
        vibrato_amp=style_gen.vibrato_amp,
        vibrato_period=style_gen.vibrato_period,
    )


def _report_stage_start(progress, stage):
    if progress is not None:
        progress.stage(stage, done=0, total=None, note="", elapsed=0.0)


def run(input_path, *, separate_vocals, separator_name, content_recognizer_model,
        retry, chunking,
        use_n_morph, intensity_curve, silence_on, openness, style_gen,
        style_name, model_name, forced_aligner, sofa_aligner,
        english_katakana_method=_DEFAULT_ENGLISH_KATAKANA_METHOD,
        keep_intermediate_dir=None, progress=None):
    front = run_front_stage(
        input_path, separate_vocals=separate_vocals, separator=separator_name,
        content_recognizer_model=content_recognizer_model, retry=retry,
        forced_aligner=forced_aligner, sofa_aligner=sofa_aligner,
        english_katakana_method=english_katakana_method, chunking=chunking,
        keep_intermediate_dir=keep_intermediate_dir,
        on_progress=progress.stage if progress is not None else None)

    _report_stage_start(progress, "events")
    mouth_events, event_diag, mora_event_group_sizes = events.confirm_mouth_events(
        front.analysis.segments, front.analysis.rms,
        open_lo=openness.open_lo, open_hi=openness.open_hi,
        open_max=openness.open_max, intensity_curve=intensity_curve, silence_on=silence_on,
        use_n_morph=use_n_morph)

    _report_stage_start(progress, "generate")
    gen_params = _build_generation_params(openness, style_gen)
    document = morphs.build_vmd_document(mouth_events, gen_params, model_name)

    diagnostics = report.build_diagnostics(
        segments=front.analysis.segments, mouth_events=mouth_events, event_diagnostics=event_diag,
        mora_event_group_sizes=mora_event_group_sizes,
        backends={"separator": separator_name, "recognizer": content_recognizer_model.model_id,
                  "forced_aligner": forced_aligner,
                  "english_katakana_method": english_katakana_method},
        style=style_name, separated=(separate_vocals != "never"),
        duration_sec=front.duration_sec,
        keys=len(document.morph), forced_split=front.forced_split)

    return PipelineResult(
        document=document, diagnostics=diagnostics, sample_rate=front.pcm.sample_rate,
        channels=front.pcm.samples.shape[1])
