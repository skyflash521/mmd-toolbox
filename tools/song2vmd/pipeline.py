"""song2vmd パイプライン統合。

音声前段(vocal_analysis の前段実行エンジン)から song2vmd 自身の口形イベント確定(events)・
モーフ生成(morphs)までを1回のパイプライン実行として結線する。song2vmd 固有の判断(写像・閾値等)は
追加せず、既存モジュールの呼び出し順序と受け渡しに徹する。

前段の実行(S0読込・S1分離・S2認識・S3 RMS の呼び出し順序、長尺分割、失敗の分類、診断用中間生成物の
書き出し)は共有側が持つので、ここは前段へ渡す設定の組み立てと、返った共有出力を後段へ渡すことだけを行う。
"""

from dataclasses import dataclass

from lipsync import GenerationParams
from vocal_analysis import DEFAULT_ENGLISH_KATAKANA_METHOD as _DEFAULT_ENGLISH_KATAKANA_METHOD
from vocal_analysis.front_stage import run_front_stage

from . import events, morphs, report


@dataclass(frozen=True)
class PipelineResult:
    """pipeline.run() の戻り値。"""

    document: object  # VmdDocument(モーフキーのみ)
    diagnostics: report.Diagnostics
    sample_rate: int
    channels: int


def _build_generation_params(openness, style_gen):
    """presets の OpennessParams/StyleGenParams から lipsync.GenerationParams を組み立てる。

    vowel_scale は presets.resolve が --vowel-gain 乗算まで済ませた最終6要素。プリセットが持つ
    生成パラメータは明示的に渡す。モーラ境界の谷(半幅・最低間隔)はプリセットが値を持たないため
    渡さず、lipsync の生成既定をそのまま使う。
    """
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


def _report_stage(progress, stage, *, done=0, total=None, note=""):
    """段の開始を報告する(各段は開始時に最低1本のprogressを出す)。

    呼び出しは各段の実処理より前に置く。elapsed は段開始時点の経過秒(0)。
    """
    if progress is not None:
        progress.stage(stage, done=done, total=total, note=note, elapsed=0.0)


def run(input_path, *, separate_vocals, separator_name, content_recognizer_model,
        retry, chunking,
        use_n_morph, intensity_curve, silence_on, openness, style_gen,
        style_name, model_name, forced_aligner, sofa_aligner,
        english_katakana_method=_DEFAULT_ENGLISH_KATAKANA_METHOD,
        keep_intermediate_dir=None, progress=None):
    """song2vmd の音声→VMDパイプラインを実行する。

    separator_name・content_recognizer_model・retry・forced_aligner・sofa_aligner・
    english_katakana_method・chunking・keep_intermediate_dir は音声前段の実行エンジンがそのまま
    受け取る設定で、ここでは解釈しない。progress は前段の進捗の中継先で、None なら報告しない。
    """
    front = run_front_stage(
        input_path, separate_vocals=separate_vocals, separator=separator_name,
        content_recognizer_model=content_recognizer_model, retry=retry,
        forced_aligner=forced_aligner, sofa_aligner=sofa_aligner,
        english_katakana_method=english_katakana_method, chunking=chunking,
        keep_intermediate_dir=keep_intermediate_dir,
        on_progress=progress.stage if progress is not None else None)

    _report_stage(progress, "events")
    mouth_events, event_diag, mora_event_group_sizes = events.confirm_mouth_events(
        front.analysis.segments, front.analysis.rms,
        open_lo=openness.open_lo, open_hi=openness.open_hi,
        open_max=openness.open_max, intensity_curve=intensity_curve, silence_on=silence_on,
        use_n_morph=use_n_morph)

    _report_stage(progress, "generate")
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
