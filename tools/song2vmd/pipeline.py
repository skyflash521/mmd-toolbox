"""song2vmd パイプライン統合(song2vmd.md §4章・§6章)。

vocal_analysis(S0読込・S1分離・S2認識・S3 RMS)から song2vmd 自身の口形イベント確定(events)・
長尺分割(chunking)・モーフ生成(morphs)までを1回のパイプライン実行として結線する。song2vmd 固有の
判断(写像・閾値等)は追加せず、既存モジュールの呼び出し順序と受け渡しに徹する。

長尺分割(--max-duration 超過時)は、処理資源対策のため S1分離・S2認識をチャンク単位(前後1.0秒の
オーバーラップ付き)で行う。境界決定には分離前の生音声のRMSを使う(分離という重い処理を境界決定の
ためだけに追加で行わずに済むため)。RMS(S3)は「曲全体基準」で正規化する必要がある(song2vmd.md
§6.4)ため、各チャンクの分離済みボーカル音声から重複領域を除いた「核」区間(隣接チャンクとの境界から
その次の境界まで)だけを取り出して全チャンク分を連結し、その連結した曲全体のボーカル音声に対して
1回だけRMSを算出する(チャンクごとに個別正規化しない)。この連結のため、各チャンクのボーカルWAVは
vocal_analysis.io.load_audio のピーク正規化を経ずに読み込む(正規化はチャンクごとに異なる倍率を
かけてしまい、チャンク間の相対的な強弱を壊すため。曲全体を1回だけ読む単一実行経路では、一様な
倍率が全体にかかるだけなのでRMSの相対正規化(ゲイン不変)を壊さない)。
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from lipsync import GenerationParams
from vocal_analysis import io as _va_io
from vocal_analysis import recognizer as _va_recognizer
from vocal_analysis import rms as _va_rms
from vocal_analysis import separator as _va_separator
from vocal_analysis.types import AudioPcm

from . import chunking, events, morphs, report

_CHUNK_OVERLAP_SEC = 1.0  # song2vmd.md §6.6(初期値)。


class IntermediateWriteError(Exception):
    """--keep-intermediate の中間生成物書き込み失敗(song2vmd.md 5.3・11章・12.3)。"""


@dataclass(frozen=True)
class PipelineResult:
    """pipeline.run() の戻り値。"""

    document: object  # VmdDocument(モーフキーのみ)
    diagnostics: report.Diagnostics
    sample_rate: int
    channels: int


def _slice_pcm(pcm, start_sec, end_sec):
    """[start_sec, end_sec) の時刻範囲を切り出した AudioPcm を返す。"""
    sr = pcm.sample_rate
    start_idx = max(0, round(start_sec * sr))
    end_idx = min(len(pcm.samples), round(end_sec * sr))
    return AudioPcm(samples=pcm.samples[start_idx:end_idx], sample_rate=sr)


def _concat_pcm(pcms):
    """複数の AudioPcm を時間順に連結した AudioPcm を返す。"""
    sr = pcms[0].sample_rate
    samples = np.concatenate([p.samples for p in pcms], axis=0)
    return AudioPcm(samples=samples, sample_rate=sr)


def _read_pcm_raw(path):
    """ピーク正規化を経ずに音声ファイルを読み込む(チャンクの核区間連結専用)。

    vocal_analysis.io.load_audio は毎回ピーク正規化(目標値固定)を適用するため、チャンクごとに
    個別に読み込むと倍率がチャンクごとに異なり、曲全体基準のRMS(song2vmd.md 6.4)が壊れる。
    """
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return AudioPcm(samples=samples, sample_rate=sample_rate)


def _build_generation_params(openness, style_gen):
    """presets の OpennessParams/StyleGenParams から lipsync.GenerationParams を組み立てる
    (song2vmd.md §8.1・§8.2)。

    vowel_scale は presets.resolve が --vowel-gain 乗算まで済ませた最終6要素。プリセットが持つ
    生成パラメータ一式を明示的に渡し、lipsync の生成既定への暗黙依存を残さない(song2vmd.md §8.1)。
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
    """段の開始を報告する(song2vmd.md 12.1「各段は開始時に最低1本のprogressを出す」)。

    呼び出しは各段の実処理より前に置く。elapsed は段開始時点の経過秒(0)。
    """
    if progress is not None:
        progress.stage(stage, done=done, total=total, note=note, elapsed=0.0)


def _save_intermediate(keep_intermediate_dir, pcm, vocal_pcm, segments):
    """--keep-intermediate 指定時に中間生成物を保存する(song2vmd.md 5.2・5.3)。

    S0正規化PCM(input_normalized.wav)・分離後ボーカルWAV(vocal.wav。長尺分割時は核区間を
    連結した曲全体分)・S2認識結果(segments.json)を、指定ディレクトリへ保存する。書き込み失敗
    (権限・ディスク等のI/O失敗)は IntermediateWriteError として送出する(song2vmd.md 11章・12.3)。
    `sf.write` はlibsndfileが開くため失敗を `OSError` でなく `sf.SoundFileError` 系で送出する
    (`mkdir`/`write_text` の失敗は `OSError`)ため、両方を捕捉する。
    """
    try:
        directory = Path(keep_intermediate_dir)
        directory.mkdir(parents=True, exist_ok=True)
        sf.write(directory / "input_normalized.wav", pcm.samples, pcm.sample_rate)
        sf.write(directory / "vocal.wav", vocal_pcm.samples, vocal_pcm.sample_rate)
        (directory / "segments.json").write_text(
            json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, sf.SoundFileError) as e:
        raise IntermediateWriteError(str(e)) from e


def run(input_path, *, separate_vocals, separator_name, content_recognizer_model, max_duration_sec,
        use_n_morph, intensity_curve, silence_on, openness, style_gen,
        style_name, model_name, forced_aligner, sofa_aligner,
        keep_intermediate_dir=None, progress=None):
    """song2vmd の音声→VMDパイプラインを実行する(song2vmd.md 4章・6章)。

    content_recognizer_model は vocal_analysis.recognizer.recognize が受け取る
    ContentRecognizerModel(vocal_analysis.md §5.2・§8.3)。forced_aligner・sofa_aligner は同じ
    recognize が受け取るS2強制アライメント段のバックエンド選択(vocal_analysis.md §5.3・§8.3)。
    keep_intermediate_dir を渡すと中間生成物(正規化PCM・分離後ボーカルWAV・認識結果)をその
    ディレクトリへ保存する(5.2の--keep-intermediate)。省略時(既定None)は何も保存しない。
    """
    _report_stage(progress, "load")
    pcm = _va_io.load_audio(input_path)
    duration_sec = len(pcm.samples) / pcm.sample_rate

    if max_duration_sec <= 0 or duration_sec <= max_duration_sec:
        segments, rms_envelope, vocal_pcm = _run_single(
            pcm, separate_vocals, content_recognizer_model, forced_aligner, sofa_aligner, progress)
    else:
        segments, rms_envelope, vocal_pcm = _run_chunked(
            pcm, duration_sec, separate_vocals, content_recognizer_model, max_duration_sec,
            forced_aligner, sofa_aligner, progress)

    if keep_intermediate_dir is not None:
        _save_intermediate(keep_intermediate_dir, pcm, vocal_pcm, segments)

    _report_stage(progress, "events")
    mouth_events, event_diag = events.confirm_mouth_events(
        segments, rms_envelope, open_lo=openness.open_lo, open_hi=openness.open_hi,
        open_max=openness.open_max, intensity_curve=intensity_curve, silence_on=silence_on,
        use_n_morph=use_n_morph)

    _report_stage(progress, "generate")
    gen_params = _build_generation_params(openness, style_gen)
    document = morphs.build_vmd_document(mouth_events, gen_params, model_name)

    diagnostics = report.build_diagnostics(
        segments=segments, mouth_events=mouth_events, event_diagnostics=event_diag,
        backends={"separator": separator_name, "recognizer": content_recognizer_model.model_id,
                  "forced_aligner": forced_aligner},
        style=style_name, separated=(separate_vocals != "never"), duration_sec=duration_sec,
        keys=len(document.morph))

    return PipelineResult(
        document=document, diagnostics=diagnostics, sample_rate=pcm.sample_rate,
        channels=pcm.samples.shape[1])


def _run_single(pcm, separate_vocals, content_recognizer_model, forced_aligner, sofa_aligner, progress):
    """長尺分割なしの単一実行(song2vmd.md 6.6の対象外の通常経路)。"""
    _report_stage(progress, "separate")
    vocal_path = _va_separator.separate(pcm, separate_vocals)
    _report_stage(progress, "recognize")
    segments = _va_recognizer.recognize(
        vocal_path, content_recognizer_model=content_recognizer_model,
        forced_aligner=forced_aligner, sofa_aligner=sofa_aligner)
    _report_stage(progress, "rms")
    vocal_pcm = _va_io.load_audio(vocal_path)
    rms_envelope = _va_rms.compute_rms(vocal_pcm)
    return segments, rms_envelope, vocal_pcm


def _run_chunked(pcm, duration_sec, separate_vocals, content_recognizer_model, max_duration_sec,
                  forced_aligner, sofa_aligner, progress):
    """長尺分割ありの実行(song2vmd.md 6.6)。境界決定は分離前の生音声RMSを使う。"""
    raw_rms = _va_rms.compute_rms(pcm)
    boundaries = chunking.find_chunk_boundaries(
        duration_sec, raw_rms.times_sec, raw_rms.values, max_duration_sec=max_duration_sec)
    edges = [0.0] + boundaries + [duration_sec]
    n = len(edges) - 1

    chunk_offsets_sec = []
    chunk_segments_list = []
    vocal_core_chunks = []
    for i in range(n):
        core_start, core_end = edges[i], edges[i + 1]
        pad_start = max(0.0, core_start - _CHUNK_OVERLAP_SEC) if i > 0 else 0.0
        pad_end = min(duration_sec, core_end + _CHUNK_OVERLAP_SEC) if i < n - 1 else duration_sec
        chunk_offsets_sec.append(pad_start)

        chunk_pcm = _slice_pcm(pcm, pad_start, pad_end)
        # done は「このチャンクを始める時点までに完了したチャンク数」(0始まり。song2vmd.md 12.1の
        # 「処理済み」の語義)。1個目のチャンクを始める時点(i=0)ではまだ0個も完了していない。
        _report_stage(progress, "separate", done=i, total=n)
        vocal_path = _va_separator.separate(chunk_pcm, separate_vocals)
        _report_stage(progress, "recognize", done=i, total=n)
        chunk_segments_list.append(
            _va_recognizer.recognize(
                vocal_path, content_recognizer_model=content_recognizer_model,
                forced_aligner=forced_aligner, sofa_aligner=sofa_aligner))

        chunk_vocal_pcm = _read_pcm_raw(vocal_path)
        vocal_core_chunks.append(_slice_pcm(chunk_vocal_pcm, core_start - pad_start, core_end - pad_start))

    merged_segments = chunking.merge_chunk_segments(chunk_segments_list, chunk_offsets_sec, boundaries)
    whole_vocal_pcm = _concat_pcm(vocal_core_chunks)
    _report_stage(progress, "rms")
    rms_envelope = _va_rms.compute_rms(whole_vocal_pcm)
    return merged_segments, rms_envelope, whole_vocal_pcm
