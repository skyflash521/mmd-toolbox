"""song2vpr パイプライン統合。

音声前段(vocal_analysis の前段実行エンジン)を呼び、song2vpr 固有の後段が使う共有出力をまとめて返す。
前段の実行(S0読込・S1分離・S2認識・S3 RMS の呼び出し順序、長尺分割、失敗の分類、診断用中間生成物の
書き出し)は共有側が持つので、ここは前段へ渡す設定の受け渡しと、返った共有出力の取り出しに徹する。
"""

from dataclasses import dataclass
from pathlib import Path

from vocal_analysis import AnalysisResult, AudioPcm, RmsEnvelope, Segment
from vocal_analysis.front_stage import run_front_stage


@dataclass(frozen=True)
class PipelineResult:
    """pipeline.run() の戻り値。

    ボーカルWAVはパスだけを持つ。前段が既に読んだ PCM を vocal_pcm で受け取るので、後段はこのパスを
    読み直さない(読み直しの失敗分類を二重に持たないため)。分離前の pcm は、テンポと拍子の推定が
    分離前の音を使うので波形そのものを保持する。
    """

    vocal_wav: Path
    vocal_pcm: AudioPcm
    segments: list[Segment]
    rms: RmsEnvelope
    pcm: AudioPcm
    duration_sec: float
    forced_split: bool
    # 診断へ出す、音声前段へ渡したバックエンド選択の設定。そのまま載せ、その処理が実際に
    # 走ったかは問わない。
    backends: dict


def _from_front_stage(front, backends) -> PipelineResult:
    analysis: AnalysisResult = front.analysis
    return PipelineResult(
        vocal_wav=analysis.vocal_wav, vocal_pcm=front.vocal_pcm, segments=analysis.segments,
        rms=analysis.rms, pcm=front.pcm, duration_sec=front.duration_sec,
        forced_split=front.forced_split, backends=backends)


def run(input_path, *, separate_vocals, separator_name, content_recognizer_model, retry,
        forced_aligner, sofa_aligner, english_katakana_method, chunking,
        keep_intermediate_dir=None, progress=None) -> PipelineResult:
    """音声前段を1回実行し、後段が使う共有出力を返す。

    separate_vocals から english_katakana_method まで、および chunking・keep_intermediate_dir は
    音声前段の実行エンジンがそのまま受け取る設定で、ここでは解釈しない。progress は前段の進捗の
    中継先で、None なら報告しない。
    """
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
