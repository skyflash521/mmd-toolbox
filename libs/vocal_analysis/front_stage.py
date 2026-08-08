"""S0〜S3 を通しで実行する前段実行エンジン。

呼び出し順序・失敗の分類・進捗の中継・長尺分割・診断用中間生成物の書き出しをここへ集め、利用先は
実行ポリシーと保存先を渡すだけで同じ前段を使えるようにする。
"""

import atexit
import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import chunking as _chunking
from . import io as _io
from . import recognizer as _recognizer
from . import rms as _rms
from . import separator as _separator
from .types import AnalysisResult, AudioPcm


class IntermediateWriteError(Exception):
    """診断用中間生成物の書き込み失敗。"""


class IntermediateReadError(Exception):
    """内部生成ファイル(分離後ボーカルWAV)の読み直し失敗。

    利用者入力の読み込み失敗と同じ例外で送出すると、利用先が利用者入力の不備として報告して
    しまうため、対象ファイルを持つ別の例外に分ける。
    """

    def __init__(self, message, *, path):
        super().__init__(message)
        self.path = path


class StageExecutionError(Exception):
    """外部推論(分離・認識)の呼び出しから漏れた、分類の定まっていない例外。

    どの工程で失敗したかを stage に持ち、利用先が工程失敗として報告できるようにする。
    """

    def __init__(self, message, *, stage):
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class FrontStageResult:
    """run_front_stage() の戻り値。"""

    analysis: AnalysisResult
    pcm: AudioPcm
    vocal_pcm: AudioPcm
    duration_sec: float
    forced_split: bool


# 呼び出し元で分類が定まっている例外。工程失敗へ写像すると、それぞれの終了コードの分岐を
# 上書きしてしまうのでそのまま送出させる。KeyboardInterrupt は Exception の派生でないため
# ここに挙げなくても通る。
_CLASSIFIED_STAGE_EXCEPTIONS = (
    _separator.SeparationError,
    _recognizer.RecognitionError,
    _io.AudioLoadError,
    IntermediateReadError,
    IntermediateWriteError,
)


def _run_inference(stage, func, *args, map_failures=True, **kwargs):
    """外部推論の呼び出しを1回だけ包み、分類の定まっていない例外を工程失敗へ写像する。

    包むのは呼び出しそのものに限る。前後の処理(音量解析・内部生成ファイルの読み直し等)まで
    含めると、それらの失敗が工程の失敗に化ける。map_failures が偽なら写像せずそのまま呼ぶ
    (その呼び出しが実際には外部推論を行わない場合に使う)。
    """
    if not map_failures:
        return func(*args, **kwargs)
    try:
        return func(*args, **kwargs)
    except _CLASSIFIED_STAGE_EXCEPTIONS:
        raise
    except _ProgressCallbackFailed as e:
        raise e.cause from None
    except Exception as e:
        # 取得・実行のどちらで失敗したかは、分類されずに漏れてきた例外からは判別できないので
        # 名乗らない(判別できる失敗は各段の専用例外の文言が示す)。
        raise StageExecutionError(f"{type(e).__name__}: {e}", stage=stage) from e


class _ProgressCallbackFailed(Exception):
    """進捗コールバックが送出した例外を、推論の失敗と区別して運ぶための内部の入れ物。"""

    def __init__(self, cause):
        super().__init__()
        self.cause = cause


def _read_intermediate(path, reader):
    """内部生成ファイルを reader で読み、失敗を IntermediateReadError へ包む。

    読み手(ピーク正規化ありの読み込みと生読み込み)によらず、失敗の意味は同じ内部生成ファイルの
    読み直し失敗なので、送出する例外も同じにする。捕捉するのは読み込み自体の失敗を表す例外だけで、
    実装の不具合を表す例外(型の誤り・メモリ不足等)は呼び出し元が想定外として扱えるようそのまま通す。
    """
    try:
        return reader(path)
    except (_io.AudioLoadError, sf.SoundFileError, OSError) as e:
        raise IntermediateReadError(str(e), path=path) from e


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

    load_audio は毎回ピーク正規化(目標値固定)を適用するため、チャンクごとに個別に読み込むと
    倍率がチャンクごとに異なり、曲全体基準のRMSが壊れる。
    """
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return AudioPcm(samples=samples, sample_rate=sample_rate)


def _report_stage(on_progress, stage, *, done=0, total=None, note=""):
    """段の開始を報告する(各段は開始時に最低1本の進捗を出す)。

    呼び出しは各段の実処理より前に置く。elapsed は段開始時点の経過秒(0)。
    """
    if on_progress is not None:
        on_progress(stage, done=done, total=total, note=note, elapsed=0.0)


def _model_download_progress(on_progress, stage, *, done, total):
    """separate()・recognize() の on_progress へ渡すコールバックを組み立てる。

    モデル初回取得のダウンロード進捗文言、および分離・書き起こし・アライメントの各実処理が
    進行中であることを示す文言を、呼び出し時点の完了数/総数(分割時はチャンク進捗)を保ったまま
    報告の note へ反映する(通知のたびに 0/None へ巻き戻さない)。elapsed はこのコールバックを
    組み立てた時点(直前の段の開始報告と同時)からの実経過秒とする(段開始からの経過秒として扱う)。
    報告先が無ければ None を返し、存在しない進捗表示への橋渡しコールバックを作らない。

    コールバック自身の失敗は、推論の失敗と区別できるよう専用の入れ物で包んで運ぶ(工程失敗へ
    写像させないため)。
    """
    if on_progress is None:
        return None

    start = time.monotonic()

    def relay(note):
        try:
            on_progress(stage, done=done, total=total, note=note,
                        elapsed=time.monotonic() - start)
        except Exception as e:
            raise _ProgressCallbackFailed(e) from None

    return relay


def _save_intermediate(keep_intermediate_dir, pcm, vocal_pcm, segments):
    """診断用中間生成物を保存する。

    S0正規化PCM(input_normalized.wav)・分離後ボーカルWAV(vocal.wav。分割時は核区間を連結した
    曲全体分)・S2認識結果(segments.json)を、指定ディレクトリへ保存する。書き込み失敗(権限・
    ディスク等のI/O失敗)は IntermediateWriteError として送出する。`sf.write` は libsndfile が
    開くため失敗を `OSError` でなく `sf.SoundFileError` 系で送出する(`mkdir`/`write_text` の
    失敗は `OSError`)ため、両方を捕捉する。
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


def run_front_stage(input_path, *, separate_vocals, separator, content_recognizer_model,
                    retry, forced_aligner, sofa_aligner, english_katakana_method,
                    chunking=None, keep_intermediate_dir=None, on_progress=None):
    """S0〜S3 を通しで実行し、共有出力とその周辺情報を返す。

    separate_vocals から english_katakana_method までは S1・S2 の公開関数がそのまま受け取る設定で、
    本関数は解釈せず渡す。chunking は長尺分割の実行ポリシー(None で分割しない)、
    keep_intermediate_dir は診断用中間生成物の保存先(None で保存しない)、on_progress は
    (stage_id, done, total, note, elapsed) を受ける進捗の報告先(None で報告しない)。
    """
    _report_stage(on_progress, "load")
    pcm = _io.load_audio(input_path)
    duration_sec = len(pcm.samples) / pcm.sample_rate

    if chunking is None or chunking.max_duration_sec <= 0 or duration_sec <= chunking.max_duration_sec:
        vocal_wav, segments, rms_envelope, vocal_pcm = _run_single(
            pcm, separate_vocals, separator, content_recognizer_model, retry,
            forced_aligner, sofa_aligner, english_katakana_method, on_progress)
        forced_split = False
    else:
        vocal_wav, segments, rms_envelope, vocal_pcm, forced_split = _run_chunked(
            pcm, duration_sec, separate_vocals, separator, content_recognizer_model,
            retry, chunking, forced_aligner, sofa_aligner, english_katakana_method, on_progress)

    if keep_intermediate_dir is not None:
        _save_intermediate(keep_intermediate_dir, pcm, vocal_pcm, segments)

    return FrontStageResult(
        analysis=AnalysisResult(vocal_wav=vocal_wav, segments=segments, rms=rms_envelope),
        pcm=pcm, vocal_pcm=vocal_pcm, duration_sec=duration_sec, forced_split=forced_split)


def _run_single(pcm, separate_vocals, separator, content_recognizer_model, retry,
                forced_aligner, sofa_aligner, english_katakana_method, on_progress):
    """分割なしの単一実行。"""
    _report_stage(on_progress, "separate")
    vocal_path = _run_inference(
        "separate", _separator.separate, pcm, separate_vocals,
        separator=separator,
        # 分離しない指定では委譲先が外部推論を一切行わないので、その呼び出しの失敗は工程失敗でない。
        map_failures=separate_vocals != "never",
        on_progress=_model_download_progress(on_progress, "separate", done=0, total=None))
    _report_stage(on_progress, "recognize")
    segments = _run_inference(
        "recognize", _recognizer.recognize, vocal_path,
        content_recognizer_model=content_recognizer_model,
        retry=retry, forced_aligner=forced_aligner, sofa_aligner=sofa_aligner,
        english_katakana_method=english_katakana_method,
        on_progress=_model_download_progress(on_progress, "recognize", done=0, total=None))
    _report_stage(on_progress, "rms")
    vocal_pcm = _read_intermediate(vocal_path, _io.load_audio)
    rms_envelope = _rms.compute_rms(vocal_pcm)
    return vocal_path, segments, rms_envelope, vocal_pcm


def _write_whole_vocal_wav(vocal_pcm):
    """連結した曲全体のボーカルPCMを非公開の内部一時領域へ書き出し、そのパスを返す。

    分割時の各チャンクの分離結果は曲全体のWAVを持たないが、共有出力のボーカルWAVは省略できない
    単一のパスなので、分割しない実行と同じ意味のパスをここで作る。この領域の破棄は正常終了時の
    自動削除に任せ、呼び出し元は削除のタイミングを制御できない。
    """
    work_dir = Path(tempfile.mkdtemp(prefix="vocal_analysis_front_"))
    atexit.register(shutil.rmtree, work_dir, ignore_errors=True)
    path = work_dir / "vocal.wav"
    sf.write(path, vocal_pcm.samples, vocal_pcm.sample_rate)
    return path


def _run_chunked(pcm, duration_sec, separate_vocals, separator, content_recognizer_model,
                 retry, chunking, forced_aligner, sofa_aligner,
                 english_katakana_method, on_progress):
    """分割ありの実行。境界決定は分離前の生音声RMSを使う。"""
    raw_rms = _rms.compute_rms(pcm)
    boundary_pairs = _chunking.find_chunk_boundaries(
        duration_sec, raw_rms.times_sec, raw_rms.values,
        max_duration_sec=chunking.max_duration_sec,
        search_window_sec=chunking.search_window_sec)
    boundaries = [b for b, _ in boundary_pairs]
    forced_split = any(f for _, f in boundary_pairs)
    edges = [0.0] + boundaries + [duration_sec]
    n = len(edges) - 1

    chunk_offsets_sec = []
    chunk_segments_list = []
    vocal_core_chunks = []
    for i in range(n):
        core_start, core_end = edges[i], edges[i + 1]
        pad_start = max(0.0, core_start - chunking.overlap_sec) if i > 0 else 0.0
        pad_end = min(duration_sec, core_end + chunking.overlap_sec) if i < n - 1 else duration_sec
        chunk_offsets_sec.append(pad_start)

        chunk_pcm = _slice_pcm(pcm, pad_start, pad_end)
        # done は「このチャンクを始める時点までに完了したチャンク数」(0始まり)。
        # 1個目のチャンクを始める時点(i=0)ではまだ0個も完了していない。
        _report_stage(on_progress, "separate", done=i, total=n)
        vocal_path = _run_inference(
            "separate", _separator.separate, chunk_pcm, separate_vocals,
            separator=separator,
            map_failures=separate_vocals != "never",
            on_progress=_model_download_progress(on_progress, "separate", done=i, total=n))
        _report_stage(on_progress, "recognize", done=i, total=n)
        chunk_segments_list.append(
            _run_inference(
                "recognize", _recognizer.recognize, vocal_path,
                content_recognizer_model=content_recognizer_model,
                retry=retry, forced_aligner=forced_aligner, sofa_aligner=sofa_aligner,
                english_katakana_method=english_katakana_method,
                on_progress=_model_download_progress(on_progress, "recognize", done=i, total=n)))

        chunk_vocal_pcm = _read_intermediate(vocal_path, _read_pcm_raw)
        vocal_core_chunks.append(_slice_pcm(chunk_vocal_pcm, core_start - pad_start, core_end - pad_start))

    merged_segments = _chunking.merge_chunk_segments(chunk_segments_list, chunk_offsets_sec, boundaries)
    whole_vocal_pcm = _concat_pcm(vocal_core_chunks)
    _report_stage(on_progress, "rms")
    rms_envelope = _rms.compute_rms(whole_vocal_pcm)
    whole_vocal_wav = _write_whole_vocal_wav(whole_vocal_pcm)
    return whole_vocal_wav, merged_segments, rms_envelope, whole_vocal_pcm, forced_split
