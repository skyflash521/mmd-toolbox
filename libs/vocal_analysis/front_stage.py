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
    pass


class IntermediateReadError(Exception):
    def __init__(self, message, *, path):
        super().__init__(message)
        self.path = path


class StageExecutionError(Exception):
    def __init__(self, message, *, stage):
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class FrontStageResult:
    """pcm は入力を読み込んで正規化した分離前の PCM。vocal_pcm は分離後ボーカルで、分割時は各チャンクの
    核区間を連結した曲全体分。forced_split は無音が見つからず目標境界で強制分割したことが1回でもあったか。"""

    analysis: AnalysisResult
    pcm: AudioPcm
    vocal_pcm: AudioPcm
    duration_sec: float
    forced_split: bool


_CLASSIFIED_STAGE_EXCEPTIONS = (
    _separator.SeparationError,
    _recognizer.RecognitionError,
    _io.AudioLoadError,
    IntermediateReadError,
    IntermediateWriteError,
)


def _run_inference(stage, func, *args, map_failures=True, **kwargs):
    if not map_failures:
        return func(*args, **kwargs)
    try:
        return func(*args, **kwargs)
    except _CLASSIFIED_STAGE_EXCEPTIONS:
        raise
    except _ProgressCallbackFailed as e:
        raise e.cause from None
    except Exception as e:
        raise StageExecutionError(f"{type(e).__name__}: {e}", stage=stage) from e


class _ProgressCallbackFailed(Exception):
    def __init__(self, cause):
        super().__init__()
        self.cause = cause


def _read_intermediate(path, reader):
    try:
        return reader(path)
    except (_io.AudioLoadError, sf.SoundFileError, OSError) as e:
        raise IntermediateReadError(str(e), path=path) from e


def _slice_pcm(pcm, start_sec, end_sec):
    sr = pcm.sample_rate
    start_idx = max(0, round(start_sec * sr))
    end_idx = min(len(pcm.samples), round(end_sec * sr))
    return AudioPcm(samples=pcm.samples[start_idx:end_idx], sample_rate=sr)


def _concat_pcm(pcms):
    sr = pcms[0].sample_rate
    samples = np.concatenate([p.samples for p in pcms], axis=0)
    return AudioPcm(samples=samples, sample_rate=sr)


def _read_pcm_without_peak_normalization(path):
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return AudioPcm(samples=samples, sample_rate=sample_rate)


def _report_stage_start(on_progress, stage, *, done=0, total=None, note=""):
    if on_progress is not None:
        on_progress(stage, done=done, total=total, note=note, elapsed=0.0)


def _model_download_progress(on_progress, stage, *, done, total):
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
    try:
        directory = Path(keep_intermediate_dir)
        directory.mkdir(parents=True, exist_ok=True)
        sf.write(directory / "input_normalized.wav", pcm.samples, pcm.sample_rate)
        sf.write(directory / "vocal.wav", vocal_pcm.samples, vocal_pcm.sample_rate)
        (directory / "segments.json").write_text(
            json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2), encoding="utf-8")
    # sf.write の書き込み失敗は OSError ではなく SoundFileError になる。
    except (OSError, sf.SoundFileError) as e:
        raise IntermediateWriteError(str(e)) from e


def run_front_stage(input_path, *, separate_vocals, separator, content_recognizer_model,
                    retry, forced_aligner, sofa_aligner, english_katakana_method,
                    chunking=None, keep_intermediate_dir=None, on_progress=None):
    """on_progress は on_progress(stage_id, *, done, total, note, elapsed) の形で呼ばれる。"""
    _report_stage_start(on_progress, "load")
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
    _report_stage_start(on_progress, "separate")
    vocal_path = _run_inference(
        "separate", _separator.separate, pcm, separate_vocals,
        separator=separator,
        map_failures=separate_vocals != "never",
        on_progress=_model_download_progress(on_progress, "separate", done=0, total=None))
    _report_stage_start(on_progress, "recognize")
    segments = _run_inference(
        "recognize", _recognizer.recognize, vocal_path,
        content_recognizer_model=content_recognizer_model,
        retry=retry, forced_aligner=forced_aligner, sofa_aligner=sofa_aligner,
        english_katakana_method=english_katakana_method,
        on_progress=_model_download_progress(on_progress, "recognize", done=0, total=None))
    _report_stage_start(on_progress, "rms")
    vocal_pcm = _read_intermediate(vocal_path, _io.load_audio)
    rms_envelope = _rms.compute_rms(vocal_pcm)
    return vocal_path, segments, rms_envelope, vocal_pcm


def _write_whole_vocal_wav(vocal_pcm):
    work_dir = Path(tempfile.mkdtemp(prefix="vocal_analysis_front_"))
    atexit.register(shutil.rmtree, work_dir, ignore_errors=True)
    path = work_dir / "vocal.wav"
    sf.write(path, vocal_pcm.samples, vocal_pcm.sample_rate)
    return path


def _run_chunked(pcm, duration_sec, separate_vocals, separator, content_recognizer_model,
                 retry, chunking, forced_aligner, sofa_aligner,
                 english_katakana_method, on_progress):
    unseparated_rms = _rms.compute_rms(pcm)
    boundary_pairs = _chunking.find_chunk_boundaries(
        duration_sec, unseparated_rms.times_sec, unseparated_rms.values,
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
        completed_chunks = i
        _report_stage_start(on_progress, "separate", done=completed_chunks, total=n)
        vocal_path = _run_inference(
            "separate", _separator.separate, chunk_pcm, separate_vocals,
            separator=separator,
            map_failures=separate_vocals != "never",
            on_progress=_model_download_progress(on_progress, "separate", done=completed_chunks, total=n))
        _report_stage_start(on_progress, "recognize", done=completed_chunks, total=n)
        chunk_segments_list.append(
            _run_inference(
                "recognize", _recognizer.recognize, vocal_path,
                content_recognizer_model=content_recognizer_model,
                retry=retry, forced_aligner=forced_aligner, sofa_aligner=sofa_aligner,
                english_katakana_method=english_katakana_method,
                on_progress=_model_download_progress(on_progress, "recognize", done=completed_chunks, total=n)))

        chunk_vocal_pcm = _read_intermediate(vocal_path, _read_pcm_without_peak_normalization)
        vocal_core_chunks.append(_slice_pcm(chunk_vocal_pcm, core_start - pad_start, core_end - pad_start))

    merged_segments = _chunking.merge_chunk_segments(chunk_segments_list, chunk_offsets_sec, boundaries)
    whole_vocal_pcm = _concat_pcm(vocal_core_chunks)
    _report_stage_start(on_progress, "rms")
    rms_envelope = _rms.compute_rms(whole_vocal_pcm)
    whole_vocal_wav = _write_whole_vocal_wav(whole_vocal_pcm)
    return whole_vocal_wav, merged_segments, rms_envelope, whole_vocal_pcm, forced_split
