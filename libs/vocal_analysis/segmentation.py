import math

import numpy as np
from scipy.signal import resample_poly

SILENCE_FRAME_SEC = 0.1
SILENCE_THRESHOLD_DB_BELOW_PEAK = 30.0
SILENCE_MIN_RUN_SEC = 0.6
SEGMENT_MIN_SEC = 1.5
SEGMENT_MAX_SEC = 25.0
TRIM_MARGIN_SEC = 0.1


def downmix_to_mono(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=1)


def resample_to_target(mono: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return mono
    gcd = math.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd
    return resample_poly(mono, up, down).astype(np.float32)


def frame_rms(mono: np.ndarray, sample_rate: int, frame_sec: float) -> np.ndarray:
    frame_len = max(1, round(frame_sec * sample_rate))
    num_frames = len(mono) // frame_len
    if num_frames == 0:
        return np.array([], dtype=np.float64)
    trimmed = mono[: num_frames * frame_len].astype(np.float64).reshape(num_frames, frame_len)
    return np.sqrt(np.mean(np.square(trimmed), axis=1))


def silence_threshold(rms_per_frame: np.ndarray) -> float:
    if rms_per_frame.size == 0:
        return 0.0
    peak = float(np.percentile(rms_per_frame, 95))
    return peak * (10 ** (-SILENCE_THRESHOLD_DB_BELOW_PEAK / 20.0))


def detect_silence_split_points(mono: np.ndarray, sample_rate: int) -> list[float]:
    rms_per_frame = frame_rms(mono, sample_rate, SILENCE_FRAME_SEC)
    if rms_per_frame.size == 0:
        return []
    threshold = silence_threshold(rms_per_frame)
    min_run_frames = max(1, round(SILENCE_MIN_RUN_SEC / SILENCE_FRAME_SEC))

    split_points: list[float] = []
    run_start: int | None = None
    for i, value in enumerate(rms_per_frame):
        is_silent = value <= threshold
        if is_silent and run_start is None:
            run_start = i
        elif not is_silent and run_start is not None:
            if i - run_start >= min_run_frames:
                split_points.append((run_start + i) / 2 * SILENCE_FRAME_SEC)
            run_start = None
    if run_start is not None and rms_per_frame.size - run_start >= min_run_frames:
        split_points.append((run_start + rms_per_frame.size) / 2 * SILENCE_FRAME_SEC)
    return split_points


def build_segment_bounds(duration_sec: float, split_points: list[float]) -> list[tuple[float, float]]:
    boundaries = [0.0, *sorted(split_points), duration_sec]
    last_index = len(boundaries) - 1
    merged: list[tuple[float, float]] = []
    start = boundaries[0]
    for index in range(1, len(boundaries)):
        end = boundaries[index]
        if end - start >= SEGMENT_MIN_SEC or index == last_index:
            merged.append((start, end))
            start = end

    result: list[tuple[float, float]] = []
    for seg_start, seg_end in merged:
        span = seg_end - seg_start
        if span <= SEGMENT_MAX_SEC:
            result.append((seg_start, seg_end))
            continue
        piece_count = math.ceil(span / SEGMENT_MAX_SEC)
        piece_len = span / piece_count
        result.extend(
            (seg_start + i * piece_len, seg_start + (i + 1) * piece_len) for i in range(piece_count)
        )
    return result


def format_time_range(start_sec: float, end_sec: float) -> str:

    def _mmss(sec: float) -> str:
        total = int(sec)
        return f"{total // 60}:{total % 60:02d}"

    return f"{_mmss(start_sec)}-{_mmss(end_sec)}"


def is_segment_silent(segment_samples: np.ndarray, threshold: float) -> bool:
    if segment_samples.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(segment_samples.astype(np.float64)))))
    return rms <= threshold


def voiced_trim_bounds(segment_samples: np.ndarray, sample_rate: int, threshold: float) -> tuple[int, int]:
    rms_per_frame = frame_rms(segment_samples, sample_rate, SILENCE_FRAME_SEC)
    voiced = np.nonzero(rms_per_frame > threshold)[0]
    if voiced.size == 0:
        return 0, len(segment_samples)
    frame_len = max(1, round(SILENCE_FRAME_SEC * sample_rate))
    margin = round(TRIM_MARGIN_SEC * sample_rate)
    lo = max(0, int(voiced[0]) * frame_len - margin)
    hi = min(len(segment_samples), (int(voiced[-1]) + 1) * frame_len + margin)
    return lo, hi
