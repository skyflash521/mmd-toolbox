from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vpr.rests import rest_intervals
from vpr.types import Part, TempoEvent

from .phonemes import xsampa_vowel_letter

_HTK_100NS_UNITS_PER_SECOND = 1e7

_VOWEL_SYMBOLS = frozenset({"a", "i", "u", "e", "o"})
_SILENCE_SYMBOLS = frozenset({"pau", "br"})
# 参照ラベルは sy・ty・zy に当たる音を sh・ch・j で表記する。
_CONSONANT_SYMBOLS = frozenset({
    "b", "by", "ch", "cl", "d", "f", "g", "gy", "h", "hy", "j", "k", "ky",
    "m", "my", "n", "N", "ny", "p", "py", "r", "ry", "s", "sh", "t", "ts",
    "v", "w", "y", "z",
})
_EXCLUDED_SYMBOLS = frozenset({"xx"})


class UnknownReferenceSymbolError(Exception):
    pass


class MonophoneLabelFormatError(Exception):
    pass


@dataclass
class CategorySegment:
    """category は a/i/u/e/o・"c"・"sil" のいずれかで、採点から除く区間は None。"""

    category: str | None
    start_sec: float
    end_sec: float


def reference_symbol_to_category(symbol: str) -> str | None:
    """戻り値は CategorySegment.category の値。写像表に無い記号は UnknownReferenceSymbolError。"""
    if symbol in _VOWEL_SYMBOLS:
        return symbol
    if symbol in _SILENCE_SYMBOLS:
        return "sil"
    if symbol in _CONSONANT_SYMBOLS:
        return "c"
    if symbol in _EXCLUDED_SYMBOLS:
        return None
    raise UnknownReferenceSymbolError(
        f"参照ラベルの写像表に無い音素記号です: {symbol!r}(写像表を補ってください)"
    )


def _parse_monophone_label_lines(
    path: Path, lines: list[str], time_scale: float
) -> list[CategorySegment]:
    segments = []
    for line in lines:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise MonophoneLabelFormatError(
                f"{path}: 「開始 終了 音素」の3列形式ではない行です: {line!r}"
            )
        start_str, end_str, symbol = fields
        segments.append(
            CategorySegment(
                category=reference_symbol_to_category(symbol),
                start_sec=float(start_str) / time_scale,
                end_sec=float(end_str) / time_scale,
            )
        )
    return segments


def parse_seconds_monophone_label(path: str | Path) -> list[CategorySegment]:
    path = Path(path)
    return _parse_monophone_label_lines(path, path.read_text(encoding="utf-8").splitlines(), time_scale=1.0)


def parse_htk100ns_monophone_label(path: str | Path) -> list[CategorySegment]:
    path = Path(path)
    return _parse_monophone_label_lines(
        path, path.read_text(encoding="utf-8").splitlines(), time_scale=_HTK_100NS_UNITS_PER_SECOND
    )


def clip_segments_to_audio_duration(
    segments: list[CategorySegment], audio_duration_sec: float
) -> list[CategorySegment]:
    result = []
    for seg in segments:
        start = max(seg.start_sec, 0.0)
        end = min(seg.end_sec, audio_duration_sec)
        if end > start:
            result.append(CategorySegment(category=seg.category, start_sec=start, end_sec=end))
    return result


def remove_invalid_time_segments(segments: list[CategorySegment]) -> list[CategorySegment]:
    """2つ以上の区間が重なる時間範囲は、どの区間からも取り除いて隙間にする。"""
    valid = [seg for seg in segments if seg.end_sec > seg.start_sec]
    if not valid:
        return []

    boundaries = sorted({seg.start_sec for seg in valid} | {seg.end_sec for seg in valid})
    result: list[CategorySegment] = []
    owner_of_last: int | None = None
    for lo, hi in zip(boundaries, boundaries[1:], strict=False):
        if hi <= lo:
            continue
        mid = (lo + hi) / 2
        covering = [i for i, seg in enumerate(valid) if seg.start_sec <= mid < seg.end_sec]
        if len(covering) != 1:
            owner_of_last = None
            continue
        owner = covering[0]
        if result and owner_of_last == owner and result[-1].end_sec == lo:
            result[-1] = CategorySegment(category=valid[owner].category, start_sec=result[-1].start_sec, end_sec=hi)
        else:
            result.append(CategorySegment(category=valid[owner].category, start_sec=lo, end_sec=hi))
        owner_of_last = owner
    return result


_MIDI_MISMATCH_THRESHOLD_SEC = 0.3


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((iv for iv in intervals if iv[1] > iv[0]), key=lambda iv: iv[0])
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(
    base: tuple[float, float], subtract: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    start, end = base
    remaining: list[tuple[float, float]] = []
    cursor = start
    for sub_start, sub_end in subtract:
        if sub_end <= cursor or sub_start >= end:
            continue
        if sub_start > cursor:
            remaining.append((cursor, min(sub_start, end)))
        cursor = max(cursor, sub_end)
        if cursor >= end:
            break
    if cursor < end:
        remaining.append((cursor, end))
    return remaining


def _intersect_intervals(
    base: tuple[float, float], others: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    start, end = base
    result: list[tuple[float, float]] = []
    for other_start, other_end in others:
        lo = max(start, other_start)
        hi = min(end, other_end)
        if hi > lo:
            result.append((lo, hi))
    return result


def find_midi_mismatch_ranges(
    segments: list[CategorySegment], midi_notes: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """母音のうちノートに覆われない部分と、sil のうちノートに覆われる部分で、一定時間以上続くものを返す。"""
    merged_notes = _merge_intervals(midi_notes)
    mismatches: list[tuple[float, float]] = []
    for seg in segments:
        if seg.category in _VOWEL_SYMBOLS:
            candidates = _subtract_intervals((seg.start_sec, seg.end_sec), merged_notes)
        elif seg.category == "sil":
            candidates = _intersect_intervals((seg.start_sec, seg.end_sec), merged_notes)
        else:
            continue
        for start, end in candidates:
            if end - start >= _MIDI_MISMATCH_THRESHOLD_SEC:
                mismatches.append((start, end))
    return mismatches


def exclude_ranges_from_segments(
    segments: list[CategorySegment], ranges_to_exclude: list[tuple[float, float]]
) -> list[CategorySegment]:
    merged_exclusions = _merge_intervals(ranges_to_exclude)
    result: list[CategorySegment] = []
    for seg in segments:
        for start, end in _subtract_intervals((seg.start_sec, seg.end_sec), merged_exclusions):
            result.append(CategorySegment(category=seg.category, start_sec=start, end_sec=end))
    return result


def ticks_to_seconds(tick: int, tempos: list[TempoEvent], resolution: int) -> float:
    """resolution は四分音符あたりの tick 数。"""
    ordered = sorted(tempos, key=lambda t: t.tick)
    seconds = 0.0
    for i, tempo in enumerate(ordered):
        next_tick = ordered[i + 1].tick if i + 1 < len(ordered) else None
        segment_end = tick if next_tick is None or tick <= next_tick else next_tick
        ticks_in_segment = segment_end - tempo.tick
        seconds += (ticks_in_segment / resolution) * (60.0 / tempo.bpm)
        if next_tick is None or tick <= next_tick:
            break
    return seconds


def generate_vpr_reference_segments(
    part: Part, tempos: list[TempoEvent], resolution: int
) -> list[CategorySegment]:
    segments: list[CategorySegment] = []
    previous_phoneme: str | None = None
    for note in part.notes:
        resolved = previous_phoneme if not note.phonemes or note.phonemes == ["-"] else note.phonemes[-1]
        if resolved is not None:
            previous_phoneme = resolved
        if resolved is None:
            continue
        vowel = xsampa_vowel_letter(resolved)
        category = vowel if vowel is not None else "c"
        segments.append(
            CategorySegment(
                category=category,
                start_sec=ticks_to_seconds(note.start_tick, tempos, resolution),
                end_sec=ticks_to_seconds(note.start_tick + note.duration_tick, tempos, resolution),
            )
        )

    end_tick = max((note.start_tick + note.duration_tick for note in part.notes), default=0)
    for rest_start_tick, rest_end_tick in rest_intervals(part.notes, end_tick):
        segments.append(
            CategorySegment(
                category="sil",
                start_sec=ticks_to_seconds(rest_start_tick, tempos, resolution),
                end_sec=ticks_to_seconds(rest_end_tick, tempos, resolution),
            )
        )

    segments.sort(key=lambda seg: seg.start_sec)
    merged: list[CategorySegment] = []
    for seg in segments:
        if merged and merged[-1].category == seg.category and merged[-1].end_sec == seg.start_sec:
            merged[-1] = CategorySegment(category=seg.category, start_sec=merged[-1].start_sec, end_sec=seg.end_sec)
        else:
            merged.append(seg)
    return merged


_SCORING_FRAME_SEC = 0.01
_NON_VOWEL_SCORED_CATEGORIES = frozenset({"c", "sil"})


def _frame_categories(segments: list[CategorySegment], duration_sec: float) -> list[str | None]:
    ordered = sorted(segments, key=lambda seg: seg.start_sec)
    num_frames = round(duration_sec / _SCORING_FRAME_SEC)
    categories: list[str | None] = []
    idx = 0
    for i in range(num_frames):
        t = (i + 0.5) * _SCORING_FRAME_SEC
        while idx < len(ordered) and ordered[idx].end_sec <= t:
            idx += 1
        if idx < len(ordered) and ordered[idx].start_sec <= t:
            categories.append(ordered[idx].category)
        else:
            categories.append(None)
    return categories


def compute_vowel_accuracy(
    predicted: list[CategorySegment], reference: list[CategorySegment], duration_sec: float
) -> float:
    ref_frames = _frame_categories(reference, duration_sec)
    pred_frames = _frame_categories(predicted, duration_sec)
    vowel_indices = [i for i, category in enumerate(ref_frames) if category in _VOWEL_SYMBOLS]
    matches = sum(1 for i in vowel_indices if pred_frames[i] == ref_frames[i])
    return matches / len(vowel_indices)


def compute_over_opening_rate(
    predicted: list[CategorySegment], reference: list[CategorySegment], duration_sec: float
) -> float:
    ref_frames = _frame_categories(reference, duration_sec)
    pred_frames = _frame_categories(predicted, duration_sec)
    non_vowel_indices = [i for i, category in enumerate(ref_frames) if category in _NON_VOWEL_SCORED_CATEGORIES]
    bled = sum(1 for i in non_vowel_indices if pred_frames[i] in _VOWEL_SYMBOLS)
    return bled / len(non_vowel_indices)


def _overlap_sec(a: CategorySegment, b: CategorySegment) -> float:
    return max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))


def _better_assignment(
    a: tuple[float, tuple[tuple[int, int], ...]], b: tuple[float, tuple[tuple[int, int], ...]]
) -> tuple[float, tuple[tuple[int, int], ...]]:
    if a[0] != b[0]:
        return a if a[0] > b[0] else b
    return a if a[1] <= b[1] else b


def match_segments(
    predicted: list[CategorySegment], reference: list[CategorySegment]
) -> tuple[list[tuple[CategorySegment, CategorySegment]], list[CategorySegment], list[CategorySegment]]:
    """predicted・reference はそれぞれ区間どうしが重ならないこと(remove_invalid_time_segments を通したもの)。
    戻り値は (対応した (基準区間, 予測区間) の組, 対応の無い基準区間, 対応の無い予測区間)。"""
    ref_vowels = sorted(
        (seg for seg in reference if seg.category in _VOWEL_SYMBOLS),
        key=lambda seg: (seg.start_sec, seg.end_sec),
    )
    pred_vowels = sorted(
        (seg for seg in predicted if seg.category in _VOWEL_SYMBOLS),
        key=lambda seg: (seg.start_sec, seg.end_sec),
    )
    n, m = len(ref_vowels), len(pred_vowels)

    dp: list[list[tuple[float, tuple[tuple[int, int], ...]]]] = [[(0.0, ())] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0]
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = _better_assignment(dp[i - 1][j], dp[i][j - 1])
            ref_seg, pred_seg = ref_vowels[i - 1], pred_vowels[j - 1]
            overlap = _overlap_sec(ref_seg, pred_seg)
            if ref_seg.category == pred_seg.category and overlap > 0:
                prev_total, prev_pairs = dp[i - 1][j - 1]
                candidate = (prev_total + overlap, prev_pairs + ((i - 1, j - 1),))
                best = _better_assignment(best, candidate)
            dp[i][j] = best

    _, pairs = dp[n][m]
    matched_ref_indices = {ref_idx for ref_idx, _ in pairs}
    matched_pred_indices = {pred_idx for _, pred_idx in pairs}
    matched = [(ref_vowels[ref_idx], pred_vowels[pred_idx]) for ref_idx, pred_idx in pairs]
    undetected = [seg for idx, seg in enumerate(ref_vowels) if idx not in matched_ref_indices]
    excess = [seg for idx, seg in enumerate(pred_vowels) if idx not in matched_pred_indices]
    return matched, undetected, excess


def compute_boundary_deviation(
    matched_pairs: list[tuple[CategorySegment, CategorySegment]],
) -> tuple[float, float] | None:
    """戻り値は開始時刻差のミリ秒の (中央値, 95パーセンタイル)。対応が1組も無ければ None。"""
    if not matched_pairs:
        return None
    deviations_ms = [
        abs(predicted.start_sec - reference.start_sec) * 1000.0 for reference, predicted in matched_pairs
    ]
    median, p95 = np.percentile(deviations_ms, [50, 95], method="linear")
    return float(median), float(p95)


@dataclass
class SongMetrics:
    vowel_accuracy: float
    over_opening_rate: float
    boundary_deviation: tuple[float, float] | None
    undetected_count: int
    excess_count: int
    reference_vowel_count: int


def compute_song_metrics(
    predicted: list[CategorySegment], reference: list[CategorySegment], duration_sec: float
) -> SongMetrics:
    matched, undetected, excess = match_segments(predicted, reference)
    return SongMetrics(
        vowel_accuracy=compute_vowel_accuracy(predicted, reference, duration_sec),
        over_opening_rate=compute_over_opening_rate(predicted, reference, duration_sec),
        boundary_deviation=compute_boundary_deviation(matched),
        undetected_count=len(undetected),
        excess_count=len(excess),
        reference_vowel_count=len(matched) + len(undetected),
    )
