import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from lipsync import ApertureClass, ConsonantClass, MouthEvent, MouthShape
from vocal_analysis.phonemes import espeak_ipa_to_vowel

FRAME_RATE = 30.0

_BILABIAL_PHONEMES = frozenset({"m", "mʲ", "b", "bʲ", "p", "pʲ"})
_MORAIC_NASAL_PHONEME = "ɴ"

_ROUNDED_PHONEMES = frozenset({"ɸ", "w"})
_SPREAD_PHONEMES = frozenset({"ɕ", "tɕ", "dʑ", "ɲ", "ç"})
_PALATALIZED_SUFFIX = "ʲ"

_FIRM_CLOSURE_PHONEMES = frozenset({"t", "d", "n", "ts", "ɲ"})
_NARROW_CHANNEL_PHONEMES = frozenset({"s", "z", "ɕ", "tɕ", "dʑ", "ç", "j"})
_SLIGHT_CLOSURE_PHONEMES = frozenset({"k", "ɡ", "ɾ", "kʲ", "ɡʲ"})
_APERTURE_RANK = {
    ApertureClass.NONE: 0,
    ApertureClass.SLIGHT_CLOSURE: 1,
    ApertureClass.NARROW_CHANNEL: 2,
    ApertureClass.FIRM_CLOSURE: 3,
}

_VOWEL_SHAPES = {"a": MouthShape.A, "i": MouthShape.I, "u": MouthShape.U, "e": MouthShape.E, "o": MouthShape.O}

_LOW_DYNAMICS_THRESHOLD_DB = 12.0
_MORA_CENTER_FRACTION = 0.6
_ONSET_WINDOW_SEC = 0.06
_ONSET_SLOPE_UNIT_SEC = 0.010
_ONSET_SLOPE_PER_10MS = 0.15
_ONSET_SMOOTH_WINDOW_SEC = 0.03
_ONSET_MIN_SPAN_SEC = 1.0 / FRAME_RATE
_WEAK_CONFIDENCE_THRESHOLD = 0.5
_WEAK_RMS_WITH_CONFIDENCE = 0.3
_WEAK_RMS_WITHOUT_CONFIDENCE = 0.2
_WEAK_SCALE = 0.5

_LONG_MORA_THRESHOLD_SEC = 1.0
_SUBWINDOW_TARGET_SEC = 0.3
_GAP_CLOSE_RUN_SEC = 0.2

_OPEN_RENORM_P_LO = 10.0
_OPEN_RENORM_P_HI = 90.0
_DEGENERATE_NORMALIZED_RMS = 0.5

_UnitKind = Literal["vowel", "bilabial", "n", "gap", "silence"]
_MORA_KINDS = frozenset({"vowel", "n"})


def _consonant_class(phoneme):
    if phoneme is None:
        return ConsonantClass.NONE
    if phoneme in _ROUNDED_PHONEMES:
        return ConsonantClass.ROUNDED
    if phoneme in _SPREAD_PHONEMES or phoneme.endswith(_PALATALIZED_SUFFIX):
        return ConsonantClass.SPREAD
    return ConsonantClass.NEUTRAL


def _aperture_class_of_phoneme(phoneme):
    if phoneme in _FIRM_CLOSURE_PHONEMES:
        return ApertureClass.FIRM_CLOSURE
    if phoneme in _NARROW_CHANNEL_PHONEMES:
        return ApertureClass.NARROW_CHANNEL
    if phoneme in _SLIGHT_CLOSURE_PHONEMES:
        return ApertureClass.SLIGHT_CLOSURE
    return ApertureClass.NONE


def _strongest_aperture_class(consonant_run):
    strongest = ApertureClass.NONE
    for phoneme in consonant_run:
        cls = _aperture_class_of_phoneme(phoneme)
        if _APERTURE_RANK[cls] > _APERTURE_RANK[strongest]:
            strongest = cls
    return strongest


@dataclass
class _Unit:
    kind: _UnitKind
    start_sec: float
    end_sec: float
    letter: str | None = None
    confidence: float | None = None
    head_consonant_phoneme: str | None = None
    aperture_consonant_run: tuple[str, ...] = ()
    nucleus_start_sec: float | None = None
    nucleus_end_sec: float | None = None
    from_gap_continuation: bool = False

    def nucleus_span(self):
        start = self.nucleus_start_sec if self.nucleus_start_sec is not None else self.start_sec
        end = self.nucleus_end_sec if self.nucleus_end_sec is not None else self.end_sec
        return start, end


@dataclass(frozen=True)
class EventDiagnostics:
    weak_vowels: int
    low_dynamics: bool
    merged_morae: int


def _nearest_index(times, t):
    idx = int(np.searchsorted(times, t))
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    before, after = times[idx - 1], times[idx]
    return idx - 1 if (t - before) <= (after - t) else idx


def _rms_window_average(rms, start_sec, end_sec):
    times, values = rms.times_sec, rms.values
    if len(times) == 0:
        return 0.0
    lo = int(np.searchsorted(times, start_sec, side="left"))
    hi = int(np.searchsorted(times, end_sec, side="left"))
    if hi > lo:
        return float(np.mean(values[lo:hi]))
    return float(values[_nearest_index(times, (start_sec + end_sec) / 2.0)])


def _mora_rms(rms, start_sec, end_sec):
    span = end_sec - start_sec
    margin = span * (1.0 - _MORA_CENTER_FRACTION) / 2.0
    return _rms_window_average(rms, start_sec + margin, end_sec - margin)


def _mora_representative_rms(rms, unit):
    nucleus_start, nucleus_end = unit.nucleus_span()
    value = _mora_rms(rms, nucleus_start, nucleus_end)
    if (unit.start_sec, unit.end_sec) != (nucleus_start, nucleus_end):
        value = max(value, _mora_rms(rms, unit.start_sec, unit.end_sec))
    return value


def _subwindow_count(duration_sec):
    n = math.floor(duration_sec / _SUBWINDOW_TARGET_SEC + 0.5)
    return max(2, n)


def _split_into_subwindows(start_sec, end_sec):
    count = _subwindow_count(end_sec - start_sec)
    width = (end_sec - start_sec) / count
    return [(start_sec + i * width, start_sec + (i + 1) * width) for i in range(count)]


def _classify_phonetic(segments, use_n_morph):
    units = []
    absorbed_consonant_start = None
    last_consonant_phoneme = None
    aperture_run = []
    for seg in segments:
        if seg.type == "vowel":
            start = absorbed_consonant_start if absorbed_consonant_start is not None else seg.start_sec
            letter = espeak_ipa_to_vowel(seg.phoneme)
            if letter is None:
                units.append(_Unit("gap", start, seg.end_sec))
            else:
                units.append(_Unit(
                    "vowel", start, seg.end_sec, letter=letter,
                    confidence=seg.confidence, head_consonant_phoneme=last_consonant_phoneme,
                    aperture_consonant_run=tuple(aperture_run),
                    nucleus_start_sec=seg.start_sec, nucleus_end_sec=seg.end_sec,
                ))
            aperture_run = []
            absorbed_consonant_start = None
            last_consonant_phoneme = None
        elif seg.type == "consonant":
            last_consonant_phoneme = seg.phoneme
            if seg.phoneme in _BILABIAL_PHONEMES:
                start = absorbed_consonant_start if absorbed_consonant_start is not None else seg.start_sec
                units.append(_Unit("bilabial", start, seg.end_sec))
                absorbed_consonant_start = None
                aperture_run = []
            elif seg.phoneme == _MORAIC_NASAL_PHONEME:
                start = absorbed_consonant_start if absorbed_consonant_start is not None else seg.start_sec
                kind = "n" if use_n_morph else "silence"
                units.append(_Unit(
                    kind, start, seg.end_sec,
                    nucleus_start_sec=seg.start_sec, nucleus_end_sec=seg.end_sec,
                ))
                absorbed_consonant_start = None
                last_consonant_phoneme = None
                aperture_run = []
            else:
                if absorbed_consonant_start is None:
                    absorbed_consonant_start = seg.start_sec
                aperture_run.append(seg.phoneme)
        else:
            start = absorbed_consonant_start if absorbed_consonant_start is not None else seg.start_sec
            units.append(_Unit("gap", start, seg.end_sec))
            absorbed_consonant_start = None
            last_consonant_phoneme = None
            aperture_run = []
    if absorbed_consonant_start is not None and segments:
        units.append(_Unit("gap", absorbed_consonant_start, segments[-1].end_sec))
    return units


def _gap_close_time(rms, start_sec, end_sec, silence_on):
    times, values = rms.times_sec, rms.values
    lo = int(np.searchsorted(times, start_sec, side="left"))
    hi = int(np.searchsorted(times, end_sec, side="left"))
    run_start_index = None
    for i in range(lo, hi):
        if values[i] <= silence_on:
            if run_start_index is None:
                run_start_index = i
            if times[i] - times[run_start_index] >= _GAP_CLOSE_RUN_SEC:
                break
        else:
            run_start_index = None
    if run_start_index is None:
        return None
    return start_sec if run_start_index == lo else float(times[run_start_index])


def _resolve_silence(units, rms, silence_on, low_dynamics):
    result = []
    open_shape: tuple[str, str | None] | None = None
    n = len(units)
    for i, u in enumerate(units):
        if u.kind == "gap":
            if i == 0 or i == n - 1 or open_shape is None:
                result.append(_Unit("silence", u.start_sec, u.end_sec))
                open_shape = None
                continue
            close_at = None if low_dynamics else _gap_close_time(rms, u.start_sec, u.end_sec, silence_on)
            kind, letter = open_shape
            if close_at is None:
                result.append(_Unit(kind, u.start_sec, u.end_sec, letter=letter,
                                    from_gap_continuation=True))
            elif close_at <= u.start_sec:
                result.append(_Unit("silence", u.start_sec, u.end_sec))
                open_shape = None
            else:
                result.append(_Unit(kind, u.start_sec, close_at, letter=letter,
                                    from_gap_continuation=True))
                result.append(_Unit("silence", close_at, u.end_sec))
                open_shape = None
        elif u.kind == "vowel":
            if not low_dynamics and _mora_representative_rms(rms, u) <= silence_on:
                result.append(_Unit("silence", u.start_sec, u.end_sec))
                open_shape = None
            else:
                result.append(u)
                open_shape = ("vowel", u.letter)
        elif u.kind == "n":
            result.append(u)
            open_shape = ("n", None)
        else:
            result.append(u)
            open_shape = None
    return result


def _merge_adjacent(units) -> tuple[list[_Unit], int]:
    merged = []
    merged_morae = 0
    for u in units:
        can_merge = (
            merged and merged[-1].kind == u.kind
            and (u.kind != "vowel" or (merged[-1].letter == u.letter and u.head_consonant_phoneme is None))
        )
        if can_merge:
            prev = merged[-1]
            if u.kind in _MORA_KINDS and not u.from_gap_continuation:
                merged_morae += 1
            prev_nucleus_start, _ = prev.nucleus_span()
            _, u_nucleus_end = u.nucleus_span()
            merged[-1] = replace(
                prev, end_sec=u.end_sec,
                nucleus_start_sec=prev_nucleus_start, nucleus_end_sec=u_nucleus_end,
                aperture_consonant_run=prev.aperture_consonant_run + u.aperture_consonant_run,
            )
        else:
            merged.append(u)
    return merged, merged_morae


def _smoothed_rms(rms):
    times, values = rms.times_sec, rms.values
    if len(times) < 2:
        return times, values
    hop = float(np.median(np.diff(times)))
    if hop <= 0:
        return times, values
    window_samples = max(1, round(_ONSET_SMOOTH_WINDOW_SEC / hop))
    half_window = max(1, (window_samples - 1) // 2)
    kernel = np.ones(2 * half_window + 1) / (2 * half_window + 1)
    # np.convolve の mode="same" は端をゼロ埋めするので、平坦な信号の端に偽の傾きが立つ。
    padded = np.pad(values, half_window, mode="edge")
    return times, np.convolve(padded, kernel, mode="valid")


def _find_onset(times, smoothed, center_sec):
    if len(times) < 2:
        return None
    lo, hi = center_sec - _ONSET_WINDOW_SEC, center_sec + _ONSET_WINDOW_SEC
    candidates = []
    for i in range(1, len(times)):
        if times[i] < lo or times[i] > hi:
            continue
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        slope = (smoothed[i] - smoothed[i - 1]) / (dt / _ONSET_SLOPE_UNIT_SEC)
        if slope > _ONSET_SLOPE_PER_10MS:
            candidates.append(times[i])
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t - center_sec))


def _refine_onsets(units, rms):
    times, smoothed = _smoothed_rms(rms)
    result = list(units)
    for i, u in enumerate(result):
        if u.kind != "vowel":
            continue
        onset = _find_onset(times, smoothed, u.start_sec)
        if onset is None:
            continue
        lower_bound = result[i - 1].start_sec + _ONSET_MIN_SPAN_SEC if i > 0 else float("-inf")
        upper_bound = u.end_sec - _ONSET_MIN_SPAN_SEC
        if lower_bound > upper_bound:
            continue
        onset = min(max(onset, lower_bound), upper_bound)
        if onset == u.start_sec:
            continue
        if i > 0:
            prev = result[i - 1]
            _, prev_nucleus_end = prev.nucleus_span()
            result[i - 1] = replace(prev, end_sec=onset, nucleus_end_sec=min(prev_nucleus_end, onset))
        result[i] = replace(u, start_sec=onset, nucleus_start_sec=onset)
    return result


def _map_open_amount(rms_value, *, open_lo, open_hi, open_max, intensity_curve):
    raw = rms_value ** intensity_curve
    clamped = min(max(raw, open_lo), open_hi)
    return min(clamped, open_max)


def _map_open_amount_continuous(normalized, *, open_lo, open_hi, open_max, intensity_curve):
    cap = min(open_hi, open_max)
    floor = min(open_lo, cap)
    return floor + (cap - floor) * (normalized ** intensity_curve)


def _is_weak_vowel(confidence, rms_value):
    if confidence is not None:
        return confidence < _WEAK_CONFIDENCE_THRESHOLD and rms_value < _WEAK_RMS_WITH_CONFIDENCE
    return rms_value < _WEAK_RMS_WITHOUT_CONFIDENCE


def _percentile_bounds(values):
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    p_lo = np.percentile(arr, _OPEN_RENORM_P_LO, method="linear")
    p_hi = np.percentile(arr, _OPEN_RENORM_P_HI, method="linear")
    return p_lo, p_hi


def _normalize_with_bounds(value, p_lo, p_hi):
    if p_hi == p_lo:
        return _DEGENERATE_NORMALIZED_RMS
    return float(np.clip((value - p_lo) / (p_hi - p_lo), 0.0, 1.0))


def confirm_mouth_events(segments, rms, *, open_lo, open_hi, open_max, intensity_curve, silence_on,
                          use_n_morph=False) -> tuple[list[MouthEvent], EventDiagnostics, list[int]]:
    """MouthEvent の時刻は 30fps のフレーム単位。3件目は、母音的口形の各モーラが連続して何件の
    MouthEvent になったかの列。"""
    if not segments:
        return [], EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=0), []

    low_dynamics = rms.dynamic_range_db < _LOW_DYNAMICS_THRESHOLD_DB
    units = _classify_phonetic(segments, use_n_morph)
    units = _resolve_silence(units, rms, silence_on, low_dynamics)
    units, merged_before_onsets = _merge_adjacent(units)
    units = _refine_onsets(units, rms)
    units, merged_after_onsets = _merge_adjacent(units)
    merged_morae = merged_before_onsets + merged_after_onsets

    mora_rms_raw = [
        _mora_representative_rms(rms, u) if u.kind in _MORA_KINDS else None for u in units
    ]
    p_lo, p_hi = _percentile_bounds([v for v in mora_rms_raw if v is not None])

    mouth_events = []
    mora_event_group_sizes = []
    weak_vowels = 0
    for u, mora_rms in zip(units, mora_rms_raw, strict=True):
        if u.kind in _MORA_KINDS:
            is_weak = u.kind == "vowel" and _is_weak_vowel(u.confidence, mora_rms)
            if is_weak:
                weak_vowels += 1
            if u.kind == "vowel":
                shape = _VOWEL_SHAPES[u.letter]
                head_consonant_class = _consonant_class(u.head_consonant_phoneme)
                head_aperture_class = _strongest_aperture_class(u.aperture_consonant_run)
            else:
                shape = MouthShape.N
                head_consonant_class = ConsonantClass.NONE
                head_aperture_class = ApertureClass.NONE

            duration = u.end_sec - u.start_sec
            if duration >= _LONG_MORA_THRESHOLD_SEC:
                bounds = _split_into_subwindows(u.start_sec, u.end_sec)
            else:
                bounds = [(u.start_sec, u.end_sec)]

            for i, (sub_start, sub_end) in enumerate(bounds):
                sub_rms = mora_rms if len(bounds) == 1 else _mora_rms(rms, sub_start, sub_end)
                normalized = _normalize_with_bounds(sub_rms, p_lo, p_hi)
                if len(bounds) > 1:
                    open_amount = _map_open_amount_continuous(
                        normalized, open_lo=open_lo, open_hi=open_hi,
                        open_max=open_max, intensity_curve=intensity_curve,
                    )
                else:
                    open_amount = _map_open_amount(
                        normalized, open_lo=open_lo, open_hi=open_hi,
                        open_max=open_max, intensity_curve=intensity_curve,
                    )
                if is_weak:
                    open_amount *= _WEAK_SCALE
                if i == 0:
                    consonant_class, aperture_class = head_consonant_class, head_aperture_class
                else:
                    consonant_class, aperture_class = ConsonantClass.NONE, ApertureClass.NONE
                mouth_events.append(MouthEvent(
                    shape=shape, start=sub_start * FRAME_RATE, end=sub_end * FRAME_RATE,
                    open_amount=open_amount, consonant_class=consonant_class,
                    aperture_class=aperture_class,
                ))
            mora_event_group_sizes.append(len(bounds))
        else:
            shape = MouthShape.BILABIAL if u.kind == "bilabial" else MouthShape.SILENCE
            mouth_events.append(MouthEvent(
                shape=shape, start=u.start_sec * FRAME_RATE, end=u.end_sec * FRAME_RATE,
                open_amount=0.0, consonant_class=ConsonantClass.NONE, aperture_class=ApertureClass.NONE,
            ))

    return mouth_events, EventDiagnostics(
        weak_vowels=weak_vowels, low_dynamics=low_dynamics, merged_morae=merged_morae
    ), mora_event_group_sizes
