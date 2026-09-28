"""区間はフレームの添字で持ち、秒へ写すのは最後の1回に限る。

秒どうしの一致で隣接を判定すると、計算順の違う実数の丸めで休符を挟まない音符を隣接と見なさないことがある。
"""

from dataclasses import dataclass, field

import numpy as np

_MIN_NOTE_DURATION_SEC = 0.08

_PITCH_CHANGE_HOLD_SEC = 0.34

MORAIC_NASAL = "ɴ"

_MAX_UNATTACHED_START_AFTER_NUCLEUS_SEC = 2.0


@dataclass
class Diagnostics:
    suppressed_notes: int = 0


@dataclass(frozen=True)
class SplitResult:
    notes: "list[Note]" = field(default_factory=list)
    syllable_segments: "list[list]" = field(default_factory=list)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


@dataclass(frozen=True)
class Note:
    start_sec: float
    end_sec: float
    midi: int
    syllable: int


def _is_syllable_nucleus(segment):
    return segment.type == "vowel" or (segment.type == "consonant"
                                       and segment.phoneme == MORAIC_NASAL)


def _syllable_index_per_frame(times_sec, segments):
    syllable_index = np.full(len(times_sec), -1)
    is_consonant = np.zeros(len(times_sec), dtype=bool)

    nuclei = [s for s in segments if _is_syllable_nucleus(s)]
    for number, segment in enumerate(nuclei):
        covered = (times_sec >= segment.start_sec) & (times_sec < segment.end_sec)
        syllable_index[covered] = number
    for segment in segments:
        if segment.type == "consonant" and not _is_syllable_nucleus(segment):
            is_consonant |= (times_sec >= segment.start_sec) & (times_sec < segment.end_sec)

    ends = np.array([segment.end_sec for segment in nuclei], dtype=float)
    following_nucleus = np.searchsorted(ends, times_sec, side="right")
    following_nucleus = np.where(following_nucleus < len(nuclei), following_nucleus, -1)
    starts = np.array([segment.start_sec for segment in nuclei], dtype=float)
    preceding_nucleus = np.searchsorted(starts, times_sec, side="right") - 1

    syllable_index = np.where((syllable_index < 0) & is_consonant, following_nucleus,
                              syllable_index)
    for i in range(len(syllable_index)):
        if syllable_index[i] < 0:
            syllable_index[i] = max(int(preceding_nucleus[i]),
                                    int(syllable_index[i - 1]) if i > 0 else -1)
    return syllable_index, is_consonant


def _segments_per_syllable(segments, nuclei):
    result = [[] for _ in nuclei]
    next_nucleus = 0
    leading_consonants = []
    for segment in segments:
        if next_nucleus < len(nuclei) and segment is nuclei[next_nucleus]:
            result[next_nucleus] = [*leading_consonants, segment]
            leading_consonants = []
            next_nucleus += 1
        elif segment.type == "consonant":
            leading_consonants.append(segment)
    if leading_consonants and result:
        result[-1].extend(leading_consonants)
    return result


def _voiced_spans(track, syllable_index, frame_sec):
    voiced = np.asarray(track.voiced)
    values = np.nan_to_num(np.asarray(track.midi, dtype=float), nan=0.0)
    rounded = np.where(voiced, np.rint(values), np.nan)
    held_frames = max(1, int(np.ceil(_PITCH_CHANGE_HOLD_SEC / frame_sec)))

    spans = []
    start = None
    held = None

    def close(stop):
        window = np.rint(values[start:stop]).astype(int)
        counts = np.bincount(window - window.min())
        # argmax は同数の最初の要素を返す。
        spans.append([start, stop, int(window.min() + counts.argmax()),
                      int(syllable_index[start])])

    for i in range(len(voiced)):
        if not voiced[i]:
            if start is not None:
                close(i)
                start = None
            continue
        if start is None:
            start, held = i, int(rounded[i])
            continue
        stop = i + held_frames
        lasts = (rounded[i] != held and stop <= len(voiced)
                 and bool(voiced[i:stop].all())
                 and bool((rounded[i:stop] == rounded[i]).all()))
        if lasts or syllable_index[i] != syllable_index[i - 1]:
            close(i)
            start, held = i, int(rounded[i])
    if start is not None:
        close(len(voiced))
    return spans


def _extend_to_leading_consonants(spans, syllable_index, is_consonant):
    for position, span in enumerate(spans):
        lo = span[0]
        previous_end = spans[position - 1][1] if position > 0 else 0
        while (lo > previous_end and is_consonant[lo - 1]
               and syllable_index[lo - 1] == syllable_index[span[0]]):
            lo -= 1
        span[0] = lo
    return spans


def _absorb_short_spans(spans, frame_sec):
    result = [list(span) for span in spans]
    changed = True
    while changed:
        changed = False
        for i, (lo, hi, midi, syllable) in enumerate(result):
            if (hi - lo) * frame_sec >= _MIN_NOTE_DURATION_SEC:
                continue
            before = (i - 1 if i > 0 and result[i - 1][1] == lo and result[i - 1][3] == syllable
                      else None)
            after = (i + 1 if i + 1 < len(result) and result[i + 1][0] == hi
                     and result[i + 1][3] == syllable else None)
            if before is None and after is None:
                continue

            if before is not None and result[before][2] == midi:
                target = before
            elif after is not None and result[after][2] == midi:
                target = after
            elif before is None:
                target = after
            elif after is None:
                target = before
            else:
                distance_before = abs(result[before][2] - midi)
                distance_after = abs(result[after][2] - midi)
                target = before if distance_before <= distance_after else after

            if target == before:
                result[target][1] = hi
            else:
                result[target][0] = lo
            del result[i]
            changed = True
            break
    return result


def _same_syllable_runs(spans):
    result = []
    for span in spans:
        if result and result[-1][-1][1] == span[0] and result[-1][-1][3] == span[3]:
            result[-1].append(span)
        else:
            result.append([span])
    return result


def _suppress_unattached(spans, times, nuclei):
    kept = []
    for run in _same_syllable_runs(spans):
        syllable = run[0][3]
        start_sec = float(times[run[0][0]])
        if syllable < 0:
            continue
        if start_sec - nuclei[syllable].end_sec > _MAX_UNATTACHED_START_AFTER_NUCLEUS_SEC:
            continue
        kept += run
    suppressed_count = len(spans) - len(kept)
    return kept, suppressed_count


def split(track, segments, *, frame_sec=None) -> SplitResult:
    """切れ目なく続く音符は、前の end_sec と次の start_sec が同じ値になる。"""
    times = np.asarray(track.times_sec)
    if len(times) < 2:
        return SplitResult()
    if frame_sec is None:
        frame_sec = float(times[1] - times[0])

    nuclei = [s for s in segments if _is_syllable_nucleus(s)]
    syllable_index, is_consonant = _syllable_index_per_frame(times, segments)
    spans = _voiced_spans(track, syllable_index, frame_sec)
    spans = _extend_to_leading_consonants(spans, syllable_index, is_consonant)
    spans = _absorb_short_spans(spans, frame_sec)
    spans, suppressed = _suppress_unattached(spans, times, nuclei)

    result = [Note(start_sec=float(times[lo]),
                   end_sec=float(times[hi]) if hi < len(times)
                   else float(times[hi - 1]) + frame_sec,
                   midi=midi, syllable=syllable)
              for lo, hi, midi, syllable in spans]
    return SplitResult(notes=result,
                       syllable_segments=_segments_per_syllable(segments, nuclei),
                       diagnostics=Diagnostics(suppressed_notes=suppressed))
