from collections import Counter
from dataclasses import dataclass, replace

from lipsync import MouthEvent, MouthShape
from vpr import Note, TempoEvent

from .mapping import note_mouth_events
from .phonemes import PhonemeCategory, categorize
from .timing import tick_to_frame

_VOWEL_LIKE = frozenset(
    {MouthShape.A, MouthShape.I, MouthShape.U, MouthShape.E, MouthShape.O, MouthShape.N}
)

_LEGATO_MAX_FRAMES = 8.0


def _classify_gap(
    left_shape: MouthShape | None,
    right_shape: MouthShape,
    gap_len: float,
    legato_max_frames: float,
) -> MouthShape:
    if left_shape not in _VOWEL_LIKE or right_shape not in _VOWEL_LIKE:
        return MouthShape.SILENCE
    if gap_len > legato_max_frames:
        return MouthShape.SILENCE
    return MouthShape.LEGATO_GAP


@dataclass(frozen=True)
class OverlapDiagnostics:
    """excluded は、開始位置が同じで重複した音符と、切り詰めで長さが0以下になった音符を除外した数。
    truncated は、終端を後続の音符の開始位置へ切り詰めて採用した音符の数。"""

    excluded: int
    truncated: int


@dataclass(frozen=True)
class EventDiagnostics:
    """vowel_undetermined は、音素から自前の口形イベントが1つも決まらなかった音符の数。これらの音符は直前の
    口形を続け、直前の口形が無ければ閉口になる。non_event_symbols は、自前の口形イベントを作らない音素記号
    (その他の子音と未知の記号)ごとの出現数。"""

    vowel_undetermined: int
    non_event_symbols: dict[str, int]


def resolve_overlaps(notes: list[Note]) -> tuple[list[Note], OverlapDiagnostics]:
    """notes は collect_notes が返す並びであること。"""
    deduped: list[Note] = []
    excluded = 0
    for note in notes:
        if deduped and deduped[-1].start_tick == note.start_tick:
            excluded += 1
            continue
        deduped.append(note)

    adopted: list[Note] = []
    truncated = 0
    for i, note in enumerate(deduped):
        original_end = note.start_tick + note.duration_tick
        end = original_end
        if i + 1 < len(deduped):
            end = min(end, deduped[i + 1].start_tick)
        if end <= note.start_tick:
            excluded += 1
            continue
        if end < original_end:
            truncated += 1
        adopted.append(replace(note, duration_tick=end - note.start_tick))
    return adopted, OverlapDiagnostics(excluded, truncated)


def build_mouth_events(
    adopted_notes: list[Note],
    tempos: list[TempoEvent],
    resolution: int,
    use_n_morph: bool = False,
    open_by_note: list[float] | None = None,
    legato_max_frames: float = _LEGATO_MAX_FRAMES,
) -> tuple[list[MouthEvent], EventDiagnostics]:
    """open_by_note は adopted_notes と同じ並び。"""
    result: list[MouthEvent] = []
    held_shape: MouthShape | None = None
    cursor = 0.0
    vowel_undetermined = 0
    non_event_symbols: Counter[str] = Counter()
    for i, note in enumerate(adopted_notes):
        for phoneme in note.phonemes:
            if categorize(phoneme) is PhonemeCategory.OTHER:
                non_event_symbols[phoneme] += 1
        note_open = open_by_note[i] if open_by_note is not None else 0.0
        start = tick_to_frame(note.start_tick, tempos, resolution)
        end = tick_to_frame(note.start_tick + note.duration_tick, tempos, resolution)
        note_events = note_mouth_events(note.phonemes, start, end, use_n_morph)
        if start > cursor:
            right_shape = (
                note_events[0].shape
                if note_events is not None
                else (held_shape if held_shape is not None else MouthShape.SILENCE)
            )
            gap_shape = _classify_gap(held_shape, right_shape, start - cursor, legato_max_frames)
            result.append(MouthEvent(gap_shape, cursor, start))
            if gap_shape is MouthShape.SILENCE:
                held_shape = MouthShape.SILENCE
        if note_events is None:
            vowel_undetermined += 1
            shape = held_shape if held_shape is not None else MouthShape.SILENCE
            open_amount = note_open if shape in _VOWEL_LIKE else 0.0
            result.append(MouthEvent(shape, start, end, open_amount))
        else:
            for event in note_events:
                open_amount = note_open if event.shape in _VOWEL_LIKE else 0.0
                result.append(replace(event, open_amount=open_amount))
            held_shape = note_events[-1].shape
        cursor = end
    return result, EventDiagnostics(vowel_undetermined, dict(non_event_symbols))
