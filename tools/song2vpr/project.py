import math
from dataclasses import dataclass, field
from typing import NamedTuple

from vpr import Note, Part, TempoEvent, TimeSignature, Track, VoiceBank, VprProject

from .lyrics import CONTINUATION, SungNote

_MIDI_NOTE_RANGE = (0, 127)

_MIN_SINGABLE_SEC = 0.028

_PLACEHOLDER_VOICE = VoiceBank(comp_id="BHKCEKSYNBXG3HB2", name="HATSUNE_MIKU_V6_ORIGINAL")


@dataclass
class Diagnostics:
    note_count: int = 0
    pitch_clamped_notes: int = 0
    quantized_merged_notes: int = 0
    quantized_stretched_notes: int = 0
    short_notes: int = 0


@dataclass
class BuildResult:
    project: VprProject
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


class _TickSpan(NamedTuple):
    start_tick: int
    end_tick: int
    longest_by_seconds: SungNote
    syllable_head: SungNote | None


def _ticks_per_bar(tempo) -> int:
    return tempo.numerator * tempo.resolution * 4 // tempo.denominator


def _clamp_pitch(midi, diagnostics):
    clamped = max(_MIDI_NOTE_RANGE[0], min(_MIDI_NOTE_RANGE[1], midi))
    if clamped != midi:
        diagnostics.pitch_clamped_notes += 1
    return clamped


def _merge_same_start_tick(quantized, diagnostics) -> list[_TickSpan]:
    merged = []
    for start_tick, end_tick, note in quantized:
        syllable_head = None if note.lyric == CONTINUATION else note
        if merged and merged[-1].start_tick == start_tick:
            kept_start, kept_end, longest, kept_head = merged[-1]
            if note.end_sec - note.start_sec > longest.end_sec - longest.start_sec:
                longest = note
            merged[-1] = _TickSpan(kept_start, max(kept_end, end_tick), longest,
                                   kept_head if kept_head is not None else syllable_head)
            diagnostics.quantized_merged_notes += 1
        else:
            merged.append(_TickSpan(start_tick, end_tick, note, syllable_head))
    return merged


def _stretch_zero_length(merged, diagnostics) -> list[_TickSpan]:
    spans = []
    for start_tick, end_tick, longest, syllable_head in merged:
        if end_tick <= start_tick:
            end_tick = start_tick + 1
            diagnostics.quantized_stretched_notes += 1
        spans.append(_TickSpan(start_tick, end_tick, longest, syllable_head))
    return spans


def _min_singable_ticks(tempo) -> int:
    return math.ceil(_MIN_SINGABLE_SEC * tempo.bpm * tempo.resolution / 60.0 - 1e-9)


def _redistribute_contiguous_run(run, floor):
    total = run[-1].start_tick + run[-1].duration_tick - run[0].start_tick
    surplus = total - len(run) * floor
    weights = [max(0, note.duration_tick - floor) for note in run]
    weight_total = sum(weights)
    integer_shares = [divmod(surplus * weight, weight_total) for weight in weights]
    durations = [floor + quotient for quotient, _ in integer_shares]
    by_largest_remainder = sorted(range(len(run)), key=lambda i: (-integer_shares[i][1], i))
    for i in by_largest_remainder[:total - sum(durations)]:
        durations[i] += 1

    redistributed = []
    position = run[0].start_tick
    for note, duration in zip(run, durations, strict=True):
        redistributed.append(Note(
            start_tick=position, duration_tick=duration, pitch=note.pitch,
            lyric=note.lyric, velocity=note.velocity, phonemes=note.phonemes,
            is_protected=note.is_protected))
        position += duration
    return redistributed


def _ensure_singable(notes, floor, diagnostics):
    contiguous_runs = []
    for note in notes:
        if (contiguous_runs
                and contiguous_runs[-1][-1].start_tick + contiguous_runs[-1][-1].duration_tick
                == note.start_tick):
            contiguous_runs[-1].append(note)
        else:
            contiguous_runs.append([note])

    result = []
    for index, run in enumerate(contiguous_runs):
        if not any(note.lyric != CONTINUATION and note.duration_tick < floor for note in run):
            result.extend(run)
            continue
        if len(run) == 1:
            note = run[0]
            limit = floor
            if index + 1 < len(contiguous_runs):
                limit = min(floor, contiguous_runs[index + 1][0].start_tick - note.start_tick)
            duration = max(note.duration_tick, limit)
            if duration < floor:
                diagnostics.short_notes += 1
            result.append(Note(start_tick=note.start_tick, duration_tick=duration,
                               pitch=note.pitch, lyric=note.lyric, velocity=note.velocity,
                               phonemes=note.phonemes, is_protected=note.is_protected))
            continue
        total = run[-1].start_tick + run[-1].duration_tick - run[0].start_tick
        if total < len(run) * floor:
            diagnostics.short_notes += sum(1 for note in run if note.duration_tick < floor)
            result.extend(run)
            continue
        result.extend(_redistribute_contiguous_run(run, floor))
    return result


def build(notes, tempo, *, name: str) -> BuildResult:
    """notes の各音符は互いに区間が重ならないこと。"""
    diagnostics = Diagnostics()
    quantized = [(tempo.to_tick(note.start_sec), tempo.to_tick(note.end_sec), note)
                 for note in sorted(notes, key=lambda note: note.start_sec)]
    spans = _stretch_zero_length(_merge_same_start_tick(quantized, diagnostics), diagnostics)

    written = [Note(start_tick=start_tick, duration_tick=end_tick - start_tick,
                    pitch=_clamp_pitch(longest.midi, diagnostics), velocity=longest.velocity,
                    lyric=(syllable_head or longest).lyric,
                    phonemes=list((syllable_head or longest).phonemes),
                    is_protected=(syllable_head or longest).is_protected)
               for start_tick, end_tick, longest, syllable_head in spans]
    written = _ensure_singable(written, _min_singable_ticks(tempo), diagnostics)

    diagnostics.note_count = len(written)

    duration_tick = (written[-1].start_tick + written[-1].duration_tick
                     if written else _ticks_per_bar(tempo))
    part = Part(name=name, start_tick=0, duration_tick=duration_tick, voice=_PLACEHOLDER_VOICE,
                notes=written)

    project = VprProject(
        resolution=tempo.resolution,
        tempos=[TempoEvent(tick=0, bpm=tempo.bpm)],
        time_signatures=[TimeSignature(tick=0, numerator=tempo.numerator,
                                       denominator=tempo.denominator)],
        tracks=[Track(name=name, parts=[part])],
        title=name,
    )
    return BuildResult(project=project, diagnostics=diagnostics)
