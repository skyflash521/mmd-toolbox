from vpr import Note, TempoEvent

_FPS = 30.0
_SECONDS_PER_MINUTE = 60.0


def _seconds_per_tick(bpm: float, resolution: int) -> float:
    return _SECONDS_PER_MINUTE / (bpm * resolution)


def tick_to_seconds(tick: int, tempos: list[TempoEvent], resolution: int) -> float:
    events = sorted(tempos, key=lambda e: e.tick)
    seconds = 0.0
    region_start = 0
    for i, event in enumerate(events):
        spt = _seconds_per_tick(event.bpm, resolution)
        next_tick = events[i + 1].tick if i + 1 < len(events) else None
        if next_tick is None or tick <= next_tick:
            return seconds + (tick - region_start) * spt
        seconds += (next_tick - region_start) * spt
        region_start = next_tick
    return seconds


def tick_to_frame(tick: int, tempos: list[TempoEvent], resolution: int) -> float:
    return tick_to_seconds(tick, tempos, resolution) * _FPS


def _sounding_seconds(note: Note, tempos: list[TempoEvent], resolution: int) -> float:
    end = note.start_tick + note.duration_tick
    return tick_to_seconds(end, tempos, resolution) - tick_to_seconds(
        note.start_tick, tempos, resolution
    )


def note_effective_bpm(
    note: Note, tempos: list[TempoEvent], resolution: int
) -> float | None:
    if note.duration_tick <= 0:
        return None
    seconds = _sounding_seconds(note, tempos, resolution)
    if seconds <= 0.0:
        return None
    beats = note.duration_tick / resolution
    return _SECONDS_PER_MINUTE * beats / seconds


def representative_bpm(
    adopted_notes: list[Note],
    tempos: list[TempoEvent],
    resolution: int,
    *,
    default_bpm: float = 120.0,
) -> float:
    bpms_and_weights: list[tuple[float, float]] = []
    for note in adopted_notes:
        bpm = note_effective_bpm(note, tempos, resolution)
        if bpm is None or bpm <= 0.0:
            continue
        weight = _sounding_seconds(note, tempos, resolution) * _FPS
        if weight <= 0.0:
            continue
        bpms_and_weights.append((bpm, weight))
    if not bpms_and_weights:
        return default_bpm
    bpms_and_weights.sort(key=lambda pair: pair[0])
    half = sum(weight for _, weight in bpms_and_weights) / 2.0
    cumulative = 0.0
    for bpm, weight in bpms_and_weights[:-1]:
        cumulative += weight
        if cumulative >= half:
            return bpm
    return bpms_and_weights[-1][0]
