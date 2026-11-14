from dataclasses import dataclass

from lipsync import MouthShape

_MORA_SHAPES = frozenset({MouthShape.A, MouthShape.I, MouthShape.U, MouthShape.E, MouthShape.O, MouthShape.N})
_CLOSED_SHAPES = frozenset({MouthShape.BILABIAL, MouthShape.SILENCE})

_RUN_FIELD_ORDER = (
    "output", "keys", "backends", "style", "separated", "phonemes", "morae", "merged_morae",
    "weak_vowels", "coverage", "closed_ranges", "max_opening", "duration_sec",
)


@dataclass(frozen=True)
class MoraReport:
    shape: str
    open_amount: float
    hold_frames: float


@dataclass(frozen=True)
class Diagnostics:
    backends: dict
    style: str
    separated: bool
    phonemes: int
    morae: int
    merged_morae: int
    weak_vowels: int
    coverage: float
    closed_ranges: int
    max_opening: float
    duration_sec: float
    keys: int
    mora_details: tuple[MoraReport, ...]
    low_dynamics: bool
    forced_split: bool


def _group_mora_events(mouth_events, mora_event_group_sizes):
    groups = []
    sizes = iter(mora_event_group_sizes)
    i, n = 0, len(mouth_events)
    while i < n:
        if mouth_events[i].shape in _MORA_SHAPES:
            count = next(sizes)
            groups.append(mouth_events[i:i + count])
            i += count
        else:
            i += 1
    return groups


def build_diagnostics(*, segments, mouth_events, event_diagnostics, mora_event_group_sizes, backends,
                       style, separated, duration_sec, keys, forced_split) -> Diagnostics:
    """mora_event_group_sizes は mouth_events の母音的口形のうち、連続して1モーラを成す件数の列。"""
    phonemes = sum(1 for s in segments if s.type in ("vowel", "consonant"))
    gap_duration = sum(s.end_sec - s.start_sec for s in segments if s.type == "gap")
    coverage = 1.0 - gap_duration / duration_sec if duration_sec > 0 else 0.0
    mora_groups = _group_mora_events(mouth_events, mora_event_group_sizes)
    morae = len(mora_groups)
    closed_ranges = sum(1 for e in mouth_events if e.shape in _CLOSED_SHAPES)
    max_opening = max((e.open_amount for e in mouth_events), default=0.0)
    mora_details = tuple(
        MoraReport(
            shape=group[0].shape.value,
            open_amount=sum(e.open_amount for e in group) / len(group),
            hold_frames=sum(e.end - e.start for e in group),
        )
        for group in mora_groups
    )
    return Diagnostics(
        backends=dict(backends), style=style, separated=separated, phonemes=phonemes, morae=morae,
        merged_morae=event_diagnostics.merged_morae, weak_vowels=event_diagnostics.weak_vowels,
        coverage=coverage, closed_ranges=closed_ranges,
        max_opening=max_opening, duration_sec=duration_sec, keys=keys, mora_details=mora_details,
        low_dynamics=event_diagnostics.low_dynamics, forced_split=forced_split,
    )


def render_report_text(diag: Diagnostics, params: dict) -> str:
    lines = [
        f"separator: {diag.backends.get('separator')}",
        f"recognizer: {diag.backends.get('recognizer')}",
        f"forced_aligner: {diag.backends.get('forced_aligner')}",
        f"english_katakana_method: {diag.backends.get('english_katakana_method')}",
        f"style: {diag.style}",
    ]
    for name, value in params.items():
        lines.append(f"{name}: {value}")
    lines.append(f"separated: {diag.separated}")
    lines.append(f"phonemes: {diag.phonemes}")
    lines.append(f"morae: {diag.morae}")
    lines.append(f"coverage: {diag.coverage:.4f}")
    for i, mora in enumerate(diag.mora_details, start=1):
        lines.append(
            f"mora[{i}]: shape={mora.shape} open_amount={mora.open_amount:.4f} "
            f"hold_frames={mora.hold_frames:.2f}"
        )
    lines.append(f"merged_morae: {diag.merged_morae}")
    lines.append(f"weak_vowels: {diag.weak_vowels}")
    lines.append(f"closed_ranges: {diag.closed_ranges}")
    lines.append(f"max_opening: {diag.max_opening:.4f}")
    lines.append(f"keys: {diag.keys}")
    lines.append(f"duration_sec: {diag.duration_sec:.3f}")
    return "\n".join(lines) + "\n"


def result_run_fields(diag: Diagnostics, output) -> dict:
    values = {
        "output": output, "keys": diag.keys, "backends": dict(diag.backends), "style": diag.style,
        "separated": diag.separated, "phonemes": diag.phonemes, "morae": diag.morae,
        "merged_morae": diag.merged_morae, "weak_vowels": diag.weak_vowels,
        "coverage": diag.coverage,
        "closed_ranges": diag.closed_ranges, "max_opening": diag.max_opening,
        "duration_sec": diag.duration_sec,
    }
    return {name: values[name] for name in _RUN_FIELD_ORDER}


def result_inspect_fields(diag: Diagnostics, *, input_kind, sample_rate, channels) -> dict:
    fields = result_run_fields(diag, output=None)
    fields.update(input_kind=input_kind, sample_rate=sample_rate, channels=channels)
    return fields
