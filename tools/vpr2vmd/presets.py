from dataclasses import dataclass

from lipsync import GenerationParams

from .openness import GAMMA_DEFAULT


@dataclass(frozen=True)
class OpennessParams:
    lo: float
    hi: float
    open_max: float
    default_open: float
    gamma: float


@dataclass(frozen=True)
class _Preset:
    """vowel_scale の成分順は (a, i, u, e, o, n)。"""

    lo: float
    hi: float
    open_max: float
    attack: int
    release: int
    coartic_overlap: int
    anticipation: int
    min_hold: int
    exaggeration: float
    vowel_scale: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    triangle_min: float = 2.0
    vibrato_threshold: int = 18
    vibrato_amp: float = 0.05
    vibrato_period: int = 15
    legato_valley_shallow: float = 0.4
    legato_valley_deep: float = 0.2
    legato_valley_slope: float = 0.025
    mora_valley_frames: float = 12.0
    mora_valley_min_gap_frames: float = 4.0


_PRESETS: dict[str, _Preset] = {
    "pop": _Preset(
        lo=0.30, hi=0.75, open_max=0.90, attack=2, release=2,
        coartic_overlap=6, anticipation=11, min_hold=3, exaggeration=1.0,
        vowel_scale=(1.30, 1.20, 1.70, 0.80, 1.70, 1.00),
        vibrato_threshold=10, vibrato_amp=0.05, vibrato_period=22,
        legato_valley_shallow=0.45, legato_valley_deep=0.30, legato_valley_slope=0.02,
    ),
    "ballad": _Preset(
        lo=0.20, hi=0.55, open_max=0.70, attack=3, release=3,
        coartic_overlap=3, anticipation=1, min_hold=4, exaggeration=0.8,
    ),
    "powerful": _Preset(
        lo=0.40, hi=0.95, open_max=0.97, attack=1, release=1,
        coartic_overlap=2, anticipation=2, min_hold=3, exaggeration=1.3,
    ),
    "whisper": _Preset(
        lo=0.10, hi=0.35, open_max=0.50, attack=2, release=2,
        coartic_overlap=2, anticipation=1, min_hold=3, exaggeration=0.7,
    ),
    "rap": _Preset(
        lo=0.30, hi=0.70, open_max=0.85, attack=1, release=1,
        coartic_overlap=1, anticipation=1, min_hold=2, exaggeration=1.0,
    ),
}


def resolve(
    style: str,
    open_max: float | None = None,
    default_open: float | None = None,
) -> tuple[OpennessParams, GenerationParams]:
    preset = _PRESETS[style]
    resolved_open_max = open_max if open_max is not None else preset.open_max
    resolved_default_open = (
        default_open if default_open is not None else (preset.lo + preset.hi) / 2
    )
    openness = OpennessParams(
        lo=preset.lo,
        hi=preset.hi,
        open_max=resolved_open_max,
        default_open=resolved_default_open,
        gamma=GAMMA_DEFAULT,
    )
    params = GenerationParams(
        open_cap=resolved_open_max,
        vowel_scale=preset.vowel_scale,
        attack_frames=preset.attack,
        release_frames=preset.release,
        min_hold_frames=preset.min_hold,
        triangle_min_frames=preset.triangle_min,
        coartic_overlap_max=preset.coartic_overlap,
        anticipation_frames=preset.anticipation,
        legato_valley_shallow=preset.legato_valley_shallow,
        legato_valley_deep=preset.legato_valley_deep,
        legato_valley_slope=preset.legato_valley_slope,
        mora_valley_frames=preset.mora_valley_frames,
        mora_valley_min_gap_frames=preset.mora_valley_min_gap_frames,
        exaggeration=preset.exaggeration,
        vibrato_threshold=preset.vibrato_threshold,
        vibrato_amp=preset.vibrato_amp,
        vibrato_period=preset.vibrato_period,
    )
    return openness, params
