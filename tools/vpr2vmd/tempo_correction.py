from dataclasses import replace

from lipsync import GenerationParams

_MIN_HOLD_FLOOR = 1
_ATTACK_RELEASE_FLOOR = 2


def _shrunk(value: int, scale: float, floor: int) -> int:
    return min(value, max(floor, round(value * scale)))


def apply_tempo_correction(
    params: GenerationParams,
    representative_bpm: float,
    *,
    ref_bpm: float = 120.0,
    s_min: float = 0.5,
) -> GenerationParams:
    frames_per_beat = (60.0 / representative_bpm) * 30.0
    ref_frames_per_beat = (60.0 / ref_bpm) * 30.0
    scale = max(s_min, min(frames_per_beat / ref_frames_per_beat, 1.0))
    return replace(
        params,
        min_hold_frames=_shrunk(params.min_hold_frames, scale, _MIN_HOLD_FLOOR),
        attack_frames=_shrunk(params.attack_frames, scale, _ATTACK_RELEASE_FLOOR),
        release_frames=_shrunk(params.release_frames, scale, _ATTACK_RELEASE_FLOOR),
    )
