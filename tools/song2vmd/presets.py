from dataclasses import dataclass

_PRESETS = {
    "pop": {
        "open_lo": 0.30, "open_hi": 0.75, "attack": 2, "release": 2,
        "coarticulation": 6, "anticipation": 11, "min_hold": 1, "exaggeration": 1.0, "open_max": 0.90,
        "vowel_scale": (1.30, 1.20, 1.70, 0.80, 1.70, 1.00),
        "triangle_min": 2.0,
        "vibrato_threshold": 10, "vibrato_amp": 0.05, "vibrato_period": 22,
        "legato_valley_shallow": 0.45, "legato_valley_deep": 0.30, "legato_valley_slope": 0.02,
    },
    "ballad": {
        "open_lo": 0.20, "open_hi": 0.55, "attack": 3, "release": 3,
        "coarticulation": 3, "anticipation": 1, "min_hold": 4, "exaggeration": 0.8, "open_max": 0.70,
        "vowel_scale": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "triangle_min": 2.0,
        "vibrato_threshold": 18, "vibrato_amp": 0.05, "vibrato_period": 15,
        "legato_valley_shallow": 0.4, "legato_valley_deep": 0.2, "legato_valley_slope": 0.025,
    },
    "powerful": {
        "open_lo": 0.40, "open_hi": 0.95, "attack": 1, "release": 1,
        "coarticulation": 2, "anticipation": 2, "min_hold": 3, "exaggeration": 1.3, "open_max": 0.97,
        "vowel_scale": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "triangle_min": 2.0,
        "vibrato_threshold": 18, "vibrato_amp": 0.05, "vibrato_period": 15,
        "legato_valley_shallow": 0.4, "legato_valley_deep": 0.2, "legato_valley_slope": 0.025,
    },
    "whisper": {
        "open_lo": 0.10, "open_hi": 0.35, "attack": 2, "release": 2,
        "coarticulation": 2, "anticipation": 1, "min_hold": 3, "exaggeration": 0.7, "open_max": 0.50,
        "vowel_scale": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "triangle_min": 2.0,
        "vibrato_threshold": 18, "vibrato_amp": 0.05, "vibrato_period": 15,
        "legato_valley_shallow": 0.4, "legato_valley_deep": 0.2, "legato_valley_slope": 0.025,
    },
    "rap": {
        "open_lo": 0.30, "open_hi": 0.70, "attack": 1, "release": 1,
        "coarticulation": 1, "anticipation": 1, "min_hold": 2, "exaggeration": 1.0, "open_max": 0.85,
        "vowel_scale": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "triangle_min": 2.0,
        "vibrato_threshold": 18, "vibrato_amp": 0.05, "vibrato_period": 15,
        "legato_valley_shallow": 0.4, "legato_valley_deep": 0.2, "legato_valley_slope": 0.025,
    },
}

STYLE_NAMES = tuple(_PRESETS)


@dataclass(frozen=True)
class OpennessParams:
    open_lo: float
    open_hi: float
    open_max: float


@dataclass(frozen=True)
class StyleGenParams:
    """vowel_scale の成分順は (a, i, u, e, o, ん)。"""

    attack_frames: int
    release_frames: int
    coartic_overlap_max: int
    anticipation_frames: int
    min_hold_frames: int
    exaggeration: float
    vowel_scale: tuple[float, float, float, float, float, float]
    triangle_min_frames: float
    vibrato_threshold: int
    vibrato_amp: float
    vibrato_period: int
    legato_valley_shallow: float
    legato_valley_deep: float
    legato_valley_slope: float


def resolve(style, *, open_max=None, coarticulation=None, anticipation=None, min_hold=None,
            vowel_gain=(1.0, 1.0, 1.0, 1.0, 1.0)) -> tuple[OpennessParams, StyleGenParams]:
    """None の引数はプリセット値を使う。vowel_gain の成分順は (a, i, u, e, o)。"""
    preset = _PRESETS[style]
    openness = OpennessParams(
        open_lo=preset["open_lo"],
        open_hi=preset["open_hi"],
        open_max=open_max if open_max is not None else preset["open_max"],
    )
    preset_scale = preset["vowel_scale"]
    vowel_scale = tuple(p * g for p, g in zip(preset_scale[:5], vowel_gain, strict=True)) + (preset_scale[5],)
    gen = StyleGenParams(
        attack_frames=preset["attack"],
        release_frames=preset["release"],
        coartic_overlap_max=coarticulation if coarticulation is not None else preset["coarticulation"],
        anticipation_frames=anticipation if anticipation is not None else preset["anticipation"],
        min_hold_frames=min_hold if min_hold is not None else preset["min_hold"],
        exaggeration=preset["exaggeration"],
        vowel_scale=vowel_scale,
        triangle_min_frames=preset["triangle_min"],
        vibrato_threshold=preset["vibrato_threshold"],
        vibrato_amp=preset["vibrato_amp"],
        vibrato_period=preset["vibrato_period"],
        legato_valley_shallow=preset["legato_valley_shallow"],
        legato_valley_deep=preset["legato_valley_deep"],
        legato_valley_slope=preset["legato_valley_slope"],
    )
    return openness, gen


def describe_values():
    return {
        name: {
            "open_max": preset["open_max"],
            "coarticulation": preset["coarticulation"],
            "anticipation": preset["anticipation"],
            "min_hold": preset["min_hold"],
        }
        for name, preset in _PRESETS.items()
    }
