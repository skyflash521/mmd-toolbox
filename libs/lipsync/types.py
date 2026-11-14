from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MouthShape(Enum):
    A = "a"
    I = "i"
    U = "u"
    E = "e"
    O = "o"
    N = "n"
    BILABIAL = "bilabial"
    SILENCE = "silence"
    LEGATO_GAP = "legato_gap"


class ConsonantClass(Enum):
    NONE = "none"
    NEUTRAL = "neutral"
    ROUNDED = "rounded"
    SPREAD = "spread"


class ApertureClass(Enum):
    NONE = "none"
    FIRM_CLOSURE = "firm_closure"
    NARROW_CHANNEL = "narrow_channel"
    SLIGHT_CLOSURE = "slight_closure"


@dataclass
class MouthEvent:
    """start・end の単位は 30fps のフレーム。open_amount は 0〜1。

    consonant_class・aperture_class は、1つのモーラを複数の連続イベントで表すとき先頭のイベントにだけ付け、
    後続のイベントは NONE にする。
    """

    shape: MouthShape
    start: float
    end: float
    open_amount: float = 0.0
    consonant_class: ConsonantClass = ConsonantClass.NONE
    aperture_class: ApertureClass = ApertureClass.NONE


@dataclass
class GenerationParams:
    """vowel_scale の成分順は (a, i, u, e, o, n)。mora_valley_frames・mora_valley_min_gap_frames は非負。"""

    open_cap: float = 0.8
    vowel_scale: tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    attack_frames: int = 2
    release_frames: int = 2
    min_hold_frames: int = 3
    triangle_min_frames: float = 2.0
    coartic_overlap_max: int = 2
    anticipation_frames: int = 1
    legato_valley_shallow: float = 0.4
    legato_valley_deep: float = 0.2
    legato_valley_slope: float = 0.025
    exaggeration: float = 1.0
    vibrato_threshold: int = 18
    vibrato_amp: float = 0.05
    vibrato_period: int = 15
    mora_valley_frames: float = 12.0
    mora_valley_min_gap_frames: float = 4.0
