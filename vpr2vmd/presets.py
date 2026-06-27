"""口パクスタイルプリセットの解決(vpr2vmd.md §4)。

`--style` のプリセット名と、任意の上書き(`--open-max`/`--default-open`)から、開き量写像の
パラメータ(openness)と lipsync の生成パラメータ(GenerationParams)を決定論的に解決する。
プリセットごとの具体値は本モジュールが出発点値として持つ(実データで調整しうる)。
"""

from dataclasses import dataclass

from lipsync import GenerationParams

GAMMA = 0.6  # ベロシティ→開き量の累乗指数(全スタイル共通の初期値)


@dataclass(frozen=True)
class OpennessParams:
    """ベロシティ→開き量写像のパラメータ(openness.open_amounts へ渡す)。"""

    lo: float
    hi: float
    open_max: float
    default_open: float
    gamma: float


@dataclass(frozen=True)
class _Preset:
    """プリセット1件の出発点値(開き量レンジ・上限・タイミング・誇張)。"""

    lo: float
    hi: float
    open_max: float
    attack: int
    release: int
    coartic_overlap: int
    anticipation: int
    min_hold: int
    exaggeration: float


# スタイルごとの出発点値(開き量レンジ・上限・タイミング・誇張。実データで調整しうる)。
# _Preset(lo, hi, open_max, attack, release, coartic_overlap, anticipation, min_hold, exaggeration)
_PRESETS: dict[str, _Preset] = {
    "pop": _Preset(0.30, 0.75, 0.90, 2, 2, 2, 1, 3, 1.0),
    "ballad": _Preset(0.20, 0.55, 0.70, 3, 3, 3, 1, 4, 0.8),
    "powerful": _Preset(0.40, 0.95, 0.97, 1, 1, 2, 2, 3, 1.3),
    "whisper": _Preset(0.10, 0.35, 0.50, 2, 2, 2, 1, 3, 0.7),
    "rap": _Preset(0.30, 0.70, 0.85, 1, 1, 1, 1, 2, 1.0),
}


def resolve(
    style: str,
    open_max: float | None = None,
    default_open: float | None = None,
) -> tuple[OpennessParams, GenerationParams]:
    """スタイル名と任意の上書きから開き量写像・生成パラメータを解決する(vpr2vmd.md §4)。

    `open_max` を渡すとそのスタイルの既定上限を上書きし、開き量写像の上限と lipsync の
    `open_cap` の両方に効く。`default_open` を渡すと一様ベロシティ時の既定開き量を上書きし、
    未指定なら開き量レンジの中央 `(lo + hi) / 2`。
    """
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
        gamma=GAMMA,
    )
    params = GenerationParams(
        open_cap=resolved_open_max,
        attack_frames=preset.attack,
        release_frames=preset.release,
        min_hold_frames=preset.min_hold,
        coartic_overlap_max=preset.coartic_overlap,
        anticipation_frames=preset.anticipation,
        exaggeration=preset.exaggeration,
    )
    return openness, params
