"""リップモーションスタイルプリセットの解決。

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
    """プリセット1件の出発点値(開き量レンジ・上限・タイミング・連続感・誇張)。

    末尾の連続感パラメータ(三角形下限・伸び表現・レガート谷)は既定を `lipsync` の
    `GenerationParams` 既定相当に置き、視覚で詰めたスタイルだけが上書きする。これにより各スタイルが
    渡す生成パラメータが presets で完結する(`GenerationParams` 既定への暗黙依存を残さない)。
    """

    lo: float
    hi: float
    open_max: float
    attack: int
    release: int
    coartic_overlap: int
    anticipation: int
    min_hold: int
    exaggeration: float
    # 母音的口形別(a, i, u, e, o, n)の開き量倍率。既定は全口形 1 倍(母音別の差をつけない)。
    vowel_scale: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    triangle_min: float = 2.0
    vibrato_threshold: int = 18
    vibrato_amp: float = 0.05
    vibrato_period: int = 15
    legato_valley_shallow: float = 0.4
    legato_valley_deep: float = 0.2
    legato_valley_slope: float = 0.025


# スタイルごとの出発点値(開き量レンジ・上限・タイミング・連続感・誇張。実データで調整しうる)。
# _Preset(lo, hi, open_max, attack, release, coartic_overlap, anticipation, min_hold, exaggeration, ...)
# pop は視覚チューニング(MMD目視)で定めた標準値(緩やかな先行準備・広い協調調音・レガート谷・伸び表現を
# 含む)。他スタイルは連続感パラメータを既定のまま据え置き、必要になったら個別に視覚で詰める。
_PRESETS: dict[str, _Preset] = {
    "pop": _Preset(
        0.30, 0.75, 0.90, 2, 2, 6, 11, 3, 1.0,
        vowel_scale=(1.30, 1.20, 1.70, 0.80, 1.70, 1.00),
        vibrato_threshold=10, vibrato_amp=0.05, vibrato_period=22,
        legato_valley_shallow=0.45, legato_valley_deep=0.30, legato_valley_slope=0.02,
    ),
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
    """スタイル名と任意の上書きから開き量写像・生成パラメータを解決する。

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
        exaggeration=preset.exaggeration,
        vibrato_threshold=preset.vibrato_threshold,
        vibrato_amp=preset.vibrato_amp,
        vibrato_period=preset.vibrato_period,
    )
    return openness, params
