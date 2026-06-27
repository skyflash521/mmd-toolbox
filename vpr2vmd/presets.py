"""口パクスタイルプリセットの解決(vpr2vmd.md §4)。

`--style` のプリセット名と、任意の上書き(`--open-max`/`--default-open`)から、開き量写像の
パラメータ(openness)と lipsync の生成パラメータ(GenerationParams)を決定論的に解決する。
プリセットごとの具体値は実装計画 §5.3 の出発点に従う。
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


def resolve(
    style: str,
    open_max: float | None = None,
    default_open: float | None = None,
) -> tuple[OpennessParams, GenerationParams]:
    raise NotImplementedError("P-4c-1 プリセット解決")
