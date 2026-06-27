"""ベロシティ→開き量の写像(vpr2vmd.md §3 入口処理)。

vpr の音符ベロシティ(0–127、常在)を各モーラの開き量(`MouthEvent.open_amount`、0〜1)へ
決定論的に写像する。写像式・既定の扱いの正本は実装計画 §5.3。開き量は lipsync の合成保持値の
スケールに使われ、合成総量の上限クランプ(open_cap)は lipsync 側が行う
([lipsync 仕様](../lipsync/lipsync.md) §4.1)。
"""

GAMMA_DEFAULT = 0.6  # 累乗カーブの既定指数(弱音側を持ち上げる)


def velocity_to_open(
    velocity: int,
    *,
    lo: float,
    hi: float,
    open_max: float,
    gamma: float = GAMMA_DEFAULT,
) -> float:
    raise NotImplementedError("P-3 開き量写像")


def open_amounts(
    velocities: list[int],
    *,
    lo: float,
    hi: float,
    open_max: float,
    default_open: float,
    gamma: float = GAMMA_DEFAULT,
) -> list[float]:
    raise NotImplementedError("P-3 開き量写像")
