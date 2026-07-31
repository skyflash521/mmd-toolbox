"""ベロシティ→開き量の写像(入口処理)。

vpr の音符ベロシティ(0–127、常在)を各モーラの開き量(`MouthEvent.open_amount`、0〜1)へ
決定論的に写像する。写像式と既定の扱いは各関数の docstring に持つ。開き量は lipsync の合成保持値の
スケールに使われ、合成総量の上限クランプ(open_cap)は lipsync 側が行う。
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
    """単一ベロシティ(0–127)を開き量へ写像する。

    `open = lo + (hi - lo) * (velocity / 127) ** gamma` を開き量レンジ `[lo, hi]` へクランプし、
    `open_max` を上限とする。
    """
    n = velocity / 127
    value = lo + (hi - lo) * (n ** gamma)
    value = min(max(value, lo), hi)  # 写像レンジ [lo, hi] へクランプ
    return min(value, open_max)  # --open-max を上限とする


def uses_default_open(velocities: list[int]) -> bool:
    """ベロシティ列が一様で、開き量が既定開き量に退化するか。

    退化の条件は open_amounts の分岐そのもの。決定経路を呼び出し側が知るための問い合わせ口で、
    同じ条件を呼び出し側へ書き写さないために置く。
    """
    return bool(velocities) and min(velocities) == max(velocities)


def open_amounts(
    velocities: list[int],
    *,
    lo: float,
    hi: float,
    open_max: float,
    default_open: float,
    gamma: float = GAMMA_DEFAULT,
) -> list[float]:
    """ベロシティ列を各要素の開き量へ写像する。

    全ベロシティが一様(強弱差 0)のときは写像式が定数に退化するため、全要素へ既定開き量
    `default_open` を用いる(`open_max` での頭打ちはしない。既定開き量の最終上限は lipsync の
    `open_cap` が担う)。強弱差があれば各ベロシティを `velocity_to_open` で個別に写像する。空入力は
    空列。
    """
    if not velocities:
        return []
    if uses_default_open(velocities):
        return [default_open] * len(velocities)
    return [
        velocity_to_open(v, lo=lo, hi=hi, open_max=open_max, gamma=gamma)
        for v in velocities
    ]
