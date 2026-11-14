GAMMA_DEFAULT = 0.6

_VELOCITY_MAX = 127


def velocity_to_open(
    velocity: int,
    *,
    lo: float,
    hi: float,
    open_max: float,
    gamma: float = GAMMA_DEFAULT,
) -> float:
    n = velocity / _VELOCITY_MAX
    value = lo + (hi - lo) * (n ** gamma)
    value = min(max(value, lo), hi)
    return min(value, open_max)


def uses_default_open(velocities: list[int]) -> bool:
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
    if not velocities:
        return []
    if uses_default_open(velocities):
        return [min(default_open, open_max)] * len(velocities)
    return [
        velocity_to_open(v, lo=lo, hi=hi, open_max=open_max, gamma=gamma)
        for v in velocities
    ]
