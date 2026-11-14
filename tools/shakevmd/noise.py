import hashlib

import numpy as np

DEFAULT_OCTAVES = 3
DEFAULT_PERSISTENCE = 0.5
BANDLIMIT_HZ = 8.0

_U64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def derive_seed(base_seed: int, *parts) -> int:
    h = hashlib.sha256()
    h.update(repr(base_seed).encode("utf-8"))
    for p in parts:
        b = repr(p).encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return int.from_bytes(h.digest()[:8], "big")


def effective_octaves(
    freq: float, octaves: int, bandlimit_hz: float = BANDLIMIT_HZ
) -> int:
    f = abs(freq)
    n = 0
    for i in range(max(0, octaves)):
        if f * (2.0**i) <= bandlimit_hz:
            n += 1
        else:
            break
    return n


def _clamp_message(octaves: int, n: int, bandlimit_hz: float) -> str:
    return f"オクターブを {octaves}→{n} にクランプした(実効周波数が {bandlimit_hz}Hz を超過)"


def _quintic_fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _splitmix64_gradients(seed: int, idx: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (idx.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) & _U64
        z = (z ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)) & _U64
        z = (z + np.uint64(0x9E3779B97F4A7C15)) & _U64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _U64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _U64
        z = z ^ (z >> np.uint64(31))
    return (z.astype(np.float64) / 2.0**64) * 2.0 - 1.0


def _seed_phase(seed: int) -> float:
    g = float(_splitmix64_gradients(seed, np.array([0], dtype=np.int64))[0])
    return (g + 1.0) * 512.0


def _perlin1d(seed: int, x: np.ndarray) -> np.ndarray:
    i0 = np.floor(x).astype(np.int64)
    frac = x - i0
    g0 = _splitmix64_gradients(seed, i0)
    g1 = _splitmix64_gradients(seed, i0 + 1)
    n0 = g0 * frac
    n1 = g1 * (frac - 1.0)
    u = _quintic_fade(frac)
    return n0 + u * (n1 - n0)


def octave_components(
    seed: int,
    t,
    freq: float,
    *,
    octaves: int = DEFAULT_OCTAVES,
    bandlimit_hz: float = BANDLIMIT_HZ,
) -> tuple[list, list[str]]:
    """t の単位は秒。"""
    t = np.asarray(t, dtype=float)
    freq = abs(freq)
    n = effective_octaves(freq, octaves, bandlimit_hz)
    warns: list[str] = []
    if n < octaves:
        warns.append(_clamp_message(octaves, n, bandlimit_hz))
    comps = []
    for i in range(n):
        f_i = freq * (2.0**i)
        oct_seed = derive_seed(seed, "octave", i)
        phase = _seed_phase(oct_seed)
        comps.append(_perlin1d(oct_seed, f_i * t + phase))
    return comps, warns


def band_limited_noise(
    seed: int,
    t,
    freq: float,
    *,
    octaves: int = DEFAULT_OCTAVES,
    persistence: float = DEFAULT_PERSISTENCE,
    bandlimit_hz: float = BANDLIMIT_HZ,
) -> tuple[np.ndarray, list[str]]:
    """t の単位は秒。"""
    t = np.asarray(t, dtype=float)
    freq = abs(freq)
    n = effective_octaves(freq, octaves, bandlimit_hz)
    warns: list[str] = []
    if n < octaves:
        warns.append(_clamp_message(octaves, n, bandlimit_hz))

    out = np.zeros_like(t)
    for i in range(n):
        f_i = freq * (2.0**i)
        amp = persistence**i
        oct_seed = derive_seed(seed, "octave", i)
        phase = _seed_phase(oct_seed)
        out = out + amp * _perlin1d(oct_seed, f_i * t + phase)
    return out, warns
