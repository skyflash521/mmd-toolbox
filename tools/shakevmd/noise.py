"""帯域制限付き多オクターブノイズ(shakevmd.md §6.1)。

- C1連続ノイズ(1Dグラディエント=パーリン系、quintic fade)。チャンネル/セグメント
  ごとに独立シード派生
- オクターブ合成: Σ persistence^i × noise(freq × 2^i × t)
- 帯域制限: 実効周波数(freq × 2^i)が bandlimit_hz を超えるオクターブは生成しない
  (自動クランプ+警告)。ベイクは30fps固定(ナイキスト15Hz)
"""

import hashlib

import numpy as np

# 内蔵値(プリセット/コアAPIで変更可。shakevmd.md §2.3)
DEFAULT_OCTAVES = 3
DEFAULT_PERSISTENCE = 0.5
BANDLIMIT_HZ = 8.0

_U64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def derive_seed(base_seed: int, *parts) -> int:
    """base_seed と parts から決定論的にシードを派生する(§5.3, §6.1)。

    SHA-256 で混ぜるため、実行をまたいで安定し、parts が違えば別値になる。
    """
    h = hashlib.sha256()
    h.update(repr(base_seed).encode("utf-8"))
    for p in parts:
        # repr で型を区別(int 2 と str "2" を別扱い)。長さ前置きで区切り衝突を防ぐ
        b = repr(p).encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return int.from_bytes(h.digest()[:8], "big")


def effective_octaves(
    freq: float, octaves: int, bandlimit_hz: float = BANDLIMIT_HZ
) -> int:
    """実際に生成するオクターブ数(§6.1 帯域制限)。

    freq × 2^i は i について単調増加なので、bandlimit_hz 以下である先頭からの
    連続オクターブ数を返す(freq 自体が超過すれば 0)。
    freq は Hz の大きさとして扱い、符号に依らず |freq| で判定する。
    """
    f = abs(freq)
    n = 0
    for i in range(max(0, octaves)):
        if f * (2.0**i) <= bandlimit_hz:
            n += 1
        else:
            break
    return n


def _clamp_message(octaves: int, n: int, bandlimit_hz: float) -> str:
    """帯域制限クランプの警告文言(§6.1)。疑似コード接頭は付けず記述のみ(コードは呼び出し側が付番)。"""
    return f"オクターブを {octaves}→{n} にクランプした(実効周波数が {bandlimit_hz}Hz を超過)"


def _fade(t: np.ndarray) -> np.ndarray:
    """quintic fade 6t^5-15t^4+10t^3(C2 → 合成ノイズはC1連続)。"""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _gradients(seed: int, idx: np.ndarray) -> np.ndarray:
    """格子点 idx(整数配列)の擬似乱数勾配を [-1,1) で返す(splitmix64系ハッシュ)。"""
    with np.errstate(over="ignore"):
        z = (idx.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) & _U64
        z = (z ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)) & _U64
        z = (z + np.uint64(0x9E3779B97F4A7C15)) & _U64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _U64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _U64
        z = z ^ (z >> np.uint64(31))
    return (z.astype(np.float64) / 2.0**64) * 2.0 - 1.0


def _seed_phase(seed: int) -> float:
    """シード由来の決定論的な位相オフセット [0, 1024)。

    格子(整数点)を非整数へずらし、全ストリームが整数格子点で同時にゼロに
    なる同期アーティファクトを防ぐ(§6.1。各オクターブ/チャンネルで別位相)。
    """
    g = float(_gradients(seed, np.array([0], dtype=np.int64))[0])  # [-1,1)
    return (g + 1.0) * 512.0


def _perlin1d(seed: int, x: np.ndarray) -> np.ndarray:
    """1Dグラディエントノイズ。値域は [-1,1]、quintic fade によりC1連続。"""
    i0 = np.floor(x).astype(np.int64)
    frac = x - i0
    g0 = _gradients(seed, i0)
    g1 = _gradients(seed, i0 + 1)
    n0 = g0 * frac
    n1 = g1 * (frac - 1.0)
    u = _fade(frac)
    return n0 + u * (n1 - n0)


def octave_components(
    seed: int,
    t,
    freq: float,
    *,
    octaves: int = DEFAULT_OCTAVES,
    bandlimit_hz: float = BANDLIMIT_HZ,
) -> tuple[list, list[str]]:
    """各オクターブのノイズ成分(persistence 重み無し)のリストと警告を返す(§6.2)。

    band_limited_noise が固定 persistence で合成するのに対し、こちらは合成前の
    オクターブ成分を返す。プロファイルクロスフェード(motion.profile_weights)で
    フレーム毎にオクターブ重みを変えて合成するために使う。

    戻り値: (components, warns)。components は effective_octaves 本の np.ndarray(各 len(t))。
    各成分は単一オクターブの _perlin1d(f_i·t + 位相)で値域 [-1,1]。
    Σ persistence^i × components[i] は band_limited_noise(同パラメータ) と一致する。
    """
    t = np.asarray(t, dtype=float)
    freq = abs(freq)  # Hz は大きさ
    n = effective_octaves(freq, octaves, bandlimit_hz)
    warns: list[str] = []
    if n < octaves:
        warns.append(_clamp_message(octaves, n, bandlimit_hz))
    comps = []
    for i in range(n):
        f_i = freq * (2.0**i)
        oct_seed = derive_seed(seed, "octave", i)
        phase = _seed_phase(oct_seed)   # band_limited_noise と同一の per-octave シード/位相
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
    """時刻列 t(秒)の帯域制限付き多オクターブノイズと警告コードを返す(§6.1)。"""
    t = np.asarray(t, dtype=float)
    freq = abs(freq)  # Hz は大きさ
    n = effective_octaves(freq, octaves, bandlimit_hz)
    warns: list[str] = []
    if n < octaves:
        warns.append(_clamp_message(octaves, n, bandlimit_hz))

    out = np.zeros_like(t)
    for i in range(n):
        f_i = freq * (2.0**i)
        amp = persistence**i
        oct_seed = derive_seed(seed, "octave", i)
        # シード由来の位相オフセットで格子をずらし、整数格子点での同期ゼロを防ぐ
        phase = _seed_phase(oct_seed)
        out = out + amp * _perlin1d(oct_seed, f_i * t + phase)
    return out, warns
