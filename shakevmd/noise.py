"""帯域制限付き多オクターブノイズ(shakevmd.md §6.1)。

- C1連続ノイズ(パーリン/シンプレックス系)、チャンネル/セグメントごとに独立シード派生
- オクターブ合成: Σ persistence^i × noise(freq × 2^i × t)
- 帯域制限: 実効周波数(freq × 2^i)が bandlimit_hz を超えるオクターブは生成しない
  (自動クランプ+警告)。ベイクは30fps固定(ナイキスト15Hz)
"""

import numpy as np

# 内蔵値(プリセット/コアAPIで変更可。shakevmd.md §2.3)
DEFAULT_OCTAVES = 3
DEFAULT_PERSISTENCE = 0.5
BANDLIMIT_HZ = 8.0


def derive_seed(base_seed: int, *parts) -> int:
    """base_seed と parts(チャンネル名・セグメント番号など)から決定論的にシードを派生する。

    同じ引数からは常に同じ値、異なる parts からは異なる値を返す。
    チャンネルごと・セグメントごとの位相独立(§5.3, §6.1)に使う。
    """
    raise NotImplementedError


def effective_octaves(
    freq: float, octaves: int, bandlimit_hz: float = BANDLIMIT_HZ
) -> int:
    """実際に生成するオクターブ数を返す(§6.1 帯域制限)。

    i = 0..octaves-1 のうち、実効周波数 freq × 2^i が bandlimit_hz 以下のものの数。
    freq 自体が bandlimit_hz を超える場合は 0。
    """
    raise NotImplementedError


def band_limited_noise(
    seed: int,
    t,
    freq: float,
    *,
    octaves: int = DEFAULT_OCTAVES,
    persistence: float = DEFAULT_PERSISTENCE,
    bandlimit_hz: float = BANDLIMIT_HZ,
) -> tuple[np.ndarray, list[str]]:
    """時刻列 t(秒)に対する帯域制限付き多オクターブノイズを返す。

    戻り値: (値の配列(t と同形状), 警告コードのリスト)。
    実効オクターブ数が要求オクターブ数より少ない(帯域制限でクランプ)場合、
    警告コードを1件以上含む。
    """
    raise NotImplementedError
