"""マーカー軌跡のロバスト平滑化。

カット境界で区間分割し、区間ごとに Hampel型外れ値置換でスパイクを抑え、
Savitzky-Golay で低域化し、強度ブレンドと最大変位クランプで部位別に平滑化する。
非外れ値は保持されるため定数・線形運動は鈍らない。前後の最大変位を診断に残す。
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class SmoothParams:
    window: int  # 平滑窓(奇数)
    strength: float  # 元値とのブレンド率 (0,1]
    max_disp: float  # 1フレームあたりの最大変位(MMD単位)


# 計画§7.3 の部位別既定値。実データで調整する前提で弱めに置く。
DEFAULT_PRESET = {
    "center": SmoothParams(7, 0.35, 0.15),
    "torso": SmoothParams(5, 0.30, 0.20),
    "head": SmoothParams(5, 0.25, 0.15),
    "arms": SmoothParams(5, 0.25, 0.25),
    "wrists": SmoothParams(5, 0.35, 0.30),
    "legs": SmoothParams(5, 0.25, 0.25),
    "feet": SmoothParams(5, 0.20, 0.20),
}


@dataclass
class SmoothResult:
    markers: dict[str, list[tuple[float, float, float]]]
    displacement: dict[str, float]  # マーカー毎の最大変位(診断)


def _odd_at_most(n: int) -> int:
    return n if n % 2 == 1 else n - 1


def _segment_bounds(n: int, cuts):
    """[0, n) を cuts(フレーム)で分割した (start, end) 区間列。"""
    pts = sorted({c for c in cuts if 0 < c < n})
    bounds = []
    start = 0
    for c in pts:
        bounds.append((start, c))
        start = c
    bounds.append((start, n))
    return bounds


def _hampel(arr, window, k=3.0):
    """Hampel型外れ値置換。非外れ値は不変、外れ値のみ窓中央値へ置換する。"""
    n = len(arr)
    out = arr.copy()
    half = window // 2
    for i in range(n):
        win = arr[max(0, i - half) : min(n, i + half + 1)]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        thresh = k * 1.4826 * mad  # MAD=0 のとき閾値0(窓内一定)
        if abs(arr[i] - med) > thresh and arr[i] != med:
            out[i] = med
    return out


def _smooth_axis(seg, params):
    """1区間1軸を Hampel→Savitzky-Golay→強度ブレンドした値を返す(クランプ前)。"""
    n = len(seg)
    window = _odd_at_most(min(params.window, n))
    if window < 3:
        return seg.copy()  # 短すぎる区間は素通し
    cleaned = _hampel(seg, window)
    low = savgol_filter(cleaned, window_length=window, polyorder=2)
    return seg + params.strength * (low - seg)


def _smooth_marker(series, params, cuts):
    """マーカー1本(フレーム毎3要素)を平滑化し、(平滑化系列, 最大変位)を返す。"""
    n = len(series)
    if n == 0:
        return [], 0.0
    arr = np.array(series, dtype=float)  # (n, 3)
    blended = arr.copy()
    for start, end in _segment_bounds(n, cuts):
        for ax in range(3):
            blended[start:end, ax] = _smooth_axis(arr[start:end, ax], params)

    out = arr.copy()
    max_disp = 0.0
    for i in range(n):
        delta = blended[i] - arr[i]
        mag = float(np.linalg.norm(delta))
        if mag > params.max_disp:
            delta = delta * (params.max_disp / mag)
        out[i] = arr[i] + delta
        max_disp = max(max_disp, float(np.linalg.norm(out[i] - arr[i])))

    smoothed = [(float(p[0]), float(p[1]), float(p[2])) for p in out]
    return smoothed, max_disp


def smooth(markers, categories, *, preset=None, cuts=()) -> SmoothResult:
    """マーカー軌跡を部位別プリセットで平滑化する。

    markers: マーカー名 -> フレーム毎 (x, y, z)。
    categories: マーカー名 -> 部位カテゴリ(プリセットのキー)。
    cuts: 区間分割するフレーム境界。
    """
    if preset is None:
        preset = DEFAULT_PRESET
    out_markers = {}
    displacement = {}
    for name, series in markers.items():
        params = preset[categories[name]]
        out_markers[name], displacement[name] = _smooth_marker(series, params, cuts)
    return SmoothResult(markers=out_markers, displacement=displacement)
