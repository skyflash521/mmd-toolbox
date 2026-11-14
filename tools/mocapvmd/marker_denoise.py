from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

Vec3 = tuple[float, float, float]

_HAMPEL_K = 3.0
_MAD_TO_SIGMA = 1.4826


@dataclass
class SmoothParams:
    """window は平滑窓の長さで奇数。strength は元値とのブレンド率で (0, 1]。max_disp は1フレームあたりの
    最大変位(MMD 単位)。"""

    window: int
    strength: float
    max_disp: float


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
    """displacement はマーカーごとの、平滑化の前後で位置が動いた量の全フレームでの最大値。"""

    markers: dict[str, list[Vec3]]
    displacement: dict[str, float]


def _odd_at_most(n: int) -> int:
    return n if n % 2 == 1 else n - 1


def _half_open_segments_split_at_cuts(n: int, cuts):
    pts = sorted({c for c in cuts if 0 < c < n})
    bounds = []
    start = 0
    for c in pts:
        bounds.append((start, c))
        start = c
    bounds.append((start, n))
    return bounds


def _hampel_replace_outliers_with_median(arr, window, k=_HAMPEL_K):
    n = len(arr)
    out = arr.copy()
    half = window // 2
    for i in range(n):
        win = arr[max(0, i - half) : min(n, i + half + 1)]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        thresh = k * _MAD_TO_SIGMA * mad
        if abs(arr[i] - med) > thresh and arr[i] != med:
            out[i] = med
    return out


def _smooth_axis_before_clamp(seg, params):
    n = len(seg)
    window = _odd_at_most(min(params.window, n))
    if window < 3:
        return seg.copy()
    cleaned = _hampel_replace_outliers_with_median(seg, window)
    low = savgol_filter(cleaned, window_length=window, polyorder=2)
    return seg + params.strength * (low - seg)


def _smooth_marker(series, params, cuts) -> tuple[list[Vec3], float]:
    n = len(series)
    if n == 0:
        return [], 0.0
    arr = np.array(series, dtype=float)
    blended = arr.copy()
    for start, end in _half_open_segments_split_at_cuts(n, cuts):
        for ax in range(3):
            blended[start:end, ax] = _smooth_axis_before_clamp(arr[start:end, ax], params)

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


def smooth(
    markers: dict[str, list[Vec3]],
    categories: dict[str, str],
    *,
    preset: dict[str, SmoothParams] | None = None,
    cuts: Iterable[int] = (),
) -> SmoothResult:
    if preset is None:
        preset = DEFAULT_PRESET
    out_markers = {}
    max_displacement = {}
    for name, series in markers.items():
        params = preset[categories[name]]
        out_markers[name], max_displacement[name] = _smooth_marker(series, params, cuts)
    return SmoothResult(markers=out_markers, displacement=max_displacement)
