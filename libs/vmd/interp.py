"""補間曲線評価・サンプリング。

入力キー列はフレーム昇順前提(正規化済みであること)。
"""

import math

import numpy as np

from .types import CameraKey

# カメラ補間ブロック(24バイト)内の各チャンネルのオフセット。
# 各チャンネルは ax, bx, ay, by の順で4バイト(始点X, 終点X, 始点Y, 終点Y)。
CAMERA_CHANNEL_OFFSET = {
    "pos_x": 0,
    "pos_y": 4,
    "pos_z": 8,
    "rot": 12,
    "distance": 16,
    "fov": 20,
}

_POS_INDEX = {"pos_x": 0, "pos_y": 1, "pos_z": 2}
_BONE_CHANNEL = {"pos_x": "X", "pos_y": "Y", "pos_z": "Z", "rot": "R"}


# ---------------------------------------------------------------------------
# ベジェ評価
# ---------------------------------------------------------------------------


def _bezier(s: float, c1: float, c2: float) -> float:
    """端点 0,1 固定の3次ベジェの1成分。c1,c2 は [0,1] 正規化済み制御点。"""
    u = 1.0 - s
    return 3 * u * u * s * c1 + 3 * u * s * s * c2 + s * s * s


def _solve_factor(x1: int, y1: int, x2: int, y2: int, x: float) -> float:
    """制御点 (x1,y1),(x2,y2)(0..127)・正規化時間 x∈[0,1] → 補間係数 y∈[0,1]。

    X(s)=x をニュートン法で解き、非収束時は二分法にフォールバックする。
    Xは単調増加前提(MMDの補間曲線は時間方向に単調)。
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    px1, py1, px2, py2 = x1 / 127.0, y1 / 127.0, x2 / 127.0, y2 / 127.0

    def fx(s: float) -> float:
        return _bezier(s, px1, px2)

    def dfx(s: float) -> float:
        u = 1.0 - s
        return 3.0 * (px1 * u * u + 2.0 * (px2 - px1) * u * s + (1.0 - px2) * s * s)

    s = x  # 初期値
    converged = False
    for _ in range(20):
        err = fx(s) - x
        if abs(err) < 1e-9:
            converged = True
            break
        d = dfx(s)
        if d <= 1e-12:
            break
        s -= err / d
        if s < 0.0 or s > 1.0:
            break

    if not converged or s < 0.0 or s > 1.0 or abs(fx(s) - x) > 1e-6:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if fx(mid) < x:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2.0

    return _bezier(s, py1, py2)


def _slerp(q0, q1, t: float):
    """クォータニオン (x,y,z,w) の球面線形補間。"""
    a = [float(c) for c in q0]
    b = [float(c) for c in q1]
    na = math.sqrt(sum(c * c for c in a))
    nb = math.sqrt(sum(c * c for c in b))
    a = [c / na for c in a]
    b = [c / nb for c in b]
    d = sum(x * y for x, y in zip(a, b, strict=True))
    if d < 0.0:
        b = [-c for c in b]
        d = -d
    if d > 0.9995:
        r = [x + t * (y - x) for x, y in zip(a, b, strict=True)]
        n = math.sqrt(sum(c * c for c in r))
        return tuple(c / n for c in r)
    th0 = math.acos(d)
    th = th0 * t
    s0 = math.sin(th0 - th) / math.sin(th0)
    s1 = math.sin(th) / math.sin(th0)
    return tuple(s0 * x + s1 * y for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# チャンネル別の制御点・値
# ---------------------------------------------------------------------------


def _control_points(arriving_key, channel: str, is_camera: bool):
    """区間の到達側(後側)キーから制御点 (x1,y1,x2,y2) を取り出す。"""
    if is_camera:
        off = CAMERA_CHANNEL_OFFSET[channel]
        ax, bx, ay, by = arriving_key.interpolation[off : off + 4]
        return (ax, ay, bx, by)
    return arriving_key.control_points()[_BONE_CHANNEL[channel]]


def _interp_value(k0, k1, channel: str, is_camera: bool, y: float):
    """補間係数 y で区間 [k0, k1] のチャンネル値を算出する。"""
    if channel in _POS_INDEX:
        i = _POS_INDEX[channel]
        return k0.position[i] + (k1.position[i] - k0.position[i]) * y
    if channel == "distance":
        return k0.distance + (k1.distance - k0.distance) * y
    if channel == "fov":
        return float(k0.fov) + (float(k1.fov) - float(k0.fov)) * y
    if channel == "rot":
        if is_camera:
            # オイラー各軸を共通の補間曲線で独立に線形補間(クォータニオン化しない)
            return tuple(
                k0.rotation[i] + (k1.rotation[i] - k0.rotation[i]) * y for i in range(3)
            )
        return _slerp(k0.rotation, k1.rotation, y)
    raise ValueError(f"未知のチャンネル: {channel}")


def _find_segment(keys, frame: int) -> int:
    """keys[i].frame <= frame < keys[i+1].frame となる i を返す。"""
    for i in range(len(keys) - 1):
        if keys[i].frame <= frame < keys[i + 1].frame:
            return i
    return len(keys) - 2


# ---------------------------------------------------------------------------
# 公開API
# ---------------------------------------------------------------------------


def sample(keys, channel: str, frame: int):
    """1チャンネルを1フレームで評価する。"""
    if not keys:
        raise ValueError("キー列が空")
    is_camera = isinstance(keys[0], CameraKey)

    # 範囲外・単一キーは端キーの値で一定(境界規約)
    if frame <= keys[0].frame or len(keys) == 1:
        k = keys[0]
        return _interp_value(k, k, channel, is_camera, 0.0)
    if frame >= keys[-1].frame:
        k = keys[-1]
        return _interp_value(k, k, channel, is_camera, 0.0)

    i = _find_segment(keys, frame)
    k0, k1 = keys[i], keys[i + 1]
    span = k1.frame - k0.frame
    if span == 0:
        return _interp_value(k0, k0, channel, is_camera, 0.0)
    x = (frame - k0.frame) / span
    y = _solve_factor(*_control_points(k1, channel, is_camera), x)
    return _interp_value(k0, k1, channel, is_camera, y)


def sample_range(keys, channel: str, frame_start: int, frame_end: int):
    """[frame_start, frame_end] を1フレーム間隔(両端含む)で評価する。

    スカラーチャンネルは numpy 配列、回転チャンネルは値のリストを返す。
    """
    vals = [sample(keys, channel, f) for f in range(frame_start, frame_end + 1)]
    if vals and isinstance(vals[0], tuple):
        return vals
    return np.array(vals, dtype=float)


def _bake_segment(k0, k1, fa, fb, frame_start, positions, rotations):
    """区間 [k0, k1] 内のフレーム fa..fb(両端含む)を評価して出力へ書き込む。"""
    span = k1.frame - k0.frame
    count = fb - fa + 1
    cps = k1.control_points()
    xs = [(f - k0.frame) / span for f in range(fa, fb + 1)]
    factor_cache = {}

    def _factors(cp):
        ys = factor_cache.get(cp)
        if ys is None:
            x1, y1, x2, y2 = cp
            if x1 == y1 and x2 == y2:
                ys = xs  # 線形ファストパス: y = x
            else:
                ys = [_solve_factor(x1, y1, x2, y2, x) for x in xs]
            factor_cache[cp] = ys
        return ys

    axes = []
    for i, ch in enumerate(("X", "Y", "Z")):
        v0, v1 = k0.position[i], k1.position[i]
        if v0 == v1:
            axes.append([v0] * count)  # 定数ファストパス
        else:
            ys = _factors(cps[ch])
            axes.append([v0 + (v1 - v0) * y for y in ys])
    if k0.rotation == k1.rotation:
        # 定数ファストパス(正規化後の端値。一般解も同じ値を返す)
        rot = _slerp(k0.rotation, k0.rotation, 0.0)
        rots = [rot] * count
    else:
        rots = [_slerp(k0.rotation, k1.rotation, y) for y in _factors(cps["R"])]

    base = fa - frame_start
    for j in range(count):
        positions[base + j] = (axes[0][j], axes[1][j], axes[2][j])
        rotations[base + j] = rots[j]


def bake_bone_track(keys, frame_start: int, frame_end: int):
    """ボーントラック全チャンネルを [frame_start, frame_end] で密ベイクする。

    1フレーム間隔・両端含む。positions は (x, y, z) のリスト、rotations は
    クォータニオン (x, y, z, w) のリストを返す。評価は sample と同一意味論
    (境界規約・到達側キーの補間曲線・回転は正規化して返す)。区間を単調に
    掃引し、フレームごとの区間探索を避ける。
    """
    if not keys:
        raise ValueError("キー列が空")
    total = frame_end - frame_start + 1
    if total <= 0:
        return [], []
    positions = [None] * total
    rotations = [None] * total

    def _fill(i0, i1, key):
        pos = key.position
        rot = _slerp(key.rotation, key.rotation, 0.0)
        for i in range(i0, i1):
            positions[i] = pos
            rotations[i] = rot

    # 単一キーは全域で端値一定(境界規約)
    if len(keys) == 1:
        _fill(0, total, keys[0])
        return positions, rotations

    first_f = keys[0].frame
    last_f = keys[-1].frame
    f = frame_start

    # 最初のキー以前(最初のキー自身を含む)は端キーの値で一定
    if f <= first_f:
        upto = min(first_f, frame_end)
        _fill(0, upto - frame_start + 1, keys[0])
        f = upto + 1

    # 区間掃引(first_f < f < last_f)
    seg = 0
    while f <= frame_end and f < last_f:
        while keys[seg + 1].frame <= f:
            seg += 1
        k0, k1 = keys[seg], keys[seg + 1]
        fb = min(k1.frame - 1, frame_end)
        _bake_segment(k0, k1, f, fb, frame_start, positions, rotations)
        f = fb + 1

    # 最後のキー以後は端キーの値で一定
    if f <= frame_end:
        _fill(f - frame_start, total, keys[-1])
    return positions, rotations


def sample_camera(keys, frame: int) -> dict:
    """カメラ全チャンネルを1フレームで評価する。"""
    return {
        "distance": sample(keys, "distance", frame),
        "position": (
            sample(keys, "pos_x", frame),
            sample(keys, "pos_y", frame),
            sample(keys, "pos_z", frame),
        ),
        "rotation": sample(keys, "rot", frame),
        "fov": sample(keys, "fov", frame),
    }
