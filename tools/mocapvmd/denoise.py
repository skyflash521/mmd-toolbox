import dataclasses
import math

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from vmd.cuts import detect_cuts_bone

POS_SPIKE = 0.3
ROT_SPIKE_DEG = 5.0
CUT_THRESHOLDS = (1.0, 30.0)
MIN_ACCENT_STEPS = 2
ZERO_EPS = 1e-9
_NLERP_DOT_THRESHOLD = 0.9995

RelativeFrame = int
Axis = int


@dataclasses.dataclass(frozen=True)
class NoiseDetection:
    pos_candidates: set[tuple[RelativeFrame, Axis]]
    rot_candidates: set[RelativeFrame]
    pos_spikes: set[tuple[RelativeFrame, Axis]]
    rot_spikes: set[RelativeFrame]
    pos_accent: set[tuple[RelativeFrame, Axis]]
    rot_accent: set[RelativeFrame]
    boundaries: set[RelativeFrame]
    cuts: set[RelativeFrame]


def _qnorm(q):
    return math.sqrt(sum(c * c for c in q))


def _qnormalize(q):
    return tuple(c / _qnorm(q) for c in q)


def _qconj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _qangle_deg(a, b):
    d = abs(sum(x * y for x, y in zip(a, b, strict=True)))
    d = min(1.0, d)
    return math.degrees(2.0 * math.acos(d))


def _relative_rotation_vector_rad(q_prev, q_cur):
    rel = _qnormalize(_qmul(_qconj(q_prev), q_cur))
    if rel[3] < 0.0:
        rel = tuple(-c for c in rel)
    x, y, z, w = rel
    vnorm = math.sqrt(x * x + y * y + z * z)
    if vnorm < ZERO_EPS:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vnorm, w)
    s = angle / vnorm
    return (x * s, y * s, z * s)


def _vlen(v):
    return math.sqrt(sum(c * c for c in v))


def _vdot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _slerp(a, b, t):
    d = _vdot(a, b)
    if d < 0.0:
        b = tuple(-c for c in b)
        d = -d
    if d > _NLERP_DOT_THRESHOLD:
        return _qnormalize(tuple(a[k] + t * (b[k] - a[k]) for k in range(4)))
    th0 = math.acos(d)
    th = th0 * t
    s0 = math.sin(th0 - th) / math.sin(th0)
    s1 = math.sin(th) / math.sin(th0)
    return tuple(s0 * a[k] + s1 * b[k] for k in range(4))


def _clamp_rot(orig, target, max_deg):
    ang = _qangle_deg(orig, target)
    if ang <= max_deg:
        return target
    return _slerp(orig, target, max_deg / ang)


def validate_bone_values(positions, rotations):
    """長さの不一致・非有限値・ノルムがゼロの quaternion は ValueError を送出する。"""
    if len(positions) != len(rotations):
        raise ValueError("位置と回転のフレーム数が一致しません")
    for p in positions:
        if not all(math.isfinite(c) for c in p):
            raise ValueError("位置に非有限値が含まれます")
    for q in rotations:
        if not all(math.isfinite(c) for c in q):
            raise ValueError("回転に非有限値が含まれます")
        if _qnorm(q) < ZERO_EPS:
            raise ValueError("ノルムがゼロの quaternion が含まれます")


def _validate(positions, rotations, pos_window, rot_window):
    validate_bone_values(positions, rotations)
    for w in (pos_window, rot_window):
        if w < 1 or w % 2 == 0:
            raise ValueError("窓幅は正の奇数である必要があります")


def _inclusive_segments_split_before_cuts(cuts, n):
    bounds = sorted(f for f in cuts if 0 < f < n)
    segs = []
    start = 0
    for f in bounds:
        segs.append((start, f - 1))
        start = f
    segs.append((start, n - 1))
    return segs


def _effective_window_or_zero(seg_len, window):
    w = min(window, seg_len if seg_len % 2 == 1 else seg_len - 1)
    return w if w >= 3 else 0


def _pos_reference(pos, segs, window):
    ref = pos.copy()
    for s, e in segs:
        weff = _effective_window_or_zero(e - s + 1, window)
        if not weff:
            continue
        for a in range(3):
            ref[s : e + 1, a] = median_filter(pos[s : e + 1, a], size=weff, mode="mirror")
    return ref


def _mirror_index(j, s, e):
    if e == s:
        return s
    span = e - s
    period = 2 * span
    k = (j - s) % period
    if k < 0:
        k += period
    if k > span:
        k = period - k
    return s + k


def _rot_reference(rots, segs, window):
    ref = list(rots)
    for s, e in segs:
        weff = _effective_window_or_zero(e - s + 1, window)
        if not weff:
            continue
        h = weff // 2
        for i in range(s, e + 1):
            center = rots[i]
            acc = [0.0, 0.0, 0.0, 0.0]
            for d in range(-h, h + 1):
                q = rots[_mirror_index(i + d, s, e)]
                if _vdot(q, center) < 0.0:
                    q = tuple(-c for c in q)
                for c in range(4):
                    acc[c] += q[c]
            ref[i] = _qnormalize(tuple(acc))
    return ref


def _pos_accent(pos, segs):
    frames = set()
    for s, e in segs:
        for a in range(3):
            signs = []
            for k in range(s, e):
                step = pos[k + 1, a] - pos[k, a]
                signs.append(1 if step > POS_SPIKE else (-1 if step < -POS_SPIKE else 0))
            _collect_same_sign_runs(signs, s, a, frames)
    return frames


def _rot_accent(rots, segs):
    frames = set()
    for s, e in segs:
        vecs = [_relative_rotation_vector_rad(rots[k], rots[k + 1]) for k in range(s, e)]
        large = [math.degrees(_vlen(v)) > ROT_SPIKE_DEG for v in vecs]
        k = 0
        while k < len(vecs):
            if not large[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(vecs) and large[j + 1] and _vdot(vecs[j], vecs[j + 1]) > 0.0:
                j += 1
            if j - k + 1 >= MIN_ACCENT_STEPS:
                for f in range(s + k, s + j + 2):
                    frames.add(f)
            k = j + 1
    return frames


def _collect_same_sign_runs(signs, seg_start, axis, frames):
    k = 0
    while k < len(signs):
        if signs[k] == 0:
            k += 1
            continue
        j = k
        while j + 1 < len(signs) and signs[j + 1] == signs[k]:
            j += 1
        if j - k + 1 >= MIN_ACCENT_STEPS:
            for f in range(seg_start + k, seg_start + j + 2):
                frames.add((f, axis))
        k = j + 1


def detect_noise_events(positions, rotations, *, pos_window, rot_window):
    _validate(positions, rotations, pos_window, rot_window)
    n = len(positions)
    if n == 0:
        return NoiseDetection(set(), set(), set(), set(), set(), set(), set(), set())

    pos = np.asarray(positions, dtype=float).reshape(n, 3)
    rots = [_qnormalize(tuple(float(c) for c in q)) for q in rotations]

    raw_cuts = detect_cuts_bone(0, positions, rots, CUT_THRESHOLDS)
    cuts = {f for f in raw_cuts if 0 < f < n}
    segs = _inclusive_segments_split_before_cuts(cuts, n)

    boundaries = {0, n - 1}
    for f in cuts:
        boundaries.add(f)
        boundaries.add(f - 1)

    pos_ref = _pos_reference(pos, segs, pos_window)
    rot_ref = _rot_reference(rots, segs, rot_window)

    seg_of = {}
    for s, e in segs:
        if (_effective_window_or_zero(e - s + 1, pos_window)
                or _effective_window_or_zero(e - s + 1, rot_window)):
            for i in range(s, e + 1):
                seg_of[i] = (s, e)

    pos_candidates = set()
    for s, e in segs:
        if not _effective_window_or_zero(e - s + 1, pos_window):
            continue
        for i in range(s, e + 1):
            for a in range(3):
                if abs(pos[i, a] - pos_ref[i, a]) > POS_SPIKE:
                    pos_candidates.add((i, a))
    rot_candidates = set()
    for s, e in segs:
        if not _effective_window_or_zero(e - s + 1, rot_window):
            continue
        for i in range(s, e + 1):
            if _qangle_deg(rots[i], rot_ref[i]) > ROT_SPIKE_DEG:
                rot_candidates.add(i)

    pos_accent = _pos_accent(pos, segs)
    rot_accent = _rot_accent(rots, segs)

    pos_spikes = set()
    for (i, a) in pos_candidates:
        if (i, a) in pos_accent or i in boundaries:
            continue
        s, e = seg_of.get(i, (i, i))
        if i <= s or i >= e:
            continue
        incoming = pos[i, a] - pos[i - 1, a]
        outgoing = pos[i + 1, a] - pos[i, a]
        if abs(incoming) > ZERO_EPS and abs(outgoing) > ZERO_EPS and incoming * outgoing < 0.0:
            pos_spikes.add((i, a))

    rot_spikes = set()
    for i in rot_candidates:
        if i in rot_accent or i in boundaries:
            continue
        s, e = seg_of.get(i, (i, i))
        if i <= s or i >= e:
            continue
        v_in = _relative_rotation_vector_rad(rots[i - 1], rots[i])
        v_out = _relative_rotation_vector_rad(rots[i], rots[i + 1])
        if _vlen(v_in) > ZERO_EPS and _vlen(v_out) > ZERO_EPS and _vdot(v_in, v_out) < 0.0:
            rot_spikes.add(i)

    return NoiseDetection(
        pos_candidates=pos_candidates,
        rot_candidates=rot_candidates,
        pos_spikes=pos_spikes,
        rot_spikes=rot_spikes,
        pos_accent=pos_accent,
        rot_accent=rot_accent,
        boundaries=boundaries,
        cuts=cuts,
    )


def _pos_smoothed(pos, segs, window):
    out = pos.copy()
    for s, e in segs:
        weff = _effective_window_or_zero(e - s + 1, window)
        if not weff:
            continue
        for a in range(3):
            med = median_filter(pos[s : e + 1, a], size=weff, mode="mirror")
            out[s : e + 1, a] = savgol_filter(med, weff, 2, mode="mirror")
    return out


def apply_denoise(positions, rotations, *, pos_window, rot_window, pos_strength, rot_strength):
    det = detect_noise_events(positions, rotations, pos_window=pos_window, rot_window=rot_window)
    n = len(positions)
    if n == 0:
        return [], []

    pos = np.asarray(positions, dtype=float).reshape(n, 3)
    rots = [_qnormalize(tuple(float(c) for c in q)) for q in rotations]
    segs = _inclusive_segments_split_before_cuts(det.cuts, n)

    smoothed = _pos_smoothed(pos, segs, pos_window)
    rot_mean = _rot_reference(rots, segs, rot_window)

    out_pos = []
    for i in range(n):
        vals = []
        for a in range(3):
            if i in det.boundaries or (i, a) in det.pos_accent:
                vals.append(float(pos[i, a]))
                continue
            delta = pos_strength * (smoothed[i, a] - pos[i, a])
            delta = max(-POS_SPIKE, min(POS_SPIKE, delta))
            vals.append(float(pos[i, a] + delta))
        out_pos.append(tuple(vals))

    out_rot = []
    for i in range(n):
        if i in det.boundaries or i in det.rot_accent:
            out_rot.append(rots[i])
            continue
        target = _slerp(rots[i], rot_mean[i], rot_strength)
        out_rot.append(_clamp_rot(rots[i], target, ROT_SPIKE_DEG))

    return out_pos, out_rot
