"""一般ノイズ軽減の検出層(mocapvmd.md §4.2 / 実装計画 §4.1)。

各ボーンの密サンプル(0始まり相対インデックス)から、スパイク候補・スパイク(補正対象)・
アクセント(保護)・カット/範囲端の境界を検出する。スパイクは窓基準(位置=窓中央値・回転=窓内
正規化平均)からの逸脱(外れ値)で、アクセントはフレーム間変化量(速度)で別々に判定する。
競合はアクセント優先で、最終スパイク = スパイク候補 ∩ 単発往復 − アクセント − 境界。

検出は相対インデックスで行い、カットは `mmd_toolbox.vmd.cuts.detect_cuts_bone`(同じ閾値)を
再利用してカット区間に分割し、各区間内でのみ窓計算・速度判定を行う(境界をまたがない)。
"""

import dataclasses
import math

import numpy as np
from scipy.ndimage import median_filter

from mmd_toolbox.vmd.cuts import detect_cuts_bone

POS_SPIKE = 0.3          # 位置スパイク候補/アクセント速度の閾値(各軸 MMD単位)
ROT_SPIKE_DEG = 5.0      # 回転スパイク候補/アクセント速度の閾値(度)
CUT_THRESHOLDS = (1.0, 30.0)  # カット閾値(位置 MMD単位 / 回転 度)
MIN_ACCENT_STEPS = 2     # アクセントと認める同方向連続差分の最小本数
ZERO_EPS = 1e-9          # 実質ゼロの下限(位置は絶対差、回転は回転ベクトルのノルム=ラジアン)


@dataclasses.dataclass(frozen=True)
class NoiseDetection:
    """検出結果。フレームは0始まり相対インデックス。位置イベントは (frame, axis) 集合。"""

    pos_candidates: set      # set[(frame, axis)]  窓中央値からの逸脱 > 0.3
    rot_candidates: set      # set[frame]          窓内正規化平均との角度差 > 5度
    pos_spikes: set          # set[(frame, axis)]  候補 ∩ 単発往復 − アクセント − 境界
    rot_spikes: set          # set[frame]
    pos_accent: set          # set[(frame, axis)]  速度が同方向2本以上連続する run の張るフレーム
    rot_accent: set          # set[frame]
    boundaries: set          # set[frame]          カット両側(F-1,F)+ 範囲端
    cuts: set                # set[frame]          検出したカットフレーム(F-1とFの間が境界)


# --- quaternion ユーティリティ ---------------------------------------------


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
    """2つの quaternion 間の角度距離(度)。符号反転は同一姿勢として 0 になる。"""
    d = abs(sum(x * y for x, y in zip(a, b)))
    d = min(1.0, d)
    return math.degrees(2.0 * math.acos(d))


def _rel_rot_vec(q_prev, q_cur):
    """前→現の相対回転を回転ベクトル(軸×角, ラジアン)で返す。向きが回転方向、ノルムが角度。"""
    rel = _qnormalize(_qmul(_qconj(q_prev), q_cur))
    if rel[3] < 0.0:  # 同一姿勢の符号反転を正準半球へ揃える
        rel = tuple(-c for c in rel)
    x, y, z, w = rel
    vnorm = math.sqrt(x * x + y * y + z * z)
    if vnorm < ZERO_EPS:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vnorm, w)  # rel[3]>=0 なので angle∈[0,π]
    s = angle / vnorm
    return (x * s, y * s, z * s)


def _vlen(v):
    return math.sqrt(sum(c * c for c in v))


def _vdot(a, b):
    return sum(x * y for x, y in zip(a, b))


# --- 入力検証・区間分割 ------------------------------------------------------


def _validate(positions, rotations, pos_window, rot_window):
    if len(positions) != len(rotations):
        raise ValueError("位置と回転のフレーム数が一致しません")
    for w in (pos_window, rot_window):
        if w < 1 or w % 2 == 0:
            raise ValueError("窓幅は正の奇数である必要があります")
    for p in positions:
        if not all(math.isfinite(c) for c in p):
            raise ValueError("位置に非有限値が含まれます")
    for q in rotations:
        if not all(math.isfinite(c) for c in q):
            raise ValueError("回転に非有限値が含まれます")
        if _qnorm(q) < ZERO_EPS:
            raise ValueError("ノルムがゼロの quaternion が含まれます")


def _segments(cuts, n):
    """カットフレーム集合(カット F は F-1 と F の間の境界)から連続区間 [(start,end), ...] を作る。"""
    bounds = sorted(f for f in cuts if 0 < f < n)
    segs = []
    start = 0
    for f in bounds:
        segs.append((start, f - 1))
        start = f
    segs.append((start, n - 1))
    return segs


def _effective_window(seg_len, window):
    """区間長に収まる最大の奇数(窓上限以下)。3未満なら 0(平滑化しない)を返す。"""
    w = min(window, seg_len if seg_len % 2 == 1 else seg_len - 1)
    return w if w >= 3 else 0


# --- 窓基準(位置中央値・回転正規化平均) ----------------------------------


def _pos_reference(pos, segs, window):
    """各区間・各軸の窓中央値(端はミラー反射)。実効窓<3の区間は元値(逸脱0)を返す。"""
    ref = pos.copy()
    for s, e in segs:
        weff = _effective_window(e - s + 1, window)
        if not weff:
            continue
        for a in range(3):
            ref[s : e + 1, a] = median_filter(pos[s : e + 1, a], size=weff, mode="mirror")
    return ref


def _mirror_index(j, s, e):
    """インデックス j を区間 [s,e] へミラー反射(端点を重複させない)で折り返す。"""
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
    """各区間の窓内正規化平均 quaternion(中心姿勢へ半球を揃えて成分平均)。実効窓<3は元値。"""
    ref = list(rots)
    for s, e in segs:
        weff = _effective_window(e - s + 1, window)
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


# --- アクセント(速度ベースの run) ----------------------------------------


def _pos_accent(pos, segs):
    """各区間・各軸で、|差分|>0.3 の同符号が2本以上連続する run の張るフレームを返す。"""
    frames = set()
    for s, e in segs:
        for a in range(3):
            signs = []
            for k in range(s, e):
                step = pos[k + 1, a] - pos[k, a]
                signs.append(1 if step > POS_SPIKE else (-1 if step < -POS_SPIKE else 0))
            _collect_runs(signs, s, a, frames)
    return frames


def _rot_accent(rots, segs):
    """各区間で、角度>5度かつ前後の回転ベクトルが同方向(内積>0)に2本以上連続する run のフレーム。"""
    frames = set()
    for s, e in segs:
        vecs = [_rel_rot_vec(rots[k], rots[k + 1]) for k in range(s, e)]
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


def _collect_runs(signs, seg_start, axis, frames):
    """符号列から同符号2本以上連続の run を見つけ、(frame, axis) を frames へ加える。"""
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


# --- 公開 API ---------------------------------------------------------------


def detect_noise_events(positions, rotations, *, pos_window, rot_window):
    """密サンプルからスパイク・アクセント・境界を検出して NoiseDetection を返す(§4.1)。"""
    _validate(positions, rotations, pos_window, rot_window)
    n = len(positions)
    if n == 0:
        return NoiseDetection(set(), set(), set(), set(), set(), set(), set(), set())

    pos = np.asarray(positions, dtype=float).reshape(n, 3)
    # 入力 quaternion を単位化してから全処理(角度距離・窓内平均・カット検出)へ渡す。
    # 非単位 quaternion だと内積ベースの角度判定が崩れて検出を取りこぼすため。
    rots = [_qnormalize(tuple(float(c) for c in q)) for q in rotations]

    raw_cuts = detect_cuts_bone(0, positions, rots, CUT_THRESHOLDS)
    cuts = {f for f in raw_cuts if 0 < f < n}
    segs = _segments(cuts, n)

    boundaries = {0, n - 1}
    for f in cuts:
        boundaries.add(f)
        boundaries.add(f - 1)

    pos_ref = _pos_reference(pos, segs, pos_window)
    rot_ref = _rot_reference(rots, segs, rot_window)

    seg_of = {}
    for s, e in segs:
        if _effective_window(e - s + 1, pos_window) or _effective_window(e - s + 1, rot_window):
            for i in range(s, e + 1):
                seg_of[i] = (s, e)

    # スパイク候補(窓基準からの逸脱)。実効窓<3の区間は参照=元値で逸脱0になり候補は出ない。
    pos_candidates = set()
    for s, e in segs:
        if not _effective_window(e - s + 1, pos_window):
            continue
        for i in range(s, e + 1):
            for a in range(3):
                if abs(pos[i, a] - pos_ref[i, a]) > POS_SPIKE:
                    pos_candidates.add((i, a))
    rot_candidates = set()
    for s, e in segs:
        if not _effective_window(e - s + 1, rot_window):
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
            continue  # 区間端は往復の判定材料が無い
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
        v_in = _rel_rot_vec(rots[i - 1], rots[i])
        v_out = _rel_rot_vec(rots[i], rots[i + 1])
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
