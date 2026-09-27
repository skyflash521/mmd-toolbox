import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from pmx.pose import LocalBonePose, evaluate_fk


@dataclass
class FitParams:
    """skip_threshold はフィットする最大マーカー変位の下限で、これ未満のフレームは元の姿勢を保つ。
    max_rot_deg・max_pos は補正量の絶対上限、k_rot(ラジアン/MMD 単位)・k_pos はマーカー変位に比例する
    補正量の上限の係数。w_marker・w_pose・w_vel はマーカー誤差・補正量・前フレームの補正との差の重み。
    min_improvement_ratio は、補正がマーカーを動かした量に対する正味の改善の下限で、これを下回る補正は
    捨てる。max_nfev は最小二乗の関数評価回数の上限。"""

    skip_threshold: float = 0.02
    max_rot_deg: float = 3.0
    max_pos: float = 0.2
    k_rot: float = 0.5
    k_pos: float = 1.5
    w_marker: float = 1.0
    w_pose: float = 0.1
    w_vel: float = 0.05
    min_improvement_ratio: float = 0.5
    max_nfev: int = 128


DEFAULT_FIT_PARAMS = FitParams()


@dataclass
class FitResult:
    poses: list[tuple[LocalBonePose, ...]]
    fallback_frames: int


_ROT_FITTED_ROLES = (
    "upper_body", "upper_body2", "lower_body", "neck", "head",
    "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r",
    "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "toe_l", "toe_r",
)
_POS_FITTED_ROLES = ("center", "groove")


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _rotvec_to_quat(dr):
    angle = math.sqrt(dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2)
    if angle < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    s = math.sin(angle / 2.0) / angle
    return (dr[0] * s, dr[1] * s, dr[2] * s, math.cos(angle / 2.0))


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_normalize(q):
    n = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n) if n > 0 else (0.0, 0.0, 0.0, 1.0)


def _apply(local, pos_idx, rot_idx, x):
    pos_corr = {idx: x[i * 3 : i * 3 + 3] for i, idx in enumerate(pos_idx)}
    base = len(pos_idx) * 3
    rot_corr = {idx: x[base + i * 3 : base + i * 3 + 3] for i, idx in enumerate(rot_idx)}
    out = []
    for bi, lp in enumerate(local):
        position = lp.position
        rotation = lp.rotation
        if bi in pos_corr:
            dp = pos_corr[bi]
            position = (position[0] + dp[0], position[1] + dp[1], position[2] + dp[2])
        if bi in rot_corr:
            rotation = _quat_normalize(_quat_mul(_rotvec_to_quat(rot_corr[bi]), rotation))
        out.append(LocalBonePose(position=position, rotation=rotation))
    return tuple(out)


def _clamp_each_bone_correction(x, n_pos, n_rot, pos_limit, rot_limit):
    x = list(x)

    def clamp_seg(off, limit):
        seg = x[off : off + 3]
        mag = math.sqrt(seg[0] ** 2 + seg[1] ** 2 + seg[2] ** 2)
        if mag > limit and mag > 0.0:
            scale = limit / mag
            x[off] *= scale
            x[off + 1] *= scale
            x[off + 2] *= scale

    for i in range(n_pos):
        clamp_seg(i * 3, pos_limit)
    base = n_pos * 3
    for i in range(n_rot):
        clamp_seg(base + i * 3, rot_limit)
    return x


def fit(profile, dense_pose, smoothed_markers, *, params=DEFAULT_FIT_PARAMS):
    model = profile.model
    bindings = profile.marker_bindings
    required = profile.required_bones
    marker_names = list(bindings)
    pos_idx = [required[r] for r in _POS_FITTED_ROLES if r in required]
    rot_idx = [required[r] for r in _ROT_FITTED_ROLES if r in required]
    n_params = (len(pos_idx) + len(rot_idx)) * 3

    fitted = []
    fallback = 0
    prev_x = np.zeros(n_params)
    for f, local in enumerate(dense_pose):
        world0 = evaluate_fk(model, local)
        fk0 = {m: world0[bindings[m].bone].position for m in marker_names}
        target = {m: smoothed_markers[m][f] for m in marker_names}
        disps = [_dist(fk0[m], target[m]) for m in marker_names]
        max_disp = max(disps) if disps else 0.0
        if max_disp < params.skip_threshold:
            fitted.append(local)
            prev_x = np.zeros(n_params)
            continue

        rot_limit = min(math.radians(params.max_rot_deg), params.k_rot * max_disp)
        pos_limit = min(params.max_pos, params.k_pos * max_disp)

        def residual(xv, _prev=prev_x):
            corrected = _apply(local, pos_idx, rot_idx, xv)
            world = evaluate_fk(model, corrected)
            res = []
            for m in marker_names:
                p = world[bindings[m].bone].position
                t = target[m]
                w = params.w_marker * bindings[m].weight
                res.extend((w * (p[0] - t[0]), w * (p[1] - t[1]), w * (p[2] - t[2])))
            res.extend(params.w_pose * xi for xi in xv)
            res.extend(
                params.w_vel * (xi - pi)
                for xi, pi in zip(xv, _prev, strict=True)
            )
            return res

        sol = least_squares(
            residual, np.zeros(n_params), method="lm", max_nfev=params.max_nfev
        )
        x = _clamp_each_bone_correction(sol.x, len(pos_idx), len(rot_idx), pos_limit, rot_limit)
        corrected = _apply(local, pos_idx, rot_idx, x)

        world = evaluate_fk(model, corrected)
        err_before = sum(_dist(fk0[m], target[m]) for m in marker_names)
        err_after = sum(
            _dist(world[bindings[m].bone].position, target[m]) for m in marker_names
        )
        improvement = err_before - err_after
        marker_movement = sum(
            _dist(world[bindings[m].bone].position, fk0[m]) for m in marker_names
        )
        if improvement > 0.0 and improvement >= params.min_improvement_ratio * marker_movement:
            fitted.append(corrected)
            prev_x = np.asarray(x)
        else:
            fitted.append(local)
            prev_x = np.zeros(n_params)
            fallback += 1

    return FitResult(poses=fitted, fallback_frames=fallback)
