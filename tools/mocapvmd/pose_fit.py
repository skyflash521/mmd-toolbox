"""姿勢フィット(最小実装)。

平滑化マーカーへ近づくよう、限定ボーン(センター/グルーブの位置、標準回転ロール
の回転)を単フレーム最小二乗で小さく補正する。マーカー変位が微小なフレームは元
姿勢を保持し、補正量は絶対上限とマーカー変位比例上限の小さい方でper-boneクランプ
し、補正後にマーカー誤差が改善しないフレームは元姿勢へフォールバックする。
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from pmx.pose import LocalBonePose, evaluate_fk


@dataclass
class FitParams:
    skip_threshold: float = 0.02  # 最大マーカー変位がこれ未満ならフィットしない
    max_rot_deg: float = 3.0  # 回転補正の絶対上限(度)
    max_pos: float = 0.2  # 位置補正の絶対上限
    k_rot: float = 0.5  # 回転補正の変位比例係数(rad/単位)
    k_pos: float = 1.5  # 位置補正の変位比例係数
    w_marker: float = 1.0
    w_pose: float = 0.1  # 元姿勢保持(補正量)の重み
    w_vel: float = 0.05  # 前フレーム補正との差分ペナルティの重み
    min_improvement_ratio: float = 0.5  # 補正がマーカーを動かした量に対する正味改善の下限(これ未満は過大補正として破棄)
    max_nfev: int = 128  # 最小二乗の関数評価上限(性能のため数反復に制限)


DEFAULT_FIT_PARAMS = FitParams()


@dataclass
class FitResult:
    poses: list  # フィット後のローカル姿勢列(フレーム毎の tuple[LocalBonePose, ...])
    fallback_frames: int  # 改善せず元姿勢に戻したフレーム数(診断)


# 回転を変数にする標準ロール(付与/捩/IK/指は変数にしない)
_ROT_ROLES = (
    "upper_body", "upper_body2", "lower_body", "neck", "head",
    "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r",
    "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "toe_l", "toe_r",
)
# 位置を変数にする標準ロール
_POS_ROLES = ("center", "groove")


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _rotvec_to_quat(dr):
    """回転ベクトル(軸×角)→クォータニオン (x,y,z,w)。"""
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
    """補正ベクトル x を元ローカル姿勢へ適用した tuple[LocalBonePose, ...]。"""
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


def _clamp(x, n_pos, n_rot, pos_limit, rot_limit):
    """各ボーンの補正量を上限以下にスケールダウンする。"""
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
    """密ローカル姿勢を平滑化マーカーへ近づけるよう小さく補正する。"""
    model = profile.model
    bindings = profile.marker_bindings
    required = profile.required_bones
    marker_names = list(bindings)
    pos_idx = [required[r] for r in _POS_ROLES if r in required]
    rot_idx = [required[r] for r in _ROT_ROLES if r in required]
    n_params = (len(pos_idx) + len(rot_idx)) * 3

    fitted = []
    fallback = 0
    prev_x = np.zeros(n_params)  # 前フレームの採用補正量(差分ペナルティ用)
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
            res.extend(params.w_pose * xi for xi in xv)  # 元姿勢保持
            res.extend(params.w_vel * (xi - pi) for xi, pi in zip(xv, _prev))  # 前フレーム差分
            return res

        sol = least_squares(
            residual, np.zeros(n_params), method="lm", max_nfev=params.max_nfev
        )
        x = _clamp(sol.x, len(pos_idx), len(rot_idx), pos_limit, rot_limit)
        corrected = _apply(local, pos_idx, rot_idx, x)

        world = evaluate_fk(model, corrected)
        err_before = sum(_dist(fk0[m], target[m]) for m in marker_names)
        err_after = sum(
            _dist(world[bindings[m].bone].position, target[m]) for m in marker_names
        )
        improvement = err_before - err_after
        # 補正がマーカーを動かした総量。
        corr_effect = sum(
            _dist(world[bindings[m].bone].position, fk0[m]) for m in marker_names
        )
        # 改善あり、かつマーカーを動かした量の大半が目標方向への改善である場合のみ採用
        # (動かした量に対し改善が小さい=冗長な過大補正は破棄して元姿勢を使う。§8.4)。
        if improvement > 0.0 and improvement >= params.min_improvement_ratio * corr_effect:
            fitted.append(corrected)
            prev_x = np.asarray(x)
        else:
            fitted.append(local)
            prev_x = np.zeros(n_params)
            fallback += 1

    return FitResult(poses=fitted, fallback_frames=fallback)
