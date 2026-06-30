"""表現空間ノイズ除去のオーケストレーション。

入力ボーンキーを、モデルプロファイルのFK・マーカー平滑化・姿勢フィットを通して
密キー列へ変換する。モデルが扱い入力にあるボーンを密キー化(線形補間)し、モデル外
ボーン(指など)は原キーのまま通す。後段の足IK安定化・疎化へ同じ密キー形式で渡せる。
"""

import math

from pmx.pose import evaluate_fk, sample_local_poses
from vmd.reduce import BONE_LINEAR_INTERP
from vmd.types import BoneKey

from . import marker_denoise, pose_fit
from .markers import extract_markers
from .model_profile import load_mocap_profile


def _quat_angle_deg(q1, q0):
    """2つの quaternion 間の角度距離(度)。符号反転(q と -q は同一姿勢)は 0 に近づく。"""
    dot = abs(sum(a * b for a, b in zip(q1, q0)))
    dot = min(1.0, dot)
    return math.degrees(2.0 * math.acos(dot))


def _collect_diagnostics(out, profile, model, pmx_path, n, dense, world, fitted,
                         raw_markers, smoothed_markers):
    """診断素データ(マーカー数・必須ボーン検証・平滑化前後の変位・fit改善)を out に書く。

    マーカー変位は平滑化前(FK元姿勢=raw)と平滑化後の距離。fit のマーカー誤差は平滑化目標との
    距離で、補正後は採用姿勢を再FK評価して測る(採用しないフレームは元姿勢のままなので悪化しない)。
    補正量は元ローカル姿勢と採用姿勢の差で、回転は角度距離(度)、位置は center/groove のみ測る。
    """
    bindings = profile.marker_bindings
    marker_names = list(bindings)

    by_marker = {}
    all_disp = []
    for m in marker_names:
        ds = [math.dist(raw_markers[m][f], smoothed_markers[m][f]) for f in range(n)]
        by_marker[m] = {
            "max": max(ds) if ds else 0.0,
            "mean": (sum(ds) / len(ds)) if ds else 0.0,
        }
        all_disp.extend(ds)

    pos_role_idx = {
        profile.required_bones[r] for r in ("center", "groove") if r in profile.required_bones
    }
    err_before = []
    err_after = []
    max_rot_deg = 0.0
    max_center = 0.0
    for f in range(n):
        wb = world[f]
        wa = evaluate_fk(model, fitted.poses[f])
        eb = [math.dist(wb[bindings[m].bone].position, smoothed_markers[m][f]) for m in marker_names]
        ea = [math.dist(wa[bindings[m].bone].position, smoothed_markers[m][f]) for m in marker_names]
        err_before.append(sum(eb) / len(eb) if eb else 0.0)
        err_after.append(sum(ea) / len(ea) if ea else 0.0)
        for bi, (dl, fl) in enumerate(zip(dense[f], fitted.poses[f])):
            ang = _quat_angle_deg(dl.rotation, fl.rotation)
            if ang > max_rot_deg:
                max_rot_deg = ang
            if bi in pos_role_idx:
                pd = math.dist(dl.position, fl.position)
                if pd > max_center:
                    max_center = pd

    out.update({
        "enabled": True,
        "pmx": pmx_path,
        "frames": n,
        "markers": {"available": len(marker_names), "required_bones_ok": True},
        "marker_displacement": {
            "max": max(all_disp) if all_disp else 0.0,
            "mean": (sum(all_disp) / len(all_disp)) if all_disp else 0.0,
            "by_marker": by_marker,
        },
        "fit": {
            "frames": n,
            "fallback_frames": fitted.fallback_frames,
            "mean_error_before": (sum(err_before) / len(err_before)) if err_before else 0.0,
            "mean_error_after": (sum(err_after) / len(err_after)) if err_after else 0.0,
            "max_bone_delta_deg": max_rot_deg,
            "max_center_delta": max_center,
        },
    })


def apply_pose_denoise(bone_keys, *, pmx_path=None, preset=None, fit_params=None,
                       diagnostics_out=None):
    """ボーンキー列に表現空間ノイズ除去を適用し、新しいボーンキー列を返す。

    diagnostics_out に dict を渡すと、マーカー数・必須ボーン検証・平滑化前後のマーカー変位・
    姿勢フィットの改善量などの診断素データを書き込む(レポート層へ渡す。§12)。
    """
    if not bone_keys:
        return []

    profile = load_mocap_profile(pmx_path)
    model = profile.model

    tracks = {}
    for k in bone_keys:
        tracks.setdefault(k.name, []).append(k)
    for keys in tracks.values():
        keys.sort(key=lambda k: k.frame)

    f0 = min(k.frame for k in bone_keys)
    f1 = max(k.frame for k in bone_keys)
    frames = range(f0, f1 + 1)

    # 密ローカル姿勢は一度だけ作り、FK・姿勢フィットで再利用する(再サンプルしない)。
    dense = [sample_local_poses(model, tracks, f) for f in frames]
    world = [evaluate_fk(model, lp) for lp in dense]

    traj = extract_markers(profile, world)
    categories = {m: mb.category for m, mb in profile.marker_bindings.items()}
    smoothed = marker_denoise.smooth(traj.markers, categories, preset=preset)
    fit_kwargs = {} if fit_params is None else {"params": fit_params}
    fitted = pose_fit.fit(profile, dense, smoothed.markers, **fit_kwargs)

    if diagnostics_out is not None:
        _collect_diagnostics(
            diagnostics_out, profile, model, pmx_path, len(frames),
            dense, world, fitted, traj.markers, smoothed.markers,
        )

    # モデルが扱い、かつ入力にあるボーンだけを密キー化する。
    processed = {b.name for b in model.bones} & set(tracks)
    name_raw = {name: tracks[name][0].name_raw for name in processed}

    out = []
    for f_idx, frame in enumerate(frames):
        poses = fitted.poses[f_idx]
        for bi, bone in enumerate(model.bones):
            if bone.name in processed:
                lp = poses[bi]
                out.append(
                    BoneKey(
                        name_raw=name_raw[bone.name],
                        frame=frame,
                        position=lp.position,
                        rotation=lp.rotation,
                        interpolation=BONE_LINEAR_INTERP,
                    )
                )

    # モデル外ボーン(指など)は原キーをそのまま通す。
    for k in bone_keys:
        if k.name not in processed:
            out.append(k)

    return out
