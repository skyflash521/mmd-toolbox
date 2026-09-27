import math

from pmx.pose import evaluate_fk, sample_local_poses
from vmd.reduce import BONE_LINEAR_INTERP
from vmd.types import BoneKey

from . import marker_denoise, pose_fit
from .markers import extract_markers
from .model_profile import load_mocap_profile


def _quat_angle_deg(q1, q0):
    dot = abs(sum(a * b for a, b in zip(q1, q0, strict=True)))
    dot = min(1.0, dot)
    return math.degrees(2.0 * math.acos(dot))


def _collect_diagnostics(out, profile, model, pmx_path, n, dense, world, fitted,
                         raw_markers, smoothed_markers):
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

    position_fitted_bone_indices = {
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
        for bi, (dl, fl) in enumerate(zip(dense[f], fitted.poses[f], strict=True)):
            ang = _quat_angle_deg(dl.rotation, fl.rotation)
            if ang > max_rot_deg:
                max_rot_deg = ang
            if bi in position_fitted_bone_indices:
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
                       diagnostics_out: dict | None = None):
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

    model_bones_in_input = {b.name for b in model.bones} & set(tracks)
    name_raw = {name: tracks[name][0].name_raw for name in model_bones_in_input}

    out = []
    for f_idx, frame in enumerate(frames):
        poses = fitted.poses[f_idx]
        for bi, bone in enumerate(model.bones):
            if bone.name in model_bones_in_input:
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

    for k in bone_keys:
        if k.name not in model_bones_in_input:
            out.append(k)

    return out
