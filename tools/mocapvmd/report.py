import math

from mocapvmd import classify, denoise, footik, presets


def _quat_angle_deg(q1, q0):
    dot = abs(sum(a * b for a, b in zip(q1, q0, strict=True)))
    dot = min(1.0, dot)
    return math.degrees(2.0 * math.acos(dot))


def _max_speed_and_angular_speed_deg_per_frame(keys) -> tuple[float, float]:
    max_speed = 0.0
    max_ang = 0.0
    for a, b in zip(keys, keys[1:], strict=False):
        gap = b.frame - a.frame
        if gap <= 0:
            continue
        max_speed = max(max_speed, math.dist(a.position, b.position) / gap)
        max_ang = max(max_ang, _quat_angle_deg(a.rotation, b.rotation) / gap)
    return max_speed, max_ang


def _spike_and_protected_counts(keys, clean_strength, category):
    if len(keys) < 2:
        return 0, 0
    params = presets.resolve_cleaning(clean_strength, category)
    try:
        det = denoise.detect_noise_events(
            [k.position for k in keys],
            [k.rotation for k in keys],
            pos_window=params["pos_window"],
            rot_window=params["rot_window"],
        )
    except ValueError:
        return 0, 0
    spike_frames = {f for f, _ in det.pos_candidates} | set(det.rot_candidates)
    protected = set(det.boundaries) | {f for f, _ in det.pos_accent} | set(det.rot_accent)
    return len(spike_frames), len(protected)


def _stabilization(bone_keys, clean_strength, denoise_on, suppression):
    order = []
    groups = {}
    for k in bone_keys:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)

    tracks = {}
    for name in order:
        ks = sorted(groups[name], key=lambda k: k.frame)
        category = classify.classify(name)
        if category not in ("foot_ik", "toe_ik") or len(ks) < 2:
            continue
        positions = [k.position for k in ks]
        rotations = [k.rotation for k in ks]
        try:
            denoise.validate_bone_values(positions, rotations)
        except ValueError:
            continue
        if denoise_on:
            params = presets.resolve_cleaning(clean_strength, category)
            positions, _ = denoise.apply_denoise(
                positions, rotations,
                pos_window=params["pos_window"], rot_window=params["rot_window"],
                pos_strength=params["pos_strength"], rot_strength=params["rot_strength"],
            )
        tracks[name] = (category, [k.frame for k in ks], positions)
    if not tracks:
        return {}
    return footik.stabilize_foot_ik(tracks, suppression)


def _reduction_rate(input_count, output_count):
    if input_count <= 0:
        return 0.0
    return 1.0 - output_count / input_count


def build_report(bone_keys, preset="medium", clean_strength=1.0, denoise=True, foot_ik_stabilize=True,
                 reduction: dict[str, dict] | None = None, suppression=1.0,
                 pose_denoise: dict | None = None):
    order = []
    groups = {}
    for k in bone_keys:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)

    stab = _stabilization(bone_keys, clean_strength, denoise, suppression) if foot_ik_stabilize else {}

    bones = []
    foot_ik = []
    toe_ik = []
    all_frames = []
    for name in order:
        keys = sorted(groups[name], key=lambda k: k.frame)
        all_frames.extend(k.frame for k in keys)
        category = classify.classify(name)
        max_speed, max_ang = _max_speed_and_angular_speed_deg_per_frame(keys)
        spike_candidates, protected_frames = _spike_and_protected_counts(keys, clean_strength, category)
        entry = {
            "name": name,
            "category": category,
            "input_keys": len(keys),
            "frame_first": keys[0].frame,
            "frame_last": keys[-1].frame,
            "max_speed": max_speed,
            "max_ang_speed_deg": max_ang,
            "spike_candidates": spike_candidates,
            "protected_frames": protected_frames,
            "cleaning": presets.resolve_cleaning(clean_strength, category),
        }
        if name in stab:
            ts = stab[name]
            entry["grounding_candidates"] = len(ts.grounding.candidate_frames)
            entry["grounding_segments"] = [
                [keys[s.start].frame, keys[s.end].frame] for s in ts.grounding.segments
            ]
            entry["max_change"] = ts.max_change
            entry["mean_change"] = ts.mean_change
            entry["lock_applied_ratio"] = ts.lock_applied_ratio
            entry["clamp_warnings"] = len(ts.warnings)
        if reduction is not None and name in reduction:
            r = reduction[name]
            entry["reduction"] = {
                "output_keys": r["output_keys"],
                "reduction_rate": _reduction_rate(r["input_keys"], r["output_keys"]),
                "tol_pos": r["tol_pos"],
                "tol_rot": r["tol_rot"],
                "cuts": r["cuts"],
                "errors": r["errors"],
            }
        bones.append(entry)
        if category == "foot_ik":
            foot_ik.append(name)
        elif category == "toe_ik":
            toe_ik.append(name)

    result = {
        "preset": preset,
        "clean_strength": clean_strength,
        "denoise": denoise,
        "foot_ik_stabilize": foot_ik_stabilize,
        "foot_slide_suppression": suppression,
        "reduce": reduction is not None,
        "range": [min(all_frames), max(all_frames)] if all_frames else [],
        "bones": bones,
        "foot_ik_candidates": foot_ik,
        "toe_ik_candidates": toe_ik,
    }
    if pose_denoise:
        result["pose_denoise"] = pose_denoise
    return result


def format_dry_run(report):
    lines = [
        f"preset: {report['preset']}",
        f"clean_strength: {report.get('clean_strength')}",
        f"denoise: {'on' if report['denoise'] else 'off'}",
        f"foot_ik_stabilize: {'on' if report.get('foot_ik_stabilize') else 'off'}",
        f"foot_slide_suppression: {report.get('foot_slide_suppression')}",
        f"reduce: {'on' if report.get('reduce') else 'off'}",
        f"range: {report['range']}",
    ]
    for b in report["bones"]:
        c = b["cleaning"]
        line = (
            f"{b['name']} [{b['category']}] keys={b['input_keys']} "
            f"frames=[{b['frame_first']},{b['frame_last']}] "
            f"max_speed={b['max_speed']:.4g} max_rot={b['max_ang_speed_deg']:.4g}deg "
            f"spikes={b['spike_candidates']} protected={b['protected_frames']} "
            f"clean_pos={c['pos_strength']:.4g} clean_rot={c['rot_strength']:.4g} "
            f"win=[{c['pos_window']},{c['rot_window']}]"
        )
        if "grounding_segments" in b:
            line += (
                f" ground_seg={len(b['grounding_segments'])}"
                f" lock_rate={b['lock_applied_ratio']:.2f}"
                f" max_chg={b['max_change']:.4g}"
                f" warn={b['clamp_warnings']}"
            )
        if "reduction" in b:
            r = b["reduction"]
            e = r["errors"]
            err_pos = max(e["pos_x"], e["pos_y"], e["pos_z"])
            line += (
                f" out_keys={r['output_keys']} red={r['reduction_rate'] * 100:.1f}%"
                f" cuts={r['cuts']} tol=[{r['tol_pos']:.4g},{r['tol_rot']:.4g}]"
                f" err_pos={err_pos:.4g} err_rot={e['rot_deg']:.4g}deg"
            )
        lines.append(line)
    lines.append(f"足IK候補: {report['foot_ik_candidates']}")
    lines.append(f"つま先IK候補: {report['toe_ik_candidates']}")
    if "pose_denoise" in report:
        pd = report["pose_denoise"]
        mk = pd["markers"]
        md = pd["marker_displacement"]
        ft = pd["fit"]
        src = f"pmx={pd['pmx']}" if pd.get("pmx") else "既定モデルプロファイル"
        lines.append(f"pose_denoise: {src} frames={pd['frames']}")
        lines.append(
            f"  markers: available={mk['available']} required_bones_ok={mk['required_bones_ok']}"
        )
        lines.append(f"  marker_disp: max={md['max']:.4g} mean={md['mean']:.4g}")
        lines.append(
            f"  fit: error {ft['mean_error_before']:.4g} -> {ft['mean_error_after']:.4g}"
            f" fallback={ft['fallback_frames']}"
            f" max_rot={ft['max_bone_delta_deg']:.4g}deg max_center={ft['max_center_delta']:.4g}"
        )
    return "\n".join(lines)
