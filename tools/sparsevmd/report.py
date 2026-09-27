def reduction_rate(input_count, output_count):
    if input_count <= 0:
        return 0.0
    return 1.0 - output_count / input_count


def _track_entry(input_count, output_count):
    return {
        "input_keys": input_count,
        "output_keys": output_count,
        "reduction_rate": reduction_rate(input_count, output_count),
    }


def build_report(
    *, target, camera, bones, selected_bones, ranges, keep_frames,
    camera_errors=None, bone_errors=None, camera_diag=None, bone_diag=None,
    reduced=True,
):
    """camera は (入力キー数, 出力キー数) か None、bones は {ボーン名: (入力キー数, 出力キー数)}。
    bone_errors・bone_diag はボーン名をキーにした dict。
    """
    selected_bones = set(selected_bones or ())
    bone_errors = bone_errors or {}
    bone_diag = bone_diag or {}
    cam = _track_entry(*camera) if camera is not None else None
    if cam is not None and camera_errors is not None:
        cam["errors"] = camera_errors
    if cam is not None and camera_diag is not None:
        cam["diagnostics"] = camera_diag
    bone_list = []
    for name, (inp, out) in (bones or {}).items():
        entry = {"name": name, **_track_entry(inp, out), "selected": name in selected_bones}
        if name in bone_errors:
            entry["errors"] = bone_errors[name]
        if name in bone_diag:
            entry["diagnostics"] = bone_diag[name]
        bone_list.append(entry)
    result = {
        "target": target,
        "camera": cam,
        "bones": bone_list,
        "ranges": list(ranges or []),
        "keep_frames": list(keep_frames or []),
    }
    if not reduced:
        result["note"] = "削減対象なし"
    return result


def _rate_pct(entry):
    return f"{entry['reduction_rate'] * 100:.1f}%"


def _format_errors(errors):
    parts = [f"{k}={v:.4g}" for k, v in errors.items()]
    return "max error: " + " ".join(parts)


def _format_diag(diag):
    lines = []
    if diag.get("cuts"):
        lines.append(f"  cuts: {diag['cuts']}")
    if diag.get("seam_rewrites"):
        lines.append(f"  seam rewrites: {diag['seam_rewrites']}")
    for v in diag.get("verify") or ():
        lines.append(
            f"  verify [{v['range'][0]}, {v['range'][1]}]: "
            f"iterations={v['iterations']} added_total={v['added_total']}"
        )
    return lines


def format_dry_run(report):
    lines = [f"target: {report['target']}"]
    if "note" in report:
        lines.append(report["note"])
    cam = report["camera"]
    if cam is not None:
        lines.append(
            f"camera: {cam['input_keys']} -> {cam['output_keys']} keys ({_rate_pct(cam)} reduced)"
        )
        if "errors" in cam:
            lines.append(f"  {_format_errors(cam['errors'])}")
        if "diagnostics" in cam:
            lines.extend(_format_diag(cam["diagnostics"]))
    for b in report["bones"]:
        state = "selected" if b["selected"] else "excluded"
        lines.append(
            f"bone {b['name']}: {b['input_keys']} -> {b['output_keys']} keys "
            f"({_rate_pct(b)}) [{state}]"
        )
        if "errors" in b:
            lines.append(f"  {_format_errors(b['errors'])}")
        if "diagnostics" in b:
            lines.extend(_format_diag(b["diagnostics"]))
    lines.append(f"ranges: {report['ranges']}")
    lines.append(f"keep_frames: {report['keep_frames']}")
    return "\n".join(lines)
