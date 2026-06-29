"""dry-run 用の処理計画・診断レポート(mocapvmd.md §4.4)。

ボーン一覧・分類結果・キー数・フレーム範囲・足IK/つま先IK候補と、各トラックの最大フレーム間
速度・最大回転角速度・スパイク候補数・保護フレーム数をまとめる。速度はトラックを時系列順に並べ、
連続キーの差をキー間フレーム差で1フレームあたりへ正規化した最大値とする。回転角は quaternion の
角度距離で測り、符号反転(q と -q は同一姿勢)を見かけの大角速度にしない。スパイク候補・保護
フレームは種別窓での検出(denoise)に基づく。
"""

import math

from mocapvmd import classify, denoise, footik, presets


def _quat_angle_deg(q1, q0):
    """2つの quaternion 間の角度距離(度)。符号反転は同一姿勢として 0 に近づく。"""
    dot = abs(sum(a * b for a, b in zip(q1, q0)))
    dot = min(1.0, dot)
    return math.degrees(2.0 * math.acos(dot))


def _track_diagnostics(keys):
    """時系列順のキー列から (最大速度, 最大角速度) を 1フレームあたりで返す。"""
    max_speed = 0.0
    max_ang = 0.0
    for a, b in zip(keys, keys[1:]):
        gap = b.frame - a.frame
        if gap <= 0:
            continue  # 同一フレームの重複キーはゼロ除算を避けて飛ばす
        max_speed = max(max_speed, math.dist(a.position, b.position) / gap)
        max_ang = max(max_ang, _quat_angle_deg(a.rotation, b.rotation) / gap)
    return max_speed, max_ang


def _spike_protected_counts(keys, clean_strength, category):
    """トラックのスパイク候補フレーム数と保護フレーム数を返す(§4.4)。

    種別の窓で検出し、スパイク候補は位置・回転候補フレームの和集合、保護フレームは境界(カット両側・
    範囲端)と位置・回転アクセントの和集合のフレーム数。キー1個以下、または値が検証を通らないトラックは
    (0, 0) を返す。
    """
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
    """foot_ik/toe_ik トラックを接地安定化し、name -> TrackStabilization を返す(§4.4)。

    パイプライン(一般ノイズ軽減→足IK安定化)と同じ順序で診断を出すため、denoise_on のときは
    クリーニング(apply_denoise)後の位置で安定化する。clean_strength はクリーニングに、横滑り抑制 S は
    接地検出・ロックに使う。値が検証を通らない・キー1個以下のトラックは対象外とする。
    """
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
    """キー削減率 = 1 - 出力/入力(§4.4)。入力0は0(ゼロ除算しない)。"""
    if input_count <= 0:
        return 0.0
    return 1.0 - output_count / input_count


def build_report(bone_keys, preset="medium", clean_strength=1.0, denoise=True, foot_ik_stabilize=True,
                 reduction=None, suppression=1.0, pose_denoise=None):
    """ボーンキー列(VmdDocument.bone、順不同でよい)から診断レポート dict を組み立てる(§4.4)。

    名前ごとにトラック化して初出順に並べ、各トラックを時系列順に整列してから診断する。preset は疎化の
    許容誤差プリセット名(表示用。§5.3)、clean_strength はクリーニング強度の倍率(§5.2)。各ボーンには、
    clean_strength で解決したクリーニングパラメータ(presets.resolve_cleaning の戻り)を付けてチューニングを
    確認できるようにする(§4.5)。

    reduction(reduce.reduce_bones の diagnostics_out。トラック名 -> {input_keys, output_keys,
    tol_pos, tol_rot, cuts, errors})を渡すと、トップレベルに reduce フラグ(疎化したか= reduction を
    渡したか)を、該当ボーンに reduction セクション(出力キー数・削減率(入出力から派生)・適用許容・
    検出カット数・最大再生誤差)を付ける(§4.4)。疎化の実行は呼び出し側(CLI)が行い、本関数は表示のみ。

    pose_denoise(pose_denoise.apply_pose_denoise の diagnostics_out)を渡すと、トップレベルに
    pose_denoise セクションをそのまま載せる(§12)。表現空間ノイズ除去の実行は呼び出し側(CLI)が行う。
    """
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
        max_speed, max_ang = _track_diagnostics(keys)
        spike_candidates, protected_frames = _spike_protected_counts(keys, clean_strength, category)
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
            # 接地区間は0始まり相対サンプルインデックス。整列済みキー列で絶対VMDフレーム番号へ戻す。
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
    """dry-run のテキスト要約を返す(§4.4)。適用プリセットと、各ボーンの診断値・解決済み
    クリーニングパラメータ・IK候補を表示する。"""
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
