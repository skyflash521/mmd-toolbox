"""処理計画・診断レポート(mocapvmd.md §4.4)。

ボーン一覧・分類結果・キー数・フレーム範囲・足IK/つま先IK候補と、各トラックの最大フレーム間
速度・最大回転角速度・スパイク候補数・保護フレーム数をまとめる。速度はトラックを時系列順に並べ、
連続キーの差をキー間フレーム差で1フレームあたりへ正規化した最大値とする。回転角は quaternion の
角度距離で測り、符号反転(q と -q は同一姿勢)を見かけの大角速度にしない。スパイク候補・保護
フレームは種別窓での検出(denoise)に基づく。
"""

import json
import math

from mocapvmd import classify, denoise, presets


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


def _spike_protected_counts(keys, preset, category):
    """トラックのスパイク候補フレーム数と保護フレーム数を返す(§4.4)。

    種別の窓で検出し、スパイク候補は位置・回転候補フレームの和集合、保護フレームは境界(カット両側・
    範囲端)と位置・回転アクセントの和集合のフレーム数。キー1個以下、または値が検証を通らないトラックは
    (0, 0) を返す。
    """
    if len(keys) < 2:
        return 0, 0
    params = presets.resolve_cleaning(preset, category)
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


def build_report(bone_keys, preset="balanced", denoise=True):
    """ボーンキー列(VmdDocument.bone、順不同でよい)から診断レポート dict を組み立てる(§4.4)。

    名前ごとにトラック化して初出順に並べ、各トラックを時系列順に整列してから診断する。
    各ボーンには、選択プリセットで解決したクリーニングパラメータ(presets.resolve_cleaning の戻り)を
    付けてチューニングを確認できるようにする(§4.5)。
    """
    order = []
    groups = {}
    for k in bone_keys:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)

    bones = []
    foot_ik = []
    toe_ik = []
    all_frames = []
    for name in order:
        keys = sorted(groups[name], key=lambda k: k.frame)
        all_frames.extend(k.frame for k in keys)
        category = classify.classify(name)
        max_speed, max_ang = _track_diagnostics(keys)
        spike_candidates, protected_frames = _spike_protected_counts(keys, preset, category)
        bones.append(
            {
                "name": name,
                "category": category,
                "input_keys": len(keys),
                "frame_first": keys[0].frame,
                "frame_last": keys[-1].frame,
                "max_speed": max_speed,
                "max_ang_speed_deg": max_ang,
                "spike_candidates": spike_candidates,
                "protected_frames": protected_frames,
                "cleaning": presets.resolve_cleaning(preset, category),
            }
        )
        if category == "foot_ik":
            foot_ik.append(name)
        elif category == "toe_ik":
            toe_ik.append(name)

    return {
        "preset": preset,
        "denoise": denoise,
        "range": [min(all_frames), max(all_frames)] if all_frames else [],
        "bones": bones,
        "foot_ik_candidates": foot_ik,
        "toe_ik_candidates": toe_ik,
    }


def write_json(report, path):
    """レポートを JSON で書き出す(日本語は非エスケープ)。失敗時は例外(CLIで終了コード3)。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def format_dry_run(report):
    """dry-run のテキスト要約を返す(§4.4)。適用プリセットと、各ボーンの診断値・解決済み
    クリーニングパラメータ・IK候補を表示する。"""
    lines = [
        f"preset: {report['preset']}",
        f"denoise: {'on' if report['denoise'] else 'off'}",
        f"range: {report['range']}",
    ]
    for b in report["bones"]:
        c = b["cleaning"]
        lines.append(
            f"{b['name']} [{b['category']}] keys={b['input_keys']} "
            f"frames=[{b['frame_first']},{b['frame_last']}] "
            f"max_speed={b['max_speed']:.4g} max_rot={b['max_ang_speed_deg']:.4g}deg "
            f"spikes={b['spike_candidates']} protected={b['protected_frames']} "
            f"clean_pos={c['pos_strength']:.4g} clean_rot={c['rot_strength']:.4g} "
            f"win=[{c['pos_window']},{c['rot_window']}]"
        )
    lines.append(f"足IK候補: {report['foot_ik_candidates']}")
    lines.append(f"つま先IK候補: {report['toe_ik_candidates']}")
    return "\n".join(lines)
