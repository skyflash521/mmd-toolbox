"""レポート出力(dry-run統計・JSON・CSV)(sparsevmd.md §2.7)。

削減の入出力キー数・削減率・選択ボーン・範囲・keep-frame をまとめ、dry-run の
テキスト表示、JSON 出力、CSV プレビューを提供する。書き込み失敗は例外を送出し、
CLI が終了コード3にする。

本MVPの範囲外(後続): 不連続検出位置・最大誤差・分割理由・per-frame サンプル値。
いずれも削減内部の診断情報で、レポートに載せるには reduce_*_track がトラック別の
診断を戻り値で surface する拡張が要る。本MVPはキー数・削減率・選択・範囲・keep を扱う。
"""

import csv
import json


def reduction_rate(input_count, output_count):
    """削減率 = 1 - 出力/入力。入力0は0(ゼロ除算しない)。"""
    if input_count <= 0:
        return 0.0
    return 1.0 - output_count / input_count


def _track_entry(input_count, output_count):
    return {
        "input_keys": input_count,
        "output_keys": output_count,
        "reduction_rate": reduction_rate(input_count, output_count),
    }


def build_report(*, target, camera, bones, selected_bones, ranges, keep_frames):
    """レポート dict を組み立てる(§2.7)。

    camera は (input, output) または None。bones は {name: (input, output)}。
    非選択ボーンも保持カウントで列挙する(§3.2)。
    """
    selected_bones = set(selected_bones or ())
    cam = _track_entry(*camera) if camera is not None else None
    bone_list = []
    for name, (inp, out) in (bones or {}).items():
        entry = {"name": name, **_track_entry(inp, out), "selected": name in selected_bones}
        bone_list.append(entry)
    return {
        "target": target,
        "camera": cam,
        "bones": bone_list,
        "ranges": list(ranges or []),
        "keep_frames": list(keep_frames or []),
    }


def _rate_pct(entry):
    return f"{entry['reduction_rate'] * 100:.1f}%"


def format_dry_run(report):
    """dry-run のテキスト要約を返す(§2.7)。"""
    lines = [f"target: {report['target']}"]
    cam = report["camera"]
    if cam is not None:
        lines.append(
            f"camera: {cam['input_keys']} -> {cam['output_keys']} keys ({_rate_pct(cam)} reduced)"
        )
    for b in report["bones"]:
        state = "selected" if b["selected"] else "excluded"
        lines.append(
            f"bone {b['name']}: {b['input_keys']} -> {b['output_keys']} keys "
            f"({_rate_pct(b)}) [{state}]"
        )
    lines.append(f"ranges: {report['ranges']}")
    lines.append(f"keep_frames: {report['keep_frames']}")
    return "\n".join(lines)


def write_json(report, path):
    """レポートを JSON で書き出す(日本語は非エスケープ)。失敗時は例外(CLIで終了コード3)。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


_CSV_HEADER = ["track", "input_keys", "output_keys", "reduction_rate", "selected"]


def write_csv(report, path):
    """トラック別の入出力キー数・削減率・選択状態を CSV で書き出す。失敗時は例外。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_CSV_HEADER)
        cam = report["camera"]
        if cam is not None:
            w.writerow(["camera", cam["input_keys"], cam["output_keys"],
                        f"{cam['reduction_rate']:.6f}", ""])
        for b in report["bones"]:
            w.writerow([b["name"], b["input_keys"], b["output_keys"],
                        f"{b['reduction_rate']:.6f}", "true" if b["selected"] else "false"])
