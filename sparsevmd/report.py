"""レポート出力(dry-run統計・JSON・CSV)(sparsevmd.md §2.7)。

削減の入出力キー数・削減率・選択ボーン・範囲・keep-frame をまとめ、dry-run の
テキスト表示、JSON 出力(build_report の dict)、フレーム毎の CSV プレビューを提供する。
書き込み失敗は例外を送出し、CLI が終了コード3にする。

dry-run / JSON(build_report の dict): キー数・削減率・選択・範囲・keep に加え、§7.2 の
軸ごと最大絶対誤差(camera_errors / bone_errors)、§2.7/§6.3 の診断(camera_diag /
bone_diag = 不連続検出位置 cuts・分割理由 splits・継ぎ目書き換え seam_rewrites)を載せる。
誤差・診断は reduce 側(measure_*_errors / reduce_*_track の diagnostics)が surface する。

--preview-csv(write_preview_csv): フレーム毎の入力/出力サンプル値と誤差(§2.7)。行は
{track, frame, channel, input, output, error}。CLI が出力を再サンプリングして組み立てる。
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


def build_report(
    *, target, camera, bones, selected_bones, ranges, keep_frames,
    camera_errors=None, bone_errors=None, camera_diag=None, bone_diag=None,
    reduced=True,
):
    """レポート dict を組み立てる(§2.7, §7.2)。

    camera は (input, output) または None。bones は {name: (input, output)}。
    非選択ボーンも保持カウントで列挙する(§3.2)。camera_errors は軸別最大誤差 dict、
    bone_errors は {name: 誤差dict}。指定時のみ各エントリに "errors" を載せる(§7.2)。
    camera_diag / bone_diag は cuts・splits・seam_rewrites の診断 dict({name:dict} for bone)。
    指定時のみ各エントリに "diagnostics" を載せる(§2.7 不連続検出位置・分割理由、§6.3 継ぎ目)。
    reduced が False(全トラック1キー以下・範囲空など削減対象なし)のとき note を載せる(§2.2/§3.1)。
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
    """誤差 dict を簡潔な1行表現にする(§7.2 の軸ごと最大絶対誤差)。"""
    parts = [f"{k}={v:.4g}" for k, v in errors.items()]
    return "max error: " + " ".join(parts)


def _format_diag(diag):
    """診断の不連続検出位置・継ぎ目書き換えを簡潔な行にする(§2.7, §6.3)。"""
    lines = []
    if diag.get("cuts"):
        lines.append(f"  cuts: {diag['cuts']}")
    if diag.get("seam_rewrites"):
        lines.append(f"  seam rewrites: {diag['seam_rewrites']}")
    return lines


def format_dry_run(report):
    """dry-run のテキスト要約を返す(§2.7, §7.2)。"""
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


def write_json(report, path):
    """レポートを JSON で書き出す(日本語は非エスケープ)。失敗時は例外(CLIで終了コード3)。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


_PREVIEW_HEADER = ["track", "frame", "channel", "input", "output", "error"]


def write_preview_csv(rows, path):
    """フレーム毎の入力/出力サンプル値と誤差を CSV 出力する(§2.7 --preview-csv)。失敗時は例外。

    rows は {track, frame, channel, input, output, error} の dict 列(各値はスカラー)。
    入力/出力/誤差は小数6桁で整形する。トラック別の要約・誤差・分割理由は --report-json 側。
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_PREVIEW_HEADER)
        for r in rows:
            w.writerow([
                r["track"], r["frame"], r["channel"],
                f"{r['input']:.6f}", f"{r['output']:.6f}", f"{r['error']:.6f}",
            ])
