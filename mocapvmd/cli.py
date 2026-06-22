"""mocapvmd CLI(mocapvmd.md §3)。

引数解析 → VMD読み(mmd_toolbox.vmd.io)→ 全ボーンの一般ノイズ軽減(クリーニング)→
足IK・つま先IKの接地安定化 → VMD書き。ボーン選択は持たず、一般ノイズ軽減は全ボーン、足IK安定化は
分類 foot_ik / toe_ik のボーンに適用する。対象外セクション(モーフ・カメラ・照明・セルフ影)は無加工
で透過する。処理後は密キーを線形補間で出力する。

終了コード: 0 正常 / 1 入力不正(VMDでない・値が非有限等)/ 2 引数エラー / 3 出力書き込み失敗。
"""

import argparse
import dataclasses
import os
import sys

from mmd_toolbox.vmd import io
from mmd_toolbox.vmd.reduce import BONE_LINEAR_INTERP
from mmd_toolbox.vmd.types import BoneKey

from . import classify, denoise, footik, presets, report


def _build_parser():
    p = argparse.ArgumentParser(prog="mocapvmd", allow_abbrev=False)
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--preset", choices=presets.PRESET_NAMES, default="balanced")
    p.add_argument("--denoise", dest="denoise", action="store_true", default=True)
    p.add_argument("--no-denoise", dest="denoise", action="store_false")
    p.add_argument("--foot-ik-stabilize", dest="foot_ik_stabilize", action="store_true", default=True)
    p.add_argument("--no-foot-ik-stabilize", dest="foot_ik_stabilize", action="store_false")
    p.add_argument("--report-json", dest="report_json")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


def _default_output(input_path):
    base, _ = os.path.splitext(input_path)
    return base + "_mocap.vmd"


def _same_path(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


def _clean_bones(bone_keys, preset):
    """全ボーンを種別別パラメータで一般ノイズ軽減し、密キー(線形補間)で返す(§4.1, §4.2)。

    各トラックを名前ごとに時系列順へまとめ、クリーニング後の密サンプルを線形補間キーとして組み直す
    (§3.3 のクリーニング後の密キー形式)。キー1個以下のトラックは平滑化できないため逐語保持する。
    """
    order = []
    groups = {}
    for k in bone_keys:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)

    out = []
    for name in order:
        ks = sorted(groups[name], key=lambda k: k.frame)
        positions = [k.position for k in ks]
        rotations = [k.rotation for k in ks]
        # 値の健全性は全トラックで検証する(1キーなど平滑化できないトラックも入力不正は弾く)。
        denoise.validate_bone_values(positions, rotations)
        if len(ks) < 2:
            out.extend(ks)
            continue
        params = presets.resolve_cleaning(preset, classify.classify(name))
        cpos, crot = denoise.apply_denoise(
            positions,
            rotations,
            pos_window=params["pos_window"],
            rot_window=params["rot_window"],
            pos_strength=params["pos_strength"],
            rot_strength=params["rot_strength"],
        )
        name_raw = ks[0].name_raw
        for i, k in enumerate(ks):
            out.append(BoneKey(name_raw, k.frame, cpos[i], crot[i], BONE_LINEAR_INTERP))
    out.sort(key=lambda k: (k.name_raw, k.frame))
    return out


def _stabilize_bones(bone_keys, preset):
    """分類 foot_ik / toe_ik の密トラックに接地安定化を適用し、密キー(線形補間)で返す(§4.3)。

    左右ペアリング・接地検出・接地ロックは footik.stabilize_foot_ik に委譲する。foot_ik / toe_ik 以外の
    ボーンと、キー1個以下のトラックは逐語透過する。回転は接地ロック対象外なので元の値を保つ。
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
        if category in ("foot_ik", "toe_ik") and len(ks) >= 2:
            tracks[name] = (category, [k.frame for k in ks], [k.position for k in ks])
    if not tracks:
        return bone_keys

    stabilized = footik.stabilize_foot_ik(tracks, preset)

    out = []
    for name in order:
        ks = sorted(groups[name], key=lambda k: k.frame)
        if name in stabilized:
            locked = stabilized[name].locked_positions
            name_raw = ks[0].name_raw
            for i, k in enumerate(ks):
                out.append(BoneKey(name_raw, k.frame, locked[i], k.rotation, BONE_LINEAR_INTERP))
        else:
            out.extend(ks)
    out.sort(key=lambda k: (k.name_raw, k.frame))
    return out


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)

    # 入力パス検証(§3.3)。不在・非通常ファイルは引数エラー。
    if not os.path.isfile(args.input):
        return 2

    # 出力先・上書きガード(§3.2)。入力と同一パスへの出力は --overwrite が必要。
    output = args.output if args.output is not None else _default_output(args.input)
    if not args.overwrite and _same_path(output, args.input):
        return 2

    # 入力読み込み(VMDでない等 → 入力不正)。
    try:
        doc, read_warnings = io.read(args.input)
    except Exception:
        return 1

    # 読み込み時の警告(デコード不能な名前フィールド等)を surface する。
    # 同一(コード・セクション・メッセージ)はキー毎の重複を避けて1行にまとめる。
    seen_warn = set()
    for w in read_warnings:
        key = (w.code, w.section, w.message)
        if key in seen_warn:
            continue
        seen_warn.add(key)
        where = f"({w.section})" if w.section else ""
        print(f"警告: {w.message}{where}", file=sys.stderr)

    # 診断レポート(dry-run 表示・report-json 出力)。どのボーンにどの処理が適用される予定かを
    # 出力を変更せずに確認できる。
    if args.dry_run or args.report_json:
        rep = report.build_report(doc.bone, args.preset, denoise=args.denoise)
        if args.dry_run:
            print(report.format_dry_run(rep))
        if args.report_json:
            try:
                report.write_json(rep, args.report_json)
            except OSError:
                return 3

    # dry-run は出力を書かずに終える。
    if args.dry_run:
        return 0

    # 一般ノイズ軽減を全ボーンへ適用する(--no-denoise 時はボーンを逐語透過)。対象外セクションは
    # いずれの場合も無加工で透過する。値が非有限・ゼロノルム quaternion 等の入力不正は終了コード1。
    if args.denoise:
        try:
            new_bone = _clean_bones(doc.bone, args.preset)
        except ValueError:
            return 1
    else:
        new_bone = doc.bone
    # 足IK安定化は一般ノイズ軽減の後に、分類 foot_ik / toe_ik のボーンへ適用する(§4.3 / §6)。
    if args.foot_ik_stabilize:
        new_bone = _stabilize_bones(new_bone, args.preset)
    out_doc = dataclasses.replace(doc, bone=new_bone)
    try:
        io.write_file(out_doc, output)
    except Exception:
        return 3
    return 0
