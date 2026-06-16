"""sparsevmd CLI(コアの薄いラッパー。sparsevmd.md §2, §9)。

引数解析 → VMD読み(mmd_toolbox.vmd.io)→ ボーン選択・範囲解決 → トラック削減
(reduce.reduce_camera_track / reduce_bone_track)→ VMD書き。
終了コード(§9): 0 正常 / 1 入力不正 / 2 引数エラー / 3 出力書き込み失敗 /
4 strict で許容誤差を満たせない。

--curve-mode は bezier(既定)/linear。bezier は各区間を1本のベジェ曲線で表現して
キーを削減し制御点を出力に格納、linear は線形補間ブロック固定で削減する。
"""

import argparse
import dataclasses
import os
import sys
from collections import Counter

from mmd_toolbox.vmd import io

from . import presets, ranges, report, selection
from .cuts import parse_cut_threshold_bone, parse_cut_threshold_camera
from .reduce import (
    StrictError,
    measure_bone_errors,
    measure_camera_errors,
    reduce_bone_track,
    reduce_camera_track,
)

# 個別許容誤差オプション → Tolerances フィールド。
_TOL_ARGS = {
    "bone_pos_tol": "bone_pos",
    "bone_rot_tol": "bone_rot",
    "camera_pos_tol": "camera_pos",
    "camera_rot_tol": "camera_rot",
    "camera_distance_tol": "camera_distance",
    "camera_fov_tol": "camera_fov",
}


def _nonneg_int(text):
    """非負整数(フレーム番号)。負値・非整数は引数エラー(§2.6)。"""
    v = int(text)  # 非整数は ValueError → argparse が exit 2
    if v < 0:
        raise argparse.ArgumentTypeError(f"フレーム番号は非負: {text!r}")
    return v


def _build_parser():
    p = argparse.ArgumentParser(prog="sparsevmd", allow_abbrev=False)
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--target", choices=("camera", "bone", "all"), default="all")
    # ボーン選択。
    p.add_argument("--bone", dest="bone", action="append", default=[])
    p.add_argument("--bone-glob", dest="bone_glob", action="append", default=[])
    p.add_argument("--bone-group", dest="bone_group", action="append", default=[])
    p.add_argument("--bone-file", dest="bone_file")
    p.add_argument("--exclude-bone", dest="exclude_bone", action="append", default=[])
    p.add_argument("--exclude-bone-glob", dest="exclude_bone_glob", action="append", default=[])
    p.add_argument("--exclude-bone-group", dest="exclude_bone_group", action="append", default=[])
    p.add_argument("--list-bones", dest="list_bones", action="store_true")
    p.add_argument("--range", dest="ranges", action="append", type=ranges.parse_range)
    # プリセット・許容誤差。
    p.add_argument("--preset", choices=presets.PRESET_NAMES, default="balanced")
    for arg in _TOL_ARGS:
        p.add_argument("--" + arg.replace("_", "-"), dest=arg, type=float)
    # フィット制御。
    p.add_argument("--max-segment-frames", dest="max_segment_frames", type=int, default=180)
    p.add_argument("--min-segment-frames", dest="min_segment_frames", type=int, default=1)
    p.add_argument("--curve-mode", dest="curve_mode", choices=("bezier", "linear"), default="bezier")
    p.add_argument("--strict", action="store_true")
    # カット・不連続。
    p.add_argument(
        "--cut-threshold-camera",
        dest="cut_threshold_camera",
        type=parse_cut_threshold_camera,
        default=(5.0, 20.0, 5.0),
    )
    p.add_argument(
        "--cut-threshold-bone",
        dest="cut_threshold_bone",
        type=parse_cut_threshold_bone,
        default=(1.0, 30.0),
    )
    p.add_argument("--no-cut-detect", dest="no_cut_detect", action="store_true")
    p.add_argument("--keep-frame", dest="keep_frames", action="append", type=_nonneg_int, default=[])
    # レポート・運用。
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--report-json", dest="report_json")
    p.add_argument("--preview-csv", dest="preview_csv")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _default_output(input_path):
    base, _ = os.path.splitext(input_path)
    return base + "_sparse.vmd"


def _same_path(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


def _build_selectors(args):
    """CLI 引数とボーンファイルから (includes, excludes) を組み立てる(§2.2)。"""
    includes = [selection.Selector("name", v) for v in args.bone]
    includes += [selection.Selector("glob", v) for v in args.bone_glob]
    includes += [selection.Selector("group", v) for v in args.bone_group]
    excludes = [selection.Selector("name", v) for v in args.exclude_bone]
    excludes += [selection.Selector("glob", v) for v in args.exclude_bone_glob]
    excludes += [selection.Selector("group", v) for v in args.exclude_bone_group]
    if args.bone_file:
        text = open(args.bone_file, encoding="utf-8").read()
        finc, fexc = selection.parse_bone_file(text)
        includes += finc
        excludes += fexc
    return includes, excludes


def _has_bone_selection(args):
    return bool(
        args.bone
        or args.bone_glob
        or args.bone_group
        or args.bone_file
        or args.exclude_bone
        or args.exclude_bone_glob
        or args.exclude_bone_group
    )


def _bone_names_in_order(bone_keys):
    """出現順を保ったボーン名(name_raw デコード)の一覧。"""
    seen = []
    seen_raw = set()
    for k in bone_keys:
        if k.name_raw not in seen_raw:
            seen_raw.add(k.name_raw)
            seen.append(k.name)
    return seen


def _bone_keys_by_name(bone_keys):
    groups = {}
    for k in bone_keys:
        groups.setdefault(k.name, []).append(k)
    return groups


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)

    # 入力パス検証(§2.2)。
    if not os.path.isfile(args.input):
        return 2
    # --bone-file パス検証(§2.2)。不在・非通常ファイルは引数エラー。
    if args.bone_file is not None and not os.path.isfile(args.bone_file):
        return 2

    # フィット制御の検証(§2.5)。
    if args.max_segment_frames < 1 or args.min_segment_frames < 1:
        return 2
    if args.min_segment_frames > args.max_segment_frames:
        return 2

    # 許容誤差解決(明示 > プリセット。§2.3/§2.4)。
    overrides = {
        field: getattr(args, arg)
        for arg, field in _TOL_ARGS.items()
        if getattr(args, arg) is not None
    }
    try:
        tols = presets.resolve_tolerances(args.preset, overrides)
    except ValueError:
        return 2

    # --target camera とボーン選択の同時指定はエラー(§2.2)。ただし --list-bones は検査モード。
    if args.target == "camera" and _has_bone_selection(args) and not args.list_bones:
        return 2

    # 出力先・上書きガード(§2.2)。list-bones は出力しないので不要。
    output = args.output if args.output is not None else _default_output(args.input)
    if not args.list_bones and not args.overwrite and _same_path(output, args.input):
        return 2

    # 入力読み込み(VMDでない等 → 入力不正 §9 コード1)。
    try:
        doc, _warnings = io.read(args.input)
    except Exception:
        return 1

    includes, excludes = _build_selectors(args)

    # --list-bones: ボーン名・キー数・選択状態を表示して終了(§2.7)。
    if args.list_bones:
        return _list_bones(doc, includes, excludes)

    cut_kw = dict(
        keep_frames=args.keep_frames,
        no_cut_detect=args.no_cut_detect,
        min_seg=args.min_segment_frames,
        max_seg=args.max_segment_frames,
        strict=args.strict,
        curve_mode=args.curve_mode,
    )

    # ボーン選択を先に解決する。明示 --bone 名が不在なら SelectionError → コード2
    # (ボーンセクションが空の場合を含む。§2.2)。これは §3.1 の空セクション コード1 より
    # 優先する(明示名の引数エラーが勝つ)。
    bone_names = _bone_names_in_order(doc.bone)
    if args.target in ("bone", "all") and _has_bone_selection(args):
        try:
            sel = selection.resolve_selection(bone_names, includes, excludes)
        except selection.SelectionError as e:
            # エラーで終了する前に、蓄積済みの不一致警告を出力する(§2.2)。
            for w in e.warnings:
                print("警告: " + w, file=sys.stderr)
            return 2
        for w in sel.warnings:
            print("警告: " + w, file=sys.stderr)
        selected = set(sel.selected)
    else:
        selected = set(bone_names)

    # 対象セクションにキーが無い(§3.1)。target all は片側のみでも続行。
    if args.target == "camera" and not doc.camera:
        return 1
    if args.target == "bone" and not doc.bone:
        return 1
    if args.target == "all" and not doc.camera and not doc.bone:
        return 1

    do_camera = args.target in ("camera", "all") and bool(doc.camera)
    do_bone = args.target in ("bone", "all") and bool(doc.bone)

    # 対象トラック全体の最小/最大フレームでグローバル範囲を1回だけ展開する(§2.2)。
    target_frames = []
    if do_camera:
        target_frames += [k.frame for k in doc.camera]
    if do_bone:
        target_frames += [k.frame for k in doc.bone if k.name in selected]
    try:
        global_ranges = _global_ranges(args.ranges, target_frames)
    except ranges.RangeError:
        return 2

    want_report = args.dry_run or args.report_json or args.preview_csv
    new_camera = doc.camera
    new_bone = doc.bone
    camera_errors = None
    bone_errors = None
    try:
        if do_camera:
            cam = _sorted_camera(doc.camera)
            cam_ranges = ranges.intersect(global_ranges, cam[0].frame, cam[-1].frame)
            new_camera = reduce_camera_track(
                cam, cam_ranges, tols, cut_thresholds=args.cut_threshold_camera, **cut_kw
            )
            if want_report:
                camera_errors = measure_camera_errors(cam, new_camera, cam_ranges)
        if do_bone:
            new_bone = _reduce_bones(
                doc.bone, selected, global_ranges, tols, args.cut_threshold_bone, cut_kw
            )
            if want_report:
                bone_errors = _measure_bone_errors(doc.bone, new_bone, selected, global_ranges)
    except StrictError:
        return 4

    # レポート(dry-run 統計・JSON・CSV)。dry-run でも report/preview は書き出す(§2.7)。
    if want_report:
        rep = report.build_report(
            target=args.target,
            camera=(len(doc.camera), len(new_camera)) if do_camera else None,
            bones=_bone_io_counts(doc.bone, new_bone) if do_bone else None,
            selected_bones=selected,
            ranges=global_ranges,
            keep_frames=args.keep_frames,
            camera_errors=camera_errors,
            bone_errors=bone_errors,
        )
        if args.dry_run:
            print(report.format_dry_run(rep))
        try:
            if args.report_json:
                report.write_json(rep, args.report_json)
            if args.preview_csv:
                report.write_csv(rep, args.preview_csv)
        except OSError:
            return 3  # レポート書き込み失敗(§9)

    if args.dry_run:
        return 0

    out_doc = dataclasses.replace(doc, camera=new_camera, bone=new_bone)
    try:
        io.write_file(out_doc, output)
    except Exception:
        return 3
    return 0


def _bone_io_counts(in_keys, out_keys):
    """ボーン名ごとの (入力キー数, 出力キー数) を返す(レポート用)。"""
    in_counts = Counter(k.name for k in in_keys)
    out_counts = Counter(k.name for k in out_keys)
    return {name: (in_counts[name], out_counts.get(name, 0)) for name in in_counts}


def _sorted_camera(camera):
    return sorted(camera, key=lambda k: k.frame)


def _global_ranges(parsed_ranges, target_frames):
    """対象トラック全体の最小/最大で --range を1回展開する(§2.2)。

    --range 未指定なら全体[min,max]。target_frames が空なら空リスト。
    """
    if not target_frames:
        return []
    gmin, gmax = min(target_frames), max(target_frames)
    if not parsed_ranges:
        return [(gmin, gmax)]
    return ranges.expand_and_normalize(parsed_ranges, gmin, gmax)


def _reduce_bones(bone_keys, selected, global_ranges, tols, cut_thresholds, cut_kw):
    """選択ボーンを削減し非選択ボーンは保持して、全ボーンキー列を返す(§3.2)。

    各トラックの実処理範囲はグローバル範囲とトラック区間の積集合(§2.2)。
    """
    groups = _bone_keys_by_name(bone_keys)
    out = []
    for name, keys in groups.items():
        ks = sorted(keys, key=lambda k: k.frame)
        if name in selected:
            track_ranges = ranges.intersect(global_ranges, ks[0].frame, ks[-1].frame)
            out.extend(
                reduce_bone_track(
                    ks, track_ranges, tols, cut_thresholds=cut_thresholds, **cut_kw
                )
            )
        else:
            out.extend(ks)  # 非選択トラックは保持(§3.2)
    # ボーン名(生バイト)・フレーム順に安定ソート(§3.2)。
    out.sort(key=lambda k: (k.name_raw, k.frame))
    return out


def _measure_bone_errors(in_bone, out_bone, selected, global_ranges):
    """選択ボーンごとに出力 vs 元サンプルの軸別最大誤差を測る(§7.2)。{name: 誤差dict}。"""
    in_groups = _bone_keys_by_name(in_bone)
    out_groups = _bone_keys_by_name(out_bone)
    errors = {}
    for name in selected:
        src = sorted(in_groups.get(name, []), key=lambda k: k.frame)
        out = sorted(out_groups.get(name, []), key=lambda k: k.frame)
        if not src or not out:
            continue
        track_ranges = ranges.intersect(global_ranges, src[0].frame, src[-1].frame)
        errors[name] = measure_bone_errors(src, out, track_ranges)
    return errors


def _list_bones(doc, includes, excludes):
    """ボーン名・キー数・選択状態を表示して 0 を返す(§2.7)。"""
    names = _bone_names_in_order(doc.bone)
    counts = {}
    for k in doc.bone:
        counts[k.name] = counts.get(k.name, 0) + 1

    selected = set()
    if names:
        try:
            result = selection.resolve_selection(names, includes, excludes)
            selected = set(result.selected)
            for w in result.warnings:
                print("警告: " + w, file=sys.stderr)
        except selection.SelectionError as e:
            # 選択不能でも一覧表示は行う(検査モード)。蓄積済みの不一致警告と理由を出す。
            for w in e.warnings:
                print("警告: " + w, file=sys.stderr)
            print("警告: " + str(e), file=sys.stderr)

    for name in names:
        state = "selected" if name in selected else "excluded"
        print(f"{name}\tkeys={counts[name]}\t{state}")
    return 0
