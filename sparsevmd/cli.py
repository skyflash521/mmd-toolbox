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

import math

from mmd_toolbox.vmd import interp, io

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


class _Progress:
    """削減処理の経過を stderr の1行に上書き表示する(§2.7)。

    フェーズ単位で start→update→finish と使う。stderr が端末でない場合
    (リダイレクト・パイプ・テスト捕捉)は無効化し、通常の出力・警告を汚さない。
    表示は付帯的なものなので、書き込み失敗(エンコード不能等)では削減処理を止めない。
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self._label = ""
        self._total = 0
        self._width = 0

    def start(self, label, total):
        self._label = label
        self._total = total
        self._width = 0
        self.update(0)

    def update(self, done, note=""):
        if not self.enabled:
            return
        pct = 100.0 * done / self._total if self._total else 100.0
        line = f"{self._label} {pct:3.0f}% ({done}/{self._total})"
        if note:
            line += f" {note}"
        pad = max(0, self._width - len(line))
        try:
            sys.stderr.write("\r" + line + " " * pad)
            sys.stderr.flush()
        except (OSError, ValueError, UnicodeError):
            self.enabled = False
            return
        self._width = len(line)

    def finish(self):
        if not self.enabled or not self._width:
            return
        try:
            sys.stderr.write("\n")
            sys.stderr.flush()
        except (OSError, ValueError):
            pass
        self._width = 0


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
        with open(args.bone_file, encoding="utf-8") as f:
            text = f.read()
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


def _undecodable_bone_names(bone_keys):
    """CP932 でデコードできないボーン名フィールドの表示名(置換文字入り)の集合(§2.2)。

    これらの名前は `--bone` / `--exclude-bone` の name 一致では使えない(§2.2)ため、
    selection に渡して name 種別の照合から除外させる。
    """
    undecodable = set()
    for k in bone_keys:
        head = k.name_raw.split(b"\x00", 1)[0]
        try:
            head.decode("cp932")
        except UnicodeDecodeError:
            undecodable.add(k.name)
    return undecodable


def _bone_keys_by_name(bone_keys):
    groups = {}
    for k in bone_keys:
        groups.setdefault(k.name, []).append(k)
    return groups


def _bone_reduced(bone_keys, selected, global_ranges):
    """選択ボーンのうち、キー2件以上かつ有効処理範囲が空でないものが1つでもあるか(§2.2/§3.1)。"""
    groups = _bone_keys_by_name(bone_keys)
    for name, keys in groups.items():
        if name not in selected or len(keys) < 2:
            continue
        ks = sorted(keys, key=lambda k: k.frame)
        if ranges.intersect(global_ranges, ks[0].frame, ks[-1].frame):
            return True
    return False


def _log_diagnostics(camera_diag, bone_diag):
    """verbose 時に不連続検出位置・継ぎ目書き換え・分割理由を stderr に出す(§2.7/§6.3)。"""
    def emit(label, d):
        if not d:
            return
        if d.get("cuts"):
            print(f"詳細[{label}]: 不連続検出位置 {d['cuts']}", file=sys.stderr)
        if d.get("seam_rewrites"):
            print(f"詳細[{label}]: 継ぎ目書き換え {d['seam_rewrites']}", file=sys.stderr)
        if d.get("splits"):
            print(f"詳細[{label}]: 分割 {len(d['splits'])} 件", file=sys.stderr)

    emit("camera", camera_diag)
    for name, d in (bone_diag or {}).items():
        emit(f"bone {name}", d)


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
        doc, read_warnings = io.read(args.input)
    except Exception:
        return 1

    # 読み込み時の警告(デコード不能な名前フィールド等)を surface する(§2.2)。
    # 同一(コード・セクション・メッセージ)はキー毎の重複を避けて1行にまとめる。
    seen_warn = set()
    for w in read_warnings:
        key = (w.code, w.section, w.message)
        if key in seen_warn:
            continue
        seen_warn.add(key)
        where = f"({w.section})" if w.section else ""
        print(f"警告: {w.message}{where}", file=sys.stderr)

    # 対象セクションを内部作業ビューで正規化する(フレーム順ソート・同一キー後勝ち。§3.1)。
    # 対象外セクションは無加工で保持される。
    norm_sections = []
    if args.target in ("camera", "all"):
        norm_sections.append("camera")
    if args.target in ("bone", "all"):
        norm_sections.append("bone")
    doc, _ = io.normalize(doc, sections=norm_sections)

    # --bone-file の読み込み・解析失敗(UTF-8 デコード不能等)は引数エラー(§2.2/§9 コード2)。
    try:
        includes, excludes = _build_selectors(args)
    except (UnicodeDecodeError, OSError, ValueError):
        return 2

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
    undecodable = _undecodable_bone_names(doc.bone)
    if args.target in ("bone", "all") and _has_bone_selection(args):
        try:
            sel = selection.resolve_selection(
                bone_names, includes, excludes, undecodable=undecodable
            )
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

    # 全削減範囲外の keep-frame は警告して無視する(§2.6)。
    for f in args.keep_frames:
        if not any(lo <= f <= hi for lo, hi in global_ranges):
            print(f"警告: keep-frame {f} は削減範囲外のため無視します", file=sys.stderr)

    want_report = args.dry_run or args.report_json or args.preview_csv
    want_diag = want_report or args.verbose  # verbose は診断を stderr ログに出す(§2.7/§6.3)
    new_camera = doc.camera
    new_bone = doc.bone
    camera_errors = None
    bone_errors = None
    camera_diag = None
    bone_diag = None
    preview_rows = []
    # 削減中の処理経過を stderr に表示する(対話端末時のみ。§2.7)。
    reporter = _Progress(enabled=sys.stderr.isatty())
    did_reduce = False
    try:
        if do_camera:
            cam = _sorted_camera(doc.camera)
            cam_ranges = ranges.intersect(global_ranges, cam[0].frame, cam[-1].frame)
            if len(cam) >= 2:
                if cam_ranges:  # 有効範囲が空なら実際には削減されない(§2.2/§3.1)
                    did_reduce = True
                camera_diag = {} if want_diag else None
                cam_total = sum(f1 - f0 for f0, f1 in cam_ranges)
                reporter.start("カメラ削減", cam_total)
                new_camera = reduce_camera_track(
                    cam, cam_ranges, tols, cut_thresholds=args.cut_threshold_camera,
                    diagnostics=camera_diag,
                    progress=lambda done, total, note="": reporter.update(done, note),
                    **cut_kw
                )
                reporter.finish()
            else:
                new_camera = doc.camera  # 1 キー以下は削減不能として逐語保持(§3.1/§3.2)
            if want_report:
                camera_errors = measure_camera_errors(cam, new_camera, cam_ranges)
            if args.preview_csv:
                preview_rows.extend(_camera_preview_rows(cam, new_camera, cam_ranges))
        if do_bone:
            bone_diag = {} if want_diag else None
            new_bone = _reduce_bones(
                doc.bone, selected, global_ranges, tols, args.cut_threshold_bone, cut_kw,
                diagnostics_out=bone_diag, reporter=reporter,
            )
            if _bone_reduced(doc.bone, selected, global_ranges):
                did_reduce = True
            if want_report:
                bone_errors = _measure_bone_errors(doc.bone, new_bone, selected, global_ranges)
            if args.preview_csv:
                preview_rows.extend(
                    _bone_preview_rows(doc.bone, new_bone, selected, global_ranges)
                )
    except StrictError:
        return 4

    # verbose: 不連続検出位置・分割理由・継ぎ目書き換えを stderr に出す(§2.7/§6.3)。
    if args.verbose:
        _log_diagnostics(camera_diag, bone_diag)

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
            camera_diag=camera_diag,
            bone_diag=bone_diag,
            reduced=did_reduce,
        )
        if args.dry_run:
            print(report.format_dry_run(rep))
        try:
            if args.report_json:
                report.write_json(rep, args.report_json)
            if args.preview_csv:
                report.write_preview_csv(preview_rows, args.preview_csv)
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


def _reduce_bones(bone_keys, selected, global_ranges, tols, cut_thresholds, cut_kw,
                  diagnostics_out=None, reporter=None):
    """選択ボーンを削減し非選択ボーンは保持して、全ボーンキー列を返す(§3.2)。

    各トラックの実処理範囲はグローバル範囲とトラック区間の積集合(§2.2)。diagnostics_out に
    dict を渡すと、選択ボーンごとに {name: 診断dict} を埋める(§2.7/§6.3)。reporter を渡すと
    削減対象ボーン1件ごとに処理経過を表示する(§2.7)。
    """
    groups = _bone_keys_by_name(bone_keys)
    total = sum(1 for name, keys in groups.items() if name in selected and len(keys) >= 2)
    if reporter is not None and total:
        reporter.start("ボーン削減", total)
    done = 0
    out = []
    for name, keys in groups.items():
        ks = sorted(keys, key=lambda k: k.frame)
        if name in selected and len(ks) >= 2:
            track_ranges = ranges.intersect(global_ranges, ks[0].frame, ks[-1].frame)
            diag = {} if diagnostics_out is not None else None
            out.extend(
                reduce_bone_track(
                    ks, track_ranges, tols, cut_thresholds=cut_thresholds,
                    diagnostics=diag, **cut_kw
                )
            )
            if diagnostics_out is not None:
                diagnostics_out[name] = diag
            done += 1
            if reporter is not None:
                reporter.update(done, name)
        else:
            # 非選択トラック、および選択でもキー1件以下(削減不能)は逐語保持(§3.1/§3.2)。
            out.extend(ks)
    if reporter is not None and total:
        reporter.finish()
    # ボーン名(生バイト)・フレーム順に安定ソート(§3.2)。
    out.sort(key=lambda k: (k.name_raw, k.frame))
    return out


def _preview_row(track, frame, channel, inp, outp):
    inp = float(inp)
    outp = float(outp)
    return {
        "track": track, "frame": frame, "channel": channel,
        "input": inp, "output": outp, "error": abs(inp - outp),
    }


def _camera_preview_rows(source, output, ranges):
    """カメラのフレーム毎サンプル行を返す(§2.7 --preview-csv)。回転はオイラー度。"""
    rows = []
    for f0, f1 in ranges:
        for f in range(f0, f1 + 1):
            s = interp.sample_camera(source, f)
            o = interp.sample_camera(output, f)
            for i, ax in enumerate(("pos_x", "pos_y", "pos_z")):
                rows.append(_preview_row("camera", f, ax, s["position"][i], o["position"][i]))
            rows.append(_preview_row("camera", f, "distance", s["distance"], o["distance"]))
            rows.append(_preview_row("camera", f, "fov", s["fov"], o["fov"]))
            for i, ax in enumerate(("rot_x", "rot_y", "rot_z")):
                rows.append(
                    _preview_row("camera", f, ax,
                                 math.degrees(s["rotation"][i]), math.degrees(o["rotation"][i]))
                )
    return rows


def _bone_preview_rows(in_bone, out_bone, selected, global_ranges):
    """選択ボーンごとのフレーム毎サンプル行を返す(§2.7)。回転は quaternion 成分。"""
    in_groups = _bone_keys_by_name(in_bone)
    out_groups = _bone_keys_by_name(out_bone)
    rows = []
    for name in selected:
        src = sorted(in_groups.get(name, []), key=lambda k: k.frame)
        out = sorted(out_groups.get(name, []), key=lambda k: k.frame)
        if not src or not out:
            continue
        track_ranges = ranges.intersect(global_ranges, src[0].frame, src[-1].frame)
        for f0, f1 in track_ranges:
            for f in range(f0, f1 + 1):
                for ax in ("pos_x", "pos_y", "pos_z"):
                    rows.append(
                        _preview_row(name, f, ax, interp.sample(src, ax, f), interp.sample(out, ax, f))
                    )
                sr = interp.sample(src, "rot", f)
                orr = interp.sample(out, "rot", f)
                for i, ax in enumerate(("rot_x", "rot_y", "rot_z", "rot_w")):
                    rows.append(_preview_row(name, f, ax, sr[i], orr[i]))
    return rows


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

    # ボーン0件でも選択子の解決を試み、未一致選択子の警告を出す(§2.7)。0件かつ選択子なしなら無警告。
    selected = set()
    try:
        result = selection.resolve_selection(
            names, includes, excludes, undecodable=_undecodable_bone_names(doc.bone)
        )
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
