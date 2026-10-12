import dataclasses
import os
import sys
import time
from collections import Counter
from collections.abc import Callable

from cli_events import (
    MachineArgumentParser,
    add_common_arguments,
    default_output_path,
    guard_output,
    run_tool,
)
from cli_options import CompoundValidator, RangeValidator, describe_options
from cli_progress import progress
from vmd import io

from . import __version__, presets, ranges, report, selection
from .reduce import (
    StrictError,
    measure_bone_errors,
    measure_camera_errors,
    reduce_bone_track,
    reduce_camera_track,
)

_TOL_OPTION_TO_FIELD = {
    "bone_pos_tol": "bone_pos",
    "bone_rot_tol": "bone_rot",
    "camera_pos_tol": "camera_pos",
    "camera_rot_tol": "camera_rot",
    "camera_distance_tol": "camera_distance",
    "camera_fov_tol": "camera_fov",
}

_TOL_HELP = {
    "bone_pos_tol": "ボーン位置の最大許容誤差(MMD距離単位)。明示値はプリセットに優先",
    "bone_rot_tol": "ボーン回転の最大角度誤差(度)。明示値はプリセットに優先",
    "camera_pos_tol": "カメラ中心位置の最大許容誤差(MMD距離単位)。明示値はプリセットに優先",
    "camera_rot_tol": "カメラ回転の最大角度誤差(度)。明示値はプリセットに優先",
    "camera_distance_tol": "カメラ距離の最大許容誤差(MMD距離単位)。明示値はプリセットに優先",
    "camera_fov_tol": "視野角の最大許容誤差(度)。整数度保存のため 0.5 以上。明示値はプリセットに優先",
}

_TOL = RangeValidator(value_type="float", minimum=0)
_FOV_TOL = RangeValidator(value_type="float", minimum=0.5)
_SEGMENT_FRAMES = RangeValidator(value_type="int", minimum=1)
_KEEP_FRAME = RangeValidator(value_type="int", minimum=0)
_RANGE = CompoundValidator(
    format="START:END",
    elements=[(name, RangeValidator(value_type="int", minimum=0, allow_empty=True))
              for name in ("START", "END")],
    relation=("START と END の少なくとも一方が必要で、START<=END が必要",
              lambda v: (v[0] is not None or v[1] is not None)
              and (v[0] is None or v[1] is None or v[0] <= v[1])),
)
_CUT_THRESHOLD_CAMERA = CompoundValidator(
    format="POS,ROT,DIST", separator=",",
    elements=[(name, RangeValidator(value_type="float", minimum=0)) for name in ("POS", "ROT", "DIST")])
_CUT_THRESHOLD_BONE = CompoundValidator(
    format="POS,ROT", separator=",",
    elements=[(name, RangeValidator(value_type="float", minimum=0)) for name in ("POS", "ROT")])

_D_GROUPS = {"choices": ["core", "arms", "legs", "fingers", "ik", "mocap"]}

_MIN_REDUCIBLE_KEYS = 2

_D_TYPE = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "target": ("enum", {"choices": ["camera", "bone", "all"]}),
    "bone": ("str", None),
    "bone_glob": ("str", None),
    "bone_group": ("enum", _D_GROUPS),
    "bone_file": ("str", None),
    "exclude_bone": ("str", None),
    "exclude_bone_glob": ("str", None),
    "exclude_bone_group": ("enum", _D_GROUPS),
    "list_bones": ("flag", None),
    "ranges": (None, None),
    "preset": ("enum", {"choices": list(presets.PRESET_NAMES)}),
    "bone_pos_tol": ("float", _TOL.constraint),
    "bone_rot_tol": ("float", _TOL.constraint),
    "camera_pos_tol": ("float", _TOL.constraint),
    "camera_rot_tol": ("float", _TOL.constraint),
    "camera_distance_tol": ("float", _TOL.constraint),
    "camera_fov_tol": ("float", _FOV_TOL.constraint),
    "max_segment_frames": (None, None),
    "min_segment_frames": (None, None),
    "curve_mode": ("enum", {"choices": ["bezier", "linear"]}),
    "strict": ("flag", None),
    "cut_threshold_camera": (None, None),
    "cut_threshold_bone": (None, None),
    "cut_detect": ("flag", None),
    "keep_frames": (None, None),
    "dry_run": ("flag", None),
    "verbose": ("flag", None),
    "quiet": ("flag", None),
}


def _build_parser():
    p = MachineArgumentParser(prog="sparsevmd", allow_abbrev=False)
    p.add_argument("input", nargs="?", help="入力VMDファイル")
    add_common_arguments(
        p, version=__version__, output_help="出力先(既定: <入力名>_sparse.vmd)",
        help_overrides={
            "--describe": "オプション定義とプリセット一覧の result を出して終了する"
                          "(VMD を読まない・入力不要の自己記述)",
            "--dry-run": "出力VMDを書かずに統計を表示する(引数検証は実施する)",
            "--verbose": "詳細ログを出す(通常時は標準出力、--machine併用時は標準エラー)",
            "--quiet": "進捗のライブ表示を抑制する(警告・統計・終了コードは抑制しない)",
        })
    p.add_argument("--target", choices=("camera", "bone", "all"), default="all",
                   help="削減対象セクション: camera / bone / all(既定 all=camera+bone)")
    p.add_argument("--bone", dest="bone", action="append", default=[],
                   help="指定ボーンのみ処理(反復可)。未指定なら全ボーン")
    p.add_argument("--bone-glob", dest="bone_glob", action="append", default=[],
                   help="glob に一致するボーンを処理対象に追加(反復可)")
    p.add_argument("--bone-group", dest="bone_group", action="append", default=[],
                   help="よく使うボーン集合を処理対象に追加(反復可)")
    p.add_argument("--bone-file", dest="bone_file",
                   help="ボーン選択ルールを UTF-8 テキストから読み込む")
    p.add_argument("--exclude-bone", dest="exclude_bone", action="append", default=[],
                   help="指定ボーンを処理対象から除外(反復可)")
    p.add_argument("--exclude-bone-glob", dest="exclude_bone_glob", action="append", default=[],
                   help="glob に一致するボーンを処理対象から除外(反復可)")
    p.add_argument("--exclude-bone-group", dest="exclude_bone_group", action="append", default=[],
                   help="よく使うボーン集合を処理対象から除外(反復可)")
    p.add_argument("--list-bones", dest="list_bones", action="store_true",
                   help="ボーン名・キー数・選択状態を表示して終了する")
    p.add_argument("--range", dest="ranges", action="append", type=_RANGE,
                   help="削減範囲 START:END(反復可)。START/END の一方は省略可(両方省略は不可)。"
                        "両方指定するときは START<=END")
    p.add_argument("--preset", choices=presets.PRESET_NAMES, default="balanced",
                   help="品質プリセット: precise / balanced / aggressive(既定 balanced)")
    for arg in _TOL_OPTION_TO_FIELD:
        p.add_argument("--" + arg.replace("_", "-"), dest=arg, type=float, help=_TOL_HELP[arg])
    p.add_argument("--max-segment-frames", dest="max_segment_frames", type=_SEGMENT_FRAMES, default=None,
                   help="出力キー間隔の上限。未指定なら無制限。指定する場合は 1 以上")
    p.add_argument("--min-segment-frames", dest="min_segment_frames", type=_SEGMENT_FRAMES, default=1,
                   help="分割しない最小区間長(既定 1)")
    p.add_argument("--curve-mode", dest="curve_mode", choices=("bezier", "linear"), default="bezier",
                   help="出力補間曲線: bezier / linear(既定 bezier)")
    p.add_argument("--strict", action="store_true",
                   help="指定誤差を満たせない場合に高密度キー保持で続行せずエラー(終了コード 4)")
    p.add_argument(
        "--cut-threshold-camera",
        dest="cut_threshold_camera",
        type=_CUT_THRESHOLD_CAMERA,
        default=(5.0, 20.0, 5.0),
        help="カメラの不連続検出閾値 POS,ROT,DIST(既定 5.0,20.0,5.0)",
    )
    p.add_argument(
        "--cut-threshold-bone",
        dest="cut_threshold_bone",
        type=_CUT_THRESHOLD_BONE,
        default=(1.0, 30.0),
        help="ボーンの不連続検出閾値 POS,ROT(既定 1.0,30.0)",
    )
    p.add_argument("--cut-detect", dest="cut_detect", action="store_true", default=True,
                   help="閾値による自動境界検出を有効化(既定 on)。--no-cut-detect の対の明示形")
    p.add_argument("--no-cut-detect", dest="cut_detect", action="store_false",
                   help="閾値による自動境界検出を無効化。--cut-detect の対")
    p.add_argument("--keep-frame", dest="keep_frames", action="append", type=_KEEP_FRAME, default=[],
                   help="指定フレームを必ずキーとして保持(反復可)")
    return p


def _build_selectors(args):
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
    seen = []
    seen_raw = set()
    for k in bone_keys:
        if k.name_raw not in seen_raw:
            seen_raw.add(k.name_raw)
            seen.append(k.name)
    return seen


def _undecodable_bone_names(bone_keys):
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


def _is_reducible(name, selected, key_count):
    return name in selected and key_count >= _MIN_REDUCIBLE_KEYS


def _has_reducible_bone_in_ranges(bone_keys, selected, global_ranges):
    groups = _bone_keys_by_name(bone_keys)
    for name, keys in groups.items():
        if not _is_reducible(name, selected, len(keys)):
            continue
        ks = sorted(keys, key=lambda k: k.frame)
        if ranges.intersect(global_ranges, ks[0].frame, ks[-1].frame):
            return True
    return False


def _log_diagnostics(camera_diag, bone_diag, file=None):
    def emit(label, d):
        if not d:
            return
        if d.get("cuts"):
            print(f"詳細[{label}]: 不連続検出位置 {d['cuts']}", file=file)
        if d.get("seam_rewrites"):
            print(f"詳細[{label}]: 継ぎ目書き換え {d['seam_rewrites']}", file=file)
        if d.get("splits"):
            print(f"詳細[{label}]: 分割 {len(d['splits'])} 件", file=file)
        for v in d.get("verify") or ():
            print(
                f"詳細[{label}]: 出力後検証 範囲[{v['range'][0]},{v['range'][1]}] "
                f"反復{v['iterations']} 追加{v['added_total']} "
                f"bad={v['bad_counts']} added={v['added_counts']}",
                file=file,
            )

    emit("camera", camera_diag)
    for name, d in (bone_diag or {}).items():
        emit(f"bone {name}", d)


def _describe(parser):
    return {"options": describe_options(parser, _D_TYPE), "presets": _describe_presets()}


def _describe_presets():
    out = []
    for name in presets.PRESET_NAMES:
        tols = presets.resolve_tolerances(name)
        values = {opt: getattr(tols, field) for opt, field in _TOL_OPTION_TO_FIELD.items()}
        out.append({"name": name, "values": values})
    return out


_BEAT_INTERVAL = 0.1


def _machine_progress(emitter, stage) -> Callable[..., None]:
    """戻り値は (done, total, note="") を受けるコールバック。"""
    start = time.monotonic()
    emitter.progress(stage=stage, done=0, total=None, note="", elapsed=0.0)

    def cb(done, total, note=""):
        emitter.progress(stage=stage, done=done, total=total, note=note,
                         elapsed=time.monotonic() - start)

    return cb


def _build_inspect(args, doc, do_camera, do_bone, selected, global_ranges,
                   new_camera, new_bone, camera_errors, bone_errors, camera_diag, bone_diag, reduced):
    sections = [name for name, present in (
        ("camera", doc.camera), ("bone", doc.bone), ("morph", doc.morph),
        ("light", doc.light), ("self_shadow", doc.self_shadow), ("ik_property", doc.ik_property),
    ) if present]
    frames = []
    if do_camera:
        frames += [k.frame for k in doc.camera]
    if do_bone:
        frames += [k.frame for k in doc.bone]
    frame_range = [min(frames), max(frames)] if frames else None
    duration = (max(frames) / 30.0) if frames else None

    camera = None
    if do_camera:
        camera = {
            "input_keys": len(doc.camera),
            "output_keys": len(new_camera),
            "errors": camera_errors,
            "cuts": (camera_diag or {}).get("cuts") or [],
        }
    bones = None
    if do_bone:
        io_counts = _bone_io_counts(doc.bone, new_bone)
        bones = []
        for name in _bone_names_in_order(doc.bone):
            inp, out = io_counts.get(name, (0, 0))
            reducible = _is_reducible(name, selected, inp)
            bones.append({
                "name": name,
                "selected": name in selected,
                "input_keys": inp,
                "output_keys": out,
                "errors": (bone_errors or {}).get(name) if reducible else None,
                "cuts": (((bone_diag or {}).get(name)) or {}).get("cuts", []) if reducible else None,
            })
    return {
        "output": None,
        "target": args.target,
        "sections": sections,
        "keys": {"camera": len(doc.camera), "bone": len(doc.bone)},
        "frame_range": frame_range,
        "duration_sec": duration,
        "ranges": [[s, e] for s, e in global_ranges],
        "keep_frames": sorted(set(args.keep_frames)),
        "reduced": reduced,
        "camera": camera,
        "bones": bones,
    }


def main(argv=None) -> int:
    return run_tool(argv, build_parser=_build_parser, describe=_describe, run=_run)


def _run(args, ctx):
    if args.input is None:
        return ctx.fail("bad_argument", "入力VMDファイル(input)が必要", 2, field="input")
    if not os.path.isfile(args.input):
        return ctx.fail("input_not_file", f"入力が存在しないか通常ファイルでない: {args.input}", 1, field="input")
    if args.bone_file is not None and not os.path.isfile(args.bone_file):
        return ctx.fail("bone_file_not_file",
                    f"--bone-file が存在しないか通常ファイルでない: {args.bone_file}", 1, field="--bone-file")

    if args.max_segment_frames is not None and args.min_segment_frames > args.max_segment_frames:
        return ctx.fail("segment_bounds_conflict",
                    f"--min-segment-frames({args.min_segment_frames})が "
                    f"--max-segment-frames({args.max_segment_frames})を超えている", 2)

    overrides = {
        field: getattr(args, arg)
        for arg, field in _TOL_OPTION_TO_FIELD.items()
        if getattr(args, arg) is not None
    }
    try:
        tols = presets.resolve_tolerances(args.preset, overrides)
    except ValueError as e:
        return ctx.fail("bad_tolerance", str(e), 2)

    if args.target == "camera" and _has_bone_selection(args) and not args.list_bones:
        return ctx.fail("target_selection_conflict",
                    "--target camera とボーン選択オプションは同時指定できない", 2)

    output = args.output if args.output is not None else default_output_path(args.input, "_sparse.vmd")
    if not args.list_bones:
        rc = guard_output(output, overwrite=args.overwrite, fail=ctx.fail)
        if rc is not None:
            return rc

    try:
        doc, read_warnings = io.read(args.input)
    except Exception as e:
        return ctx.fail("not_vmd", f"入力を VMD として読めない: {type(e).__name__}: {e}", 1, field="input")

    seen_warn = set()
    for w in read_warnings:
        key = (w.code, w.section, w.message)
        if key in seen_warn:
            continue
        seen_warn.add(key)
        ctx.warn(w.code, w.message, section=w.section)

    norm_sections = []
    if args.target in ("camera", "all"):
        norm_sections.append("camera")
    if args.target in ("bone", "all"):
        norm_sections.append("bone")
    doc, _ = io.normalize(doc, sections=norm_sections)

    try:
        includes, excludes = _build_selectors(args)
    except (UnicodeDecodeError, OSError, ValueError) as e:
        return ctx.fail("bad_bone_file", f"--bone-file を読めない: {type(e).__name__}: {e}", 1,
                    field="--bone-file")

    if args.list_bones:
        return _list_bones(doc, includes, excludes, ctx)

    cut_kw = dict(
        keep_frames=args.keep_frames,
        no_cut_detect=not args.cut_detect,
        min_seg=args.min_segment_frames,
        max_seg=args.max_segment_frames,
        strict=args.strict,
        curve_mode=args.curve_mode,
    )

    bone_names = _bone_names_in_order(doc.bone)
    undecodable = _undecodable_bone_names(doc.bone)
    if args.target in ("bone", "all") and _has_bone_selection(args):
        if not bone_names:
            return ctx.fail("no_target_keys", "ボーン選択を指定したがボーンキーが無い", 1, field="input")
        try:
            sel = selection.resolve_selection(
                bone_names, includes, excludes, undecodable=undecodable
            )
        except selection.SelectionError as e:
            for w in e.warnings:
                ctx.warn("selector_unmatched", w)
            return ctx.fail("bone_selection_invalid", str(e), 2)
        for w in sel.warnings:
            ctx.warn("selector_unmatched", w)
        selected = set(sel.selected)
    else:
        selected = set(bone_names)

    if args.target == "camera" and not doc.camera:
        return ctx.fail("no_target_keys", "--target camera だがカメラキーが無い", 1, field="input")
    if args.target == "bone" and not doc.bone:
        return ctx.fail("no_target_keys", "--target bone だがボーンキーが無い", 1, field="input")
    if args.target == "all" and not doc.camera and not doc.bone:
        return ctx.fail("no_target_keys", "対象セクション(camera/bone)にキーが無い", 1, field="input")

    do_camera = args.target in ("camera", "all") and bool(doc.camera)
    do_bone = args.target in ("bone", "all") and bool(doc.bone)

    target_frames = []
    if do_camera:
        target_frames += [k.frame for k in doc.camera]
    if do_bone:
        target_frames += [k.frame for k in doc.bone if k.name in selected]
    try:
        global_ranges = _global_ranges(args.ranges, target_frames)
    except ranges.RangeError as e:
        return ctx.fail("range_invalid", f"--range: {e}", 2, field="--range")

    for f in args.keep_frames:
        if not any(lo <= f <= hi for lo, hi in global_ranges):
            msg = f"keep-frame {f} は削減範囲外のため無視します"
            ctx.warn("keep_frame_ignored", msg)

    want_report = args.dry_run
    want_diag = want_report or args.verbose
    new_camera = doc.camera
    new_bone = doc.bone
    camera_errors = None
    bone_errors = None
    camera_diag = None
    bone_diag = None
    reporter = ctx.track_progress(progress.ProgressReporter(
        sys.stderr, enabled=(sys.stderr.isatty() and not args.quiet and ctx.emitter is None)))
    did_reduce = False
    try:
        if do_camera:
            cam = _sorted_camera(doc.camera)
            cam_ranges = ranges.intersect(global_ranges, cam[0].frame, cam[-1].frame)
            cam_cb = _machine_progress(ctx.emitter, "camera") if ctx.emitter is not None else None
            if len(cam) >= 2:
                if cam_ranges:
                    did_reduce = True
                camera_diag = {} if want_diag else None
                if ctx.emitter is None:
                    reporter.stage("キーフレーム圧縮")
                    cam_cb = lambda done, total, note="": reporter.update(  # noqa: E731
                        done, total, note=f"カメラ {note}" if note else "カメラ")
                new_camera = reduce_camera_track(
                    cam, cam_ranges, tols, cut_thresholds=args.cut_threshold_camera,
                    diagnostics=camera_diag, progress=cam_cb, **cut_kw
                )
            else:
                new_camera = doc.camera
            if want_report:
                camera_errors = measure_camera_errors(cam, new_camera, cam_ranges)
        if do_bone:
            bone_diag = {} if want_diag else None
            bone_cb = _machine_progress(ctx.emitter, "bone") if ctx.emitter is not None else None
            new_bone = _reduce_bones(
                doc.bone, selected, global_ranges, tols, args.cut_threshold_bone, cut_kw,
                diagnostics_out=bone_diag,
                reporter=(reporter if ctx.emitter is None else None), on_progress=bone_cb,
            )
            if _has_reducible_bone_in_ranges(doc.bone, selected, global_ranges):
                did_reduce = True
            if want_report:
                bone_errors = _measure_bone_errors(doc.bone, new_bone, selected, global_ranges)
    except StrictError:
        return ctx.fail("strict_tolerance_unmet", "--strict 指定で許容誤差を満たせない", 4)

    if args.verbose:
        reporter.close()
        _log_diagnostics(camera_diag, bone_diag, file=(sys.stderr if ctx.emitter is not None else None))

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
        if args.dry_run and ctx.emitter is None:
            reporter.close()
            print(report.format_dry_run(rep))

    if args.dry_run:
        if ctx.emitter is not None:
            ctx.emitter.result(mode="inspect", **_build_inspect(
                args, doc, do_camera, do_bone, selected, global_ranges, new_camera, new_bone,
                camera_errors, bone_errors, camera_diag, bone_diag, did_reduce))
        return 0

    out_doc = dataclasses.replace(doc, camera=new_camera, bone=new_bone)
    try:
        io.write_file(out_doc, output)
    except Exception as e:
        return ctx.fail("write_failed", f"出力の書き込みに失敗: {type(e).__name__}: {e}", 3,
                    field="--output", path=output)

    if ctx.emitter is None:
        reporter.close()
        reporter.summary(f"完了 {output}")
    else:
        ctx.emitter.result(
            mode="reduce", output=output, target=args.target,
            camera={"input_keys": len(doc.camera), "output_keys": len(new_camera)} if do_camera else None,
            bone={"input_keys": len(doc.bone), "output_keys": len(new_bone)} if do_bone else None,
            reduced=did_reduce)
    return 0


def _bone_io_counts(in_keys, out_keys):
    in_counts = Counter(k.name for k in in_keys)
    out_counts = Counter(k.name for k in out_keys)
    return {name: (in_counts[name], out_counts.get(name, 0)) for name in in_counts}


def _sorted_camera(camera):
    return sorted(camera, key=lambda k: k.frame)


def _global_ranges(parsed_ranges, target_frames):
    if not target_frames:
        return []
    gmin, gmax = min(target_frames), max(target_frames)
    if not parsed_ranges:
        return [(gmin, gmax)]
    return ranges.expand_and_normalize(parsed_ranges, gmin, gmax)


def _reduce_bones(bone_keys, selected, global_ranges, tols, cut_thresholds, cut_kw,
                  diagnostics_out=None, reporter=None, on_progress=None):
    groups = _bone_keys_by_name(bone_keys)
    total = sum(1 for name, keys in groups.items() if _is_reducible(name, selected, len(keys)))
    if reporter is not None and total:
        reporter.stage("キーフレーム圧縮")
    done = 0
    last_beat = 0.0
    out = []
    for name, keys in groups.items():
        ks = sorted(keys, key=lambda k: k.frame)
        if _is_reducible(name, selected, len(ks)):
            if reporter is not None:
                reporter.update(done, total, note=name)
            track_ranges = ranges.intersect(global_ranges, ks[0].frame, ks[-1].frame)
            diag = {} if diagnostics_out is not None else None
            track_cb = None
            if reporter is not None or on_progress is not None:
                def track_cb(frames_done, frames_total, phase="", name=name, done=done):
                    nonlocal last_beat
                    note = f"{name} {frames_done}/{frames_total}" + (f" {phase}" if phase else "")
                    if reporter is not None:
                        reporter.update(done, total, note=note)
                    now = time.monotonic()
                    if on_progress is not None and now - last_beat >= _BEAT_INTERVAL:
                        last_beat = now
                        on_progress(done, total, name)
            out.extend(
                reduce_bone_track(
                    ks, track_ranges, tols, cut_thresholds=cut_thresholds,
                    diagnostics=diag, c1_smoothing=True, progress=track_cb, **cut_kw
                )
            )
            if diagnostics_out is not None:
                diagnostics_out[name] = diag
            done += 1
            if on_progress is not None:
                on_progress(done, total, name)
        else:
            out.extend(ks)
    out.sort(key=lambda k: (k.name_raw, k.frame))
    return out


def _measure_bone_errors(in_bone, out_bone, selected, global_ranges):
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


def _list_bones(doc, includes, excludes, ctx):
    names = _bone_names_in_order(doc.bone)
    counts = {}
    for k in doc.bone:
        counts[k.name] = counts.get(k.name, 0) + 1

    selected = set()
    try:
        result = selection.resolve_selection(
            names, includes, excludes, undecodable=_undecodable_bone_names(doc.bone)
        )
        selected = set(result.selected)
        for w in result.warnings:
            ctx.warn("selector_unmatched", w)
    except selection.SelectionError as e:
        for w in e.warnings:
            ctx.warn("selector_unmatched", w)
        ctx.warn("selection_unresolved", str(e))

    if ctx.emitter is not None:
        ctx.emitter.result(mode="list_bones", bones=[
            {"name": name, "keys": counts[name], "selected": name in selected} for name in names
        ])
    else:
        for name in names:
            state = "selected" if name in selected else "excluded"
            print(f"{name}\tkeys={counts[name]}\t{state}")
    return 0
