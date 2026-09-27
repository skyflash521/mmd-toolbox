import argparse
import dataclasses
import os
import sys
import time
from collections import Counter
from collections.abc import Callable

from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    argparse_error_field,
    emit_failure,
    install_sigbreak_handler,
)
from cli_progress import progress
from vmd import io

from . import __version__, presets, ranges, report, selection
from .cuts import parse_cut_threshold_bone, parse_cut_threshold_camera
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

_D_NONNEG = {"min": 0, "max": None, "exclusive_min": False}
_D_FOV = {"min": 0.5, "max": None, "exclusive_min": False}
_D_AT_LEAST_1 = {"min": 1, "max": None, "exclusive_min": False}
_D_GROUPS = {"choices": ["core", "arms", "legs", "fingers", "ik", "mocap"]}

_MIN_REDUCIBLE_KEYS = 2


def _compound(fmt, *name_type_min: tuple[str, str, float]):
    return {"format": fmt,
            "fields": [{"name": n, "type": t, "min": m, "max": None, "exclusive_min": False}
                       for n, t, m in name_type_min]}


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
    "ranges": ("compound", _compound("START:END", ("START", "int", 0), ("END", "int", 0))),
    "preset": ("enum", {"choices": list(presets.PRESET_NAMES)}),
    "bone_pos_tol": ("float", _D_NONNEG),
    "bone_rot_tol": ("float", _D_NONNEG),
    "camera_pos_tol": ("float", _D_NONNEG),
    "camera_rot_tol": ("float", _D_NONNEG),
    "camera_distance_tol": ("float", _D_NONNEG),
    "camera_fov_tol": ("float", _D_FOV),
    "max_segment_frames": ("int", _D_AT_LEAST_1),
    "min_segment_frames": ("int", _D_AT_LEAST_1),
    "curve_mode": ("enum", {"choices": ["bezier", "linear"]}),
    "strict": ("flag", None),
    "cut_threshold_camera": ("compound", _compound(
        "POS,ROT,DIST", ("POS", "float", 0), ("ROT", "float", 0), ("DIST", "float", 0))),
    "cut_threshold_bone": ("compound", _compound(
        "POS,ROT", ("POS", "float", 0), ("ROT", "float", 0))),
    "cut_detect": ("flag", None),
    "keep_frames": ("int", _D_NONNEG),
    "dry_run": ("flag", None),
    "verbose": ("flag", None),
    "quiet": ("flag", None),
}


def _nonneg_int(text):
    v = int(text)
    if v < 0:
        raise argparse.ArgumentTypeError(f"フレーム番号は非負: {text!r}")
    return v


def _build_parser():
    p = MachineArgumentParser(prog="sparsevmd", allow_abbrev=False)
    p.add_argument("input", nargs="?", help="入力VMDファイル")
    p.add_argument("-o", "--output", help="出力先(既定: <入力名>_sparse.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
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
    p.add_argument("--range", dest="ranges", action="append", type=ranges.parse_range,
                   help="削減範囲 START:END(反復可)。START/END の一方は省略可")
    p.add_argument("--preset", choices=presets.PRESET_NAMES, default="balanced",
                   help="品質プリセット: precise / balanced / aggressive(既定 balanced)")
    for arg in _TOL_OPTION_TO_FIELD:
        p.add_argument("--" + arg.replace("_", "-"), dest=arg, type=float, help=_TOL_HELP[arg])
    p.add_argument("--max-segment-frames", dest="max_segment_frames", type=int, default=None,
                   help="出力キー間隔の上限。未指定なら無制限。指定する場合は 1 以上")
    p.add_argument("--min-segment-frames", dest="min_segment_frames", type=int, default=1,
                   help="分割しない最小区間長(既定 1)")
    p.add_argument("--curve-mode", dest="curve_mode", choices=("bezier", "linear"), default="bezier",
                   help="出力補間曲線: bezier / linear(既定 bezier)")
    p.add_argument("--strict", action="store_true",
                   help="指定誤差を満たせない場合に高密度キー保持で続行せずエラー(終了コード 4)")
    p.add_argument(
        "--cut-threshold-camera",
        dest="cut_threshold_camera",
        type=parse_cut_threshold_camera,
        default=(5.0, 20.0, 5.0),
        help="カメラの不連続検出閾値 POS,ROT,DIST(既定 5.0,20.0,5.0)",
    )
    p.add_argument(
        "--cut-threshold-bone",
        dest="cut_threshold_bone",
        type=parse_cut_threshold_bone,
        default=(1.0, 30.0),
        help="ボーンの不連続検出閾値 POS,ROT(既定 1.0,30.0)",
    )
    p.add_argument("--cut-detect", dest="cut_detect", action="store_true", default=True,
                   help="閾値による自動境界検出を有効化(既定 on)。--no-cut-detect の対の明示形")
    p.add_argument("--no-cut-detect", dest="cut_detect", action="store_false",
                   help="閾値による自動境界検出を無効化。--cut-detect の対")
    p.add_argument("--keep-frame", dest="keep_frames", action="append", type=_nonneg_int, default=[],
                   help="指定フレームを必ずキーとして保持(反復可)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力VMDを書かずに統計を表示する(引数検証は実施する)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="詳細ログを出す(通常時は標準出力、--machine併用時は標準エラー)")
    p.add_argument("--quiet", dest="quiet", action="store_true",
                   help="進捗のライブ表示を抑制する(警告・統計・終了コードは抑制しない)")
    p.add_argument("--machine", action="store_true",
                   help="出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・"
                        "標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない")
    p.add_argument("--describe", action="store_true",
                   help="オプション定義とプリセット一覧の result を出して終了する"
                        "(VMD を読まない・入力不要の自己記述)")
    p.add_argument("--version", action="version", version=f"sparsevmd {__version__}",
                   help="バージョンを表示して終了する")
    return p


def _default_output(input_path):
    base, _ = os.path.splitext(input_path)
    return base + "_sparse.vmd"


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


def _describe_options(parser):
    options = []
    for action in parser._actions:
        dest = action.dest
        if dest not in _D_TYPE:
            continue
        type_, constraint = _D_TYPE[dest]
        if dest == "input":
            name = "input"
        else:
            pos = [s for s in action.option_strings if s.startswith("--") and not s.startswith("--no-")]
            if not pos:
                continue
            name = pos[0]
        repeat = isinstance(action, argparse._AppendAction)
        options.append({
            "name": name,
            "type": type_,
            "constraint": constraint,
            "default": None if repeat else action.default,
            "help": action.help,
            "repeat": repeat,
        })
    return options


def _describe_presets():
    out = []
    for name in presets.PRESET_NAMES:
        tols = presets.resolve_tolerances(name)
        values = {opt: getattr(tols, field) for opt, field in _TOL_OPTION_TO_FIELD.items()}
        out.append({"name": name, "values": values})
    return out


def _machine_progress(emitter, stage) -> Callable[..., None]:
    """戻り値は (done, total, note="") を受けるコールバック。"""
    start = time.monotonic()
    emitter.progress(stage=stage, done=0, total=None, note="", elapsed=0.0)

    def cb(done, total, note=""):
        emitter.progress(stage=stage, done=done, total=total, note=note,
                         elapsed=time.monotonic() - start)

    return cb


def _emit_selector_unmatched(emitter, message):
    if emitter is not None:
        emitter.warning(code="selector_unmatched", message=message, section=None)
    else:
        print("warning: selector_unmatched: " + message, file=sys.stderr)


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


def _fail(emitter, code, message, exit_code, *, field=None, path=None):
    return emit_failure(emitter, code=code, message=message, exit_code=exit_code,
                        field=field, path=path)


def main(argv=None) -> int:
    """戻り値は終了コード。"""
    emitter = None
    try:
        if argv is None:
            argv = sys.argv[1:]

        machine = "--machine" in argv
        describe = "--describe" in argv
        emitter = EventEmitter(sys.stdout.buffer) if (machine or describe) else None

        def fail(code, message, exit_code, *, field=None, path=None):
            return _fail(emitter, code, message, exit_code, field=field, path=path)

        install_sigbreak_handler()
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(errors="backslashreplace")
            except Exception:
                pass
        parser = _build_parser()
        try:
            args = parser.parse_args(argv)
        except ArgumentParseError as e:
            return fail("bad_argument", e.message, 2, field=argparse_error_field(e.message))
        except SystemExit as e:
            # argparse は --help と --version の表示後に SystemExit を送出する。
            code = e.code
            return code if isinstance(code, int) else (0 if code is None else 2)

        if args.describe:
            emitter.result(mode="describe", options=_describe_options(parser),
                           presets=_describe_presets())
            return 0
        if args.input is None:
            return fail("bad_argument", "入力VMDファイル(input)が必要", 2, field="input")

        return _run(args, emitter, fail)
    except KeyboardInterrupt:
        return _fail(emitter, "cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return _fail(emitter, "internal_error", f"{type(e).__name__}: {e}", 1)


def _run(args, emitter, fail):
    if not os.path.isfile(args.input):
        return fail("input_not_file", f"入力が存在しないか通常ファイルでない: {args.input}", 1, field="input")
    if args.bone_file is not None and not os.path.isfile(args.bone_file):
        return fail("bone_file_not_file",
                    f"--bone-file が存在しないか通常ファイルでない: {args.bone_file}", 1, field="--bone-file")

    if args.min_segment_frames < 1:
        return fail("bad_argument", f"--min-segment-frames は 1 以上が必要: {args.min_segment_frames}",
                    2, field="--min-segment-frames")
    if args.max_segment_frames is not None and args.max_segment_frames < 1:
        return fail("bad_argument", f"--max-segment-frames は 1 以上が必要: {args.max_segment_frames}",
                    2, field="--max-segment-frames")
    if args.max_segment_frames is not None and args.min_segment_frames > args.max_segment_frames:
        return fail("segment_bounds_conflict",
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
        return fail("bad_tolerance", str(e), 2)

    if args.target == "camera" and _has_bone_selection(args) and not args.list_bones:
        return fail("target_selection_conflict",
                    "--target camera とボーン選択オプションは同時指定できない", 2)

    output = args.output if args.output is not None else _default_output(args.input)
    if not args.list_bones and os.path.isdir(output):
        return fail("output_is_directory",
                    f"出力先がディレクトリです(ファイルパスを指定): {output}",
                    2, field="--output", path=output)
    if not args.list_bones and not args.overwrite and os.path.exists(output):
        return fail("output_exists",
                    f"出力先に既存ファイルがあります。上書きには --overwrite が必要: {output}", 2, field="--output")

    try:
        doc, read_warnings = io.read(args.input)
    except Exception as e:
        return fail("not_vmd", f"入力を VMD として読めない: {type(e).__name__}: {e}", 1, field="input")

    seen_warn = set()
    for w in read_warnings:
        key = (w.code, w.section, w.message)
        if key in seen_warn:
            continue
        seen_warn.add(key)
        if emitter is not None:
            emitter.warning(code=w.code, message=w.message,
                            section=[w.section] if w.section else None)
        else:
            where = f"({w.section})" if w.section else ""
            print(f"warning: {w.code}: {w.message}{where}", file=sys.stderr)

    norm_sections = []
    if args.target in ("camera", "all"):
        norm_sections.append("camera")
    if args.target in ("bone", "all"):
        norm_sections.append("bone")
    doc, _ = io.normalize(doc, sections=norm_sections)

    try:
        includes, excludes = _build_selectors(args)
    except (UnicodeDecodeError, OSError, ValueError) as e:
        return fail("bad_bone_file", f"--bone-file を読めない: {type(e).__name__}: {e}", 1,
                    field="--bone-file")

    if args.list_bones:
        return _list_bones(doc, includes, excludes, emitter=emitter)

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
            return fail("no_target_keys", "ボーン選択を指定したがボーンキーが無い", 1, field="input")
        try:
            sel = selection.resolve_selection(
                bone_names, includes, excludes, undecodable=undecodable
            )
        except selection.SelectionError as e:
            for w in e.warnings:
                _emit_selector_unmatched(emitter, w)
            return fail("bone_selection_invalid", str(e), 2)
        for w in sel.warnings:
            _emit_selector_unmatched(emitter, w)
        selected = set(sel.selected)
    else:
        selected = set(bone_names)

    if args.target == "camera" and not doc.camera:
        return fail("no_target_keys", "--target camera だがカメラキーが無い", 1, field="input")
    if args.target == "bone" and not doc.bone:
        return fail("no_target_keys", "--target bone だがボーンキーが無い", 1, field="input")
    if args.target == "all" and not doc.camera and not doc.bone:
        return fail("no_target_keys", "対象セクション(camera/bone)にキーが無い", 1, field="input")

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
        return fail("range_invalid", f"--range: {e}", 2, field="--range")

    for f in args.keep_frames:
        if not any(lo <= f <= hi for lo, hi in global_ranges):
            msg = f"keep-frame {f} は削減範囲外のため無視します"
            if emitter is not None:
                emitter.warning(code="keep_frame_ignored", message=msg, section=None)
            else:
                print(f"warning: keep_frame_ignored: {msg}", file=sys.stderr)

    want_report = args.dry_run
    want_diag = want_report or args.verbose
    new_camera = doc.camera
    new_bone = doc.bone
    camera_errors = None
    bone_errors = None
    camera_diag = None
    bone_diag = None
    reporter = progress.ProgressReporter(
        sys.stderr, enabled=(sys.stderr.isatty() and not args.quiet and emitter is None))
    did_reduce = False
    try:
        try:
            if do_camera:
                cam = _sorted_camera(doc.camera)
                cam_ranges = ranges.intersect(global_ranges, cam[0].frame, cam[-1].frame)
                cam_cb = _machine_progress(emitter, "camera") if emitter is not None else None
                if len(cam) >= 2:
                    if cam_ranges:
                        did_reduce = True
                    camera_diag = {} if want_diag else None
                    if emitter is None:
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
                bone_cb = _machine_progress(emitter, "bone") if emitter is not None else None
                new_bone = _reduce_bones(
                    doc.bone, selected, global_ranges, tols, args.cut_threshold_bone, cut_kw,
                    diagnostics_out=bone_diag,
                    reporter=(reporter if emitter is None else None), on_progress=bone_cb,
                )
                if _has_reducible_bone_in_ranges(doc.bone, selected, global_ranges):
                    did_reduce = True
                if want_report:
                    bone_errors = _measure_bone_errors(doc.bone, new_bone, selected, global_ranges)
        except StrictError:
            reporter.close()
            return fail("strict_tolerance_unmet", "--strict 指定で許容誤差を満たせない", 4)

        if args.verbose:
            reporter.close()
            _log_diagnostics(camera_diag, bone_diag, file=(sys.stderr if emitter is not None else None))

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
            if args.dry_run and emitter is None:
                reporter.close()
                print(report.format_dry_run(rep))

        if args.dry_run:
            if emitter is not None:
                emitter.result(mode="inspect", **_build_inspect(
                    args, doc, do_camera, do_bone, selected, global_ranges, new_camera, new_bone,
                    camera_errors, bone_errors, camera_diag, bone_diag, did_reduce))
            return 0

        out_doc = dataclasses.replace(doc, camera=new_camera, bone=new_bone)
        try:
            io.write_file(out_doc, output)
        except Exception as e:
            reporter.close()
            return fail("write_failed", f"出力の書き込みに失敗: {type(e).__name__}: {e}", 3,
                        field="--output", path=output)

        if emitter is None:
            reporter.close()
            reporter.summary(f"完了 {output}")
        else:
            emitter.result(
                mode="reduce", output=output, target=args.target,
                camera={"input_keys": len(doc.camera), "output_keys": len(new_camera)} if do_camera else None,
                bone={"input_keys": len(doc.bone), "output_keys": len(new_bone)} if do_bone else None,
                reduced=did_reduce)
        return 0
    finally:
        reporter.close()


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
    out = []
    for name, keys in groups.items():
        ks = sorted(keys, key=lambda k: k.frame)
        if _is_reducible(name, selected, len(ks)):
            if reporter is not None:
                reporter.update(done, total, note=name)
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


def _list_bones(doc, includes, excludes, emitter=None):
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
            _emit_selector_unmatched(emitter, w)
    except selection.SelectionError as e:
        for w in e.warnings:
            _emit_selector_unmatched(emitter, w)
        if emitter is not None:
            emitter.warning(code="selection_unresolved", message=str(e), section=None)
        else:
            print("warning: selection_unresolved: " + str(e), file=sys.stderr)

    if emitter is not None:
        emitter.result(mode="list_bones", bones=[
            {"name": name, "keys": counts[name], "selected": name in selected} for name in names
        ])
    else:
        for name in names:
            state = "selected" if name in selected else "excluded"
            print(f"{name}\tkeys={counts[name]}\t{state}")
    return 0
