import dataclasses
import math
import os
import sys
import time

from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    argparse_error_field,
    emit_failure,
    install_sigbreak_handler,
)
from cli_progress_router import ProgressRouter
from pmx.types import PmxFormatError
from vmd import interp, io
from vmd.reduce import BONE_LINEAR_INTERP
from vmd.types import BoneKey

from . import __version__, classify, denoise, footik, presets, reduce, report
from .model_profile import MocapModelProfileError
from .pose_denoise import apply_pose_denoise

_STAGE_LABELS = {
    "denoise": "ノイズ軽減",
    "foot_ik": "足IK接地安定化",
    "reduce": "キーフレーム圧縮",
}


def _build_parser():
    p = MachineArgumentParser(prog="mocapvmd", allow_abbrev=False)
    p.add_argument("--version", action="version", version=f"mocapvmd {__version__}",
                   help="バージョンを表示して終了する")
    p.add_argument("--machine", action="store_true",
                   help="出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・"
                        "標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない")
    p.add_argument("--describe", action="store_true",
                   help="オプション定義とプリセット一覧を JSON Lines の result で出力して終了する"
                        "(VMD を読まない・入力不要の自己記述)")
    p.add_argument("input", nargs="?", help="入力VMDファイル")
    p.add_argument("-o", "--output", help="出力先(既定: <入力名>_mocap.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
    p.add_argument("--preset", choices=presets.PRESET_NAMES, default="medium",
                   help="キーフレーム圧縮の許容誤差プリセット(速度観点): slower/slow/medium/fast/faster。"
                        "種別スケールの基準値")
    p.add_argument("--clean-strength", dest="clean_strength", type=float, default=1.0,
                   help="ノイズ軽減の効き量の倍率(ブレンド率に掛ける。0で無加工相当、上げるほど強く平準化する。窓幅は据え置き)")
    p.add_argument("--denoise", dest="denoise", action="store_true", default=True,
                   help="ノイズ軽減を有効化(既定 on)")
    p.add_argument("--no-denoise", dest="denoise", action="store_false",
                   help="ノイズ軽減を無効化")
    p.add_argument("--denoise-mode", dest="denoise_mode", choices=("bone", "pose"), default="bone",
                   help="ノイズ軽減の方式: bone(ボーン単位)/ pose(MMDで実際に見える動き=表現空間で評価)")
    p.add_argument("--pmx", dest="pmx", default=None,
                   help="pose 方式が参照するモデルPMX(未指定時は内蔵の既定モデルプロファイル)")
    p.add_argument("--foot-ik-stabilize", dest="foot_ik_stabilize", action="store_true", default=True,
                   help="足IK接地安定化を有効化(既定 on)")
    p.add_argument("--no-foot-ik-stabilize", dest="foot_ik_stabilize", action="store_false",
                   help="足IK接地安定化を無効化")
    p.add_argument("--foot-slide-suppression", dest="foot_slide_suppression", type=float, default=1.0,
                   help="足IK接地中の横滑り抑制の強さ(0〜1)。既定1.0は接地中の横滑りを除去する")
    p.add_argument("--reduce-error-bone-pos", dest="reduce_error_bone_pos", type=float, default=None,
                   help="キーフレーム圧縮の位置許容誤差の基準値(MMD単位)。明示値はプリセットに優先。種別スケールを掛ける")
    p.add_argument("--reduce-error-bone-rot", dest="reduce_error_bone_rot", type=float, default=None,
                   help="キーフレーム圧縮の回転許容誤差の基準値(度)。明示値はプリセットに優先。種別スケールを掛ける")
    p.add_argument("--curve-mode", dest="curve_mode", choices=("bezier", "linear"), default="bezier",
                   help="出力補間曲線: bezier / linear")
    p.add_argument("--reduce", dest="reduce", action="store_true", default=True,
                   help="キーフレーム圧縮を有効化(既定 on)。少ないキー+ベジェ補間で出力する")
    p.add_argument("--no-reduce", dest="reduce", action="store_false",
                   help="キーフレームを圧縮せず、圧縮前の密キー(線形補間)を出力する(診断・比較用)")
    p.add_argument("--list-bones", dest="list_bones", action="store_true",
                   help="ボーン一覧と分類結果を表示して終了する")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力せず処理計画と診断を表示する(引数検証は実施する)")
    p.add_argument("--quiet", dest="quiet", action="store_true",
                   help="進捗表示を抑制する(警告・診断・終了コードは抑制しない)")
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                   help="通常実行でも 4.4 のレポートを標準出力へ表示する(出力VMDは書く)")
    return p


_DESCRIBE_NONNEGATIVE = {"min": 0, "max": None, "exclusive_min": False}
_DESCRIBE_UNIT_INTERVAL = {"min": 0, "max": 1, "exclusive_min": False}
_DESCRIBED_TYPE_AND_CONSTRAINT_BY_DEST = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "preset": ("enum", {"choices": list(presets.PRESET_NAMES)}),
    "clean_strength": ("float", _DESCRIBE_NONNEGATIVE),
    "denoise": ("flag", None),
    "denoise_mode": ("enum", {"choices": ["bone", "pose"]}),
    "pmx": ("str", None),
    "foot_ik_stabilize": ("flag", None),
    "foot_slide_suppression": ("float", _DESCRIBE_UNIT_INTERVAL),
    "reduce_error_bone_pos": ("float", _DESCRIBE_NONNEGATIVE),
    "reduce_error_bone_rot": ("float", _DESCRIBE_NONNEGATIVE),
    "curve_mode": ("enum", {"choices": ["bezier", "linear"]}),
    "reduce": ("flag", None),
    "list_bones": ("flag", None),
    "dry_run": ("flag", None),
    "quiet": ("flag", None),
    "verbose": ("flag", None),
}


def _describe_options(parser):
    options = []
    for action in parser._actions:
        dest = action.dest
        if dest not in _DESCRIBED_TYPE_AND_CONSTRAINT_BY_DEST:
            continue
        type_, constraint = _DESCRIBED_TYPE_AND_CONSTRAINT_BY_DEST[dest]
        if dest == "input":
            name = "input"
        else:
            positive_long_forms = [
                s for s in action.option_strings if s.startswith("--") and not s.startswith("--no-")
            ]
            if not positive_long_forms:
                continue
            name = positive_long_forms[0]
        options.append({
            "name": name,
            "type": type_,
            "constraint": constraint,
            "default": action.default,
            "help": action.help,
        })
    return options


def _describe_presets():
    out = []
    for name in presets.PRESET_NAMES:
        pos, rot = presets.reduction_base(name)
        out.append({"name": name,
                    "values": {"reduce_error_bone_pos": pos, "reduce_error_bone_rot": rot}})
    return out


def _default_output(input_path):
    base, _ = os.path.splitext(input_path)
    return base + "_mocap.vmd"


def _surface_warnings(read_warnings, machine, emitter):
    seen_warn = set()
    for w in read_warnings:
        key = (w.code, w.section, w.message)
        if key in seen_warn:
            continue
        seen_warn.add(key)
        if machine:
            emitter.warning(code=w.code, message=w.message,
                            section=[w.section] if w.section else None)
        else:
            where = f"({w.section})" if w.section else ""
            print(f"warning: {w.code}: {w.message}{where}", file=sys.stderr)


def _bone_names_in_first_seen_order(bone_keys):
    order = []
    seen = set()
    for k in bone_keys:
        if k.name not in seen:
            seen.add(k.name)
            order.append(k.name)
    return order


def _list_bones_text(bone_keys):
    return "\n".join(
        f"{name} [{classify.classify(name)}]" for name in _bone_names_in_first_seen_order(bone_keys)
    )


def _validate_bones(bone_keys):
    denoise.validate_bone_values([k.position for k in bone_keys], [k.rotation for k in bone_keys])


def _densify_bones(bone_keys):
    order = []
    later_key_by_frame = {}
    for k in bone_keys:
        if k.name not in later_key_by_frame:
            later_key_by_frame[k.name] = {}
            order.append(k.name)
        later_key_by_frame[k.name][k.frame] = k

    out = []
    for name in order:
        by_frame = later_key_by_frame[name]
        ks = [by_frame[f] for f in sorted(by_frame)]
        if len(ks) < 2:
            out.extend(ks)
            continue
        positions, rotations = interp.bake_bone_track(ks, ks[0].frame, ks[-1].frame)
        name_raw = ks[0].name_raw
        for i, f in enumerate(range(ks[0].frame, ks[-1].frame + 1)):
            out.append(BoneKey(name_raw, f, positions[i], rotations[i], BONE_LINEAR_INTERP))
    out.sort(key=lambda k: (k.name_raw, k.frame))
    return out


def _clean_bones(bone_keys, clean_strength):
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
        if len(ks) < 2:
            out.extend(ks)
            continue
        params = presets.resolve_cleaning(clean_strength, classify.classify(name))
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


def _stabilize_bones(bone_keys, suppression):
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

    stabilized = footik.stabilize_foot_ik(tracks, suppression)

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


def _build_inspect(args, doc, reduction_diag, pose_diag):
    order = []
    groups = {}
    for k in doc.bone:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)
    all_frames = [k.frame for k in doc.bone]
    frame_range = [min(all_frames), max(all_frames)] if all_frames else None
    duration = (max(all_frames) / 30.0) if all_frames else None
    bones = []
    for name in order:
        frames = [k.frame for k in groups[name]]
        bones.append({
            "name": name,
            "category": classify.classify(name),
            "keys": len(groups[name]),
            "frame_range": [min(frames), max(frames)],
        })
    sections = [
        name for name, present in (
            ("bone", doc.bone), ("morph", doc.morph), ("camera", doc.camera),
            ("light", doc.light), ("self_shadow", doc.self_shadow), ("ik_property", doc.ik_property),
        ) if present
    ]
    pose = None
    if pose_diag:
        fit = pose_diag["fit"]
        md = pose_diag["marker_displacement"]
        mk = pose_diag["markers"]
        pose = {
            "pmx": pose_diag["pmx"],
            "frames": int(pose_diag["frames"]),
            "markers": {"available": int(mk["available"]),
                        "required_bones_ok": bool(mk["required_bones_ok"])},
            "marker_displacement": {"max": float(md["max"]), "mean": float(md["mean"])},
            "fit": {
                "mean_error_before": float(fit["mean_error_before"]),
                "mean_error_after": float(fit["mean_error_after"]),
                "fallback_frames": int(fit["fallback_frames"]),
                "max_bone_delta_deg": float(fit["max_bone_delta_deg"]),
                "max_center_delta": float(fit["max_center_delta"]),
            },
        }
    return {
        "output": None,
        "input_kind": "bone",
        "keys": len(doc.bone),
        "frame_range": frame_range,
        "duration_sec": duration,
        "sections": sections,
        "preset": args.preset,
        "clean_strength": args.clean_strength,
        "denoise": args.denoise,
        "denoise_mode": args.denoise_mode,
        "foot_ik_stabilize": args.foot_ik_stabilize,
        "foot_slide_suppression": args.foot_slide_suppression,
        "curve_mode": args.curve_mode,
        "reduce": args.reduce,
        "bones": bones,
        "reduction": reduction_diag,
        "pose_denoise": pose,
    }


def _fail(emitter, code, message, exit_code, *, field=None, path=None):
    return emit_failure(emitter, code=code, message=message, exit_code=exit_code,
                        field=field, path=path)


def main(argv=None) -> int:
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
                # ロケール符号化(cp932 等)で表せない文字を stderr へ書くと UnicodeEncodeError になる。
                sys.stderr.reconfigure(errors="backslashreplace")
            except Exception:
                pass
        parser = _build_parser()
        try:
            args = parser.parse_args(argv)
        except ArgumentParseError as e:
            return fail("bad_argument", e.message, 2, field=argparse_error_field(e.message))
        except SystemExit as e:
            code = e.code
            return code if isinstance(code, int) else (0 if code is None else 2)

        if args.describe:
            emitter.result(mode="describe", options=_describe_options(parser), presets=_describe_presets())
            return 0
        if args.input is None:
            return fail("bad_argument", "入力VMDファイル(input)が必要", 2, field="input")

        return _run(args, machine, emitter, fail)
    except KeyboardInterrupt:
        return _fail(emitter, "cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return _fail(emitter, "internal_error", f"{type(e).__name__}: {e}", 1)


def _run(args, machine, emitter, fail):
    if not os.path.isfile(args.input):
        return fail("input_not_file", f"入力が存在しないか通常ファイルでない: {args.input}", 1, field="input")

    if args.list_bones:
        try:
            doc, read_warnings = io.read(args.input)
        except Exception as e:
            return fail("not_vmd", f"入力を VMD として読めない: {type(e).__name__}: {e}", 1, field="input")
        _surface_warnings(read_warnings, machine, emitter)
        if machine:
            emitter.result(mode="list_bones", bones=[
                {"name": n, "category": classify.classify(n)} for n in _bone_names_in_first_seen_order(doc.bone)
            ])
        else:
            print(_list_bones_text(doc.bone))
        return 0

    output = args.output if args.output is not None else _default_output(args.input)
    if os.path.isdir(output):
        return fail("output_is_directory",
                    f"出力先がディレクトリです(ファイルパスを指定): {output}",
                    2, field="--output", path=output)
    if not args.overwrite and os.path.exists(output):
        return fail("output_exists",
                    f"出力先に既存ファイルがあります。上書きには --overwrite が必要: {output}",
                    2, field="--output")

    for name, v in (("--reduce-error-bone-pos", args.reduce_error_bone_pos),
                    ("--reduce-error-bone-rot", args.reduce_error_bone_rot)):
        if v is not None and (not math.isfinite(v) or v < 0.0):
            return fail("bad_argument", f"{name} は有限の非負値が必要: {v}", 2, field=name)

    if not math.isfinite(args.clean_strength) or args.clean_strength < 0.0:
        return fail("bad_argument", f"--clean-strength は有限の非負値が必要: {args.clean_strength}",
                    2, field="--clean-strength")

    s = args.foot_slide_suppression
    if not math.isfinite(s) or not 0.0 <= s <= 1.0:
        return fail("bad_argument", f"--foot-slide-suppression は 0〜1 の有限値が必要: {s}",
                    2, field="--foot-slide-suppression")

    if args.denoise and args.denoise_mode == "pose" and args.pmx is not None:
        if not os.path.isfile(args.pmx):
            return fail("pmx_not_file", f"--pmx が存在しないか通常ファイルでない: {args.pmx}",
                        1, field="--pmx")

    try:
        doc, read_warnings = io.read(args.input)
    except Exception as e:
        return fail("not_vmd", f"入力を VMD として読めない: {type(e).__name__}: {e}", 1, field="input")
    _surface_warnings(read_warnings, machine, emitter)

    try:
        _validate_bones(doc.bone)
    except ValueError as e:
        return fail("invalid_bone_values",
                    f"ボーン値が不正(非有限・ゼロノルム quaternion): {e}", 1, field="input")

    dense_bone = _densify_bones(doc.bone)

    reporter = ProgressRouter(machine=machine, quiet=args.quiet, emitter=emitter,
                              stream=sys.stderr, labels=_STAGE_LABELS)
    want_report = args.dry_run or args.verbose
    reduction_diag = {} if (args.reduce and want_report) else None
    pose_diag = {} if (args.denoise and args.denoise_mode == "pose" and want_report) else None
    try:
        if args.denoise:
            reporter.stage("denoise")
            if args.denoise_mode == "pose":
                try:
                    new_bone = apply_pose_denoise(dense_bone, pmx_path=args.pmx, diagnostics_out=pose_diag)
                except PmxFormatError as e:
                    return fail("not_pmx", f"PMX 形式が不正: {type(e).__name__}: {e}", 1, field="--pmx")
                except MocapModelProfileError as e:
                    field = "--pmx" if args.pmx is not None else None
                    return fail("model_profile_invalid", f"モデルプロファイルが不正: {e}", 1, field=field)
            else:
                new_bone = _clean_bones(dense_bone, args.clean_strength)
        else:
            new_bone = dense_bone
        denoised_bone = new_bone
        if args.foot_ik_stabilize:
            reporter.stage("foot_ik")
            new_bone = _stabilize_bones(new_bone, args.foot_slide_suppression)
        cleaned_bone = new_bone
        if args.reduce:
            reduce_start = time.monotonic()
            reporter.stage("reduce")

            def reduce_cb(done, total):
                reporter.stage("reduce", done=done, total=total,
                               elapsed=time.monotonic() - reduce_start)

            new_bone = reduce.reduce_bones(
                new_bone,
                args.preset,
                override_pos=args.reduce_error_bone_pos,
                override_rot=args.reduce_error_bone_rot,
                curve_mode=args.curve_mode,
                diagnostics_out=reduction_diag,
                progress=reduce_cb,
            )
    finally:
        reporter.close()

    if not machine and want_report:
        rep = report.build_report(
            dense_bone,
            args.preset,
            clean_strength=args.clean_strength,
            denoise=args.denoise,
            foot_ik_stabilize=args.foot_ik_stabilize,
            reduction=reduction_diag,
            suppression=args.foot_slide_suppression,
            pose_denoise=pose_diag,
            cleaned_bone_keys=cleaned_bone,
            denoised_bone_keys=denoised_bone,
        )
        print(report.format_dry_run(rep))

    if args.dry_run:
        if machine:
            emitter.result(mode="inspect", **_build_inspect(args, doc, reduction_diag, pose_diag))
        return 0

    out_doc = dataclasses.replace(doc, bone=new_bone)
    try:
        io.write_file(out_doc, output)
    except Exception as e:
        return fail("write_failed", f"出力の書き込みに失敗: {type(e).__name__}: {e}",
                    3, field="--output", path=output)

    if machine:
        emitter.result(mode="process", output=output,
                       input_keys=len(doc.bone), output_keys=len(new_bone))
    reporter.summary(f"完了 {output}")
    return 0
