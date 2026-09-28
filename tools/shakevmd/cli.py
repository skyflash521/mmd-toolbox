import argparse
import math
import os
import sys
import time

from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    emit_failure,
    install_sigbreak_handler,
)
from cli_progress_router import ProgressRouter
from shakevmd import __version__, cuts, presets
from shakevmd.bake import RangeOverlapError, bake
from shakevmd.warn import ShakeWarning
from vmd import interp, io
from vmd.reduce import Tolerances, reduce_camera_track

_HARD_DEFAULTS = {
    "amp_rot": 0.8, "amp_pos": 0.05, "rot_weights": (1.0, 1.0, 0.3),
    "freq": 1.2, "motion_damp": 1.0, "settle": 0.0, "cut_threshold": (5.0, 20.0),
    "fade": 0.7,
}

_BONE_TOLERANCES_UNUSED_FOR_CAMERA = {"bone_pos": 1.0, "bone_rot": 30.0}
_SMOOTH_TOLERANCES = Tolerances(
    **_BONE_TOLERANCES_UNUSED_FOR_CAMERA,
    camera_pos=0.10, camera_rot=0.25, camera_distance=0.10, camera_fov=1.00,
)
_CUT_DISTANCE_THRESHOLD_UNUSED_WITHOUT_DETECTION = 5.0
_SMOOTH_MAX_SEG = 5

_STAGE_LABELS = {"bake": "ベイク", "smooth": "スムージング"}


def _finite_float(text):
    v = float(text)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"有限な数値が必要: {text!r}")
    return v


def _nonneg_float(text):
    v = _finite_float(text)
    if v < 0:
        raise argparse.ArgumentTypeError(f"非負の数値が必要: {text!r}")
    return v


def _positive_float(text):
    v = _finite_float(text)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"正の数値が必要: {text!r}")
    return v


def _parse_range(text) -> tuple[int | None, int | None]:
    if text.count(":") != 1:
        raise argparse.ArgumentTypeError(f"範囲は START:END 形式: {text!r}")
    s_str, e_str = text.split(":")

    def part(v):
        if v == "":
            return None
        try:
            n = int(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"範囲端は整数: {v!r}") from None
        if n < 0:
            raise argparse.ArgumentTypeError(f"フレーム番号は非負: {v!r}")
        return n

    s, e = part(s_str), part(e_str)
    if s is not None and e is not None and s > e:
        raise argparse.ArgumentTypeError(f"範囲は START<=END: {text!r}")
    return (s, e)


def _parse_rot_weights(text) -> tuple[float, float, float]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--rot-weights は P,Y,R の3要素: {text!r}")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--rot-weights は数値3要素: {text!r}") from None
    if not all(math.isfinite(v) for v in vals):
        raise argparse.ArgumentTypeError(f"--rot-weights は有限値: {text!r}")
    return vals


def _parse_cut_threshold(text) -> tuple[float, float]:
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--cut-threshold は 位置,角度 の2要素: {text!r}")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--cut-threshold は数値2要素: {text!r}") from None
    if not all(math.isfinite(v) for v in vals):
        raise argparse.ArgumentTypeError(f"--cut-threshold は有限値: {text!r}")
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError(f"--cut-threshold は非負(感度閾値): {text!r}")
    return vals


def _parse_impulse(text) -> tuple[int, float, float]:
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--impulse は F:S:D の3要素: {text!r}")
    f_str, s_str, d_str = parts
    try:
        f, s, d = int(f_str), float(s_str), float(d_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--impulse は F:S:D(F=整数, S/D=数値): {text!r}") from None
    if f < 0:
        raise argparse.ArgumentTypeError(f"--impulse の F(フレーム)は非負: {text!r}")
    if not (math.isfinite(s) and math.isfinite(d)):
        raise argparse.ArgumentTypeError(f"--impulse の S/D は有限値: {text!r}")
    if s < 0:
        raise argparse.ArgumentTypeError(f"--impulse の S(強さ)は非負: {text!r}")
    if d <= 0:
        raise argparse.ArgumentTypeError(f"--impulse の D(減衰秒)は正: {text!r}")
    return (f, s, d)


def _build_parser() -> argparse.ArgumentParser:
    p = MachineArgumentParser(prog="shakevmd", allow_abbrev=False)
    p.add_argument("--version", action="version", version=f"shakevmd {__version__}",
                   help="バージョンを表示して終了する")
    p.add_argument("--machine", action="store_true",
                   help="出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・"
                        "標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない")
    p.add_argument("--describe", action="store_true",
                   help="オプション定義とプリセット一覧を JSON Lines の result で出力して終了する"
                        "(VMD を読まない・入力不要の自己記述)")
    p.add_argument("input", nargs="?", help="入力カメラ VMD ファイル")
    p.add_argument("-o", "--output", help="出力先(既定: <入力名>_shake.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
    p.add_argument("--range", dest="ranges", action="append", type=_parse_range,
                   metavar="START:END",
                   help="揺れ適用範囲。START/END は各々省略可。複数指定可(既定は全範囲)")
    p.add_argument("--amp-rot", type=_nonneg_float, help="回転振幅の基準値(度)")
    p.add_argument("--amp-pos", type=_nonneg_float, help="位置振幅(MMD距離単位)")
    p.add_argument("--rot-weights", type=_parse_rot_weights, metavar="P,Y,R",
                   help="Pitch/Yaw/Roll 個別重み")
    p.add_argument("--freq", type=_positive_float, help="ノイズ基本周波数(Hz)")
    p.add_argument("--seed", type=int, default=1, help="ノイズシード(既定 1・再現性)")
    p.add_argument("--fade", type=_nonneg_float, help="範囲端の自動フェード時間(秒)")
    p.add_argument("--motion-damp", type=_nonneg_float,
                   help="元モーション速度に応じた振幅減衰係数(0 で無効)")
    p.add_argument("--settle", type=_nonneg_float,
                   help="停止検出時の減衰振動の初期振幅(度・0 で無効)")
    p.add_argument("--cut-threshold", type=_parse_cut_threshold, metavar="位置,角度",
                   help="カット自動検出の感度(位置ジャンプ,角度ジャンプ)")
    p.add_argument("--impulse", dest="impulses", action="append", type=_parse_impulse,
                   metavar="F:S:D",
                   help="フレーム F に強さ S・減衰 D 秒の衝撃を加算(複数指定可)")
    p.add_argument("--preset", choices=presets.PRESET_NAMES,
                   help="公開引数を一括設定するプリセット(個別引数の明示指定が優先)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力せず統計を表示する(引数検証は実施する)")
    p.add_argument("-v", "--verbose", action="store_true", help="詳細ログを出す")
    p.add_argument("--smooth", default=True, action=argparse.BooleanOptionalAction,
                   help="ベイク後の密なキーをベジェ補間でなめらかに整理する"
                        "(スムージング。既定 on。--no-smooth で密キー+線形)")
    p.add_argument("--quiet", dest="quiet", action="store_true",
                   help="進捗表示を抑制する(警告・統計・終了コードは抑制しない)")
    return p


_NONNEG_FLOAT_CONSTRAINT = {"min": 0, "max": None, "exclusive_min": False}
_POSITIVE_FLOAT_CONSTRAINT = {"min": 0, "max": None, "exclusive_min": True}


def _cfield(name, type_, mn, mx, ex):
    return {"name": name, "type": type_, "min": mn, "max": mx, "exclusive_min": ex}


_D_COMPOUND = {
    "ranges": {"format": "START:END", "fields": [
        _cfield("START", "int", 0, None, False), _cfield("END", "int", 0, None, False)]},
    "rot_weights": {"format": "P,Y,R", "fields": [
        _cfield("P", "float", None, None, False), _cfield("Y", "float", None, None, False),
        _cfield("R", "float", None, None, False)]},
    "cut_threshold": {"format": "位置,角度", "fields": [
        _cfield("位置", "float", 0, None, False), _cfield("角度", "float", 0, None, False)]},
    "impulses": {"format": "F:S:D", "fields": [
        _cfield("F", "int", 0, None, False), _cfield("S", "float", 0, None, False),
        _cfield("D", "float", 0, None, True)]},
}

_D_TYPE = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "ranges": ("compound", _D_COMPOUND["ranges"]),
    "amp_rot": ("float", _NONNEG_FLOAT_CONSTRAINT),
    "amp_pos": ("float", _NONNEG_FLOAT_CONSTRAINT),
    "rot_weights": ("compound", _D_COMPOUND["rot_weights"]),
    "freq": ("float", _POSITIVE_FLOAT_CONSTRAINT),
    "seed": ("int", None),
    "fade": ("float", _NONNEG_FLOAT_CONSTRAINT),
    "motion_damp": ("float", _NONNEG_FLOAT_CONSTRAINT),
    "settle": ("float", _NONNEG_FLOAT_CONSTRAINT),
    "cut_threshold": ("compound", _D_COMPOUND["cut_threshold"]),
    "impulses": ("compound", _D_COMPOUND["impulses"]),
    "preset": ("enum", {"choices": list(presets.PRESET_NAMES)}),
    "dry_run": ("flag", None),
    "verbose": ("flag", None),
    "smooth": ("flag", None),
    "quiet": ("flag", None),
}


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
            name = next(s for s in action.option_strings
                        if s.startswith("--") and not s.startswith("--no-"))
        if dest in _HARD_DEFAULTS:
            v = _HARD_DEFAULTS[dest]
            default = list(v) if isinstance(v, tuple) else v
        else:
            default = action.default
        options.append({
            "name": name, "type": type_, "constraint": constraint,
            "default": default, "help": action.help,
        })
    return options


def _describe_presets():
    out = []
    for name in presets.PRESET_NAMES:
        values = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in presets.get_preset(name).items()
            if k not in presets.INTERNAL_PARAM_NAMES
        }
        out.append({"name": name, "values": values})
    return out


def _default_output(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return base + "_shake.vmd"


def _all_finite(keys) -> bool:
    for k in keys:
        if not math.isfinite(k.distance):
            return False
        if not all(math.isfinite(c) for c in k.position):
            return False
        if not all(math.isfinite(r) for r in k.rotation):
            return False
    return True


def _working_view(camera):
    by_frame = {}
    for k in camera:
        by_frame[k.frame] = k
    return [by_frame[f] for f in sorted(by_frame)]


def _resolve_param(name, args, preset):
    v = getattr(args, name)
    if v is not None:
        return v
    if name in preset:
        return preset[name]
    return _HARD_DEFAULTS[name]


def _shake_stats(orig_camera, baked_keys, applied, cut_pos, cut_rot):
    """戻り値は (最大振幅, 検出したカットのフレーム列)。"""
    wv = _working_view(orig_camera)
    detected_cuts = cuts.detect_cuts(wv, cut_pos, cut_rot)
    applied_frames = set()
    for a, b in applied:
        applied_frames.update(range(a, b + 1))
    max_amp = 0.0
    for k in sorted(baked_keys, key=lambda x: x.frame):
        if k.frame not in applied_frames:
            continue
        s = interp.sample_camera(wv, k.frame)
        dr = [k.rotation[i] - s["rotation"][i] for i in range(3)]
        dp = [k.position[i] - s["position"][i] for i in range(3)]
        for v in dr + dp:
            max_amp = max(max_amp, abs(v))
    return max_amp, detected_cuts


def _argparse_field(message: str):
    # argparse は起因引数を構造化して渡さないので、標準の文言から取り出す。
    if message.startswith("argument "):
        name = message[len("argument "):].split(":", 1)[0].strip()
        # argparse は同じ引数の複数の表記を "-o/--output" のように "/" で連結する。
        return name.split("/")[-1] if name.startswith("-") else name
    if message.startswith("unrecognized arguments:"):
        rest = message[len("unrecognized arguments:"):].split()
        return rest[0] if rest else None
    if message.startswith("the following arguments are required:"):
        rest = message[len("the following arguments are required:"):].strip()
        return rest.split(",")[0].strip() or None
    return None


def _smooth_keep_frames(resolved, detected_cuts) -> tuple[int, ...]:
    max_segment_grid = {f for f0, f1 in resolved for f in range(int(f0), int(f1) + 1, _SMOOTH_MAX_SEG)}
    both_sides_of_cuts = {f for c in detected_cuts for f in (c - 1, c) if f >= 0}
    return tuple(sorted(both_sides_of_cuts | max_segment_grid))


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
                sys.stderr.reconfigure(errors="backslashreplace")
            except Exception:
                pass
        parser = _build_parser()
        try:
            args = parser.parse_args(argv)
        except ArgumentParseError as e:
            return fail("bad_argument", e.message, 2, field=_argparse_field(e.message))
        except SystemExit as e:
            code = e.code
            return code if isinstance(code, int) else (0 if code is None else 2)

        if args.describe:
            emitter.result(
                mode="describe",
                options=_describe_options(parser),
                presets=_describe_presets(),
            )
            return 0
        if args.input is None:
            return fail("bad_argument", "入力カメラ VMD ファイル(input)が必要", 2, field="input")

        return _run(args, machine, emitter, fail)
    except KeyboardInterrupt:
        return _fail(emitter, "cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return _fail(emitter, "internal_error", f"{type(e).__name__}: {e}", 1)


def _run(args, machine, emitter, fail) -> int:
    output = args.output if args.output is not None else _default_output(args.input)

    if os.path.isdir(output):
        return fail("output_is_directory",
                    f"出力先がディレクトリです(ファイルパスを指定): {output}",
                    2, field="--output", path=output)

    if not args.overwrite and os.path.exists(output):
        return fail(
            "output_exists",
            f"出力先に既存ファイルがあります。上書きには --overwrite が必要: {output}",
            2, field="--output",
        )

    try:
        doc, read_warnings = io.read(args.input)
    except Exception as e:
        return fail("not_vmd", f"入力を VMD として読めない: {type(e).__name__}: {e}",
                    1, field="input")
    if not doc.camera:
        return fail("no_camera_keys", "入力にカメラキーがない", 1, field="input")

    ranges = None
    if args.ranges:
        frames = [k.frame for k in doc.camera]
        first, last = min(frames), max(frames)
        ranges = [
            (first if s is None else s, last if e is None else e)
            for (s, e) in args.ranges
        ]
        if any(s > e for (s, e) in ranges):
            return fail("range_reversed",
                        "範囲の開始が終了より後(省略端の解決後に START>END)", 2, field="--range")

    preset = presets.get_preset(args.preset) if args.preset else {}
    amp_rot = _resolve_param("amp_rot", args, preset)
    amp_pos = _resolve_param("amp_pos", args, preset)
    rot_weights = _resolve_param("rot_weights", args, preset)
    freq = _resolve_param("freq", args, preset)
    motion_damp = _resolve_param("motion_damp", args, preset)
    settle = _resolve_param("settle", args, preset)
    cut_threshold = _resolve_param("cut_threshold", args, preset)
    fade = _resolve_param("fade", args, preset)
    internal = {k: preset[k] for k in presets.INTERNAL_PARAM_NAMES if k in preset}

    reporter = ProgressRouter(machine=machine, quiet=args.quiet, emitter=emitter,
                              stream=sys.stderr, labels=_STAGE_LABELS)
    try:
        bake_start = time.monotonic()
        reporter.stage("bake")

        def bake_cb(done, total):
            reporter.stage("bake", done=done, total=total,
                           elapsed=time.monotonic() - bake_start)
        try:
            result = bake(
                doc.camera,
                ranges=ranges,
                seed=args.seed,
                amp_rot=amp_rot,
                amp_pos=amp_pos,
                rot_weights=rot_weights,
                freq=freq,
                motion_damp=motion_damp,
                settle=settle,
                fade_sec=fade,
                cut_pos_threshold=cut_threshold[0],
                cut_rot_threshold=cut_threshold[1],
                impulses=tuple(args.impulses or ()),
                progress=bake_cb,
                **internal,
            )
        except RangeOverlapError as e:
            return fail("range_overlap", str(e), 2, field="--range")
        except (ValueError, OverflowError) as e:
            return fail("value_overflow", f"値が過大でベイクが破綻した: {type(e).__name__}: {e}", 2)
        reporter.close()

        # struct は float32 への変換で inf と nan を例外なく通す。
        if not _all_finite(result.camera_keys):
            return fail("non_finite_output",
                        "焼き出力が非有限(inf/nan)になった。振幅・重みが過大", 2)

        warnings = [
            ShakeWarning(w.code, w.message, (w.section,) if w.section else None)
            for w in read_warnings
        ]
        warnings += list(result.warnings)
        sections = [
            name for name, present in (
                ("bone", doc.bone), ("morph", doc.morph), ("light", doc.light),
                ("self_shadow", doc.self_shadow), ("ik_property", doc.ik_property),
            ) if present
        ]
        if sections:
            warnings.append(ShakeWarning(
                "non_camera_sections_passthrough",
                "カメラ以外のセクションは無加工で透過した",
                tuple(sections),
            ))
        for w in warnings:
            if machine:
                emitter.warning(
                    code=w.code, message=w.message,
                    section=list(w.section) if w.section is not None else None,
                )
            else:
                print(f"warning: {w.code}: {w.message}", file=sys.stderr)

        wv_frames = [k.frame for k in _working_view(doc.camera)]
        applied = result.resolved
        max_amp, detected_cuts = _shake_stats(
            doc.camera, result.camera_keys, applied, cut_threshold[0], cut_threshold[1])

        if not machine and (args.dry_run or args.verbose):
            print(f"range: {applied}")
            print(f"keys: {len(result.camera_keys)}")
            print(f"max amplitude: {max_amp:.6g}")
            print(f"cuts: {detected_cuts}")

        if args.dry_run:
            if machine:
                emitter.result(
                    mode="inspect",
                    output=None,
                    input_kind="camera",
                    keys=len(wv_frames),
                    frame_range=[int(wv_frames[0]), int(wv_frames[-1])],
                    duration_sec=wv_frames[-1] / 30.0,
                    sections=["camera"] + list(sections),
                    applied_ranges=[[int(a), int(b)] for a, b in applied],
                    max_amplitude=float(max_amp),
                    detected_cuts=[int(c) for c in detected_cuts],
                )
            return 0

        camera_out = result.camera_keys
        if args.smooth:
            smooth_start = time.monotonic()
            reporter.stage("smooth")

            def smooth_cb(done, total, note=""):
                reporter.stage("smooth", done=done, total=total, note=note,
                               elapsed=time.monotonic() - smooth_start)
            camera_out = reduce_camera_track(
                result.camera_keys,
                result.resolved,
                _SMOOTH_TOLERANCES,
                cut_thresholds=(cut_threshold[0], cut_threshold[1],
                                _CUT_DISTANCE_THRESHOLD_UNUSED_WITHOUT_DETECTION),
                keep_frames=_smooth_keep_frames(result.resolved, detected_cuts),
                no_cut_detect=True,
                min_seg=1,
                max_seg=_SMOOTH_MAX_SEG,
                strict=False,
                curve_mode="bezier",
                force_bezier=True,
                progress=smooth_cb,
            )
            reporter.close()

        doc.camera = camera_out

        try:
            io.write_file(doc, output)
        except OverflowError:
            return fail("value_overflow", "値が大きすぎて書き込み時に float32 で溢れた", 2)
        except Exception as e:
            return fail("write_failed", f"出力の書き込みに失敗: {type(e).__name__}: {e}",
                        3, field="--output", path=output)

        if machine:
            emitter.result(
                mode="bake",
                output=output,
                keys=len(camera_out),
                applied_ranges=[[int(a), int(b)] for a, b in applied],
                max_amplitude=float(max_amp),
                detected_cuts=[int(c) for c in detected_cuts],
            )
        reporter.summary(f"完了 {output}")
        return 0
    finally:
        reporter.close()
