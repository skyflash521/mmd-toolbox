import argparse
import math
import sys
import time

from cli_events import (
    MachineArgumentParser,
    add_common_arguments,
    default_output_path,
    guard_output,
    run_tool,
)
from cli_options import CompoundValidator, RangeValidator, describe_options
from cli_progress_router import ProgressRouter
from shakevmd import __version__, cuts, presets
from shakevmd.bake import RangeOverlapError, bake
from shakevmd.warn import ShakeWarning
from vmd import VmdFormatError, interp, io
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


_NONNEG_FLOAT = RangeValidator(value_type="float", minimum=0)
_POSITIVE_FLOAT = RangeValidator(value_type="float", minimum=0, exclusive_min=True)
_FRAME_END = RangeValidator(value_type="int", minimum=0, allow_empty=True)
_ROT_WEIGHTS = CompoundValidator(
    format="P,Y,R", separator=",",
    elements=[(n, RangeValidator(value_type="float")) for n in ("P", "Y", "R")],
)
_CUT_THRESHOLD = CompoundValidator(
    format="位置,角度", separator=",",
    elements=[(n, RangeValidator(value_type="float", minimum=0)) for n in ("位置", "角度")],
)
_RANGE = CompoundValidator(
    format="START:END",
    elements=[("START", _FRAME_END), ("END", _FRAME_END)],
    relation=("範囲は START<=END",
              lambda v: v[0] is None or v[1] is None or v[0] <= v[1]),
)
_IMPULSE = CompoundValidator(
    format="F:S:D",
    elements=[
        ("F", RangeValidator(value_type="int", minimum=0)),
        ("S", RangeValidator(value_type="float", minimum=0)),
        ("D", RangeValidator(value_type="float", minimum=0, exclusive_min=True)),
    ],
)


def _build_parser() -> argparse.ArgumentParser:
    p = MachineArgumentParser(prog="shakevmd", allow_abbrev=False)
    add_common_arguments(
        p, version=__version__, output_help="出力先(既定: <入力名>_shake.vmd)",
        help_overrides={
            "--describe": "オプション定義とプリセット一覧を JSON Lines の result で出力して終了する"
                          "(VMD を読まない・入力不要の自己記述)",
            "--dry-run": "出力せず統計を表示する(引数検証は実施する)",
            "--verbose": "詳細ログを出す",
            "--quiet": "進捗表示を抑制する(警告・統計・終了コードは抑制しない)",
        })
    p.add_argument("input", nargs="?", help="入力カメラ VMD ファイル")
    p.add_argument("--range", dest="ranges", action="append", type=_RANGE,
                   metavar="START:END",
                   help="揺れ適用範囲。START/END は各々省略可。"
                        "START と END の両方を指定するときは START<=END。複数指定可(既定は全範囲)")
    p.add_argument("--amp-rot", type=_NONNEG_FLOAT, help="回転振幅の基準値(度)")
    p.add_argument("--amp-pos", type=_NONNEG_FLOAT, help="位置振幅(MMD距離単位)")
    p.add_argument("--rot-weights", type=_ROT_WEIGHTS, metavar="P,Y,R",
                   help="Pitch/Yaw/Roll 個別重み")
    p.add_argument("--freq", type=_POSITIVE_FLOAT, help="ノイズ基本周波数(Hz)")
    p.add_argument("--seed", type=int, default=1, help="ノイズシード(既定 1・再現性)")
    p.add_argument("--fade", type=_NONNEG_FLOAT, help="範囲端の自動フェード時間(秒)")
    p.add_argument("--motion-damp", type=_NONNEG_FLOAT,
                   help="元モーション速度に応じた振幅減衰係数(0 で無効)")
    p.add_argument("--settle", type=_NONNEG_FLOAT,
                   help="停止検出時の減衰振動の初期振幅(度・0 で無効)")
    p.add_argument("--cut-threshold", type=_CUT_THRESHOLD, metavar="位置,角度",
                   help="カット自動検出の感度(位置ジャンプ,角度ジャンプ)")
    p.add_argument("--impulse", dest="impulses", action="append", type=_IMPULSE,
                   metavar="F:S:D",
                   help="フレーム F に強さ S・減衰 D 秒の衝撃を加算(複数指定可)")
    p.add_argument("--preset", choices=presets.PRESET_NAMES,
                   help="公開引数を一括設定するプリセット(個別引数の明示指定が優先)")
    p.add_argument("--smooth", default=True, action=argparse.BooleanOptionalAction,
                   help="ベイク後の密なキーをベジェ補間でなめらかに整理する"
                        "(スムージング。既定 on。--no-smooth で密キー+線形)")
    return p


_D_TYPE = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "ranges": (None, None),
    "amp_rot": (None, None),
    "amp_pos": (None, None),
    "rot_weights": (None, None),
    "freq": (None, None),
    "seed": ("int", None),
    "fade": (None, None),
    "motion_damp": (None, None),
    "settle": (None, None),
    "cut_threshold": (None, None),
    "impulses": (None, None),
    "preset": ("enum", {"choices": list(presets.PRESET_NAMES)}),
    "dry_run": ("flag", None),
    "verbose": ("flag", None),
    "smooth": ("flag", None),
    "quiet": ("flag", None),
}

_DESCRIBE_DEFAULTS = {
    k: list(v) if isinstance(v, tuple) else v for k, v in _HARD_DEFAULTS.items()
}


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


def _smooth_keep_frames(resolved, detected_cuts) -> tuple[int, ...]:
    max_segment_grid = {f for f0, f1 in resolved for f in range(int(f0), int(f1) + 1, _SMOOTH_MAX_SEG)}
    both_sides_of_cuts = {f for c in detected_cuts for f in (c - 1, c) if f >= 0}
    return tuple(sorted(both_sides_of_cuts | max_segment_grid))


def _describe(parser):
    return {
        "options": describe_options(parser, _D_TYPE, defaults=_DESCRIBE_DEFAULTS),
        "presets": _describe_presets(),
    }


def main(argv=None) -> int:
    return run_tool(argv, build_parser=_build_parser, describe=_describe, run=_run)


def _run(args, ctx) -> int:
    machine = ctx.machine
    emitter = ctx.emitter
    fail = ctx.fail

    if args.input is None:
        return fail("bad_argument", "入力カメラ VMD ファイル(input)が必要", 2, field="input")

    output = (args.output if args.output is not None
              else default_output_path(args.input, "_shake.vmd"))
    guarded = guard_output(output, overwrite=args.overwrite, fail=fail)
    if guarded is not None:
        return guarded

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

    reporter = ctx.track_progress(ProgressRouter(
        machine=machine, quiet=args.quiet, emitter=emitter,
        stream=sys.stderr, labels=_STAGE_LABELS))
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
        ctx.warn(w.code, w.message, section=w.section if machine else None)

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
    except (OverflowError, VmdFormatError):
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
