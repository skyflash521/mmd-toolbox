import argparse
import inspect
import os
from dataclasses import dataclass, replace

from cli_events import (
    MachineArgumentParser,
    add_common_arguments,
    default_output_path,
    guard_output,
    run_tool,
)
from cli_options import RangeValidator, describe_options
from lipsync import generate_morph_keys
from vmd import VmdDocument, ensure_frame0_neutral_keys, normalize, write_file
from vpr import read

from . import __version__, loudness, openness, presets, timing
from .events import (
    EventDiagnostics,
    OverlapDiagnostics,
    build_mouth_events,
    resolve_overlaps,
)
from .io import TrackSelectionError, collect_notes, select_track
from .tempo_correction import apply_tempo_correction

STYLE_NAMES = ("pop", "ballad", "powerful", "whisper", "rap")

_VMD_MODEL_NAME_BYTES = 20

_DEFAULT_LEGATO_MAX = inspect.signature(build_mouth_events).parameters["legato_max_frames"].default
_DEFAULT_REF_BPM = inspect.signature(apply_tempo_correction).parameters["ref_bpm"].default
_DEFAULT_TEMPO_SCALE_MIN = inspect.signature(apply_tempo_correction).parameters["s_min"].default


def _model_name(text: str) -> str:
    try:
        encoded = text.encode("cp932")
    except UnicodeEncodeError:
        raise argparse.ArgumentTypeError(
            f"モデル名は Shift-JIS(cp932)で表現できる文字のみ: {text!r}"
        ) from None
    if len(encoded) > _VMD_MODEL_NAME_BYTES:
        raise argparse.ArgumentTypeError(
            f"モデル名は cp932 で {_VMD_MODEL_NAME_BYTES} バイト以内"
            f"({len(encoded)} バイト): {text!r}"
        )
    return text


_unit_float = RangeValidator(value_type="float", minimum=0, maximum=1)
_positive_float = RangeValidator(value_type="float", minimum=0, exclusive_min=True)
_nonneg_float = RangeValidator(value_type="float", minimum=0)
_positive_unit_float = RangeValidator(value_type="float", minimum=0, maximum=1, exclusive_min=True)
_nonneg_int = RangeValidator(value_type="int", minimum=0)
_positive_int = RangeValidator(value_type="int", minimum=1)


def _build_parser() -> argparse.ArgumentParser:
    p = MachineArgumentParser(prog="vpr2vmd", allow_abbrev=False)
    p.add_argument("input", nargs="?", help="入力 vpr ファイル")
    p.add_argument("--track",
                   help="リップモーション対象の歌唱トラック。半角数字だけの指定は 0-based の INDEX、"
                        "それ以外は Track 名(既定: 先頭トラック)")
    p.add_argument("--model-name", dest="model_name", type=_model_name,
                   default=f"vpr2vmd {__version__}",
                   help="VMD に格納するモデル名(最大 20 バイト・Shift-JIS)")
    p.add_argument("--style", choices=STYLE_NAMES, default="pop",
                   help="リップモーションスタイルプリセット(開き量レンジ・タイミング・誇張を切り替える)")
    p.add_argument("--n-morph", dest="n_morph", action="store_true", default=False,
                   help="撥音に「ん」モーフを使う(既定 off)。--no-n-morph の対の明示形")
    p.add_argument("--no-n-morph", dest="n_morph", action="store_false",
                   help="撥音に「ん」モーフを使わず無音(閉口)に倒す(既定)。--n-morph の対")
    p.add_argument("--open-max", dest="open_max", type=_unit_float,
                   help="口の開き量の上限(0.0〜1.0。既定: プリセット値)")
    p.add_argument("--default-open", dest="default_open", type=_unit_float,
                   help="声量コントローラ曲線が無く、ベロシティが一様なときの既定開き量"
                        "(0.0〜1.0。既定: 開き量レンジ中央。開き量へ用いるときに --open-max で頭打ちする)")
    p.add_argument("--legato-max", dest="legato_max", type=_positive_float,
                   help="レガート間隙とみなす間隙長の上限(フレーム・正値。既定: 8.0)")
    p.add_argument("--valley-shallow", dest="valley_shallow", type=_unit_float,
                   help="レガート谷の谷係数の上限(浅い側・0.0〜1.0。既定: プリセット値)")
    p.add_argument("--valley-deep", dest="valley_deep", type=_unit_float,
                   help="レガート谷の谷係数の下限(深い側・0.0〜1.0。既定: プリセット値)")
    p.add_argument("--valley-slope", dest="valley_slope", type=_nonneg_float,
                   help="間隙長 1 フレームあたりの谷係数の減少(0 以上。既定: プリセット値)")
    p.add_argument("--coartic-overlap", dest="coartic_overlap", type=_positive_int,
                   help="協調調音の重なり上限(=基準長・1 以上。既定: プリセット値)")
    p.add_argument("--anticipation", dest="anticipation", type=_nonneg_int,
                   help="母音口形の先行準備フレーム数(0 以上・0 で無効。既定: プリセット値)")
    p.add_argument("--ref-bpm", dest="ref_bpm", type=_positive_float,
                   help="テンポ補正の基準テンポ(正値。既定: 120)")
    p.add_argument("--tempo-scale-min", dest="tempo_scale_min", type=_positive_unit_float,
                   help="テンポ補正の下げ止まり係数(0 超〜1.0。既定: 0.5)")
    add_common_arguments(
        p, version=__version__, quiet=False, output_help="出力 VMD(既定: <入力名>.vmd)",
        help_overrides={
            "--dry-run": "出力せず処理計画と診断を表示する",
            "--verbose": "通常実行でも処理計画と診断を標準出力へ表示する(出力 VMD は書く)",
            "--describe": "オプション定義とスタイルプリセット一覧の result を出して終了する"
                          "(vpr を読まない・入力不要の自己記述)",
        },
    )
    return p


_DERIVED_FROM_VALIDATOR = (None, None)

_DESCRIBE_TYPE_BY_DEST = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "track": ("str", None),
    "model_name": ("str", None),
    "style": ("enum", {"choices": list(STYLE_NAMES)}),
    "n_morph": ("flag", None),
    "open_max": _DERIVED_FROM_VALIDATOR,
    "default_open": _DERIVED_FROM_VALIDATOR,
    "legato_max": _DERIVED_FROM_VALIDATOR,
    "valley_shallow": _DERIVED_FROM_VALIDATOR,
    "valley_deep": _DERIVED_FROM_VALIDATOR,
    "valley_slope": _DERIVED_FROM_VALIDATOR,
    "coartic_overlap": _DERIVED_FROM_VALIDATOR,
    "anticipation": _DERIVED_FROM_VALIDATOR,
    "ref_bpm": _DERIVED_FROM_VALIDATOR,
    "tempo_scale_min": _DERIVED_FROM_VALIDATOR,
    "dry_run": ("flag", None),
    "verbose": ("flag", None),
}

_FIXED_DEFAULT_BY_DEST = {
    "legato_max": _DEFAULT_LEGATO_MAX,
    "ref_bpm": _DEFAULT_REF_BPM,
    "tempo_scale_min": _DEFAULT_TEMPO_SCALE_MIN,
}


def _describe_options(parser):
    return [
        option
        for dest, entry in _DESCRIBE_TYPE_BY_DEST.items()
        for option in describe_options(parser, {dest: entry}, defaults=_FIXED_DEFAULT_BY_DEST)
    ]


def _describe_presets():
    out = []
    for name in STYLE_NAMES:
        openness_params, gen = presets.resolve(name)
        out.append({
            "name": name,
            "values": {
                "open_max": openness_params.open_max,
                "default_open": openness_params.default_open,
                "valley_shallow": gen.legato_valley_shallow,
                "valley_deep": gen.legato_valley_deep,
                "valley_slope": gen.legato_valley_slope,
                "coartic_overlap": gen.coartic_overlap_max,
                "anticipation": gen.anticipation_frames,
            },
        })
    return out


def _print_plan(args, output: str, built: "_Built") -> None:
    resolved = built.resolved_params
    print(f"input: {args.input}")
    print(f"output: {output}")
    print(f"track: {built.track_index} {built.track_name}")
    print(f"style: {args.style}")
    print(f"n-morph: {'on (撥音→ん)' if args.n_morph else 'off (撥音→無音)'}")
    print(f"model-name: {args.model_name!r}")
    print(f"open-max: {resolved['open_max']}")
    print(f"default-open: {resolved['default_open']}")
    print(f"legato-max: {resolved['legato_max']}")
    print(
        f"valley(shallow/deep/slope): "
        f"{resolved['valley_shallow']}/{resolved['valley_deep']}/{resolved['valley_slope']}"
    )
    print(f"coartic-overlap: {resolved['coartic_overlap']}")
    print(f"anticipation: {resolved['anticipation']}")
    print(f"tempo(ref-bpm/scale-min): {resolved['ref_bpm']}/{resolved['tempo_scale_min']}")
    print(f"representative-bpm: {resolved['representative_bpm']}")


@dataclass
class _Diagnostics:
    adopted_notes: int
    mouth_events: int
    morph_keys: int
    open_source: str | None
    open_amounts: list[float]
    overlap: OverlapDiagnostics
    mouth_event: EventDiagnostics


@dataclass
class _Built:
    document: VmdDocument
    diagnostics: _Diagnostics
    track_index: int
    track_name: str
    resolved_params: dict


def _build(args, ctx):
    try:
        project, warnings = read(args.input)
    except Exception as e:
        return ctx.fail("not_vpr", f"入力を vpr として読めません: {type(e).__name__}: {e}", 1, field="input")
    _surface_warnings(warnings, ctx)
    if not project.tracks:
        return ctx.fail("no_tracks", "入力 vpr にトラックがありません", 1, field="input")
    try:
        track = select_track(project, args.track)
    except TrackSelectionError as e:
        return ctx.fail("bad_track", f"--track: {e}", 2, field="--track")
    track_index = next(i for i, t in enumerate(project.tracks) if t is track)

    adopted, overlap_diag = resolve_overlaps(collect_notes(track))
    openness_params, gen_params = presets.resolve(args.style, args.open_max, args.default_open)
    rep_bpm = timing.representative_bpm(adopted, project.tempos, project.resolution)
    ref_bpm = args.ref_bpm if args.ref_bpm is not None else _DEFAULT_REF_BPM
    tempo_scale_min = args.tempo_scale_min if args.tempo_scale_min is not None else _DEFAULT_TEMPO_SCALE_MIN
    gen_params = apply_tempo_correction(gen_params, rep_bpm, ref_bpm=ref_bpm, s_min=tempo_scale_min)
    overrides = {}
    if args.coartic_overlap is not None:
        overrides["coartic_overlap_max"] = args.coartic_overlap
    if args.anticipation is not None:
        overrides["anticipation_frames"] = args.anticipation
    if args.valley_shallow is not None:
        overrides["legato_valley_shallow"] = args.valley_shallow
    if args.valley_deep is not None:
        overrides["legato_valley_deep"] = args.valley_deep
    if args.valley_slope is not None:
        overrides["legato_valley_slope"] = args.valley_slope
    if overrides:
        gen_params = replace(gen_params, **overrides)
    open_by_note = loudness.open_amounts_from_loudness(
        track.parts,
        adopted,
        lo=openness_params.lo,
        hi=openness_params.hi,
        open_max=openness_params.open_max,
        gamma=openness_params.gamma,
    )
    if open_by_note is not None:
        open_source = "dynamics"
    else:
        velocities = [note.velocity for note in adopted]
        open_source = "default" if openness.uses_default_open(velocities) else "velocity"
        open_by_note = openness.open_amounts(
            velocities,
            lo=openness_params.lo,
            hi=openness_params.hi,
            open_max=openness_params.open_max,
            default_open=openness_params.default_open,
            gamma=openness_params.gamma,
        )
    legato_max = args.legato_max if args.legato_max is not None else _DEFAULT_LEGATO_MAX
    mouth_events, event_diag = build_mouth_events(
        adopted,
        project.tempos,
        project.resolution,
        use_n_morph=args.n_morph,
        open_by_note=open_by_note,
        legato_max_frames=legato_max,
    )
    morph_keys = generate_morph_keys(mouth_events, gen_params)

    document = VmdDocument(
        model_name_raw=args.model_name.encode("cp932").ljust(_VMD_MODEL_NAME_BYTES, b"\x00"),
        morph=morph_keys,
    )
    document = ensure_frame0_neutral_keys(document, sections=("morph",))
    document, normalize_warnings = normalize(document, sections=["morph"])
    _surface_normalize_warnings(normalize_warnings, ctx)
    diagnostics = _Diagnostics(
        adopted_notes=len(adopted),
        mouth_events=len(mouth_events),
        morph_keys=len(document.morph),
        open_source=open_source if adopted else None,
        open_amounts=list(open_by_note),
        overlap=overlap_diag,
        mouth_event=event_diag,
    )
    resolved_params = {
        "open_max": openness_params.open_max,
        "default_open": openness_params.default_open,
        "legato_max": legato_max,
        "valley_shallow": gen_params.legato_valley_shallow,
        "valley_deep": gen_params.legato_valley_deep,
        "valley_slope": gen_params.legato_valley_slope,
        "coartic_overlap": gen_params.coartic_overlap_max,
        "anticipation": gen_params.anticipation_frames,
        "ref_bpm": ref_bpm,
        "tempo_scale_min": tempo_scale_min,
        "representative_bpm": rep_bpm,
    }
    return _Built(document, diagnostics, track_index, track.name, resolved_params)


def _print_diagnostics(diag: _Diagnostics) -> None:
    print("--- 診断 ---")
    print(f"採用音符数: {diag.adopted_notes}")
    print(f"口形イベント数: {diag.mouth_events}")
    print(f"モーフキー数: {diag.morph_keys}")
    print(f"開き量の決定経路: {diag.open_source if diag.open_source is not None else 'なし'}")
    if diag.open_amounts:
        lo = min(diag.open_amounts)
        hi = max(diag.open_amounts)
        avg = sum(diag.open_amounts) / len(diag.open_amounts)
        print(f"開き量(最小/最大/平均): {lo:.3f}/{hi:.3f}/{avg:.3f}")
    print(f"母音未確定: {diag.mouth_event.vowel_undetermined}")
    print(f"重複音符 除外: {diag.overlap.excluded} 切り詰め: {diag.overlap.truncated}")
    symbols = diag.mouth_event.non_event_symbols
    if symbols:
        listed = " ".join(f"{sym}({symbols[sym]})" for sym in sorted(symbols))
        print(f"イベント外記号: {listed}")


def _open_amounts_stats(amounts):
    if not amounts:
        return None
    return {"min": min(amounts), "max": max(amounts), "mean": sum(amounts) / len(amounts)}


def _build_inspect(args, built: _Built) -> dict:
    diag = built.diagnostics
    return {
        "output": None,
        "input_kind": "vpr",
        "track_index": built.track_index,
        "track_name": built.track_name,
        "style": args.style,
        "n_morph": args.n_morph,
        "model_name": args.model_name,
        "params": built.resolved_params,
        "adopted_notes": diag.adopted_notes,
        "mouth_events": diag.mouth_events,
        "morph_keys": diag.morph_keys,
        "open_source": diag.open_source,
        "open_amounts": _open_amounts_stats(diag.open_amounts),
        "vowel_undetermined": diag.mouth_event.vowel_undetermined,
        "overlap_excluded": diag.overlap.excluded,
        "overlap_truncated": diag.overlap.truncated,
        "non_event_symbols": diag.mouth_event.non_event_symbols,
    }


def _valley_bounds_inverted(args) -> bool:
    _, gen = presets.resolve(args.style, args.open_max, args.default_open)
    shallow = args.valley_shallow if args.valley_shallow is not None else gen.legato_valley_shallow
    deep = args.valley_deep if args.valley_deep is not None else gen.legato_valley_deep
    return deep > shallow


_POSITION_KEYS = ("track_index", "part_index", "note_index", "related_note_index", "tick")


def _warn(ctx, code, message, *, section=None, **positions):
    ctx.warn(code, message, section=section, **{key: positions.get(key) for key in _POSITION_KEYS})


def _surface_warnings(warnings, ctx) -> None:
    seen = set()
    for w in warnings:
        if not ctx.machine:
            key = (w.code, w.message)
            if key in seen:
                continue
            seen.add(key)
        _warn(ctx, w.code, w.message, track_index=w.track_index, part_index=w.part_index,
              note_index=w.note_index, related_note_index=w.related_note_index, tick=w.tick)


_SURFACED_NORMALIZE_CODES = frozenset({"normalize-duplicate"})


def _surface_normalize_warnings(warnings, ctx) -> None:
    seen = set()
    for w in warnings:
        if w.code not in _SURFACED_NORMALIZE_CODES:
            continue
        if not ctx.machine:
            key = (w.code, w.section, w.message)
            if key in seen:
                continue
            seen.add(key)
        _warn(ctx, w.code, w.message, section=w.section)


def main(argv=None) -> int:
    return run_tool(
        argv,
        build_parser=_build_parser,
        describe=lambda parser: {"options": _describe_options(parser), "presets": _describe_presets()},
        run=_run,
    )


def _run(args, ctx) -> int:
    if args.input is None:
        return ctx.fail("bad_argument", "入力 vpr(input)が必要です", 2, field="input")

    output = args.output if args.output is not None else default_output_path(args.input, ".vmd")

    rejected = guard_output(output, overwrite=args.overwrite, fail=ctx.fail)
    if rejected is not None:
        return rejected

    if _valley_bounds_inverted(args):
        return ctx.fail("valley_bounds_inverted",
                        "谷係数の下限が上限を超えています(--valley-deep > --valley-shallow)", 2)

    if not os.path.isfile(args.input):
        return ctx.fail("input_not_found", f"入力 vpr が見つかりません: {args.input}", 1, field="input")

    built = _build(args, ctx)
    if isinstance(built, int):
        return built
    diag = built.diagnostics

    if diag.adopted_notes == 0:
        _warn(ctx, "no_adopted_notes", "対象トラックに有効な発音がありません")

    if (args.dry_run or args.verbose) and not ctx.machine:
        _print_plan(args, output, built)
        _print_diagnostics(diag)

    if args.dry_run:
        if ctx.machine:
            ctx.emitter.result(mode="inspect", **_build_inspect(args, built))
        return 0

    try:
        write_file(built.document, output)
    except Exception as e:
        return ctx.fail("write_failed", f"出力の書き込みに失敗: {e}", 3, field="--output", path=output)

    if ctx.machine:
        ctx.emitter.result(
            mode="convert", output=output, track_index=built.track_index, track_name=built.track_name,
            morph_keys=diag.morph_keys, adopted_notes=diag.adopted_notes, mouth_events=diag.mouth_events,
        )
    return 0
