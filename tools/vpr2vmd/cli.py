import argparse
import inspect
import os
import sys
from dataclasses import dataclass, replace

from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    argparse_error_field,
    emit_failure,
    install_sigbreak_handler,
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
    p.add_argument("-o", "--output", help="出力 VMD(既定: <入力名>.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
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
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力せず処理計画と診断を表示する")
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                   help="通常実行でも処理計画と診断を標準出力へ表示する(出力 VMD は書く)")
    p.add_argument("--machine", action="store_true",
                   help="出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・"
                        "標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない")
    p.add_argument("--describe", action="store_true",
                   help="オプション定義とスタイルプリセット一覧の result を出して終了する"
                        "(vpr を読まない・入力不要の自己記述)")
    p.add_argument("--version", action="version", version=f"vpr2vmd {__version__}",
                   help="バージョンを表示して終了する")
    return p


def _default_output(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return base + ".vmd"


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

_FIXED_DEFAULT_BY_OPTION = {
    "--legato-max": _DEFAULT_LEGATO_MAX,
    "--ref-bpm": _DEFAULT_REF_BPM,
    "--tempo-scale-min": _DEFAULT_TEMPO_SCALE_MIN,
}


def _describe_options(parser):
    options = describe_options(parser, _DESCRIBE_TYPE_BY_DEST)
    for option in options:
        if option["name"] in _FIXED_DEFAULT_BY_OPTION:
            option["default"] = _FIXED_DEFAULT_BY_OPTION[option["name"]]
    return options


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


def _build(args, emitter, fail):
    try:
        project, warnings = read(args.input)
    except Exception as e:
        return fail("not_vpr", f"入力を vpr として読めません: {type(e).__name__}: {e}", 1, field="input")
    _surface_warnings(warnings, emitter)
    if not project.tracks:
        return fail("no_tracks", "入力 vpr にトラックがありません", 1, field="input")
    try:
        track = select_track(project, args.track)
    except TrackSelectionError as e:
        return fail("bad_track", f"--track: {e}", 2, field="--track")
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
    _surface_normalize_warnings(normalize_warnings, emitter)
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


def _surface_warnings(warnings, emitter) -> None:
    if emitter is not None:
        for w in warnings:
            emitter.warning(
                code=w.code, message=w.message, section=None,
                track_index=w.track_index, part_index=w.part_index, note_index=w.note_index,
                related_note_index=w.related_note_index, tick=w.tick,
            )
        return
    seen = set()
    for w in warnings:
        key = (w.code, w.message)
        if key in seen:
            continue
        seen.add(key)
        print(f"warning: {w.code}: {w.message}", file=sys.stderr)


_SURFACED_NORMALIZE_CODES = frozenset({"normalize-duplicate"})


def _surface_normalize_warnings(warnings, emitter) -> None:
    warnings = [w for w in warnings if w.code in _SURFACED_NORMALIZE_CODES]
    if emitter is not None:
        for w in warnings:
            emitter.warning(
                code=w.code, message=w.message, section=w.section,
                track_index=None, part_index=None, note_index=None,
                related_note_index=None, tick=None,
            )
        return
    seen = set()
    for w in warnings:
        key = (w.code, w.section, w.message)
        if key in seen:
            continue
        seen.add(key)
        print(f"warning: {w.code}: {w.message}({w.section})", file=sys.stderr)


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
            return fail("bad_argument", e.message, 2, field=argparse_error_field(e.message))
        except SystemExit as e:
            code = e.code
            return code if isinstance(code, int) else (0 if code is None else 2)

        if args.describe:
            emitter.result(mode="describe", options=_describe_options(parser),
                           presets=_describe_presets())
            return 0
        if args.input is None:
            return fail("bad_argument", "入力 vpr(input)が必要です", 2, field="input")

        return _run(args, emitter, fail)
    except KeyboardInterrupt:
        return _fail(emitter, "cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return _fail(emitter, "internal_error", f"{type(e).__name__}: {e}", 1)


def _run(args, emitter, fail) -> int:
    output = args.output if args.output is not None else _default_output(args.input)

    if os.path.isdir(output):
        return fail("output_is_directory",
                    f"出力先がディレクトリです(ファイルパスを指定): {output}",
                    2, field="--output", path=output)

    if not args.overwrite and os.path.exists(output):
        return fail("output_exists",
                    f"出力先に既存ファイルがあります(--overwrite が必要): {output}", 2, field="--output")

    if _valley_bounds_inverted(args):
        return fail("valley_bounds_inverted",
                    "谷係数の下限が上限を超えています(--valley-deep > --valley-shallow)", 2)

    if not os.path.isfile(args.input):
        return fail("input_not_found", f"入力 vpr が見つかりません: {args.input}", 1, field="input")

    built = _build(args, emitter, fail)
    if isinstance(built, int):
        return built
    diag = built.diagnostics

    if diag.adopted_notes == 0:
        if emitter is not None:
            emitter.warning(
                code="no_adopted_notes", message="対象トラックに有効な発音がありません", section=None,
                track_index=None, part_index=None, note_index=None, related_note_index=None, tick=None,
            )
        else:
            print("warning: no_adopted_notes: 対象トラックに有効な発音がありません", file=sys.stderr)

    if (args.dry_run or args.verbose) and emitter is None:
        _print_plan(args, output, built)
        _print_diagnostics(diag)

    if args.dry_run:
        if emitter is not None:
            emitter.result(mode="inspect", **_build_inspect(args, built))
        return 0

    try:
        write_file(built.document, output)
    except OSError as e:
        return fail("write_failed", f"出力の書き込みに失敗: {e}", 3, field="--output", path=output)

    if emitter is not None:
        emitter.result(
            mode="convert", output=output, track_index=built.track_index, track_name=built.track_name,
            morph_keys=diag.morph_keys, adopted_notes=diag.adopted_notes, mouth_events=diag.mouth_events,
        )
    return 0
