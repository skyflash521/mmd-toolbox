import argparse
import sys
from pathlib import Path

import vocal_analysis_cli as _va_cli
from cli_events import (
    MachineArgumentParser,
    add_common_arguments,
    default_output_path,
    guard_output,
    run_tool,
)
from cli_options import CompoundValidator, RangeValidator, describe_options
from cli_resource_watch import ProgressWithResourceCheck, ResourceWatch, torch_gpu_warning
from vpr import write_file as _write_vpr

from . import __version__
from . import lyrics as _lyrics
from . import notes as _notes
from . import progress as _progress
from . import project as _project
from . import report as _report
from . import tempo as _tempo
from . import warning_text as _warning_text

# pip はコンソールスクリプトを extra の有無に依らず登録するので、extra 未導入でもここへ到達する。
try:
    from vocal_analysis.recognizer import RecognitionError

    from . import pipeline as _pipeline
    from . import pitch as _pitch
except ImportError as exc:
    _MISSING_DEPENDENCY = exc
else:
    _MISSING_DEPENDENCY = None


_MIN_VPR_REPRESENTABLE_BPM = 0.01
_TEMPO_BPM = RangeValidator(value_type="float", minimum=_MIN_VPR_REPRESENTABLE_BPM)

_WHOLE_NOTE_TICKS = 4 * _tempo.RESOLUTION
_MAX_DENOMINATOR_WITH_INTEGER_BEAT_TICKS = _WHOLE_NOTE_TICKS & -_WHOLE_NOTE_TICKS
_TIME_SIGNATURE = CompoundValidator(
    format="N/D",
    separator="/",
    elements=[("N", RangeValidator(value_type="int", minimum=1)),
              ("D", RangeValidator(value_type="int", minimum=1,
                                   maximum=_MAX_DENOMINATOR_WITH_INTEGER_BEAT_TICKS))],
    relation=("分母は2の冪", lambda values: values[1] & (values[1] - 1) == 0))


def _build_parser() -> argparse.ArgumentParser:
    p = MachineArgumentParser(prog="song2vpr", allow_abbrev=False)
    p.add_argument("input", nargs="?", help="入力音声ファイル(wav/mp3等)")
    p.add_argument("--lyrics", metavar="PATH",
                   help="歌詞テキストファイル(漢字かな交じり可。UTF-8 で読む)")
    p.add_argument("--tempo", metavar="BPM", type=_TEMPO_BPM,
                   help="テンポ(四分音符を1拍とした BPM)。未指定時は音声から推定し、"
                        "推定できなければ 120 を仮置きする")
    p.add_argument("--time-signature", dest="time_signature", metavar="N/D",
                   type=_TIME_SIGNATURE,
                   help="拍子(4/4 の形式。分母は2の冪)。未指定時は 4/4 を使う")
    _va_cli.add_arguments(p)
    p.add_argument("--keep-intermediate", dest="keep_intermediate", action="store_true",
                   help="中間生成物(正規化PCM・分離後ボーカルWAV・認識結果)を残す(診断用)")
    add_common_arguments(
        p, version=__version__, output_help="出力vpr(既定: <入力名>.vpr)",
        help_overrides={
            "--dry-run": "出力せず診断を表示する(引数検証は dry-run でも実施する)",
            "--verbose": "通常実行でも --dry-run と同じ診断を標準出力へ表示する(出力vprは書く)",
            "--describe": "オプション定義のresultを出して終了する(音声を読まない・入力不要の自己記述)",
        })
    return p


_DERIVED_FROM_VALIDATOR = (None, None)

_D_TYPE = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "lyrics": ("str", None),
    "tempo": _DERIVED_FROM_VALIDATOR,
    "time_signature": _DERIVED_FROM_VALIDATOR,
    **_va_cli.describe_type_table(),
    "dry_run": ("flag", None),
    "keep_intermediate": ("flag", None),
    "verbose": ("flag", None),
    "quiet": ("flag", None),
}

def _describe(parser):
    return {"options": describe_options(parser, _D_TYPE), "presets": []}


def main(argv=None) -> int:
    """戻り値はプロセスの終了コード。"""
    return run_tool(argv, build_parser=_build_parser, describe=_describe, run=_run)


def _run(args, ctx) -> int:
    emitter = ctx.emitter
    fail = ctx.fail
    if args.input is None:
        return fail("bad_argument", "入力音声(input)が必要です", 2, field="input")

    violation = _va_cli.validate(args)
    if violation is not None:
        name, reason = violation
        return fail("bad_argument", reason, 2, field=name)

    _va_cli.apply_device(args)

    output = args.output if args.output is not None else default_output_path(args.input, ".vpr")
    guard_failure = guard_output(output, overwrite=args.overwrite, fail=fail)
    if guard_failure is not None:
        return guard_failure

    if _MISSING_DEPENDENCY is not None:
        return fail("missing_dependency",
                    _va_cli.missing_dependency_message(_MISSING_DEPENDENCY), 4)

    progress_reporter = ctx.track_progress(_va_cli.build_router(
        machine=args.machine, quiet=args.quiet, emitter=emitter, stream=sys.stderr,
        extra_labels=_progress.EXTRA_STAGE_LABELS))

    def emit_warning(code, fields):
        message, human_text = _warning_text.warning_texts(code, fields)
        if emitter is not None:
            emitter.warning(code=code, message=message, **fields)
        else:
            ctx.warn(code, human_text)

    progress = ProgressWithResourceCheck(progress_reporter, ResourceWatch(emit_warning))
    torch_warning = torch_gpu_warning(args.device)
    if torch_warning is not None:
        emit_warning(*torch_warning)

    lyrics_text = None
    if args.lyrics is not None:
        try:
            lyrics_text = Path(args.lyrics).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as e:
            return fail("lyrics_unreadable", f"歌詞ファイルを読めません: {e}", 1,
                        field="--lyrics", path=args.lyrics)

    keep_intermediate_dir = f"{output}.intermediate" if args.keep_intermediate else None

    try:
        front = _pipeline.run(
            args.input, separate_vocals=args.separate_vocals, separator_name=args.separator,
            content_recognizer_model=_va_cli.resolve_recognizer_model(args),
            retry=args.recognizer_retry, forced_aligner=args.forced_aligner,
            sofa_aligner=_va_cli.resolve_sofa_config(args),
            english_katakana_method=args.english_katakana_method,
            chunking=_va_cli.resolve_chunking_policy(args),
            keep_intermediate_dir=keep_intermediate_dir, progress=progress)
    except Exception as e:
        failure = _va_cli.front_stage_failure(
            e, keep_intermediate_dir=keep_intermediate_dir,
            extra_labels=_progress.EXTRA_STAGE_LABELS)
        if failure is None:
            raise
        extra = {} if failure.stage is None else {"stage": failure.stage}
        return fail(failure.code, failure.message, failure.exit_code,
                    field=failure.field, path=failure.path, **extra)

    if front.forced_split:
        emit_warning("forced_split", {})

    progress.stage("f0")
    track = _pitch.estimate(front.vocal_pcm)

    progress.stage("notes")
    split = _notes.split(track, front.segments)
    try:
        annotated = _lyrics.annotate(split.notes, split.syllable_segments, front.rms,
                                     lyrics_text=lyrics_text)
    except RecognitionError as e:
        return fail("stage_failed", str(e), 4, stage="notes")
    estimate = _tempo.estimate(front.pcm, tempo_bpm=args.tempo,
                               time_signature=args.time_signature)
    if estimate.tempo_defaulted:
        emit_warning("tempo_defaulted", {})
    if annotated.diagnostics.kana_reading_ineffective:
        emit_warning("kana_reading_ineffective",
                     {"unconverted_chars": annotated.diagnostics.unconverted_chars,
                      "counted_chars": annotated.diagnostics.counted_chars})

    if not annotated.notes:
        emit_warning("no_notes", {})

    built = _project.build(annotated.notes, estimate, name=Path(output).stem)

    if not args.dry_run:
        progress.stage("write")
        try:
            _write_vpr(built.project, output)
        except OSError as e:
            return fail("write_failed", f"出力を書けません: {e}", 3,
                        field="--output", path=output)
        progress_reporter.close()
        progress_reporter.summary(f"完了 {output}")

    common = dict(tempo=estimate, duration_sec=front.duration_sec,
                  separated=(args.separate_vocals != "never"), backends=front.backends,
                  split=split.diagnostics, annotation=annotated.diagnostics,
                  build=built.diagnostics)
    if args.dry_run:
        fields = _report.inspect_fields(sample_rate=front.pcm.sample_rate,
                                        channels=front.pcm.samples.shape[1], **common)
        mode = "inspect"
    else:
        fields = _report.run_fields(output=output, **common)
        mode = "run"

    if emitter is not None:
        emitter.result(mode=mode, **fields)
    elif args.dry_run or args.verbose:
        progress_reporter.close()
        print(_report.report_text(fields, lyrics_given=args.lyrics is not None))
    return 0
