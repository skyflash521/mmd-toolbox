import argparse
import os
import sys

import vocal_analysis_cli as _va_cli
from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    argparse_error_field,
    emit_failure,
    install_sigbreak_handler,
)
from cli_options import CompoundValidator, RangeValidator, describe_options
from cli_resource_watch import ProgressWithResourceCheck, ResourceWatch, torch_gpu_warning
from vmd import write_file as _vmd_write_file

from . import __version__
from . import presets as _presets
from . import progress as _progress
from . import report as _report
from . import resource_watch as _resource_watch

# pip はコンソールスクリプトを extra の有無に依らず登録するので、extra 未導入でもここへ到達する。
try:
    from vocal_analysis.front_stage import (
        IntermediateReadError,
        IntermediateWriteError,
        StageExecutionError,
    )
    from vocal_analysis.io import AudioLoadError
    from vocal_analysis.recognizer import RecognitionError
    from vocal_analysis.separator import SeparationError

    from . import pipeline as _pipeline
except ImportError as exc:
    _MISSING_DEPENDENCY = exc
else:
    _MISSING_DEPENDENCY = None

STYLE_NAMES = _presets.STYLE_NAMES

_VMD_MODEL_NAME_MAX_BYTES = 20

_LOW_DYNAMICS_MESSAGE = "曲のダイナミックレンジが小さいため、音量に基づく無音化を抑制しました"
_FORCED_SPLIT_MESSAGE = "無音が見つからず最大チャンク長で強制分割しました"


def _vmd_model_name(text: str) -> str:
    try:
        encoded = text.encode("cp932")
    except UnicodeEncodeError:
        raise argparse.ArgumentTypeError(
            f"モデル名は Shift-JIS(cp932)で表現できる文字のみ: {text!r}"
        ) from None
    if len(encoded) > _VMD_MODEL_NAME_MAX_BYTES:
        raise argparse.ArgumentTypeError(
            f"モデル名は cp932 で {_VMD_MODEL_NAME_MAX_BYTES} バイト以内"
            f"({len(encoded)} バイト): {text!r}"
        )
    return text


_unit_float = RangeValidator(value_type="float", minimum=0, maximum=1)
_positive_float = RangeValidator(value_type="float", minimum=0, exclusive_min=True)
_nonneg_int = RangeValidator(value_type="int", minimum=0)

_vowel_gain = CompoundValidator(
    format="a:i:u:e:o",
    elements=[(name, RangeValidator(value_type="float", minimum=0))
              for name in ("a", "i", "u", "e", "o")])

_silence_threshold = CompoundValidator(
    format="ON:OFF",
    elements=[("ON", _unit_float), ("OFF", _unit_float)],
    relation=("ON<OFF(下降側<上昇側)が必要", lambda values: values[0] < values[1]))


def _build_parser() -> argparse.ArgumentParser:
    p = MachineArgumentParser(prog="song2vmd", allow_abbrev=False)
    p.add_argument("input", nargs="?", help="入力音声ファイル(wav/mp3等)")
    p.add_argument("-o", "--output", help="出力VMD(既定: <入力名>.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
    p.add_argument("--model-name", dest="model_name", type=_vmd_model_name,
                   default=f"song2vmd {__version__}",
                   help="VMDに格納するモデル名(最大20バイト・Shift-JIS)")
    p.add_argument("--style", choices=STYLE_NAMES, default="pop",
                   help="歌い方スタイルプリセット(開き量レンジ・タイミングを切り替える)")
    _va_cli.add_arguments(p)
    p.add_argument("--n-morph", dest="n_morph", action="store_true", default=False,
                   help="撥音に「ん」モーフを使う(既定off)。--no-n-morphの対の明示形")
    p.add_argument("--no-n-morph", dest="n_morph", action="store_false",
                   help="撥音に「ん」モーフを使わず無音(閉口)に倒す(既定)。--n-morphの対")
    p.add_argument("--vowel-gain", dest="vowel_gain", type=_vowel_gain, default=(1.0, 1.0, 1.0, 1.0, 1.0),
                   help="母音別(あ/い/う/え/お)の開き量微調整倍率(a:i:u:e:o)。"
                        "プリセットの母音別倍率へ要素ごとに乗算する")
    p.add_argument("--open-max", dest="open_max", type=_unit_float,
                   help="口の開き量の上限(0.0〜1.0。既定: プリセット値)")
    p.add_argument("--coarticulation", dest="coarticulation", type=_nonneg_int,
                   help="隣接母音の協調調音の重なり長上限(フレーム。既定: プリセット値)")
    p.add_argument("--anticipation", dest="anticipation", type=_nonneg_int,
                   help="母音口形を音より先行させる最大フレーム数(既定: プリセット値)")
    p.add_argument("--min-hold", dest="min_hold", type=_nonneg_int,
                   help="最小保持フレーム。これより短いモーラは併合/間引き(既定: プリセット値)")
    p.add_argument("--intensity-curve", dest="intensity_curve", type=_positive_float, default=0.6,
                   help="強弱→開き量の非線形指数(累乗則)")
    p.add_argument("--silence-threshold", dest="silence_threshold", type=_silence_threshold,
                   default=(0.06, 0.10),
                   help="無音判定のヒステリシス開始/終了しきい値(ON:OFF。ON は OFF より小さい値)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力せず処理計画と診断を表示する(引数検証は dry-run でも実施する)")
    p.add_argument("--keep-intermediate", dest="keep_intermediate", action="store_true",
                   help="中間生成物(正規化PCM・分離後ボーカルWAV・認識結果)を残す(診断用)")
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                   help="通常実行でも --dry-run と同じ診断レポートを標準出力へ表示する(出力VMDは書く)")
    p.add_argument("--quiet", dest="quiet", action="store_true",
                   help="進捗表示を抑制する(警告・診断・終了コードは抑制しない)")
    p.add_argument("--machine", action="store_true",
                   help="出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・"
                        "標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない")
    p.add_argument("--describe", action="store_true",
                   help="オプション定義とプリセット一覧のresultを出して終了する"
                        "(音声を読まない・入力不要の自己記述)")
    p.add_argument("--version", action="version", version=f"song2vmd {__version__}",
                   help="バージョンを表示して終了する")
    return p


_DERIVED_FROM_VALIDATOR = (None, None)

_DESCRIBE_TYPE_TABLE = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "model_name": ("str", None),
    "style": ("enum", {"choices": list(STYLE_NAMES)}),
    **_va_cli.describe_type_table(),
    "n_morph": ("flag", None),
    "vowel_gain": _DERIVED_FROM_VALIDATOR,
    "open_max": _DERIVED_FROM_VALIDATOR,
    "coarticulation": _DERIVED_FROM_VALIDATOR,
    "anticipation": _DERIVED_FROM_VALIDATOR,
    "min_hold": _DERIVED_FROM_VALIDATOR,
    "intensity_curve": _DERIVED_FROM_VALIDATOR,
    "silence_threshold": _DERIVED_FROM_VALIDATOR,
    "dry_run": ("flag", None),
    "keep_intermediate": ("flag", None),
    "verbose": ("flag", None),
    "quiet": ("flag", None),
}


def _describe_presets():
    return [{"name": name, "values": dict(values)} for name, values in _presets.describe_values().items()]


def _default_output(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return base + ".vmd"


def _fail(emitter, code, message, exit_code, *, field=None, path=None, stage=None):
    extra = {} if stage is None else {"stage": stage}
    return emit_failure(emitter, code=code, message=message, exit_code=exit_code,
                        field=field, path=path, **extra)


def main(argv=None) -> int:
    """戻り値はプロセスの終了コード。"""
    emitter = None
    try:
        if argv is None:
            argv = sys.argv[1:]

        machine = "--machine" in argv
        describe = "--describe" in argv
        emitter = EventEmitter(sys.stdout.buffer) if (machine or describe) else None

        def fail(code, message, exit_code, *, field=None, path=None, stage=None):
            return _fail(emitter, code, message, exit_code, field=field, path=path, stage=stage)

        install_sigbreak_handler()
        # ロケール符号化で表せない文字を標準エラーへ書くと UnicodeEncodeError になる。
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
            emitter.result(mode="describe", options=describe_options(parser, _DESCRIBE_TYPE_TABLE),
                           presets=_describe_presets())
            return 0
        if args.input is None:
            return fail("bad_argument", "入力音声(input)が必要です", 2, field="input")

        violation = _va_cli.validate(args)
        if violation is not None:
            name, reason = violation
            return fail("bad_argument", reason, 2, field=name)

        _va_cli.apply_device(args)

        return _run(args, emitter, fail)
    except KeyboardInterrupt:
        return _fail(emitter, "cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return _fail(emitter, "internal_error", f"{type(e).__name__}: {e}", 1)


def _report_params(args, openness, style_gen):
    return {
        "open_lo": openness.open_lo, "open_hi": openness.open_hi, "open_max": openness.open_max,
        "intensity_curve": args.intensity_curve,
        "silence_threshold_on": args.silence_threshold[0], "silence_threshold_off": args.silence_threshold[1],
        "coarticulation": style_gen.coartic_overlap_max, "anticipation": style_gen.anticipation_frames,
        "min_hold": style_gen.min_hold_frames, "vowel_scale": style_gen.vowel_scale,
        "max_duration_sec": args.max_duration,
    }


def _run(args, emitter, fail) -> int:
    output = args.output if args.output is not None else _default_output(args.input)

    if os.path.isdir(output):
        return fail("output_is_directory",
                    f"出力先がディレクトリです(ファイルパスを指定): {output}",
                    2, field="--output", path=output)

    if not args.overwrite and os.path.exists(output):
        return fail("output_exists",
                    f"出力先に既存ファイルがあります(--overwrite が必要): {output}", 2, field="--output")

    if _MISSING_DEPENDENCY is not None:
        return fail("missing_dependency",
                    _va_cli.missing_dependency_message(_MISSING_DEPENDENCY), 4)

    content_recognizer_model = _va_cli.resolve_recognizer_model(args)
    openness, style_gen = _presets.resolve(
        args.style, open_max=args.open_max, coarticulation=args.coarticulation,
        anticipation=args.anticipation, min_hold=args.min_hold, vowel_gain=args.vowel_gain)
    progress_reporter = _progress.build_router(
        machine=emitter is not None, quiet=args.quiet, emitter=emitter, stream=sys.stderr)

    def _emit_warning(code, message, human_text, fields):
        if emitter is not None:
            emitter.warning(code=code, message=message, **fields)
        else:
            progress_reporter.close()
            print(f"warning: {code}: {human_text}", file=sys.stderr)

    def _emit_observed_warning(code, fields):
        message, human_text = _resource_watch.warning_texts(code, fields)
        _emit_warning(code, message, human_text, fields)

    progress = ProgressWithResourceCheck(
        progress_reporter, ResourceWatch(_emit_observed_warning))
    torch_warning = torch_gpu_warning(args.device)
    if torch_warning is not None:
        _emit_observed_warning(*torch_warning)
    keep_intermediate_dir = f"{output}.intermediate" if args.keep_intermediate else None

    try:
        try:
            result = _pipeline.run(
                args.input, separate_vocals=args.separate_vocals, separator_name=args.separator,
                content_recognizer_model=content_recognizer_model,
                retry=args.recognizer_retry,
                chunking=_va_cli.resolve_chunking_policy(args),
                use_n_morph=args.n_morph, intensity_curve=args.intensity_curve,
                silence_on=args.silence_threshold[0], openness=openness, style_gen=style_gen,
                style_name=args.style, model_name=args.model_name,
                forced_aligner=args.forced_aligner, sofa_aligner=_va_cli.resolve_sofa_config(args),
                english_katakana_method=args.english_katakana_method,
                keep_intermediate_dir=keep_intermediate_dir, progress=progress)
        except IntermediateReadError as e:
            progress_reporter.close()
            return fail("not_audio", str(e), 1, path=str(e.path))
        except AudioLoadError as e:
            progress_reporter.close()
            exit_code = 4 if e.reason == "decoder_missing" else 1
            return fail(e.reason, str(e), exit_code, field="input")
        except StageExecutionError as e:
            progress_reporter.close()
            return fail("stage_failed", f"{_progress.stage_label(e.stage)}に失敗しました: {e}",
                        4, stage=e.stage)
        except SeparationError as e:
            progress_reporter.close()
            return fail("stage_failed", str(e), 4, stage="separate")
        except RecognitionError as e:
            progress_reporter.close()
            return fail("stage_failed", str(e), 4, stage="recognize")
        except IntermediateWriteError as e:
            progress_reporter.close()
            return fail("write_failed", str(e), 3, field="--keep-intermediate", path=keep_intermediate_dir)

        if result.diagnostics.low_dynamics:
            _emit_warning("low_dynamics_suppressed", _LOW_DYNAMICS_MESSAGE, _LOW_DYNAMICS_MESSAGE, {})

        if result.diagnostics.forced_split:
            _emit_warning("forced_split", _FORCED_SPLIT_MESSAGE, _FORCED_SPLIT_MESSAGE, {})

        if args.dry_run:
            if emitter is not None:
                emitter.result(mode="inspect", **_report.result_inspect_fields(
                    result.diagnostics, input_kind="audio",
                    sample_rate=result.sample_rate, channels=result.channels))
            else:
                progress_reporter.close()
                params = _report_params(args, openness, style_gen)
                sys.stdout.write(_report.render_report_text(result.diagnostics, params))
            return 0

        progress.stage("write")
        try:
            _vmd_write_file(result.document, output)
        except OSError as e:
            progress_reporter.close()
            return fail("write_failed", str(e), 3, field="--output", path=output)

        progress_reporter.close()
        progress_reporter.summary(f"完了 {output}")

        if emitter is None and args.verbose:
            params = _report_params(args, openness, style_gen)
            sys.stdout.write(_report.render_report_text(result.diagnostics, params))

        if emitter is not None:
            emitter.result(mode="run", **_report.result_run_fields(result.diagnostics, output=output))
        return 0
    finally:
        progress_reporter.close()
