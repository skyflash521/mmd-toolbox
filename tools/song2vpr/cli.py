"""song2vpr CLI。

歌の音声ファイルから VOCALOID プロジェクトファイル(vpr)を1コマンドで生成する独立 CLI。本ファイルは
引数解析・検証・出力先解決・上書きガード・追加依存のガードを実装する。

終了コード: 0 正常 / 1 入力不正 / 2 引数エラー(範囲・書式・上書きガード等) /
3 出力書き込み失敗 / 4 音声前段の外部依存の失敗・歌詞のかな読みの失敗・追加依存の未導入
(復号器未検出と追加依存の未導入を除き失敗ステージを明示する) / 130 協調的な中断(Ctrl-C 等)。

`--machine`/`--describe` は構造化出力モード。標準出力を JSON Lines のイベントストリーム専用にし、
失敗も error イベントで理由を返す。イベント送出は共有基盤 cli_events を用いる。既定(非機械)の
表示・終了コードは変えない。
"""

import argparse
import os
import sys
from pathlib import Path

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
from vpr import write_file as _write_vpr

from . import __version__
from . import lyrics as _lyrics
from . import notes as _notes
from . import progress as _progress
from . import project as _project
from . import tempo as _tempo
from . import warning_text as _warning_text

# 追加依存(vocal-analysis extra)を要する取り込みだけをここへ集める。コンソールスクリプトは追加依存
# なしの導入でも登録されるため、依存が揃わない環境から起動されうる。取り込み失敗を例外のまま保持して
# モジュール自体の取り込みは成立させ、main() が理由1行へ畳めるようにする。未導入の判定はこの取り込みの
# 成否で行う。追加依存を要しない配布物内のモジュール(共有ライブラリ・自身のモジュール)は、導入の破損と
# 区別するため取り込み文をここへ入れない。
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
    from . import pitch as _pitch
except ImportError as exc:
    _MISSING_DEPENDENCY = exc
else:
    _MISSING_DEPENDENCY = None


# --tempo の下限は、vpr が格納できる粒度(BPM×100 の整数)で表せる最小の正値。上限は設けない
# (極端な値でも音符の要件は量子化の規則が保つ)。
_TEMPO = RangeValidator(value_type="float", minimum=0.01)

# --time-signature は「分子/分母」。分母は音価を表す2の冪で、1拍が整数の tick になる範囲に限る。
# 全音符の tick(4 × 分解能)を割り切る2の冪の最大が上限なので、分解能から導く(値を書くと分解能の
# 変更に追随しない)。2の冪であることは範囲の形で表せないので help に示す。
_WHOLE_NOTE_TICKS = 4 * _tempo.RESOLUTION
_MAX_DENOMINATOR = _WHOLE_NOTE_TICKS & -_WHOLE_NOTE_TICKS
_TIME_SIGNATURE = CompoundValidator(
    format="N/D",
    separator="/",
    elements=[("N", RangeValidator(value_type="int", minimum=1)),
              ("D", RangeValidator(value_type="int", minimum=1, maximum=_MAX_DENOMINATOR))],
    relation=("分母は2の冪", lambda values: values[1] & (values[1] - 1) == 0))


def _build_parser() -> argparse.ArgumentParser:
    # 使用法エラーは全経路で CLI 本体が引き取るため、SystemExit の代わりに ArgumentParseError を
    # 送出する MachineArgumentParser を使う(構造化出力モードは error イベントへ、それ以外は人間向けの
    # エラー行へ振り替える)。--help/--version は error() を経由しないので影響を受けず、SystemExit で
    # 短絡する。
    # allow_abbrev=False: 仕様外の前置き省略形を受理しない(未知/省略形は exit 2)。
    p = MachineArgumentParser(prog="song2vpr", allow_abbrev=False)
    # input は nargs="?"(--describe を入力無しで成立させるため)。describe 以外の実行では main() が欠落を検査する。
    p.add_argument("input", nargs="?", help="入力音声ファイル(wav/mp3等)")
    p.add_argument("-o", "--output", metavar="PATH", help="出力vpr(既定: <入力名>.vpr)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
    p.add_argument("--lyrics", metavar="PATH",
                   help="歌詞テキストファイル(漢字かな交じり可。UTF-8 で読む)")
    p.add_argument("--tempo", metavar="BPM", type=_TEMPO,
                   help="テンポ(四分音符を1拍とした BPM)。未指定時は音声から推定し、"
                        "推定できなければ 120 を仮置きする")
    p.add_argument("--time-signature", dest="time_signature", metavar="N/D",
                   type=_TIME_SIGNATURE,
                   help="拍子(4/4 の形式。分母は2の冪)。未指定時は音声から推定し、"
                        "推定できなければ 4/4 を仮置きする")
    _va_cli.add_arguments(p)
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力せず診断を表示する(引数検証は dry-run でも実施する)")
    p.add_argument("--keep-intermediate", dest="keep_intermediate", action="store_true",
                   help="中間生成物(正規化PCM・分離後ボーカルWAV・認識結果)を残す(診断用)")
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                   help="通常実行でも --dry-run と同じ診断を標準出力へ表示する(出力vprは書く)")
    p.add_argument("--quiet", dest="quiet", action="store_true",
                   help="進捗表示を抑制する(警告・診断・終了コードは抑制しない)")
    p.add_argument("--machine", action="store_true",
                   help="出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・"
                        "標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない")
    p.add_argument("--describe", action="store_true",
                   help="オプション定義のresultを出して終了する(音声を読まない・入力不要の自己記述)")
    p.add_argument("--version", action="version", version=f"song2vpr {__version__}",
                   help="バージョンを表示して終了する")
    return p


# --describe の型表。dest → (type, constraint)。検証子を持つ引数は型も制約もそこから取るので
# ここでは None を置く。help/default は parser の各 action から取る。共通引数群の分は共有側の
# 型情報をそのまま併せる(手書きで複製すると共有側の変更へ追随しない)。
_D_TYPE = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "lyrics": ("str", None),
    "tempo": (None, None),
    "time_signature": (None, None),
    **_va_cli.describe_type_table(),
    "dry_run": ("flag", None),
    "keep_intermediate": ("flag", None),
    "verbose": ("flag", None),
    "quiet": ("flag", None),
}


def _default_output(input_path: str) -> str:
    # 既定出力は <入力名(拡張子なし)>.vpr。元の拡張子に依らず常に .vpr。
    base, _ = os.path.splitext(input_path)
    return base + ".vpr"


def _fail(emitter, code, message, exit_code, *, field=None, path=None, stage=None):
    """失敗を報告して終了コードを返す。報告の分岐(error イベント / 人間向けのエラー行)と、
    標準出力へ書けない場合の後退は共有基盤 cli_events の emit_failure が持つ。stage は
    stage_failed のみが持つ追加キー(失敗ステージの安定id。progressと同じ語彙)で、error イベントへ
    だけ載る。main() が emitter 未確立の段階の中断・想定外例外でも呼べるよう、emitter を closure でなく
    引数に取る。"""
    extra = {} if stage is None else {"stage": stage}
    return emit_failure(emitter, code=code, message=message, exit_code=exit_code,
                        field=field, path=path, **extra)


def main(argv=None) -> int:
    """CLI エントリポイント。終了コードを返す(0/1/2/3/4/130)。"""
    # emitter は try の外側で初期化する: 下の except KeyboardInterrupt/Exception は、emitter 構築より
    # 前(argv 解決・--machine 判定 中)に中断・想定外例外が起きた場合でも参照できる必要があるため
    # (この区間の SIGINT は install_sigbreak_handler() の登録有無に関係なく既定ハンドラで常に有効)。
    emitter = None
    # main() の冒頭から本体実行までを1つの try で畳む。KeyboardInterrupt(CTRL_BREAK_EVENT の橋渡し先・
    # 通常の SIGINT の両方を含む)はどの時点で届いても取りこぼさず協調的な中断(cancelled/130)として
    # 畳み、それ以外の想定外例外はトレースバックを漏らさず internal_error(理由1行 + 終了コード1)へ畳む。
    #
    # try 内での並びに注意: emitter・fail を組んでから install_sigbreak_handler() を呼ぶ。逆順だと、
    # ハンドラ登録直後〜emitter 構築完了までの区間で中断された場合に --machine 指定でも構造化 cancelled
    # イベントを出せず人間向け1行へ後退する(その時点では emitter が未確立=None のため)。
    try:
        if argv is None:
            argv = sys.argv[1:]

        # 構造化出力モード判定。解析前に argv で先取り(引数エラー時も出力チャネルを決めるため)。
        # --describe は --machine を要さない独立メタ操作。どちらかがあれば emitter を用意する。
        # emitter はバイナリ stdout へ UTF-8 で書く(ロケール符号化非依存)。どちらも無ければ None で、
        # 失敗は人間向けのエラー行へ出る。
        machine = "--machine" in argv
        describe = "--describe" in argv
        emitter = EventEmitter(sys.stdout.buffer) if (machine or describe) else None

        def fail(code, message, exit_code, *, field=None, path=None, stage=None):
            return _fail(emitter, code, message, exit_code, field=field, path=path, stage=stage)

        # Windows の CTRL_BREAK_EVENT を下の except KeyboardInterrupt へ橋渡しする(他 OS では no-op)。
        install_sigbreak_handler()
        # 人間向け標準エラーはロケール符号化で表せない文字でも UnicodeEncodeError で落とさない。
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(errors="backslashreplace")
            except Exception:
                pass
        # ArgumentParseError/SystemExit の捕捉は引数解析だけに閉じる(_run() 以下が送出しうる
        # SystemExit まで飲み込んで exit 0/2 に押し込めないため)。
        parser = _build_parser()
        try:
            args = parser.parse_args(argv)
        except ArgumentParseError as e:
            # MachineArgumentParser は使用法エラーで例外を送出する(SystemExit の代わり)。fail() が
            # 構造化出力モードでは error イベント、それ以外では人間向けのエラー行1行へ振り替える。
            return fail("bad_argument", e.message, 2, field=argparse_error_field(e.message))
        except SystemExit as e:
            # 両モードの --help/--version(メタ操作・code 0)。使用法エラーは上の
            # ArgumentParseError で引き取るのでここには来ない。例外を握って終了コードへ変換する。
            code = e.code
            return code if isinstance(code, int) else (0 if code is None else 2)

        # 自己記述。音声を読まず options/presets の result を出して終了する独立メタ操作。
        # song2vpr はプリセットを持たないため presets は常に空配列。
        if args.describe:
            emitter.result(mode="describe", options=describe_options(parser, _D_TYPE), presets=[])
            return 0
        # input は nargs="?"(--describe を入力無しで成立させるため)。describe 以外の実行では必須。
        if args.input is None:
            return fail("bad_argument", "入力音声(input)が必要です", 2, field="input")

        # 共通引数群の組み合わせ検証(SOFA 経路の必須3項目・モデルリビジョンの単独指定)。
        violation = _va_cli.validate(args)
        if violation is not None:
            name, reason = violation
            return fail("bad_argument", reason, 2, field=name)

        # 実行デバイスの選択の適用。最初の GPU 照会より前でなければ効かないので、処理の起動前
        # のここで呼ぶ。
        _va_cli.apply_device(args)

        return _run(args, emitter, fail)
    except KeyboardInterrupt:
        return _fail(emitter, "cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return _fail(emitter, "internal_error", f"{type(e).__name__}: {e}", 1)


def _run(args, emitter, fail) -> int:
    """引数解析済みの本体(出力先の検査 → 上書きガード → 追加依存のガード)。失敗は fail() で終端する。"""
    output = args.output if args.output is not None else _default_output(args.input)

    # 出力先の検査: 既存ディレクトリは --overwrite でも書けないので、上書きの許可を促さず専用コードで
    # 先に拒否する。
    if os.path.isdir(output):
        return fail("output_is_directory",
                    f"出力先がディレクトリです(ファイルパスを指定): {output}",
                    2, field="--output", path=output)

    # 上書きガード: 出力先に既存ファイルがある場合は --overwrite 無しで拒否する。
    if not args.overwrite and os.path.exists(output):
        return fail("output_exists",
                    f"出力先に既存ファイルがあります(--overwrite が必要): {output}", 2, field="--output")

    # 追加依存が未導入なら以降はどれも実行できないため、引数の検証をすべて終えた後・処理を開始する
    # 直前のここで畳む。メタ操作(--help/--version/--describe)と引数エラーはここより前で完結し、追加
    # 依存の有無に依らずそれぞれの終了コードで終わる(オプションの定義・検証・自己記述のペイロードは
    # 追加依存を要さない取り込みだけで組み立てられる)。fail() 経由なので、構造化出力モードでは error
    # イベント、それ以外では標準エラーへの1行になる。
    if _MISSING_DEPENDENCY is not None:
        return fail("missing_dependency",
                    _va_cli.missing_dependency_message(_MISSING_DEPENDENCY), 4)

    progress_reporter = _progress.build_router(
        machine=args.machine, quiet=args.quiet, emitter=emitter, stream=sys.stderr)

    def emit_warning(code, fields):
        # 判定が返すのは安定コードと観測値だけなので、本文はここで組み立てる。人間向けの警告は
        # ライブ進捗行と混線しないよう、書く前に close でライブ行を消す(改行付きの1行として確定し、
        # 次の stage() でライブ行が下の行に再開する)。
        message, human_text = _warning_text.warning_texts(code, fields)
        if emitter is not None:
            emitter.warning(code=code, message=message, **fields)
        else:
            progress_reporter.close()
            print(f"warning: {code}: {human_text}", file=sys.stderr)

    # 全終了経路(正常終了・パイプラインの失敗・中断・想定外例外)でライブ行を必ず消す。失敗の理由行は
    # ライブ行を消した後に書く必要があるので各 except でも消すが、close は現在の段の記録を戻すだけで
    # 冪等なため、ここで重ねて呼んでも害はない。
    try:
        # 資源逼迫の判定は段の切り替わりで呼ぶ。共有側が用意する接続を使い、進捗の中継へ相乗りさせる。
        progress = ProgressWithResourceCheck(progress_reporter, ResourceWatch(emit_warning))
        # GPU を使えない構成は処理を始める前に知らせる(数分かけてから伝えても手遅れなため)。実行デバイスの
        # 選択はここより前(main)でプロセスへ適用済み。
        torch_warning = torch_gpu_warning(args.device)
        if torch_warning is not None:
            emit_warning(*torch_warning)

        # 歌詞ファイルは音声前段を始める前に読む(誤指定が分離・認識を終えてから露見しないため)。
        lyrics_text = None
        if args.lyrics is not None:
            try:
                # 先頭のバイト順マークは読み込み時に取り除く(かな読みへ持ち込まない)。
                lyrics_text = Path(args.lyrics).read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError) as e:
                progress_reporter.close()
                return fail("lyrics_unreadable", f"歌詞ファイルを読めません: {e}", 1,
                            field="--lyrics", path=args.lyrics)

        # 中間生成物は出力先の隣に <出力ファイル名>.intermediate を作って保存する。--dry-run でも
        # 抑制しない(抑制するのは最終 vpr の書き出しだけ)。
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
        except IntermediateReadError as e:
            # 内部生成ファイルの読み直し失敗。利用者入力を指す field は載せず、対象ファイルを path に載せる。
            progress_reporter.close()
            return fail("not_audio", str(e), 1, path=str(e.path))
        except AudioLoadError as e:
            # 復号器の未検出は入力の不備ではなく環境の不足なので、入力不正と別の終了コードで返す。
            progress_reporter.close()
            return fail(e.reason, str(e), 4 if e.reason == "decoder_missing" else 1, field="input")
        except StageExecutionError as e:
            # 共有側は失敗の要旨だけを持つので、どの工程かは利用者向けの工程名で示す。
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
            return fail("write_failed", str(e), 3, field="--keep-intermediate",
                        path=keep_intermediate_dir)

        if front.forced_split:
            emit_warning("forced_split", {})

        progress.stage("f0")
        track = _pitch.estimate(front.vocal_pcm)

        progress.stage("notes")
        try:
            annotated = _lyrics.annotate(_notes.split(track, front.segments), front.segments,
                                         front.rms, lyrics_text=lyrics_text)
        except RecognitionError as e:
            # かな読みは音符へ歌詞を割り当てる段の中で行うので、失敗が指す段は認識でなく音符化。
            progress_reporter.close()
            return fail("stage_failed", str(e), 4, stage="notes")
        estimate = _tempo.estimate(front.pcm, tempo_bpm=args.tempo,
                                   time_signature=args.time_signature)
        if estimate.tempo_defaulted:
            emit_warning("tempo_defaulted", {})
        if estimate.time_signature_defaulted:
            emit_warning("time_signature_defaulted", {})
        if annotated.diagnostics.kana_reading_ineffective:
            emit_warning("kana_reading_ineffective",
                         {"unconverted_chars": annotated.diagnostics.unconverted_chars,
                          "counted_chars": annotated.diagnostics.counted_chars})

        if not annotated.notes:
            emit_warning("no_notes", {})

        built = _project.build(annotated.notes, estimate, name=Path(output).stem)

        # --dry-run が抑制するのは最終 vpr の書き出しだけで、ここまでの工程は通常実行と同じに走る
        # (診断の各件数が実測値であるため)。
        if not args.dry_run:
            progress.stage("write")
            try:
                _write_vpr(built.project, output)
            except OSError as e:
                progress_reporter.close()
                return fail("write_failed", f"出力を書けません: {e}", 3,
                            field="--output", path=output)
            progress_reporter.close()
            progress_reporter.summary(f"完了 {output}")
        return 0
    finally:
        progress_reporter.close()
