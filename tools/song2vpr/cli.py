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

import vocal_analysis_cli as _va_cli
from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    argparse_error_field,
    emit_failure,
    install_sigbreak_handler,
)
from cli_options import describe_options

from . import __version__

# 追加依存(vocal-analysis extra)を要する取り込みだけをここへ集める。コンソールスクリプトは追加依存
# なしの導入でも登録されるため、依存が揃わない環境から起動されうる。取り込み失敗を例外のまま保持して
# モジュール自体の取り込みは成立させ、main() が理由1行へ畳めるようにする。未導入の判定はこの取り込みの
# 成否で行う。追加依存を要しない配布物内のモジュール(共有ライブラリ・自身のモジュール)は、導入の破損と
# 区別するため取り込み文をここへ入れない。
try:
    from vocal_analysis import front_stage as _front_stage  # noqa: F401
except ImportError as exc:
    _MISSING_DEPENDENCY = exc
else:
    _MISSING_DEPENDENCY = None


def _build_parser() -> argparse.ArgumentParser:
    # 使用法エラーは全経路で CLI 本体が引き取るため、SystemExit の代わりに ArgumentParseError を
    # 送出する MachineArgumentParser を使う(構造化出力モードは error イベントへ、それ以外は人間向けの
    # エラー行へ振り替える)。--help/--version は error() を経由しないので影響を受けず、SystemExit で
    # 短絡する。
    # allow_abbrev=False: 仕様外の前置き省略形を受理しない(未知/省略形は exit 2)。
    p = MachineArgumentParser(prog="song2vpr", allow_abbrev=False)
    # input は nargs="?"(--describe を入力無しで成立させるため)。describe 以外の実行では main() が欠落を検査する。
    p.add_argument("input", nargs="?", help="入力音声ファイル(wav/mp3等)")
    p.add_argument("-o", "--output", help="出力vpr(既定: <入力名>.vpr)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
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

    # 音声前段以降の処理経路をまだ持たないため、ガードをすべて通った実行は何も書かずに 0 を返す。
    return 0
