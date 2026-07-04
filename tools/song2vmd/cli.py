"""song2vmd CLI(song2vmd.md §5)。

日本語の歌声音声からアニメ的口パクの口パク VMD を1コマンドで生成する独立 CLI。本ファイルは
引数解析・検証・出力先解決・上書きガードと `--dry-run` の空実行を実装する。実際の変換パイプライン
(vocal_analysis による音声読み込み・分離・認識・強弱RMS算出 → 口形イベント確定 → lipsync による
モーフキー生成 → vmd.io による VMD 出力)は `_run()` に未実装(`NotImplementedError`)。

終了コード(song2vmd.md 11章): 0 正常 / 1 入力不正 / 2 引数エラー(範囲・書式・上書きガード等) /
3 出力書き込み失敗 / 4 音声前段の外部依存の失敗(失敗ステージ明示) / 130 協調的な中断(Ctrl-C 等)。

`--machine`/`--describe` は構造化出力モード(song2vmd.md 5.5・12章)。標準出力を JSON Lines の
イベントストリーム専用にし、失敗も error イベントで理由を返す。共通の契約は
[CLI インターフェース規約](../../docs/conventions/cli-interface.md)が正、イベント送出は共有基盤
[cli_events](../../libs/cli_events/cli_events.md) を用いる。既定(非機械)の表示・終了コードは変えない。
"""

import argparse
import math
import os
import sys

from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    argparse_error_field,
    error_event,
)

from . import __version__
from . import presets as _presets

# 歌い方スタイルプリセット名(song2vmd.md 8.1)。具体値の解決は presets モジュールが持つ。
STYLE_NAMES = _presets.STYLE_NAMES

# --separate-vocals の実施方針(song2vmd.md 5.2)。
SEPARATE_VOCALS_MODES = ("auto", "always", "never")

# S1/S2 の登録アダプタの安定 id(vocal_analysis.md §8.2・§8.3 が正本)。選択肢の公開は利用先 CLI の
# 責務なので、現行の採用アダプタ id をここで公開する(採用アダプタの追加・変更は vocal_analysis.md
# §8.3 を先に更新してから、この一覧を追随させる)。
SEPARATOR_NAMES = ("audio-separator-htdemucs-ft",)
RECOGNIZER_NAMES = ("whisper-ctc-forcedalign",)

# VMD ヘッダのモデル名は固定 20 バイト・Shift-JIS(song2vmd.md 5.2・9章)。
_MODEL_NAME_MAX_BYTES = 20


def _model_name(text: str) -> str:
    """--model-name を検証する。cp932 で表現でき、20 バイト以内であること。"""
    try:
        encoded = text.encode("cp932")
    except UnicodeEncodeError:
        raise argparse.ArgumentTypeError(
            f"モデル名は Shift-JIS(cp932)で表現できる文字のみ: {text!r}"
        )
    if len(encoded) > _MODEL_NAME_MAX_BYTES:
        raise argparse.ArgumentTypeError(
            f"モデル名は cp932 で {_MODEL_NAME_MAX_BYTES} バイト以内"
            f"({len(encoded)} バイト): {text!r}"
        )
    return text


def _finite_float(text: str) -> float:
    """有限な float へ変換する(inf/nan を弾く)。範囲チェックは呼び出し側の検証関数で行う。"""
    v = float(text)  # 非数値は ValueError → argparse が exit 2 にする
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"有限な数値が必要: {text!r}")
    return v


def _unit_float(text: str) -> float:
    """0.0〜1.0 の有限 float(--open-max)。開き量は lipsync の開き量上限(0〜1)に対応する。"""
    v = _finite_float(text)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"0.0〜1.0 の範囲が必要: {text!r}")
    return v


def _positive_float(text: str) -> float:
    """正の有限 float(--intensity-curve)。強弱→開き量の累乗指数は 0 以下になり得ない。"""
    v = _finite_float(text)
    if v <= 0.0:
        raise argparse.ArgumentTypeError(f"正の数値が必要: {text!r}")
    return v


def _nonneg_float(text: str) -> float:
    """0 以上の有限 float(--max-duration)。0 は長尺分割の無効化(song2vmd.md 5.2)。"""
    v = _finite_float(text)
    if v < 0.0:
        raise argparse.ArgumentTypeError(f"0 以上の数値が必要: {text!r}")
    return v


def _nonneg_int(text: str) -> int:
    """0 以上の整数(--coarticulation・--anticipation・--min-hold)。"""
    v = int(text)  # 非整数は ValueError → argparse が exit 2 にする
    if v < 0:
        raise argparse.ArgumentTypeError(f"0 以上の整数が必要: {text!r}")
    return v


def _vowel_gain(text: str) -> tuple:
    """--vowel-gain の `a:i:u:e:o` を5要素 float タプルへ解析する(song2vmd.md 5.2・8.3)。

    各要素は開き量への倍率なので非負の有限値を要求する。撥音「ん」の倍率(1.0固定)は
    このタプルに含めず、lipsync へ渡す直前に補う(song2vmd.md 8.2)。
    """
    parts = text.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(f"--vowel-gain は a:i:u:e:o の5要素: {text!r}")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--vowel-gain は数値5要素: {text!r}")
    if not all(math.isfinite(v) for v in vals):
        raise argparse.ArgumentTypeError(f"--vowel-gain は有限値: {text!r}")
    if any(v < 0.0 for v in vals):
        raise argparse.ArgumentTypeError(f"--vowel-gain は非負(倍率): {text!r}")
    return vals


def _silence_threshold(text: str) -> tuple:
    """--silence-threshold の `ON:OFF` を (on, off) へ解析する(song2vmd.md 4.3・8.1)。

    正規化RMSのヒステリシスしきい値で、いずれも 0.0〜1.0。開いている状態から閉口へ入る
    下降側(ON)は、閉じている状態から開口へ戻る上昇側(OFF)より小さくなければならない
    (下降側 < 上昇側。無音ヒステリシスの意味上の制約)。
    """
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--silence-threshold は ON:OFF の2要素: {text!r}")
    try:
        on, off = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"--silence-threshold は数値2要素: {text!r}")
    if not (math.isfinite(on) and math.isfinite(off)):
        raise argparse.ArgumentTypeError(f"--silence-threshold は有限値: {text!r}")
    if not (0.0 <= on <= 1.0 and 0.0 <= off <= 1.0):
        raise argparse.ArgumentTypeError(f"--silence-threshold は 0.0〜1.0 の範囲: {text!r}")
    if not on < off:
        raise argparse.ArgumentTypeError(
            f"--silence-threshold は ON<OFF(下降側<上昇側)が必要: {text!r}"
        )
    return (on, off)


def _build_parser(machine: bool = False) -> argparse.ArgumentParser:
    # 構造化出力モード(--machine / --describe)は使用法エラーを error イベントへ振り替えるため、
    # SystemExit の代わりに ArgumentParseError を送出する MachineArgumentParser を使う(--help/--version は
    # error() を経由しないので影響を受けず、従来どおり SystemExit で短絡する)。
    # allow_abbrev=False: 仕様外の前置き省略形を受理しない(未知/省略形は exit 2)。
    cls = MachineArgumentParser if machine else argparse.ArgumentParser
    p = cls(prog="song2vmd", allow_abbrev=False)
    # input は nargs="?"(--describe を入力無しで成立させるため)。describe 以外の実行では main() が欠落を検査する。
    p.add_argument("input", nargs="?", help="入力音声ファイル(wav/mp3等)")
    p.add_argument("-o", "--output", help="出力VMD(既定: <入力名>.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先が入力と同一パスになる指定を許可する(別パスの既存ファイルは常に上書き)")
    p.add_argument("--model-name", dest="model_name", type=_model_name, default="",
                   help="VMDに格納するモデル名(最大20バイト・Shift-JIS)")
    p.add_argument("--style", choices=STYLE_NAMES, default="pop",
                   help="歌い方スタイルプリセット(開き量レンジ・タイミングを切り替える)")
    p.add_argument("--separate-vocals", dest="separate_vocals", choices=SEPARATE_VOCALS_MODES,
                   default="auto", help="ボーカル分離の実施方針(auto/always/never)")
    p.add_argument("--separator", choices=SEPARATOR_NAMES, default=SEPARATOR_NAMES[0],
                   help="S1ボーカル分離バックエンドの選択(vocal_analysisの登録アダプタ安定id)")
    p.add_argument("--recognizer", choices=RECOGNIZER_NAMES, default=RECOGNIZER_NAMES[0],
                   help="S2音素/母音認識バックエンドの選択(vocal_analysisの登録アダプタ安定id)")
    # --n-morph / --no-n-morph は既定 on の対(song2vmd.md 5.2)。dest=n_morph を共有する。
    p.add_argument("--n-morph", dest="n_morph", action="store_true", default=True,
                   help="撥音「ん」に「ん」モーフを使う(既定on)。--no-n-morphの対の明示形")
    p.add_argument("--no-n-morph", dest="n_morph", action="store_false",
                   help="撥音「ん」に「ん」モーフを使わず無音(閉口)に倒す。--n-morphの対")
    p.add_argument("--vowel-gain", dest="vowel_gain", type=_vowel_gain, default=(1.0, 1.0, 1.0, 1.0, 1.0),
                   help="母音別(あ/い/う/え/お)の開き量倍率(a:i:u:e:o)")
    # 既定はプリセット値。未指定センチネル(None)は presets.resolve がプリセットから解決する。
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
                   default=(0.06, 0.10), help="無音判定のヒステリシス開始/終了しきい値(ON:OFF)")
    p.add_argument("--max-duration", dest="max_duration", type=_nonneg_float, default=300.0,
                   help="長尺の自動分割境界(秒。0で無効)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力せず処理計画と診断を表示する(引数検証は dry-run でも実施する)")
    p.add_argument("--keep-intermediate", dest="keep_intermediate", action="store_true",
                   help="中間生成物(正規化PCM・分離WAV・認識結果)を残す(診断用)")
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                   help="詳細ログを標準エラーへ出す")
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


# --describe(12.2)の型/制約表。dest → (type, constraint)。help/default は parser の各 action から取る。
_D_UNIT = {"min": 0, "max": 1, "exclusive_min": False}             # 0〜1(開き量)
_D_NONNEG_INT = {"min": 0, "max": None, "exclusive_min": False}    # 0以上の整数
_D_POS_FLOAT = {"min": 0, "max": None, "exclusive_min": True}      # 正の数値(強弱指数)
_D_NONNEG_FLOAT = {"min": 0, "max": None, "exclusive_min": False}  # 0以上の数値(長尺分割境界)


def _cfield(name, mn, mx, ex):
    return {"name": name, "type": "float", "min": mn, "max": mx, "exclusive_min": ex}


_D_COMPOUND = {
    "vowel_gain": {"format": "a:i:u:e:o",
                   "fields": [_cfield(n, 0, None, False) for n in ("a", "i", "u", "e", "o")]},
    "silence_threshold": {"format": "ON:OFF",
                          "fields": [_cfield("ON", 0, 1, False), _cfield("OFF", 0, 1, False)]},
}

_D_TYPE = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "model_name": ("str", None),
    "style": ("enum", {"choices": list(STYLE_NAMES)}),
    "separate_vocals": ("enum", {"choices": list(SEPARATE_VOCALS_MODES)}),
    "separator": ("enum", {"choices": list(SEPARATOR_NAMES)}),
    "recognizer": ("enum", {"choices": list(RECOGNIZER_NAMES)}),
    "n_morph": ("flag", None),
    "vowel_gain": ("compound", _D_COMPOUND["vowel_gain"]),
    "open_max": ("float", _D_UNIT),
    "coarticulation": ("int", _D_NONNEG_INT),
    "anticipation": ("int", _D_NONNEG_INT),
    "min_hold": ("int", _D_NONNEG_INT),
    "intensity_curve": ("float", _D_POS_FLOAT),
    "silence_threshold": ("compound", _D_COMPOUND["silence_threshold"]),
    "max_duration": ("float", _D_NONNEG_FLOAT),
    "dry_run": ("flag", None),
    "keep_intermediate": ("flag", None),
    "verbose": ("flag", None),
    "quiet": ("flag", None),
}


def _describe_options(parser):
    """--describe の options を parser 定義から機械導出する(12.2)。順序は add_argument 順。

    各要素は {name, type, constraint, default, help}(キー5つ)。メタ/モード操作(--describe/--version/
    --help/--machine)は _D_TYPE に無いので除外。真偽フラグの否定形(--no-n-morph)は肯定形の長形式で
    既に載るのでスキップする。
    """
    options = []
    for action in parser._actions:
        dest = action.dest
        if dest not in _D_TYPE:
            continue
        type_, constraint = _D_TYPE[dest]
        if dest == "input":
            name = "input"
        else:
            # 肯定形の長形式を採る。--no-* だけの否定形 action はスキップ(肯定形で既に載る)。
            pos = [s for s in action.option_strings if s.startswith("--") and not s.startswith("--no-")]
            if not pos:
                continue
            name = pos[0]
        options.append({
            "name": name,
            "type": type_,
            "constraint": constraint,
            "default": action.default,
            "help": action.help,
        })
    return options


def _describe_presets():
    """--describe の presets を歌い方スタイルプリセットから導出する(12.2)。

    各要素は {name, values}。values は CLI で上書き可能な公開引数名(open_max・coarticulation・
    anticipation・min_hold)→ そのプリセットが与える値のオブジェクト(song2vmd.md 8.1・presetsモジュール)。
    """
    return [{"name": name, "values": dict(values)} for name, values in _presets.describe_values().items()]


def _default_output(input_path: str) -> str:
    # song2vmd.md 5.1: 既定出力は <入力名(拡張子なし)>.vmd。元の拡張子に依らず常に .vmd。
    base, _ = os.path.splitext(input_path)
    return base + ".vmd"


def _same_path(a: str, b: str) -> bool:
    """2 パスが同一ファイルを指すか。未存在でも realpath 比較で判定する。"""
    # 実ファイルが同一かを優先(symlink・大小無視 FS でも inode で一致判定)。出力先が
    # 未存在だと samefile が立たないので、symlink 解決した realpath で比較する。
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


def main(argv=None) -> int:
    """CLI エントリポイント。終了コードを返す(0/1/2/3/4/130。song2vmd.md 11章・12章)。"""
    # 人間向け標準エラーはロケール符号化で表せない文字でも UnicodeEncodeError で落とさない
    # (CLI インターフェース規約 §10)。
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    if argv is None:
        argv = sys.argv[1:]

    # 構造化出力モード判定(規約 §3)。解析前に argv で先取り(引数エラー時も出力チャネルを決めるため)。
    # --describe は --machine を要さない独立メタ操作。どちらかがあれば emitter を用意し、
    # MachineArgumentParser で使用法エラーも error イベントへ振り替える。emitter はバイナリ stdout へ
    # UTF-8 で書く(ロケール符号化非依存)。どちらも無ければ None(従来の人間向け経路)。
    machine = "--machine" in argv
    describe = "--describe" in argv
    emitter = EventEmitter(sys.stdout.buffer) if (machine or describe) else None

    def fail(code, message, exit_code, *, field=None, path=None):
        """失敗を報告して終了コードを返す(12.3)。構造化出力モードは error イベントでストリームを終端し、
        それ以外は理由を標準エラーへ1行出す(トレースバックは出さない)。"""
        if emitter is not None:
            emitter.error(**error_event(
                code=code, message=message, exit_code=exit_code, field=field, path=path))
        else:
            print(f"error: {message}", file=sys.stderr)
        return exit_code

    parser = _build_parser(machine or describe)
    try:
        args = parser.parse_args(argv)
    except ArgumentParseError as e:
        # 構造化出力モードの MachineArgumentParser は使用法エラーで例外を送出する(SystemExit の代わり)。
        return fail("bad_argument", e.message, 2, field=argparse_error_field(e.message))
    except SystemExit as e:
        # 非機械の使用法エラー(argparse が stderr へ出力済み・code 2)と、両モードの --help/--version
        # (メタ操作・code 0)。例外を握って終了コードへ変換する。
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)

    # 自己記述(12.2)。音声を読まず options/presets の result を出して終了する独立メタ操作。
    if args.describe:
        emitter.result(mode="describe", options=_describe_options(parser),
                       presets=_describe_presets())
        return 0
    # input は nargs="?"(--describe を入力無しで成立させるため)。describe 以外の実行では必須。
    if args.input is None:
        return fail("bad_argument", "入力音声(input)が必要です", 2, field="input")

    # 引数解析後の本体。KeyboardInterrupt(Ctrl-C 等)は協調的な中断(cancelled/130)として畳み、それ以外の
    # 想定外例外はトレースバックを漏らさず internal_error(理由1行 + 終了コード1)へ畳む(11章・12.3)。
    try:
        return _run(args, emitter, fail)
    except KeyboardInterrupt:
        return fail("cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return fail("internal_error", f"{type(e).__name__}: {e}", 1)


def _run(args, emitter, fail) -> int:
    """引数解析済みの本体(検証 → 空実行 or 変換)。失敗は fail() で終端する(12.3)。

    出力先解決・上書きガード・`--dry-run` の空実行のみを実装する。音声読み込み〜VMD書き出しの
    変換パイプライン(非 dry-run の実行、および `--machine --dry-run` の入力検査 result)は未実装。
    """
    output = args.output if args.output is not None else _default_output(args.input)

    # 上書きガード(song2vmd.md 5.3): 出力先が入力と同一パスになる指定だけを --overwrite 無しで拒否する。
    # 別パスの既存出力ファイルは対象にしない。同一パス判定を存在確認より先に置く(未存在でも入力上書きは弾く)。
    if not args.overwrite and _same_path(output, args.input):
        return fail("output_overwrites_input",
                    f"出力先が入力と同一パスです(--overwrite が必要): {output}", 2, field="--output")

    # --dry-run は出力を書かずに終える(song2vmd.md 5.2の「空実行」)。機械モードは result/error の
    # ちょうど1つで終端する契約(12章・CLI インターフェース規約 §4)のため、`--machine --dry-run` の
    # 入力検査 result は未実装のため、無イベントで正常終了させず internal_error へ畳む。
    if args.dry_run:
        if emitter is not None:
            raise NotImplementedError("song2vmd: --machine --dry-run の入力検査は未実装")
        return 0

    raise NotImplementedError("song2vmd: 音声処理パイプラインは未実装")
