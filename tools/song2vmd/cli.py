"""song2vmd CLI。

日本語の歌声音声からアニメ的なリップモーション VMD を1コマンドで生成する独立 CLI。本ファイルは
引数解析・検証・出力先解決・上書きガードと、`pipeline`(vocal_analysis による音声読み込み・分離・
認識・強弱RMS算出 → 口形イベント確定 → lipsync によるモーフキー生成)の呼び出し・
`vmd.io` による VMD 出力・レポート/診断(`report`)・進捗表示(`progress`)への配線を実装する。

終了コード: 0 正常 / 1 入力不正 / 2 引数エラー(範囲・書式・上書きガード等) /
3 出力書き込み失敗 / 4 音声前段の外部依存の失敗(失敗ステージ明示)・追加依存の未導入 /
130 協調的な中断(Ctrl-C 等)。

`--machine`/`--describe` は構造化出力モード。標準出力を JSON Lines の
イベントストリーム専用にし、失敗も error イベントで理由を返す。イベント送出は共有基盤
cli_events を用いる。既定(非機械)の表示・終了コードは変えない。
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    argparse_error_field,
    emit_failure,
    install_sigbreak_handler,
)
from vmd import write_file as _vmd_write_file
from vocal_analysis import (
    DEFAULT_CONTENT_RECOGNIZER_MODEL,
    DEFAULT_ENGLISH_KATAKANA_METHOD,
    DEFAULT_FORCED_ALIGNER,
    ContentRecognizerModel,
    SofaAlignerConfig,
)

from . import __version__
from . import presets as _presets
from . import progress as _progress
from . import report as _report
from . import resource_watch as _resource_watch

# 追加依存(vocal-analysis extra)を要する取り込みだけをここへ集める。コンソールスクリプトは追加依存
# なしの導入でも登録されるため、依存が揃わない環境から起動されうる。取り込み失敗を例外のまま保持して
# モジュール自体の取り込みは成立させ、main() が理由1行へ畳めるようにする(失敗時に未定義になる名前は、
# main() が引数解析より前に終了するため参照されない)。追加依存を要しない配布物内のモジュール(共有
# ライブラリ・自身のモジュール)は、導入の破損と区別するため取り込み文をここへ入れない。
try:
    from vocal_analysis.io import AudioLoadError
    from vocal_analysis.recognizer import RecognitionError
    from vocal_analysis.separator import SeparationError

    from . import pipeline as _pipeline
except ImportError as exc:
    _MISSING_DEPENDENCY = exc
else:
    _MISSING_DEPENDENCY = None

# 歌い方スタイルプリセット名。具体値の解決は presets モジュールが持つ。
STYLE_NAMES = _presets.STYLE_NAMES

# --separate-vocals の実施方針。
SEPARATE_VOCALS_MODES = ("always", "never")

# S1 の登録アダプタの安定 id。選択肢の公開は利用先 CLI の責務なので、現行の採用アダプタ id を
# ここで公開する(vocal_analysis 側の採用アダプタの追加・変更にこの一覧を追随させる)。
# S2内容認識モデルは安定idでなく
# `--recognizer-model-id`/`--recognizer-model-revision`(ContentRecognizerModel)で選ぶ。
SEPARATOR_NAMES = ("audio-separator-htdemucs-ft",)

# S2強制アライメント段の登録アダプタの安定id。既定は
# DEFAULT_FORCED_ALIGNER(wav2vec2-ctc-forcedalign)で、SOFA選択時のみ--sofa-*系が必須になる。
FORCED_ALIGNER_NAMES = ("wav2vec2-ctc-forcedalign", "sofa-forcedalign")

# 英語カタカナ化フォールバックの変換方式。既定は DEFAULT_ENGLISH_KATAKANA_METHOD(arpakana)。
ENGLISH_KATAKANA_METHOD_NAMES = ("arpakana", "tinyllama-katakana-converter")

# 実行デバイスの選択。cpu はプロセスからGPU(CUDA)を隠して音声前段の全モデルと
# SOFAサブプロセスをCPUへ倒す(VRAM不足環境の回避手段)。既定 auto は環境から自動選択。
DEVICE_MODES = ("auto", "cpu")

# VMD ヘッダのモデル名は固定 20 バイト・Shift-JIS。
_MODEL_NAME_MAX_BYTES = 20


def _model_name(text: str) -> str:
    """--model-name を検証する。cp932 で表現でき、20 バイト以内であること。"""
    try:
        encoded = text.encode("cp932")
    except UnicodeEncodeError:
        raise argparse.ArgumentTypeError(
            f"モデル名は Shift-JIS(cp932)で表現できる文字のみ: {text!r}"
        ) from None
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
    """正の有限 float(--intensity-curve・--sofa-timeout)。累乗指数もタイムアウト秒数も 0 以下は無意味。"""
    v = _finite_float(text)
    if v <= 0.0:
        raise argparse.ArgumentTypeError(f"正の数値が必要: {text!r}")
    return v


def _nonneg_float(text: str) -> float:
    """0 以上の有限 float(--max-duration)。0 は長尺分割の無効化。"""
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
    """--vowel-gain の `a:i:u:e:o` を5要素 float タプルへ解析する。

    各要素はプリセットの母音別倍率へ乗算する微調整倍率なので非負の有限値を要求する。
    撥音はプリセット値のままで本引数の対象外。
    """
    parts = text.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(f"--vowel-gain は a:i:u:e:o の5要素: {text!r}")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--vowel-gain は数値5要素: {text!r}") from None
    if not all(math.isfinite(v) for v in vals):
        raise argparse.ArgumentTypeError(f"--vowel-gain は有限値: {text!r}")
    if any(v < 0.0 for v in vals):
        raise argparse.ArgumentTypeError(f"--vowel-gain は非負(倍率): {text!r}")
    return vals


def _silence_threshold(text: str) -> tuple:
    """--silence-threshold の `ON:OFF` を (on, off) へ解析する。

    正規化RMSのヒステリシスしきい値で、いずれも 0.0〜1.0。無音/継続の判定に使う下降側(ON)は
    上昇側(OFF)より小さくなければならない(下降側 < 上昇側。無音ヒステリシスの意味上の制約)。
    上昇側(OFF)は無音状態からの母音復帰自体の判定には使わない。
    """
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--silence-threshold は ON:OFF の2要素: {text!r}")
    try:
        on, off = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"--silence-threshold は数値2要素: {text!r}") from None
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
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
    p.add_argument("--model-name", dest="model_name", type=_model_name,
                   default=f"song2vmd {__version__}",
                   help="VMDに格納するモデル名(最大20バイト・Shift-JIS)")
    p.add_argument("--style", choices=STYLE_NAMES, default="pop",
                   help="歌い方スタイルプリセット(開き量レンジ・タイミングを切り替える)")
    p.add_argument("--separate-vocals", dest="separate_vocals", choices=SEPARATE_VOCALS_MODES,
                   default="always", help="ボーカル分離の実施方針(always/never)")
    p.add_argument("--separator", choices=SEPARATOR_NAMES, default=SEPARATOR_NAMES[0],
                   help="S1ボーカル分離バックエンドの選択(vocal_analysisの登録アダプタ安定id)")
    p.add_argument("--recognizer-model-id", dest="recognizer_model_id", default=None,
                   help=f"S2内容認識モデルの指定。未指定時は vocal_analysis の既定モデル"
                        f"(model_id={DEFAULT_CONTENT_RECOGNIZER_MODEL.model_id!r}・"
                        f"model_revision={DEFAULT_CONTENT_RECOGNIZER_MODEL.model_revision!r}固定)を使う")
    p.add_argument("--recognizer-model-revision", dest="recognizer_model_revision", default=None,
                   help="--recognizer-model-id のリビジョン指定(組で使う。--recognizer-model-id "
                        "指定時にこれを省略すると最新リビジョンを使う。--recognizer-model-id 自体を"
                        "省略した場合は本オプションは無視されず引数エラーになる)")
    # --recognizer-retry / --no-recognizer-retry は既定 on の対。dest=recognizer_retry を共有する。
    p.add_argument("--recognizer-retry", dest="recognizer_retry", action="store_true", default=True,
                   help="S2内容認識のトリガ式リトライ(エコー幻覚・反復幻覚。主モデル自身をプロンプト無しで"
                        "再認識する)を有効にする(既定on)。--no-recognizer-retryの対の明示形")
    p.add_argument("--no-recognizer-retry", dest="recognizer_retry", action="store_false",
                   help="S2内容認識のリトライを無効にする。--recognizer-retryの対")
    p.add_argument("--forced-aligner", dest="forced_aligner", choices=FORCED_ALIGNER_NAMES,
                   default=DEFAULT_FORCED_ALIGNER,
                   help="S2強制アライメント段のバックエンド選択(vocal_analysisの登録アダプタ安定id)。"
                        "sofa-forcedalign選択時は--sofa-python/--sofa-root/--sofa-checkpointが必須")
    p.add_argument("--sofa-python", dest="sofa_python", default=None,
                   help="SOFA専用venvのPython実行ファイルパス(--forced-aligner sofa-forcedalign時に必須)")
    p.add_argument("--sofa-root", dest="sofa_root", default=None,
                   help="SOFAリポジトリのルートパス(--forced-aligner sofa-forcedalign時に必須)")
    p.add_argument("--sofa-checkpoint", dest="checkpoint_path", default=None,
                   help="SOFAチェックポイント(.ckpt)ファイルパス(--forced-aligner sofa-forcedalign時に必須)")
    p.add_argument("--sofa-timeout", dest="sofa_timeout", type=_positive_float, default=300.0,
                   help="SOFAサブプロセス1回あたりのタイムアウト秒数")
    p.add_argument("--english-katakana-method", dest="english_katakana_method",
                   choices=ENGLISH_KATAKANA_METHOD_NAMES,
                   default=DEFAULT_ENGLISH_KATAKANA_METHOD,
                   help="英語カタカナ化フォールバックの変換方式選択(既定arpakana。"
                        "tinyllama-katakana-converterは生成モデルを使う選択式オプション)")
    p.add_argument("--device", choices=DEVICE_MODES, default="auto",
                   help="実行デバイスの選択。auto=環境から自動選択(GPU(CUDA)が利用可能ならGPU)、"
                        "cpu=GPUを使わずCPUで実行する(音声前段の全モデル・SOFAサブプロセスを含む。"
                        "VRAM不足環境の回避手段)")
    # --n-morph / --no-n-morph は既定 off の対。dest=n_morph を共有する。
    p.add_argument("--n-morph", dest="n_morph", action="store_true", default=False,
                   help="撥音に「ん」モーフを使う(既定off)。--no-n-morphの対の明示形")
    p.add_argument("--no-n-morph", dest="n_morph", action="store_false",
                   help="撥音に「ん」モーフを使わず無音(閉口)に倒す(既定)。--n-morphの対")
    p.add_argument("--vowel-gain", dest="vowel_gain", type=_vowel_gain, default=(1.0, 1.0, 1.0, 1.0, 1.0),
                   help="母音別(あ/い/う/え/お)の開き量微調整倍率(a:i:u:e:o)。"
                        "プリセットの母音別倍率へ要素ごとに乗算する")
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


# --describe の型/制約表。dest → (type, constraint)。help/default は parser の各 action から取る。
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
    "recognizer_model_id": ("str", None),
    "recognizer_model_revision": ("str", None),
    "recognizer_retry": ("flag", None),
    "forced_aligner": ("enum", {"choices": list(FORCED_ALIGNER_NAMES)}),
    "english_katakana_method": ("enum", {"choices": list(ENGLISH_KATAKANA_METHOD_NAMES)}),
    "device": ("enum", {"choices": list(DEVICE_MODES)}),
    "sofa_python": ("str", None),
    "sofa_root": ("str", None),
    "checkpoint_path": ("str", None),
    "sofa_timeout": ("float", _D_POS_FLOAT),
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
    """--describe の options を parser 定義から機械導出する。順序は add_argument 順。

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
    """--describe の presets を歌い方スタイルプリセットから導出する。

    各要素は {name, values}。values は CLI で上書き可能な公開引数名(open_max・coarticulation・
    anticipation・min_hold)→ そのプリセットが与える値のオブジェクト(値の正体は presets モジュール)。
    """
    return [{"name": name, "values": dict(values)} for name, values in _presets.describe_values().items()]


def _default_output(input_path: str) -> str:
    # 既定出力は <入力名(拡張子なし)>.vmd。元の拡張子に依らず常に .vmd。
    base, _ = os.path.splitext(input_path)
    return base + ".vmd"


def _fail(emitter, code, message, exit_code, *, field=None, path=None, stage=None):
    """失敗を報告して終了コードを返す。報告の分岐(error イベント / 人間向けのエラー行)と、
    標準出力へ書けない場合の後退は共有基盤 cli_events の emit_failure が持つ。stage は
    stage_failed のみが持つ追加キー(失敗ステージの安定id。progressと同じ語彙)で、error イベントへ
    だけ載る。main() が emitter 未確立の段階の中断・想定外例外でも呼べるよう、emitter を closure でなく
    引数に取る。"""
    extra = {} if stage is None else {"stage": stage}
    return emit_failure(emitter, code=code, message=message, exit_code=exit_code,
                        field=field, path=path, **extra)


def _missing_dependency_message(exc) -> str:
    """追加依存が未導入のときの理由1行。取り込めなかったモジュール名と導入コマンドを示す。"""
    # ModuleNotFoundError は不足モジュール名を name に持つ。持たない ImportError(名前の解決失敗等)は
    # 例外の文言をそのまま理由に使う。
    name = getattr(exc, "name", None)
    missing = repr(name) if name else str(exc)
    return (f"音声前段の依存パッケージ {missing} を取り込めません。"
            'song2vmd は追加依存を要します: pip install ".[vocal-analysis]"')


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
    # イベントを出せず人間向け1行へ後退する(その時点では emitter が未確立=None のため)。この並びなら、
    # ハンドラが有効になった時点で emitter は既に完成しており、以後どこで中断されても正しい経路で
    # 報告できる。
    try:
        if argv is None:
            argv = sys.argv[1:]

        # 構造化出力モード判定。解析前に argv で先取り(引数エラー時も出力チャネルを決めるため)。
        # --describe は --machine を要さない独立メタ操作。どちらかがあれば emitter を用意し、
        # MachineArgumentParser で使用法エラーも error イベントへ振り替える。emitter はバイナリ stdout へ
        # UTF-8 で書く(ロケール符号化非依存)。どちらも無ければ None(従来の人間向け経路)。
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
        # 追加依存が未導入なら本体はどの経路も実行できないため、引数解析より前にここで畳む
        # (--version/--help/--describe を含む全経路が対象)。emitter 構築後に置くことで、
        # 構造化出力モードでは error イベント、それ以外では標準エラーへの1行になる。
        if _MISSING_DEPENDENCY is not None:
            return fail("missing_dependency", _missing_dependency_message(_MISSING_DEPENDENCY), 4)

        # ArgumentParseError/SystemExit の捕捉は引数解析だけに閉じる(_run() 以下が送出しうる
        # SystemExit まで飲み込んで exit 0/2 に押し込めないため)。
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

        # 自己記述。音声を読まず options/presets の result を出して終了する独立メタ操作。
        if args.describe:
            emitter.result(mode="describe", options=_describe_options(parser),
                           presets=_describe_presets())
            return 0
        # input は nargs="?"(--describe を入力無しで成立させるため)。describe 以外の実行では必須。
        if args.input is None:
            return fail("bad_argument", "入力音声(input)が必要です", 2, field="input")

        # --forced-aligner sofa-forcedalign 選択時のみ --sofa-* を必須検証する(既定のwav2vec2経路は
        # --sofa-* が一切不要)。順序どおり走査し最初に見つかったNoneだけを報告する
        # (--recognizer-model-revision の組み合わせ検証と同じ「1回の呼び出しにつき最初の1件のみ報告」方式)。
        if args.forced_aligner == "sofa-forcedalign":
            for field, dest in (
                ("--sofa-python", "sofa_python"),
                ("--sofa-root", "sofa_root"),
                ("--sofa-checkpoint", "checkpoint_path"),
            ):
                if getattr(args, dest) is None:
                    return fail("bad_argument", f"{field} は --forced-aligner sofa-forcedalign 時に必須です",
                                2, field=field)

        # --device cpu はプロセスからGPU(CUDA)を隠して全段をCPUへ倒す。CUDAのデバイス集合は
        # プロセス内の最初の照会以降固定されるため、パイプライン起動前のここで設定しなければ効かない。
        # 環境は子プロセスへ継承されるため、SOFAサブプロセスにも同じ選択が効く。
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        return _run(args, emitter, fail)
    except KeyboardInterrupt:
        return _fail(emitter, "cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return _fail(emitter, "internal_error", f"{type(e).__name__}: {e}", 1)


def _resolve_content_recognizer_model(args):
    """--recognizer-model-id/--recognizer-model-revision から ContentRecognizerModel を
    組み立てる。--recognizer-model-id 未指定時は vocal_analysis の既定モデルを使う。"""
    if args.recognizer_model_id is None:
        return DEFAULT_CONTENT_RECOGNIZER_MODEL
    return ContentRecognizerModel(
        model_id=args.recognizer_model_id, model_revision=args.recognizer_model_revision)


def _resolve_sofa_aligner_config(args):
    """--sofa-* から SofaAlignerConfig を組み立てる。

    --forced-aligner が sofa-forcedalign 以外のときは None を返す(SofaAlignerConfig不要)。
    """
    if args.forced_aligner != "sofa-forcedalign":
        return None
    return SofaAlignerConfig(
        sofa_python=Path(args.sofa_python), sofa_root=Path(args.sofa_root),
        checkpoint_path=Path(args.checkpoint_path), timeout_sec=args.sofa_timeout)


def _report_params(args, openness, style_gen):
    """--dry-run の人間向けレポートに載せる、解決後の主要パラメータ。"""
    return {
        "open_lo": openness.open_lo, "open_hi": openness.open_hi, "open_max": openness.open_max,
        "intensity_curve": args.intensity_curve,
        "silence_threshold_on": args.silence_threshold[0], "silence_threshold_off": args.silence_threshold[1],
        "coarticulation": style_gen.coartic_overlap_max, "anticipation": style_gen.anticipation_frames,
        "min_hold": style_gen.min_hold_frames, "vowel_scale": style_gen.vowel_scale,
        "max_duration_sec": args.max_duration,
    }


def _torch_gpu_warning(device_mode):
    """GPU を使えない構成なら (code, message, human_text, fields) を返す。該当しなければ None。

    NVIDIA GPU を積んだ機材で torch の構成が原因で GPU を使えないときだけ返す。GPU の有無は
    NVIDIA ドライバが導入する nvidia-smi が PATH 上にあるかで判定する(存在を確かめるだけで実行は
    しない)。この条件が無いと、GPU を積んでいない機材と macOS で毎回警告が出る。`--device cpu` は
    利用者が CPU 実行を選んでいるので判定しない。torch を読み込めない環境でも判定しない。
    """
    if device_mode == "cpu" or shutil.which("nvidia-smi") is None:
        return None
    try:
        import torch
    except ImportError:
        return None
    version = torch.__version__
    if torch.version.cuda is None:
        message = "導入されている torch では GPU を扱えません"
        return (
            "cpu_only_torch", message,
            f"{message}(torch {version} は CPU 専用版)。CPU で処理します",
            {"torch_version": version},
        )
    if not torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES が環境から渡っている場合、利用者が見せる GPU を自分で絞った結果
        # として CUDA が使えないことがある。この構成で「別の CUDA のバージョンで入れ直せ」は
        # 効かない対処になるため判定しない(--device cpu 時は上の早期リターンで到達しない)。
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            return None
        message = "この GPU で使えない CUDA 版の torch が入っています"
        return (
            "cuda_unavailable", message,
            f"{message}(torch {version})。別の CUDA のバージョンで入れ直してください",
            {"torch_version": version},
        )
    return None


def _run(args, emitter, fail) -> int:
    """引数解析済みの本体(検証 → パイプライン実行 → 空実行/書き出し)。失敗は fail() で終端する。"""
    output = args.output if args.output is not None else _default_output(args.input)

    # 上書きガード: 出力先に既存ファイルがある場合は --overwrite 無しで拒否する。
    if not args.overwrite and os.path.exists(output):
        return fail("output_exists",
                    f"出力先に既存ファイルがあります(--overwrite が必要): {output}", 2, field="--output")

    # --recognizer-model-id 未指定なのに --recognizer-model-revision だけを指定するのは対象が無く無意味。
    if args.recognizer_model_id is None and args.recognizer_model_revision is not None:
        return fail("bad_argument",
                    "--recognizer-model-revision は --recognizer-model-id と組で指定する",
                    2, field="--recognizer-model-revision")

    content_recognizer_model = _resolve_content_recognizer_model(args)
    openness, style_gen = _presets.resolve(
        args.style, open_max=args.open_max, coarticulation=args.coarticulation,
        anticipation=args.anticipation, min_hold=args.min_hold, vowel_gain=args.vowel_gain)
    progress_reporter = _progress.ProgressReporter(
        machine=emitter is not None, quiet=args.quiet, emitter=emitter, stream=sys.stderr)

    def _emit_warning(code, message, human_text, fields):
        # 人間向けの警告はライブ進捗行と混線しないよう、書く前に close でライブ行を消す
        # (改行付きの1行として確定し、次の stage() でライブ行が下の行に再開する)。
        if emitter is not None:
            emitter.warning(code=code, message=message, **fields)
        else:
            progress_reporter.close()
            print(f"warning: {code}: {human_text}", file=sys.stderr)

    progress = _resource_watch.ProgressWithResourceCheck(
        progress_reporter, _resource_watch.ResourceWatch(_emit_warning))
    # GPU を使えない構成は処理を始める前に知らせる(数分かけてから伝えても手遅れなため)。
    torch_warning = _torch_gpu_warning(args.device)
    if torch_warning is not None:
        _emit_warning(*torch_warning)
    # 中間生成物は出力先の隣に <出力ファイル名>.intermediate/ を作って保存する。
    keep_intermediate_dir = f"{output}.intermediate" if args.keep_intermediate else None

    try:
        try:
            result = _pipeline.run(
                args.input, separate_vocals=args.separate_vocals, separator_name=args.separator,
                content_recognizer_model=content_recognizer_model,
                retry=args.recognizer_retry,
                max_duration_sec=args.max_duration,
                use_n_morph=args.n_morph, intensity_curve=args.intensity_curve,
                silence_on=args.silence_threshold[0], openness=openness, style_gen=style_gen,
                style_name=args.style, model_name=args.model_name,
                forced_aligner=args.forced_aligner, sofa_aligner=_resolve_sofa_aligner_config(args),
                english_katakana_method=args.english_katakana_method,
                keep_intermediate_dir=keep_intermediate_dir, progress=progress)
        except AudioLoadError as e:
            progress_reporter.close()
            exit_code = 4 if e.reason == "decoder_missing" else 1
            return fail(e.reason, str(e), exit_code, field="input")
        except SeparationError as e:
            progress_reporter.close()
            return fail("stage_failed", str(e), 4, stage="separate")
        except RecognitionError as e:
            progress_reporter.close()
            return fail("stage_failed", str(e), 4, stage="recognize")
        except _pipeline.IntermediateWriteError as e:
            progress_reporter.close()
            return fail("write_failed", str(e), 3, field="--keep-intermediate", path=keep_intermediate_dir)

        if result.diagnostics.low_dynamics:
            # --quiet は進捗表示だけを抑制し、警告は抑制しない。機械モードは
            # warning イベント、非機械モードは標準エラーへの1行を出す。ライブ行と
            # 警告行が同じ端末で連結・混線しないよう、書く前にライブ行を消す。
            if emitter is not None:
                emitter.warning(code="low_dynamics_suppressed",
                                message="曲のダイナミックレンジが小さいため、音量に基づく無音化を抑制しました")
            else:
                progress_reporter.close()
                print("warning: low_dynamics_suppressed: "
                      "曲のダイナミックレンジが小さいため、音量に基づく無音化を抑制しました",
                      file=sys.stderr)

        if result.diagnostics.forced_split:
            # 長尺分割で無音点が見つからず、最大チャンク長で強制分割した境界があったことを知らせる。
            # low_dynamics_suppressedと同じ扱い(--quietでも抑制しない、ライブ行を消してから出す)。
            if emitter is not None:
                emitter.warning(code="forced_split",
                                message="無音が見つからず最大チャンク長で強制分割しました")
            else:
                progress_reporter.close()
                print("warning: forced_split: 無音が見つからず最大チャンク長で強制分割しました",
                      file=sys.stderr)

        # --dry-run は出力を書かずに終える(空実行)。診断は実データから得る。
        if args.dry_run:
            if emitter is not None:
                emitter.result(mode="inspect", **_report.result_inspect_fields(
                    result.diagnostics, input_kind="audio",
                    sample_rate=result.sample_rate, channels=result.channels))
            else:
                # 標準出力へのレポートも同じ端末でライブ行と連結しうるため、書く前に消す
                # (low_dynamics 警告が無かった経路でも、ここで確実にライブ行を消す)。
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
        # 全終了経路(正常終了・--dry-run・パイプライン失敗・書き込み失敗・中断)でライブ行を必ず消す。
        # 正常終了は直前で明示的に close 済みだが、close は冪等なのでここでの再呼び出しも無害。
        progress_reporter.close()
