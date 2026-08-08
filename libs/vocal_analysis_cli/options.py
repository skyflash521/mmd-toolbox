"""音声前段を CLI から駆動するための共通引数群の定義・検証・設定への解決。"""

import os
from pathlib import Path
from typing import get_args

from cli_options import RangeValidator
from vocal_analysis import (
    DEFAULT_CONTENT_RECOGNIZER_MODEL,
    DEFAULT_ENGLISH_KATAKANA_METHOD,
    DEFAULT_FORCED_ALIGNER,
    DEFAULT_SEPARATOR,
    SEPARATOR_IDS,
    ChunkingPolicy,
    ContentRecognizerModel,
    EnglishKatakanaMethod,
    ForcedAlignerId,
    SofaAlignerConfig,
)

# 分離の実施方針と実行デバイスは、音声前段が選択肢も既定も公開していない(前者は分離を呼ぶかどうかの
# 指定でアダプタの選択ではなく、後者はプロセスの環境をどう整えるかの指定)。ここで定める。
SEPARATE_VOCALS_MODES = ("always", "never")
DEVICE_MODES = ("auto", "cpu")

# 音声前段に対応物を持つ選択肢は登録・設定型から取る(手書きで複製すると追加・変更へ追随しない)。
FORCED_ALIGNER_IDS = get_args(ForcedAlignerId)
ENGLISH_KATAKANA_METHODS = get_args(EnglishKatakanaMethod)

_SOFA_ALIGNER = "sofa-forcedalign"
_DEFAULT_SOFA_TIMEOUT_SEC = SofaAlignerConfig.__dataclass_fields__["timeout_sec"].default

# タイムアウト秒数は 0 以下が無意味。分割の目標長は 0 が分割の無効化。
_positive_float = RangeValidator(value_type="float", minimum=0, exclusive_min=True)
_nonneg_float = RangeValidator(value_type="float", minimum=0)


def add_arguments(parser):
    """共通引数群を parser へ一括登録する。一部だけを非公開にする機構は持たない。"""
    parser.add_argument("--separate-vocals", dest="separate_vocals", choices=SEPARATE_VOCALS_MODES,
                        default="always", help="ボーカル分離の実施方針(always/never)")
    parser.add_argument("--separator", choices=SEPARATOR_IDS, default=DEFAULT_SEPARATOR,
                        help="S1ボーカル分離バックエンドの選択(vocal_analysisの登録アダプタ安定id)")
    parser.add_argument("--recognizer-model-id", dest="recognizer_model_id", default=None,
                        help=f"S2内容認識モデルの指定。未指定時は vocal_analysis の既定モデル"
                             f"(model_id={DEFAULT_CONTENT_RECOGNIZER_MODEL.model_id!r}・"
                             f"model_revision={DEFAULT_CONTENT_RECOGNIZER_MODEL.model_revision!r}固定)を使う")
    parser.add_argument("--recognizer-model-revision", dest="recognizer_model_revision", default=None,
                        help="--recognizer-model-id のリビジョン指定(組で使う。--recognizer-model-id "
                             "指定時にこれを省略すると最新リビジョンを使う。--recognizer-model-id 自体を"
                             "省略した場合は本オプションは無視されず引数エラーになる)")
    # --recognizer-retry / --no-recognizer-retry は既定 on の対。dest=recognizer_retry を共有する。
    parser.add_argument("--recognizer-retry", dest="recognizer_retry", action="store_true", default=True,
                        help="S2内容認識のトリガ式リトライ(エコー幻覚・反復幻覚。主モデル自身をプロンプト無しで"
                             "再認識する)を有効にする(既定on)。--no-recognizer-retryの対の明示形")
    parser.add_argument("--no-recognizer-retry", dest="recognizer_retry", action="store_false",
                        help="S2内容認識のリトライを無効にする。--recognizer-retryの対")
    parser.add_argument("--forced-aligner", dest="forced_aligner", choices=FORCED_ALIGNER_IDS,
                        default=DEFAULT_FORCED_ALIGNER,
                        help="S2強制アライメント段のバックエンド選択(vocal_analysisの登録アダプタ安定id)。"
                             "sofa-forcedalign選択時は--sofa-python/--sofa-root/--sofa-checkpointが必須")
    parser.add_argument("--sofa-python", dest="sofa_python", default=None,
                        help="SOFA専用venvのPython実行ファイルパス(--forced-aligner sofa-forcedalign時に必須)")
    parser.add_argument("--sofa-root", dest="sofa_root", default=None,
                        help="SOFAリポジトリのルートパス(--forced-aligner sofa-forcedalign時に必須)")
    parser.add_argument("--sofa-checkpoint", dest="sofa_checkpoint", default=None,
                        help="SOFAチェックポイント(.ckpt)ファイルパス(--forced-aligner sofa-forcedalign時に必須)")
    parser.add_argument("--sofa-timeout", dest="sofa_timeout", type=_positive_float,
                        default=_DEFAULT_SOFA_TIMEOUT_SEC,
                        help="SOFAサブプロセス1回あたりのタイムアウト秒数")
    parser.add_argument("--english-katakana-method", dest="english_katakana_method",
                        choices=ENGLISH_KATAKANA_METHODS,
                        default=DEFAULT_ENGLISH_KATAKANA_METHOD,
                        help="英語カタカナ化フォールバックの変換方式選択(既定arpakana。"
                             "tinyllama-katakana-converterは生成モデルを使う選択式オプション)")
    parser.add_argument("--device", choices=DEVICE_MODES, default="auto",
                        help="実行デバイスの選択。auto=環境から自動選択(GPU(CUDA)が利用可能ならGPU)、"
                             "cpu=GPUを使わずCPUで実行する(音声前段の全モデル・SOFAサブプロセスを含む。"
                             "VRAM不足環境の回避手段)")
    parser.add_argument("--max-duration", dest="max_duration", type=_nonneg_float, default=300.0,
                        help="長尺の自動分割境界(秒。0で無効)")


def describe_type_table():
    """共通引数群についての自己記述の型情報。ツールは自分の固有引数の分と併せて渡す。"""
    return {
        "separate_vocals": ("enum", {"choices": list(SEPARATE_VOCALS_MODES)}),
        "separator": ("enum", {"choices": list(SEPARATOR_IDS)}),
        "recognizer_model_id": ("str", None),
        "recognizer_model_revision": ("str", None),
        "recognizer_retry": ("flag", None),
        "forced_aligner": ("enum", {"choices": list(FORCED_ALIGNER_IDS)}),
        "sofa_python": ("str", None),
        "sofa_root": ("str", None),
        "sofa_checkpoint": ("str", None),
        # 浮動小数は対応表を空にして、型も範囲も引数の検証子から導く。
        "sofa_timeout": (None, None),
        "english_katakana_method": ("enum", {"choices": list(ENGLISH_KATAKANA_METHODS)}),
        "device": ("enum", {"choices": list(DEVICE_MODES)}),
        "max_duration": (None, None),
    }


def validate(args):
    """単独の引数の型・範囲では表せない組み合わせの誤りを検出する。

    違反があれば (対象の引数名, 理由の本文) を返し、無ければ None を返す。1回の呼び出しにつき
    最初に見つかった1件だけを返す(複数を並べても利用者はどれから直すか選べない)。
    """
    if args.forced_aligner == _SOFA_ALIGNER:
        for name, dest in (
            ("--sofa-python", "sofa_python"),
            ("--sofa-root", "sofa_root"),
            ("--sofa-checkpoint", "sofa_checkpoint"),
        ):
            if getattr(args, dest) is None:
                return name, f"{name} は --forced-aligner {_SOFA_ALIGNER} 時に必須です"
    if args.recognizer_model_id is None and args.recognizer_model_revision is not None:
        return ("--recognizer-model-revision",
                "--recognizer-model-revision は --recognizer-model-id と組で指定してください")
    return None


def resolve_recognizer_model(args):
    """内容認識モデルの設定。モデル指定が無ければ音声前段の既定モデルをそのまま返す。"""
    if args.recognizer_model_id is None:
        return DEFAULT_CONTENT_RECOGNIZER_MODEL
    return ContentRecognizerModel(
        model_id=args.recognizer_model_id, model_revision=args.recognizer_model_revision)


def resolve_sofa_config(args):
    """SOFA 経路の設定。SOFA 経路を選んでいなければ None を返す。

    選んでいない経路の設定を作ると、指定されていない値を既定で埋めることになる。
    """
    if args.forced_aligner != _SOFA_ALIGNER:
        return None
    return SofaAlignerConfig(
        sofa_python=Path(args.sofa_python), sofa_root=Path(args.sofa_root),
        checkpoint_path=Path(args.sofa_checkpoint), timeout_sec=args.sofa_timeout)


def resolve_chunking_policy(args):
    """長尺分割の実行ポリシー。目標長が 0 のときは None(分割しない)を返す。"""
    if args.max_duration == 0:
        return None
    return ChunkingPolicy(max_duration_sec=args.max_duration)


def apply_device(args):
    """実行デバイスの選択をプロセスへ適用する。

    cpu を選んだときは CUDA_VISIBLE_DEVICES を -1 にしてプロセスから GPU を隠す。auto では何も
    しない(利用者が外から与えた設定を上書きしない)。利用できるデバイスの集合はプロセス内の最初の
    照会以降固定されるので、最初の GPU 照会より前に呼ばなければ効かない。環境変数は子プロセスへ
    継承されるので、音声前段が起動するサブプロセスにも同じ選択が効く。
    """
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def missing_dependency_message(exc):
    """追加依存が未導入のときの理由1行。取り込めなかったモジュール名と導入コマンドを示す。"""
    # ModuleNotFoundError は不足モジュール名を name に持つ。持たない ImportError(名前の解決失敗等)は
    # 例外の文言をそのまま理由に使う。
    name = getattr(exc, "name", None)
    missing = repr(name) if name else str(exc)
    return (f"音声前段の依存パッケージ {missing} を取り込めません。"
            '追加依存が要ります: pip install ".[vocal-analysis]"')
