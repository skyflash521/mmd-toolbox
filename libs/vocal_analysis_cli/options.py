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

SEPARATE_VOCALS_MODES = ("always", "never")
DEVICE_MODES = ("auto", "cpu")

FORCED_ALIGNER_IDS = get_args(ForcedAlignerId)
ENGLISH_KATAKANA_METHODS = get_args(EnglishKatakanaMethod)

_SOFA_ALIGNER = "sofa-forcedalign"
_DEFAULT_SOFA_TIMEOUT_SEC = SofaAlignerConfig.__dataclass_fields__["timeout_sec"].default

_positive_float = RangeValidator(value_type="float", minimum=0, exclusive_min=True)
_nonneg_float = RangeValidator(value_type="float", minimum=0)

_DERIVED_FROM_VALIDATOR = (None, None)


def add_arguments(parser):
    parser.add_argument("--separate-vocals", dest="separate_vocals", choices=SEPARATE_VOCALS_MODES,
                        default="always", help="ボーカル分離の実施方針(always/never)")
    parser.add_argument("--separator", choices=SEPARATOR_IDS, default=DEFAULT_SEPARATOR,
                        help="ボーカル分離の方式")
    parser.add_argument("--recognizer-model-id", dest="recognizer_model_id", default=None,
                        help=f"内容認識モデルの指定。未指定時は {DEFAULT_CONTENT_RECOGNIZER_MODEL.model_id} を使う")
    parser.add_argument("--recognizer-model-revision", dest="recognizer_model_revision", default=None,
                        help="--recognizer-model-id と組で使うリビジョンの指定。--recognizer-model-id を指定して"
                             "これを省略すると、そのモデルの最新リビジョンを使う。"
                             "--recognizer-model-id を付けずに指定すると引数エラーになる")
    parser.add_argument("--recognizer-retry", dest="recognizer_retry", action="store_true", default=True,
                        help="内容認識の結果にエコー幻覚・反復幻覚が出たら認識し直す(既定)")
    parser.add_argument("--no-recognizer-retry", dest="recognizer_retry", action="store_false",
                        help="内容認識の結果にエコー幻覚・反復幻覚が出ても認識し直さない")
    parser.add_argument("--forced-aligner", dest="forced_aligner", choices=FORCED_ALIGNER_IDS,
                        default=DEFAULT_FORCED_ALIGNER,
                        help="強制アライメントの方式。sofa-forcedalign を選ぶときは "
                             "--sofa-python・--sofa-root・--sofa-checkpoint が必須")
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
                        help="読みを得られない英単語をカタカナにする方式(既定 arpakana。"
                             "tinyllama-katakana-converter は生成モデルを使う)")
    parser.add_argument("--device", choices=DEVICE_MODES, default="auto",
                        help="実行デバイス。auto は GPU(CUDA)が使えれば GPU を使う。cpu は GPU を使わない"
                             "(SOFA を含む全モデル。VRAM が足りないときの回避手段)")
    parser.add_argument("--max-duration", dest="max_duration", type=_nonneg_float, default=300.0,
                        help="長い曲を自動で分割する長さ(秒。0 で分割しない)")


def describe_type_table():
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
        "sofa_timeout": _DERIVED_FROM_VALIDATOR,
        "english_katakana_method": ("enum", {"choices": list(ENGLISH_KATAKANA_METHODS)}),
        "device": ("enum", {"choices": list(DEVICE_MODES)}),
        "max_duration": _DERIVED_FROM_VALIDATOR,
    }


def validate(args) -> tuple[str, str] | None:
    """違反があれば (対象の引数の長形式フラグ名, 理由の本文) を、無ければ None を返す。"""
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


def resolve_recognizer_model(args) -> ContentRecognizerModel:
    if args.recognizer_model_id is None:
        return DEFAULT_CONTENT_RECOGNIZER_MODEL
    return ContentRecognizerModel(
        model_id=args.recognizer_model_id, model_revision=args.recognizer_model_revision)


def resolve_sofa_config(args) -> SofaAlignerConfig | None:
    if args.forced_aligner != _SOFA_ALIGNER:
        return None
    return SofaAlignerConfig(
        sofa_python=Path(args.sofa_python), sofa_root=Path(args.sofa_root),
        checkpoint_path=Path(args.sofa_checkpoint), timeout_sec=args.sofa_timeout)


def resolve_chunking_policy(args) -> ChunkingPolicy | None:
    if args.max_duration == 0:
        return None
    return ChunkingPolicy(max_duration_sec=args.max_duration)


def apply_device(args) -> None:
    """プロセス内で最初に GPU を照会するより前に呼ぶこと。"""
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def missing_dependency_message(exc) -> str:
    name = getattr(exc, "name", None)
    missing = repr(name) if name else str(exc)
    return (f"音声前段の依存パッケージ {missing} を取り込めません。"
            '追加依存が要ります: pip install ".[vocal-analysis]"')
