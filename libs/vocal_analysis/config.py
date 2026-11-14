from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args


@dataclass(frozen=True)
class RecognizerConfig:
    model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    model_revision: str = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
    sample_rate: int = 16000
    dtype: str = "float32"


@dataclass(frozen=True)
class ContentRecognizerModel:
    """model_revision が None なら最新リビジョンを使う。"""

    model_id: str
    model_revision: str | None = None


DEFAULT_CONTENT_RECOGNIZER_MODEL = ContentRecognizerModel(
    model_id="openai/whisper-medium",
    model_revision="abdf7c39ab9d0397620ccaea8974cc764cd0953e",
)

KANA_PROMPT = "すべて ひらがなだけで こたえてください。かんじは つかわないでください。"


@dataclass(frozen=True)
class EnglishKatakanaModel:
    model_id: str
    model_revision: str | None = None


ENGLISH_KATAKANA_MODEL = EnglishKatakanaModel(
    model_id="pyon0024/tinyllama-katakana-converter",
    model_revision="3319c206a7f62f0da2660a96a1b3395c3048cfec",
)

EnglishKatakanaMethod = Literal["arpakana", "tinyllama-katakana-converter"]
DEFAULT_ENGLISH_KATAKANA_METHOD: EnglishKatakanaMethod = "arpakana"


@dataclass(frozen=True)
class SeparatorConfig:
    model_filename: str = "htdemucs_ft.yaml"
    output_single_stem: str = "vocals"
    # Demucs の shifts は shift 平均のための追加フォワードパスの回数で、0 で無効になる。
    shifts: int = 0


SeparatorId = Literal["audio-separator-htdemucs-ft"]
SEPARATOR_IDS: tuple[SeparatorId, ...] = get_args(SeparatorId)
DEFAULT_SEPARATOR: SeparatorId = "audio-separator-htdemucs-ft"

ForcedAlignerId = Literal["wav2vec2-ctc-forcedalign", "sofa-forcedalign"]
DEFAULT_FORCED_ALIGNER: ForcedAlignerId = "wav2vec2-ctc-forcedalign"


@dataclass(frozen=True)
class SofaAlignerConfig:
    """sofa_python は SOFA の依存を導入した専用環境の Python 実行ファイル。sofa_root は SOFA リポジトリの
    ルートで、SOFA の実行時の作業ディレクトリになる。checkpoint_path は利用者が用意する SOFA の
    チェックポイント(.ckpt)。timeout_sec は SOFA サブプロセス1回あたりの上限。"""

    sofa_python: Path
    sofa_root: Path
    checkpoint_path: Path
    timeout_sec: float = 300.0


@dataclass(frozen=True)
class ChunkingPolicy:
    """max_duration_sec はチャンクの目標長で、実際のチャンクは探索窓のぶん前後してこの値を超えることがある。"""

    max_duration_sec: float
    search_window_sec: float = 5.0
    overlap_sec: float = 1.0


RECOGNIZER_CONFIG = RecognizerConfig()
SEPARATOR_CONFIG = SeparatorConfig()
