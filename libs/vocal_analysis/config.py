"""外部モデル委譲ステージの固定推論条件(vocal_analysis.md §5.1・§5.2・§8.3・§4)。

S2 の音素モデル(強制アライメント用)・内容認識モデル(既定 kana-whisper・選択可能な代替
whisper-medium+かなプロンプト)と S1 分離器の非決定要素を固定し、
S-1 測定と実装が同一条件で動くようにする。モデル id・revision は §8.3、Demucs の shift 平均無効化は
§4・§8.3 が定める固定値。実行デバイス・dtype・スレッド・乱数シードは決定論のための固定値。これらは
実装が独自に変えない(変更が要れば vocal_analysis.md を先に更新する)。

§5.1 が固定対象に挙げる条件のうち、mono への downmix・16kHz への再サンプリング方式・バッチは
S2 アダプタの変換/推論の実装内部で確定する(採用ライブラリと実測に依存するため、アダプタ実装が
その方式を定数として固定し、本 config はアダプタ非依存の目標値・実行条件だけを持つ)。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecognizerConfig:
    """S2 音素モデル(強制アライメント用。§5.2)の固定条件(§5.1・§8.3)。"""

    model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    model_revision: str = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
    sample_rate: int = 16000  # 認識器へ渡す目標サンプルレート(mono 化・再サンプリングは S2 アダプタ)
    device: str = "cpu"  # GPU 不要・決定論を優先
    dtype: str = "float32"
    num_threads: int = 1  # スレッド並列の非決定を避ける
    random_seed: int = 0


@dataclass(frozen=True)
class WhisperConfig:
    """S2 内容認識モデル(選択可能な代替アダプタ whisper-ctc-forcedalign。§5.2)の固定条件(§8.3)。"""

    model_id: str = "openai/whisper-medium"
    model_revision: str = "abdf7c39ab9d0397620ccaea8974cc764cd0953e"
    kana_prompt: str = "すべて ひらがなだけで こたえてください。かんじは つかわないでください。"


@dataclass(frozen=True)
class KanaWhisperConfig:
    """S2 内容認識モデル(既定アダプタ kana-whisper-ctc-forcedalign。§5.2)の固定条件(§8.3)。"""

    model_id: str = "sbintuitions/kana-whisper"
    model_revision: str = "88ecb3d79c5846cb4fcf76f4107b84c8fa2acd82"


@dataclass(frozen=True)
class SeparatorConfig:
    """S1 ボーカル分離器(audio-separator 経由の Demucs v4 htdemucs_ft)の固定条件(§4・§8.3)。"""

    model_filename: str = "htdemucs_ft.yaml"
    output_single_stem: str = "vocals"  # ボーカルstem以外を書き出させない(§8.3後注)
    shifts: int = 0  # shift 平均(非決定要素)を無効化


RECOGNIZER_CONFIG = RecognizerConfig()
WHISPER_CONFIG = WhisperConfig()
KANA_WHISPER_CONFIG = KanaWhisperConfig()
SEPARATOR_CONFIG = SeparatorConfig()
