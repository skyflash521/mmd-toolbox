"""外部モデル委譲ステージの固定推論条件。

S2 の音素モデル(強制アライメント用)・内容認識モデル(既定値・候補値)と S1 分離器のモデル指定
(id・revision、Demucs の shift 平均無効化)を固定する。加えて音素モデルは
実行デバイス・dtype・スレッド・乱数シードも固定し、S-1 測定と実装が同一条件で動くようにする。
内容認識モデルの実行デバイスは環境依存で自動選択し(GPUが利用可能ならGPUを使う)、この固定の
対象外。これらの固定値は実装が独自に変えない。

固定対象の条件のうち、mono への downmix・16kHz への再サンプリング方式・バッチは
S2 アダプタの変換/推論の実装内部で確定する(採用ライブラリと実測に依存するため、アダプタ実装が
その方式を定数として固定し、本 config はアダプタ非依存の目標値・実行条件だけを持つ)。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RecognizerConfig:
    """S2 音素モデル(強制アライメント用)の固定条件。

    device・dtype・num_threads・random_seed は音素モデル(強制アライメント)専用の
    決定論のための固定値であり、内容認識モデル(Whisper系)には適用しない。内容認識
    モデルの実行デバイスは環境依存で自動選択する(GPUが利用可能ならGPUを使う。recognizer.py の
    `_select_content_recognizer_device`)。内容認識の貪欲デコード(ビーム幅1・サンプリング無し)は
    サンプリング由来の乱数的非決定性を排除するが、実行デバイス・スレッド数の違いによる浮動小数点
    演算の丸め誤差までは排除しない。環境が異なれば僅差のトークン選択が割れうる。
    """

    model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    model_revision: str = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
    sample_rate: int = 16000  # 認識器へ渡す目標サンプルレート(mono化・再サンプリングはS2アダプタ。内容認識・音素モデルで共用)
    device: str = "cpu"  # 音素モデル専用。決定論を優先しGPUを使わない
    dtype: str = "float32"  # 音素モデル専用
    num_threads: int = 1  # 音素モデル専用。推論時(_compute_log_probs)に局所適用しスレッド並列の非決定を避ける
    random_seed: int = 0  # 音素モデル専用


@dataclass(frozen=True)
class ContentRecognizerModel:
    """S2 内容認識モデルの指定。model_revision を省略(None)すると最新リビジョンを使う。"""

    model_id: str
    model_revision: str | None = None


# 既定値: 歌唱データで検証済み・高速。
DEFAULT_CONTENT_RECOGNIZER_MODEL = ContentRecognizerModel(
    model_id="openai/whisper-medium",
    model_revision="abdf7c39ab9d0397620ccaea8974cc764cd0953e",
)

# 候補値: 常にかなを返す。歌唱データでの学習・評価実績は無い。
KANA_WHISPER_MODEL = ContentRecognizerModel(
    model_id="sbintuitions/kana-whisper",
    model_revision="88ecb3d79c5846cb4fcf76f4107b84c8fa2acd82",
)

# かな限定プロンプト。既定値・候補値のどちらにも同じ手順で渡す(モデルによる分岐なし)。
# 内容認識のトリガ式リトライの再認識には渡さない。
KANA_PROMPT = "すべて ひらがなだけで こたえてください。かんじは つかわないでください。"


@dataclass(frozen=True)
class EnglishOovKatakanaModel:
    """英語未知語カタカナ化フォールバックで使う変換モデルの指定。model_revisionを省略(None)
    すると最新リビジョンを使う。method="tinyllama-katakana-converter"選択時のみ使う。
    """

    model_id: str
    model_revision: str | None = None


ENGLISH_OOV_KATAKANA_MODEL = EnglishOovKatakanaModel(
    model_id="pyon0024/tinyllama-katakana-converter",
    model_revision="3319c206a7f62f0da2660a96a1b3395c3048cfec",
)

# 英語未知語カタカナ化フォールバックの変換方式。既定値"arpakana"はARPAbet音素をルールベースで
# カタカナへ変換する(生成モデル・GPU不要)。"tinyllama-katakana-converter"は
# ENGLISH_OOV_KATAKANA_MODELの生成モデルを使う(選択式オプション)。
EnglishOovKatakanaMethod = Literal["arpakana", "tinyllama-katakana-converter"]
DEFAULT_ENGLISH_OOV_KATAKANA_METHOD: EnglishOovKatakanaMethod = "arpakana"


@dataclass(frozen=True)
class SeparatorConfig:
    """S1 ボーカル分離器(audio-separator 経由の Demucs v4 htdemucs_ft)の固定条件。"""

    model_filename: str = "htdemucs_ft.yaml"
    output_single_stem: str = "vocals"  # ボーカルstem以外を書き出させない
    shifts: int = 0  # shift 平均(非決定要素)を無効化


ForcedAlignerId = Literal["wav2vec2-ctc-forcedalign", "sofa-forcedalign"]
DEFAULT_FORCED_ALIGNER: ForcedAlignerId = "wav2vec2-ctc-forcedalign"


@dataclass(frozen=True)
class SofaAlignerConfig:
    """S2 SOFA経路の実行環境指定。既定値・同梱チェックポイントは一切持たない
    (利用者保護のための方針判断。商用利用が制限されたチェックポイントを既定値にしない)。
    """

    sofa_python: Path  # 利用者が用意した専用venvのPython実行ファイルパス
    sofa_root: Path  # SOFAリポジトリのルート(infer.py実行時のcwdに使う)
    checkpoint_path: Path  # 利用者提供のSOFAチェックポイント(.ckptファイル)パス
    timeout_sec: float = 300.0  # 1回のSOFA呼び出しあたりのサブプロセスタイムアウト秒数


RECOGNIZER_CONFIG = RecognizerConfig()
SEPARATOR_CONFIG = SeparatorConfig()
