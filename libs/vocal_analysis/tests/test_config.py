"""固定推論条件のテスト(vocal_analysis.md §5.1・§8.3)。

外部モデル委譲ステージ(S1 分離・S2 認識)の非決定要素を固定し、S-1 測定と実装が同一条件で
動くようにする。モデル id・revision は §8.3 の固定値、Demucs の shift 平均無効化は §4・§8.3。
これらは実装が独自に変えない値であり、値がずれていないことをテストで固定する。
"""

import dataclasses

import pytest


def test_recognizer_model_id_and_revision_are_pinned():
    from vocal_analysis import RECOGNIZER_CONFIG

    # §8.3 で固定した採用モデルと revision。実装が独自に変えない。
    assert RECOGNIZER_CONFIG.model_id == "facebook/wav2vec2-lv-60-espeak-cv-ft"
    assert RECOGNIZER_CONFIG.model_revision == "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"


def test_recognizer_target_sample_rate_is_16khz():
    from vocal_analysis import RECOGNIZER_CONFIG

    # S2 認識器が要求する目標サンプルレート 16kHz(§5・§5.1)。mono への downmix と
    # 16kHz への再サンプリング(方式)は S2 アダプタの変換責務でそこで検証する。
    # config の可変値はこの目標レートで、mono はモデル要件で固定なので別フィールドにしない。
    assert RECOGNIZER_CONFIG.sample_rate == 16000


def test_recognizer_inference_conditions_are_deterministic():
    from vocal_analysis import RECOGNIZER_CONFIG

    # 実行デバイス/dtype・スレッド・乱数シードを固定して決定論にする(§5.1)。
    assert RECOGNIZER_CONFIG.device == "cpu"
    assert RECOGNIZER_CONFIG.dtype == "float32"
    assert RECOGNIZER_CONFIG.num_threads == 1
    assert RECOGNIZER_CONFIG.random_seed == 0


def test_whisper_model_id_revision_and_kana_prompt_are_pinned():
    from vocal_analysis import WHISPER_CONFIG

    # §8.3 で固定した選択可能な代替アダプタ(whisper-ctc-forcedalign)のモデル・revisionと、
    # かな限定プロンプトの固定文字列。実装が独自に変えない。
    assert WHISPER_CONFIG.model_id == "openai/whisper-medium"
    assert WHISPER_CONFIG.model_revision == "abdf7c39ab9d0397620ccaea8974cc764cd0953e"
    assert WHISPER_CONFIG.kana_prompt == (
        "これはすべてかなだけでかかれたぶんしょうです。かんじはいっさいつかいません。"
    )


def test_kana_whisper_model_id_and_revision_are_pinned():
    from vocal_analysis import KANA_WHISPER_CONFIG

    # §8.3 で固定した既定アダプタ(kana-whisper-ctc-forcedalign)のモデルとrevision。
    assert KANA_WHISPER_CONFIG.model_id == "sbintuitions/kana-whisper"
    assert KANA_WHISPER_CONFIG.model_revision == "88ecb3d79c5846cb4fcf76f4107b84c8fa2acd82"


def test_separator_shifts_disabled_for_determinism():
    from vocal_analysis import SEPARATOR_CONFIG

    # Demucs の shift 平均は非決定要素なので無効化する(§4・§8.3)。
    assert SEPARATOR_CONFIG.shifts == 0


def test_separator_model_and_stem_are_pinned():
    from vocal_analysis import SEPARATOR_CONFIG

    # §8.3後注: audio-separator 経由で実行する Demucs v4 htdemucs_ft と、書き出す単一stem(vocals)。
    assert SEPARATOR_CONFIG.model_filename == "htdemucs_ft.yaml"
    assert SEPARATOR_CONFIG.output_single_stem == "vocals"


def test_configs_are_frozen():
    from vocal_analysis import KANA_WHISPER_CONFIG, RECOGNIZER_CONFIG, SEPARATOR_CONFIG, WHISPER_CONFIG

    # 固定値なので再代入を禁じる(frozen dataclass)。
    with pytest.raises(dataclasses.FrozenInstanceError):
        RECOGNIZER_CONFIG.model_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        SEPARATOR_CONFIG.shifts = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        WHISPER_CONFIG.model_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        KANA_WHISPER_CONFIG.model_id = "other"
