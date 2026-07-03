"""固定推論条件のテスト(vocal_analysis.md §5.1・§8.3)。

外部モデル委譲ステージ(S1 分離・S2 認識)の非決定要素を固定し、S-1 ゲートと実装が同一条件で
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


def test_separator_shifts_disabled_for_determinism():
    from vocal_analysis import SEPARATOR_CONFIG

    # Demucs の shift 平均は非決定要素なので無効化する(§4・§8.3)。
    assert SEPARATOR_CONFIG.shifts == 0


def test_configs_are_frozen():
    from vocal_analysis import RECOGNIZER_CONFIG, SEPARATOR_CONFIG

    # 固定値なので再代入を禁じる(frozen dataclass)。
    with pytest.raises(dataclasses.FrozenInstanceError):
        RECOGNIZER_CONFIG.model_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        SEPARATOR_CONFIG.shifts = 1
