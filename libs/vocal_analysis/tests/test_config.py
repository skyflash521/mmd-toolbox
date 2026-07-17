"""固定推論条件のテスト。

外部モデル委譲ステージ(S1 分離・S2 認識)のモデル指定(id・revision)の固定値、および
dtype・Demucs の shift 平均無効化は実装が独自に変えない値であり、値がずれていないことを
テストで固定する。
"""

import dataclasses
import typing
from pathlib import Path

import pytest


def test_recognizer_model_id_and_revision_are_pinned():
    from vocal_analysis import RECOGNIZER_CONFIG

    # 採用モデルと revision の固定値。実装が独自に変えない。
    assert RECOGNIZER_CONFIG.model_id == "facebook/wav2vec2-lv-60-espeak-cv-ft"
    assert RECOGNIZER_CONFIG.model_revision == "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"


def test_recognizer_target_sample_rate_is_16khz():
    from vocal_analysis import RECOGNIZER_CONFIG

    # S2 認識器が要求する目標サンプルレート 16kHz。mono への downmix と
    # 16kHz への再サンプリング(方式)は S2 アダプタの変換責務でそこで検証する。
    # config の可変値はこの目標レートで、mono はモデル要件で固定なので別フィールドにしない。
    assert RECOGNIZER_CONFIG.sample_rate == 16000


def test_recognizer_dtype_is_pinned():
    from vocal_analysis import RECOGNIZER_CONFIG

    # 音素モデル(強制アライメント用)の dtype はモデル id・revision と同様に認識結果に影響する
    # 推論条件で、S-1 測定で品質を検証済みの固定値。変更は revision 変更と同じ手順を要する。
    assert RECOGNIZER_CONFIG.dtype == "float32"


def test_default_content_recognizer_model_id_and_revision_are_pinned():
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL

    # 既定値(openai/whisper-medium)として固定したモデル・revision。実装が独自に変えない。
    assert DEFAULT_CONTENT_RECOGNIZER_MODEL.model_id == "openai/whisper-medium"
    assert DEFAULT_CONTENT_RECOGNIZER_MODEL.model_revision == "abdf7c39ab9d0397620ccaea8974cc764cd0953e"


def test_kana_whisper_model_id_and_revision_are_pinned():
    from vocal_analysis import KANA_WHISPER_MODEL

    # 候補値(kana-whisper)として固定したモデルとrevision。
    assert KANA_WHISPER_MODEL.model_id == "sbintuitions/kana-whisper"
    assert KANA_WHISPER_MODEL.model_revision == "88ecb3d79c5846cb4fcf76f4107b84c8fa2acd82"


def test_kana_prompt_is_pinned():
    from vocal_analysis import KANA_PROMPT

    # かな限定プロンプトとして固定した文字列。既定値・候補値どちらに渡す場合も共通。
    assert KANA_PROMPT == "すべて ひらがなだけで こたえてください。かんじは つかわないでください。"


def test_content_recognizer_model_revision_defaults_to_none():
    from vocal_analysis import ContentRecognizerModel

    # revision省略時は最新リビジョンを使う。既定値・候補値以外を任意指定するときの挙動。
    custom = ContentRecognizerModel(model_id="openai/whisper-large-v3")
    assert custom.model_revision is None


def test_separator_shifts_disabled_for_speed():
    from vocal_analysis import SEPARATOR_CONFIG

    # Demucs の shift 平均は複数回の追加フォワードパスを伴い処理が遅くなるため無効化する。
    assert SEPARATOR_CONFIG.shifts == 0


def test_separator_model_and_stem_are_pinned():
    from vocal_analysis import SEPARATOR_CONFIG

    # audio-separator 経由で実行する Demucs v4 htdemucs_ft と、書き出す単一stem(vocals)。
    assert SEPARATOR_CONFIG.model_filename == "htdemucs_ft.yaml"
    assert SEPARATOR_CONFIG.output_single_stem == "vocals"


def test_sofa_aligner_config_stores_user_provided_paths():
    from vocal_analysis import SofaAlignerConfig

    # sofa_python・sofa_root・checkpoint_path はいずれも利用者提供の必須値(既定値なし)。
    config = SofaAlignerConfig(
        sofa_python=Path("/tmp/sofa-venv/python"),
        sofa_root=Path("/tmp/SOFA"),
        checkpoint_path=Path("/tmp/checkpoint.ckpt"),
    )
    assert config.sofa_python == Path("/tmp/sofa-venv/python")
    assert config.sofa_root == Path("/tmp/SOFA")
    assert config.checkpoint_path == Path("/tmp/checkpoint.ckpt")


def test_sofa_aligner_config_timeout_defaults_to_300_seconds():
    from vocal_analysis import SofaAlignerConfig

    # timeout_sec のみ実装が定める既定値(300.0秒、float型)を持つ。
    config = SofaAlignerConfig(
        sofa_python=Path("/tmp/sofa-venv/python"),
        sofa_root=Path("/tmp/SOFA"),
        checkpoint_path=Path("/tmp/checkpoint.ckpt"),
    )
    assert config.timeout_sec == 300.0
    assert isinstance(config.timeout_sec, float)

    type_hints = typing.get_type_hints(SofaAlignerConfig)
    assert type_hints["timeout_sec"] is float
    default = {f.name: f for f in dataclasses.fields(SofaAlignerConfig)}["timeout_sec"].default
    assert default == 300.0
    assert isinstance(default, float)


def test_sofa_aligner_config_required_fields_have_no_default():
    from vocal_analysis import SofaAlignerConfig

    # 本プロジェクトはSOFAのチェックポイント・実行環境の既定値を一切持たない(利用者保護の方針)。
    with pytest.raises(TypeError):
        SofaAlignerConfig()

    # 3フィールドそれぞれが個別に既定値を持たないことを確認する(いずれか1つにだけ既定値があっても
    # 上記の引数なし呼び出しはTypeErrorのままなので、フィールド単位の確認が別途要る)。
    field_by_name = {f.name: f for f in dataclasses.fields(SofaAlignerConfig)}
    type_hints = typing.get_type_hints(SofaAlignerConfig)
    for name in ("sofa_python", "sofa_root", "checkpoint_path"):
        field = field_by_name[name]
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING
        assert type_hints[name] is Path


def test_sofa_aligner_config_is_frozen():
    from vocal_analysis import SofaAlignerConfig

    config = SofaAlignerConfig(
        sofa_python=Path("/tmp/sofa-venv/python"),
        sofa_root=Path("/tmp/SOFA"),
        checkpoint_path=Path("/tmp/checkpoint.ckpt"),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.timeout_sec = 60.0


def test_configs_are_frozen():
    from vocal_analysis import (
        DEFAULT_CONTENT_RECOGNIZER_MODEL,
        KANA_WHISPER_MODEL,
        RECOGNIZER_CONFIG,
        SEPARATOR_CONFIG,
    )

    # 固定値なので再代入を禁じる(frozen dataclass)。
    with pytest.raises(dataclasses.FrozenInstanceError):
        RECOGNIZER_CONFIG.model_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        SEPARATOR_CONFIG.shifts = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONTENT_RECOGNIZER_MODEL.model_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        KANA_WHISPER_MODEL.model_id = "other"
