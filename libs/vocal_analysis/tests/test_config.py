import dataclasses
import typing
from pathlib import Path

import pytest


def test_recognizer_model_id_and_revision_are_pinned():
    from vocal_analysis import RECOGNIZER_CONFIG

    assert RECOGNIZER_CONFIG.model_id == "facebook/wav2vec2-lv-60-espeak-cv-ft"
    assert RECOGNIZER_CONFIG.model_revision == "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"


def test_recognizer_target_sample_rate_is_16khz():
    from vocal_analysis import RECOGNIZER_CONFIG

    assert RECOGNIZER_CONFIG.sample_rate == 16000


def test_recognizer_dtype_is_pinned():
    from vocal_analysis import RECOGNIZER_CONFIG

    assert RECOGNIZER_CONFIG.dtype == "float32"


def test_default_content_recognizer_model_id_and_revision_are_pinned():
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL

    assert DEFAULT_CONTENT_RECOGNIZER_MODEL.model_id == "openai/whisper-medium"
    assert DEFAULT_CONTENT_RECOGNIZER_MODEL.model_revision == "abdf7c39ab9d0397620ccaea8974cc764cd0953e"


def test_kana_prompt_is_pinned():
    from vocal_analysis import KANA_PROMPT

    assert KANA_PROMPT == "すべて ひらがなだけで こたえてください。かんじは つかわないでください。"


def test_content_recognizer_model_revision_defaults_to_none():
    from vocal_analysis import ContentRecognizerModel

    custom = ContentRecognizerModel(model_id="openai/whisper-large-v3")
    assert custom.model_revision is None


def test_separator_shift_averaging_is_disabled():
    from vocal_analysis import SEPARATOR_CONFIG

    assert SEPARATOR_CONFIG.shifts == 0


def test_separator_model_and_stem_are_pinned():
    from vocal_analysis import SEPARATOR_CONFIG

    assert SEPARATOR_CONFIG.model_filename == "htdemucs_ft.yaml"
    assert SEPARATOR_CONFIG.output_single_stem == "vocals"


def test_sofa_aligner_config_stores_user_provided_paths():
    from vocal_analysis import SofaAlignerConfig

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


def test_sofa_aligner_config_each_path_field_has_no_default():
    from vocal_analysis import SofaAlignerConfig

    with pytest.raises(TypeError):
        SofaAlignerConfig()

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


def test_chunking_policy_defaults_search_window_5s_and_overlap_1s():
    from vocal_analysis import ChunkingPolicy

    policy = ChunkingPolicy(max_duration_sec=300.0)
    assert policy.search_window_sec == 5.0
    assert policy.overlap_sec == 1.0


def test_configs_are_frozen():
    from vocal_analysis import (
        DEFAULT_CONTENT_RECOGNIZER_MODEL,
        RECOGNIZER_CONFIG,
        SEPARATOR_CONFIG,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        RECOGNIZER_CONFIG.model_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        SEPARATOR_CONFIG.shifts = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONTENT_RECOGNIZER_MODEL.model_id = "other"
