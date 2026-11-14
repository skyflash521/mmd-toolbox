import dataclasses
import typing
from pathlib import Path

import pytest


def test_separator_model_and_stem_are_pinned():
    from vocal_analysis import SEPARATOR_CONFIG

    assert SEPARATOR_CONFIG.model_filename == "htdemucs_ft.yaml"
    assert SEPARATOR_CONFIG.output_single_stem == "vocals"


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
