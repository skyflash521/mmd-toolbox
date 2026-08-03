"""音声前段の共通引数群の定義・検証・設定への解決の単体テスト。

音声前段の重い依存は取り込まないので、実モデル・GPU・ネットワークを要さない。
"""

import argparse
import os
from pathlib import Path

import pytest

from cli_options import RangeValidator, describe_options
from vocal_analysis import (
    DEFAULT_CONTENT_RECOGNIZER_MODEL,
    DEFAULT_ENGLISH_KATAKANA_METHOD,
    DEFAULT_FORCED_ALIGNER,
    DEFAULT_SEPARATOR,
    SEPARATOR_IDS,
    ChunkingPolicy,
    ContentRecognizerModel,
    SofaAlignerConfig,
)
from vocal_analysis_cli import (
    add_arguments,
    apply_device,
    describe_type_table,
    missing_dependency_message,
    resolve_chunking_policy,
    resolve_recognizer_model,
    resolve_sofa_config,
    validate,
)

_SOFA = ["--forced-aligner", "sofa-forcedalign",
         "--sofa-python", "/venv/python", "--sofa-root", "/sofa",
         "--sofa-checkpoint", "/ckpt.ckpt"]


def _parse(argv=()):
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    return parser.parse_args(list(argv))


def _actions():
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    return parser._actions


# --- 登録と格納先 --------------------------------------------------------------


def test_every_argument_is_registered_and_readable_after_parsing():
    args = _parse()
    assert vars(args) == {
        "separate_vocals": "always",
        "separator": DEFAULT_SEPARATOR,
        "recognizer_model_id": None,
        "recognizer_model_revision": None,
        "recognizer_retry": True,
        "forced_aligner": DEFAULT_FORCED_ALIGNER,
        "sofa_python": None,
        "sofa_root": None,
        "sofa_checkpoint": None,
        "sofa_timeout": SofaAlignerConfig.__dataclass_fields__["timeout_sec"].default,
        "english_katakana_method": DEFAULT_ENGLISH_KATAKANA_METHOD,
        "device": "auto",
        "max_duration": 300.0,
    }


def test_destination_names_come_from_the_long_form():
    # 格納先名は長形式からハイフンを置き換えて導く(真偽の対は肯定形の名前を共有する)。
    for action in _actions():
        long_forms = [s for s in action.option_strings if s.startswith("--")]
        positive = [s for s in long_forms if not s.startswith("--no-")]
        expected = (positive[0] if positive else long_forms[0])[2:].replace("-", "_")
        if positive:
            assert action.dest == expected
        else:
            assert action.dest == expected.removeprefix("no_")


def test_boolean_pair_shares_one_destination():
    assert _parse(["--no-recognizer-retry"]).recognizer_retry is False
    assert _parse(["--recognizer-retry"]).recognizer_retry is True


# --- 音声前段に対応物を持つ引数 --------------------------------------------------


def _choices_of(name):
    return next(a.choices for a in _actions() if name in a.option_strings)


def test_choices_and_defaults_follow_the_front_stage_registrations():
    assert tuple(_choices_of("--separator")) == tuple(SEPARATOR_IDS)
    assert _parse().separator == DEFAULT_SEPARATOR
    assert set(_choices_of("--forced-aligner")) == {"wav2vec2-ctc-forcedalign", "sofa-forcedalign"}
    assert _parse().forced_aligner == DEFAULT_FORCED_ALIGNER
    assert set(_choices_of("--english-katakana-method")) == {
        "arpakana", "tinyllama-katakana-converter"}
    assert _parse().english_katakana_method == DEFAULT_ENGLISH_KATAKANA_METHOD
    assert _parse().sofa_timeout == SofaAlignerConfig.__dataclass_fields__["timeout_sec"].default


def test_separate_vocals_and_device_choices_are_defined_here():
    assert tuple(_choices_of("--separate-vocals")) == ("always", "never")
    assert tuple(_choices_of("--device")) == ("auto", "cpu")


# --- 数値引数の受理集合と自己記述 ------------------------------------------------


@pytest.mark.parametrize("argv, dest, value", [
    (["--sofa-timeout", "0.5"], "sofa_timeout", 0.5),
    (["--max-duration", "0"], "max_duration", 0.0),
    (["--max-duration", "600"], "max_duration", 600.0),
])
def test_numeric_arguments_accept_their_range(argv, dest, value):
    assert getattr(_parse(argv), dest) == value


@pytest.mark.parametrize("argv", [
    ["--sofa-timeout", "0"],
    ["--sofa-timeout", "-1"],
    ["--sofa-timeout", "nan"],
    ["--max-duration", "-1"],
    ["--max-duration", "inf"],
])
def test_numeric_arguments_reject_outside_their_range(argv):
    with pytest.raises(SystemExit):
        _parse(argv)


def test_numeric_type_and_constraint_are_derived_from_the_validators():
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    options = {o["name"]: o for o in describe_options(parser, describe_type_table())}
    for name in ("--sofa-timeout", "--max-duration"):
        validator = next(a.type for a in parser._actions if name in a.option_strings)
        assert isinstance(validator, RangeValidator)
        assert options[name]["type"] == validator.value_type
        assert options[name]["constraint"] == validator.constraint


def test_type_table_covers_every_registered_argument():
    table = describe_type_table()
    assert {a.dest for a in _actions()} == set(table)


def test_enum_and_string_and_flag_entries_have_their_shapes():
    table = describe_type_table()
    assert table["separator"] == ("enum", {"choices": list(SEPARATOR_IDS)})
    assert table["recognizer_model_id"] == ("str", None)
    assert table["recognizer_retry"] == ("flag", None)


# --- 組み合わせ検証 --------------------------------------------------------------


def test_no_violation_returns_nothing():
    assert validate(_parse()) is None
    assert validate(_parse(_SOFA)) is None


@pytest.mark.parametrize("missing, expected", [
    ("--sofa-python", "--sofa-python"),
    ("--sofa-root", "--sofa-root"),
    ("--sofa-checkpoint", "--sofa-checkpoint"),
])
def test_sofa_requires_three_items(missing, expected):
    # オプションと値は対で並んでいるので、落とす引数はその値も一緒に落とす。
    argv = [a for i, a in enumerate(_SOFA) if _SOFA[i - 1] != missing and a != missing]
    name, reason = validate(_parse(argv))
    assert name == expected
    assert expected in reason


def test_sofa_missing_items_are_reported_one_at_a_time_in_scan_order():
    args = _parse(["--forced-aligner", "sofa-forcedalign"])
    assert validate(args)[0] == "--sofa-python"
    args = _parse(["--forced-aligner", "sofa-forcedalign", "--sofa-python", "/venv/python"])
    assert validate(args)[0] == "--sofa-root"


def test_sofa_items_are_not_required_on_the_default_route():
    assert validate(_parse(["--sofa-python", "/venv/python"])) is None


def test_revision_without_model_id_is_a_violation():
    name, reason = validate(_parse(["--recognizer-model-revision", "rev1"]))
    assert name == "--recognizer-model-revision"
    assert "--recognizer-model-id" in reason


def test_model_id_without_revision_is_allowed():
    assert validate(_parse(["--recognizer-model-id", "org/model"])) is None


def test_sofa_violation_is_reported_before_the_revision_violation():
    args = _parse(["--forced-aligner", "sofa-forcedalign",
                   "--recognizer-model-revision", "rev1"])
    assert validate(args)[0] == "--sofa-python"


# --- 設定への解決 ----------------------------------------------------------------


def test_recognizer_model_falls_back_to_the_front_stage_default():
    assert resolve_recognizer_model(_parse()) is DEFAULT_CONTENT_RECOGNIZER_MODEL


def test_recognizer_model_uses_the_given_id_and_revision():
    args = _parse(["--recognizer-model-id", "org/model", "--recognizer-model-revision", "rev1"])
    assert resolve_recognizer_model(args) == ContentRecognizerModel(
        model_id="org/model", model_revision="rev1")


def test_sofa_config_is_none_on_the_default_route():
    assert resolve_sofa_config(_parse()) is None


def test_sofa_config_carries_paths_and_timeout():
    args = _parse([*_SOFA, "--sofa-timeout", "120"])
    assert resolve_sofa_config(args) == SofaAlignerConfig(
        sofa_python=Path("/venv/python"), sofa_root=Path("/sofa"),
        checkpoint_path=Path("/ckpt.ckpt"), timeout_sec=120.0)


def test_chunking_policy_uses_the_target_duration_and_the_policy_defaults():
    policy = resolve_chunking_policy(_parse(["--max-duration", "600"]))
    assert policy == ChunkingPolicy(max_duration_sec=600.0)
    assert (policy.search_window_sec, policy.overlap_sec) == (5.0, 1.0)


def test_chunking_policy_is_none_when_splitting_is_disabled():
    assert resolve_chunking_policy(_parse(["--max-duration", "0"])) is None


# --- 実行デバイスの適用 ----------------------------------------------------------


def test_cpu_hides_the_gpu_from_the_process(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    apply_device(_parse(["--device", "cpu"]))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def test_auto_leaves_the_environment_untouched(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    apply_device(_parse())
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    apply_device(_parse())
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


# --- 追加依存の未導入 ------------------------------------------------------------


def test_missing_dependency_message_names_the_module_and_the_install_command():
    message = missing_dependency_message(ModuleNotFoundError("No module named 'torch'", name="torch"))
    assert "'torch'" in message
    assert 'pip install ".[vocal-analysis]"' in message


def test_missing_dependency_message_falls_back_to_the_exception_text():
    message = missing_dependency_message(ImportError("cannot import name 'x'"))
    assert "cannot import name 'x'" in message


def test_missing_dependency_message_does_not_name_a_tool():
    message = missing_dependency_message(ModuleNotFoundError("No module named 'torch'", name="torch"))
    # どのツールから呼ばれても同じ追加依存を指すので、ツール名は入れない。
    assert "song2vmd" not in message and "song2vpr" not in message
