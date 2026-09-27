import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import get_args

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
    EnglishKatakanaMethod,
    ForcedAlignerId,
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

_VOCAL_ANALYSIS_EXTRA_MODULES = (
    "arpakana", "audio_separator", "huggingface_hub", "librosa", "nltk", "onnxruntime",
    "psutil", "pyopenjtalk", "soundfile", "torch", "tqdm", "transformers",
)


def _parse(argv=()):
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    return parser.parse_args(list(argv))


def _actions():
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    return parser._actions


def _without_option(argv, option):
    i = argv.index(option)
    return argv[:i] + argv[i + 2:]


def test_importing_the_module_does_not_pull_in_the_extra_dependencies():
    code = ("import sys, vocal_analysis_cli; "
            f"print(sorted({{m.split('.')[0] for m in sys.modules}} & set({_VOCAL_ANALYSIS_EXTRA_MODULES!r})))")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert completed.stdout.strip() == "[]"


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


def test_destination_names_come_from_the_long_form_and_the_boolean_pair_uses_the_positive_one():
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


def _choices_of(name):
    return next(a.choices for a in _actions() if name in a.option_strings)


def test_choices_and_defaults_follow_the_front_stage_registrations():
    assert tuple(_choices_of("--separator")) == tuple(SEPARATOR_IDS)
    assert _parse().separator == DEFAULT_SEPARATOR
    assert tuple(_choices_of("--forced-aligner")) == get_args(ForcedAlignerId)
    assert _parse().forced_aligner == DEFAULT_FORCED_ALIGNER
    assert tuple(_choices_of("--english-katakana-method")) == get_args(EnglishKatakanaMethod)
    assert _parse().english_katakana_method == DEFAULT_ENGLISH_KATAKANA_METHOD
    assert _parse().sofa_timeout == SofaAlignerConfig.__dataclass_fields__["timeout_sec"].default


def test_separate_vocals_and_device_choices_are_defined_here():
    assert tuple(_choices_of("--separate-vocals")) == ("always", "never")
    assert tuple(_choices_of("--device")) == ("auto", "cpu")


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


def test_type_table_entry_follows_the_kind_of_each_argument():
    table = describe_type_table()
    for action in _actions():
        if action.choices is not None:
            expected = ("enum", {"choices": list(action.choices)})
        elif action.nargs == 0:
            expected = ("flag", None)
        elif isinstance(action.type, RangeValidator):
            expected = (None, None)
        else:
            expected = ("str", None)
        assert table[action.dest] == expected, action.dest


def test_no_violation_returns_nothing():
    assert validate(_parse()) is None
    assert validate(_parse(_SOFA)) is None


@pytest.mark.parametrize("missing", ["--sofa-python", "--sofa-root", "--sofa-checkpoint"])
def test_sofa_route_reports_each_missing_item_by_its_name(missing):
    name, reason = validate(_parse(_without_option(_SOFA, missing)))
    assert name == missing
    assert missing in reason


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


def test_missing_dependency_message_names_the_module_and_the_install_command():
    message = missing_dependency_message(ModuleNotFoundError("No module named 'torch'", name="torch"))
    assert "'torch'" in message
    assert 'pip install ".[vocal-analysis]"' in message


def test_missing_dependency_message_falls_back_to_the_exception_text():
    message = missing_dependency_message(ImportError("cannot import name 'x'"))
    assert "cannot import name 'x'" in message


def test_missing_dependency_message_does_not_name_a_tool():
    message = missing_dependency_message(ModuleNotFoundError("No module named 'torch'", name="torch"))
    assert "song2vmd" not in message and "song2vpr" not in message
