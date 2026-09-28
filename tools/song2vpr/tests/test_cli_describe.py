import json

import pytest

from cli_options import RangeValidator
from song2vpr import cli

_META_AND_MODE_OPERATIONS = {"--describe", "--version", "--help", "--machine"}

_SHARED_OPTIONS = {
    "--separate-vocals", "--separator", "--recognizer-model-id", "--recognizer-model-revision",
    "--recognizer-retry", "--forced-aligner", "--sofa-python", "--sofa-root", "--sofa-checkpoint",
    "--sofa-timeout", "--english-katakana-method", "--device", "--max-duration",
}

_OWN_OPTION_TYPE_CONSTRAINT_DEFAULT = {
    "input": ("str", None, None),
    "--output": ("str", None, None),
    "--overwrite": ("flag", None, False),
    "--lyrics": ("str", None, None),
    "--tempo": ("float", {"min": 0.01, "max": None, "exclusive_min": False}, None),
    "--time-signature": ("compound", {
        "format": "N/D",
        "fields": [
            {"name": "N", "type": "int", "min": 1, "max": None, "exclusive_min": False},
            {"name": "D", "type": "int", "min": 1, "max": 128, "exclusive_min": False},
        ],
    }, None),
    "--dry-run": ("flag", None, False),
    "--keep-intermediate": ("flag", None, False),
    "--verbose": ("flag", None, False),
    "--quiet": ("flag", None, False),
}

_RANGE_VALIDATED_OPTION_NAMES = ["--sofa-timeout", "--max-duration", "--tempo"]


def _describe_raw(capsysbinary, argv=("--describe",)):
    assert cli.main(list(argv)) == 0
    return capsysbinary.readouterr().out


def _describe(capsysbinary, argv=("--describe",)):
    text = _describe_raw(capsysbinary, argv).decode("utf-8")
    events = [json.loads(ln) for ln in text.split("\n") if ln]
    assert len(events) == 1
    return events[0]


def _options_by_name(capsysbinary):
    return {option["name"]: option for option in _describe(capsysbinary)["options"]}


def _actions_by_describe_name(parser):
    actions = {}
    for action in parser._actions:
        if action.dest == "input":
            actions["input"] = action
            continue
        positive = [s for s in action.option_strings
                    if s.startswith("--") and not s.startswith("--no-")]
        if positive:
            actions[positive[0]] = action
    return actions


def test_describe_without_input_ends_with_a_single_result(capsysbinary):
    event = _describe(capsysbinary)
    assert event["type"] == "result"
    assert event["mode"] == "describe"


def test_describe_result_has_exactly_four_keys(capsysbinary):
    event = _describe(capsysbinary)
    assert set(event) == {"type", "mode", "options", "presets"}


def test_describe_does_not_require_machine_flag(capsysbinary):
    event = _describe(capsysbinary)
    assert "options" in event and "presets" in event


def test_describe_with_nonexistent_input_path_still_describes(tmp_path, capsysbinary):
    event = _describe(capsysbinary, ("--describe", str(tmp_path / "absent.wav")))
    assert event["mode"] == "describe"


def test_describe_stdout_is_lf_terminated_json_lines_without_blank_lines(capsysbinary):
    raw = _describe_raw(capsysbinary)
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    lines = raw.decode("utf-8").split("\n")[:-1]
    assert lines and all(line for line in lines)
    for line in lines:
        json.loads(line)


def test_options_have_five_keys_each(capsysbinary):
    for option in _options_by_name(capsysbinary).values():
        assert set(option) == {"name", "type", "constraint", "default", "help"}


def test_options_are_exactly_own_and_shared_arguments_without_duplicates(capsysbinary):
    expected = set(_OWN_OPTION_TYPE_CONSTRAINT_DEFAULT) | _SHARED_OPTIONS
    options = _describe(capsysbinary)["options"]
    assert len(options) == len(expected)
    assert {option["name"] for option in options} == expected


def test_options_exclude_meta_and_mode_operations(capsysbinary):
    assert not (set(_options_by_name(capsysbinary)) & _META_AND_MODE_OPERATIONS)


def test_options_exclude_negated_boolean_flag(capsysbinary):
    assert "--no-recognizer-retry" not in _options_by_name(capsysbinary)


def test_option_help_is_non_empty(capsysbinary):
    for option in _options_by_name(capsysbinary).values():
        assert option["help"]


def test_time_signature_help_tells_the_power_of_two_denominator(capsysbinary):
    assert "2の冪" in _options_by_name(capsysbinary)["--time-signature"]["help"]


@pytest.mark.parametrize("name, expected", sorted(_OWN_OPTION_TYPE_CONSTRAINT_DEFAULT.items()))
def test_own_option_type_constraint_default(capsysbinary, name, expected):
    option = _options_by_name(capsysbinary)[name]
    assert (option["type"], option["constraint"], option["default"]) == expected


@pytest.mark.parametrize("name", _RANGE_VALIDATED_OPTION_NAMES)
def test_numeric_constraint_is_derived_from_the_validator(capsysbinary, name):
    parser = cli._build_parser()
    validator = _actions_by_describe_name(parser)[name].type
    assert isinstance(validator, RangeValidator)
    assert _options_by_name(capsysbinary)[name]["constraint"] == validator.constraint


def test_presets_is_empty_list(capsysbinary):
    assert _describe(capsysbinary)["presets"] == []


def test_type_table_covers_non_meta_args():
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._D_TYPE)


def test_describe_arg_error_without_machine_flag_ends_stdout_with_error_event(capsysbinary):
    assert cli.main(["--describe", "--max-duration", "abc"]) == 2
    text = capsysbinary.readouterr().out.decode("utf-8")
    events = [json.loads(ln) for ln in text.split("\n") if ln]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "bad_argument"
    assert events[-1]["field"] == "--max-duration"
    assert events[-1]["exit_code"] == 2
