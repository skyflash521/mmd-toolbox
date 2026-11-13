import json

import pytest

from song2vpr import cli

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


def test_describe_result_has_exactly_four_keys(capsysbinary):
    event = _describe(capsysbinary)
    assert set(event) == {"type", "mode", "options", "presets"}


def test_options_are_exactly_own_and_shared_arguments_without_duplicates(capsysbinary):
    expected = set(_OWN_OPTION_TYPE_CONSTRAINT_DEFAULT) | _SHARED_OPTIONS
    options = _describe(capsysbinary)["options"]
    assert len(options) == len(expected)
    assert {option["name"] for option in options} == expected


def test_option_help_is_non_empty(capsysbinary):
    for option in _options_by_name(capsysbinary).values():
        assert option["help"]


def test_time_signature_help_tells_the_power_of_two_denominator(capsysbinary):
    assert "2の冪" in _options_by_name(capsysbinary)["--time-signature"]["help"]


@pytest.mark.parametrize("name, expected", sorted(_OWN_OPTION_TYPE_CONSTRAINT_DEFAULT.items()))
def test_own_option_type_constraint_default(capsysbinary, name, expected):
    option = _options_by_name(capsysbinary)[name]
    assert (option["type"], option["constraint"], option["default"]) == expected


def test_presets_is_empty_list(capsysbinary):
    assert _describe(capsysbinary)["presets"] == []
