import argparse
import json
import math

import pytest

from cli_options import CompoundValidator, RangeValidator, describe_options
from song2vmd import __version__, cli

_NONNEG_INT = {"min": 0, "max": None, "exclusive_min": False}
_UNIT = {"min": 0, "max": 1, "exclusive_min": False}
_POS = {"min": 0, "max": None, "exclusive_min": True}


def _ffield(name, mn, mx, ex):
    return {"name": name, "type": "float", "min": mn, "max": mx, "exclusive_min": ex}


EXPECTED_TYPE_CONSTRAINT_DEFAULT = {
    "input": ("str", None, None),
    "--output": ("str", None, None),
    "--overwrite": ("flag", None, False),
    "--model-name": ("str", None, f"song2vmd {__version__}"),
    "--style": ("enum", {"choices": ["pop", "ballad", "powerful", "whisper", "rap"]}, "pop"),
    "--separate-vocals": ("enum", {"choices": ["always", "never"]}, "always"),
    "--separator": ("enum", {"choices": ["audio-separator-htdemucs-ft"]}, "audio-separator-htdemucs-ft"),
    "--recognizer-model-id": ("str", None, None),
    "--recognizer-model-revision": ("str", None, None),
    "--recognizer-retry": ("flag", None, True),
    "--forced-aligner": ("enum", {"choices": ["wav2vec2-ctc-forcedalign", "sofa-forcedalign"]},
                         "wav2vec2-ctc-forcedalign"),
    "--english-katakana-method": ("enum", {"choices": ["arpakana", "tinyllama-katakana-converter"]},
                                      "arpakana"),
    "--device": ("enum", {"choices": ["auto", "cpu"]}, "auto"),
    "--sofa-python": ("str", None, None),
    "--sofa-root": ("str", None, None),
    "--sofa-checkpoint": ("str", None, None),
    "--sofa-timeout": ("float", _POS, 300.0),
    "--n-morph": ("flag", None, False),
    "--vowel-gain": ("compound",
                     {"format": "a:i:u:e:o",
                      "fields": [_ffield(n, 0, None, False) for n in ("a", "i", "u", "e", "o")]},
                     [1.0, 1.0, 1.0, 1.0, 1.0]),
    "--open-max": ("float", _UNIT, None),
    "--coarticulation": ("int", _NONNEG_INT, None),
    "--anticipation": ("int", _NONNEG_INT, None),
    "--min-hold": ("int", _NONNEG_INT, None),
    "--intensity-curve": ("float", _POS, 0.6),
    "--silence-threshold": ("compound",
                           {"format": "ON:OFF",
                            "fields": [_ffield("ON", 0, 1, False), _ffield("OFF", 0, 1, False)]},
                           [0.06, 0.10]),
    "--max-duration": ("float", {"min": 0, "max": None, "exclusive_min": False}, 300.0),
    "--dry-run": ("flag", None, False),
    "--keep-intermediate": ("flag", None, False),
    "--verbose": ("flag", None, False),
    "--quiet": ("flag", None, False),
}

EXPECTED_PRESETS = {
    "pop": {"open_max": 0.90, "coarticulation": 6, "anticipation": 11, "min_hold": 1},
    "ballad": {"open_max": 0.70, "coarticulation": 3, "anticipation": 1, "min_hold": 4},
    "powerful": {"open_max": 0.97, "coarticulation": 2, "anticipation": 2, "min_hold": 3},
    "whisper": {"open_max": 0.50, "coarticulation": 2, "anticipation": 1, "min_hold": 3},
    "rap": {"open_max": 0.85, "coarticulation": 1, "anticipation": 1, "min_hold": 2},
}


def single_describe_result(capsysbinary):
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert len(events) == 1 and events[0]["type"] == "result" and events[0]["mode"] == "describe"
    return events[0]


def test_describe_stdout_is_a_single_lf_terminated_json_line(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n") and b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 1 and all(lines)
    assert "type" in json.loads(lines[0])


def test_describe_emits_result_without_input(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = single_describe_result(capsysbinary)
    assert isinstance(r["options"], list) and r["options"]
    assert isinstance(r["presets"], list) and r["presets"]
    assert set(r) == {"type", "mode", "options", "presets"}


def test_describe_works_without_machine_flag(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    assert single_describe_result(capsysbinary)["mode"] == "describe"


def test_describe_with_nonexistent_input_still_describes(capsysbinary):
    rc = cli.main(["--describe", "does_not_exist.wav"])
    assert rc == 0
    assert single_describe_result(capsysbinary)["mode"] == "describe"


def test_describe_options_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = single_describe_result(capsysbinary)
    by_name = {o["name"]: o for o in r["options"]}
    for meta in ("--describe", "--version", "--help", "--machine"):
        assert meta not in by_name
    assert "--no-n-morph" not in by_name
    assert len(r["options"]) == len(EXPECTED_TYPE_CONSTRAINT_DEFAULT)
    for o in r["options"]:
        assert set(o) == {"name", "type", "constraint", "default", "help"}
        assert isinstance(o["help"], str) and o["help"]
    assert set(by_name) == set(EXPECTED_TYPE_CONSTRAINT_DEFAULT)
    for name, (type_, constraint, default) in EXPECTED_TYPE_CONSTRAINT_DEFAULT.items():
        o = by_name[name]
        assert o["type"] == type_, name
        assert o["constraint"] == constraint, name
        assert o["default"] == default, name


def test_describe_presets_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = single_describe_result(capsysbinary)
    for p in r["presets"]:
        assert set(p) == {"name", "values"}
        assert set(p["values"]) == {"open_max", "coarticulation", "anticipation", "min_hold"}
    assert {p["name"]: p["values"] for p in r["presets"]} == EXPECTED_PRESETS


def test_describe_type_table_covers_non_meta_args():
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._DESCRIBE_TYPE_TABLE)


def test_describe_without_machine_reports_arg_error_as_error_event(capsysbinary):
    rc = cli.main(["--describe", "--max-duration", "abc"])
    assert rc == 2
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "bad_argument"
    assert events[-1]["field"] == "--max-duration" and events[-1]["exit_code"] == 2


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


def _options_by_name(parser):
    return {o["name"]: o for o in describe_options(parser, cli._DESCRIBE_TYPE_TABLE)}


def _rejects(validator, text):
    try:
        validator(text)
    except (argparse.ArgumentTypeError, ValueError):
        return True
    return False


_NUMERIC_NAMES = [n for n, (type_, c, _) in EXPECTED_TYPE_CONSTRAINT_DEFAULT.items()
                  if type_ in ("int", "float") and c is not None]
_COMPOUND_NAMES = [n for n, (type_, _, _) in EXPECTED_TYPE_CONSTRAINT_DEFAULT.items() if type_ == "compound"]

_COMPOUND_VALID_BASELINES = {
    "--vowel-gain": [1.0, 1.0, 1.0, 1.0, 1.0],
    "--silence-threshold": [0.1, 0.5],
}

_COMPOUND_BOUNDS_ACCEPTED_WITH_BASELINE = {
    "--vowel-gain": [(i, "min") for i in range(5)],
    "--silence-threshold": [(0, "min"), (1, "max")],
}


@pytest.mark.parametrize("name", _COMPOUND_NAMES)
def test_compound_field_constraint_agrees_with_argument_validation(name):
    parser = cli._build_parser()
    fields = _options_by_name(parser)[name]["constraint"]["fields"]
    validator = _actions_by_describe_name(parser)[name].type

    def value_with(index, element):
        values = list(_COMPOUND_VALID_BASELINES[name])
        values[index] = element
        return ":".join(repr(float(v)) for v in values)

    for index, field in enumerate(fields):
        minimum, maximum = field["min"], field["max"]
        if minimum is not None:
            if field["exclusive_min"]:
                assert _rejects(validator, value_with(index, minimum))
            assert _rejects(validator, value_with(index, math.nextafter(float(minimum), -math.inf)))
        if maximum is not None:
            assert _rejects(validator, value_with(index, math.nextafter(float(maximum), math.inf)))
    for index, bound in _COMPOUND_BOUNDS_ACCEPTED_WITH_BASELINE[name]:
        assert not _rejects(validator, value_with(index, fields[index][bound]))


@pytest.mark.parametrize("name", _NUMERIC_NAMES)
def test_numeric_constraint_is_derived_from_the_validator(name):
    parser = cli._build_parser()
    validator = _actions_by_describe_name(parser)[name].type
    assert isinstance(validator, RangeValidator)
    assert _options_by_name(parser)[name]["constraint"] == validator.constraint


@pytest.mark.parametrize("name", _COMPOUND_NAMES)
def test_compound_constraint_is_derived_from_the_element_validators(name):
    parser = cli._build_parser()
    validator = _actions_by_describe_name(parser)[name].type
    assert isinstance(validator, CompoundValidator)

    constraint = _options_by_name(parser)[name]["constraint"]
    assert constraint["format"] == validator.format
    assert [f["name"] for f in constraint["fields"]] == [n for n, _ in validator.elements]
    for field, (_, element) in zip(constraint["fields"], validator.elements, strict=True):
        assert isinstance(element, RangeValidator)
        assert {k: field[k] for k in ("min", "max", "exclusive_min")} == element.constraint
