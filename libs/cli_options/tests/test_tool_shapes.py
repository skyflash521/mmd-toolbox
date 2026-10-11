import argparse

import pytest

from cli_options import CompoundValidator, RangeValidator, describe_options


def _reason(validator, text):
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        validator(text)
    return str(excinfo.value)


def _field(name, type_, minimum, maximum, exclusive_min):
    return {"name": name, "type": type_, "min": minimum, "max": maximum, "exclusive_min": exclusive_min}


def _range_endpoints():
    return CompoundValidator(
        format="START:END",
        elements=[(name, RangeValidator(value_type="int", minimum=0, allow_empty=True))
                  for name in ("START", "END")],
        relation=("START<=END が必要",
                  lambda values: values[0] is None or values[1] is None or values[0] <= values[1]))


def _impulse():
    return CompoundValidator(
        format="F:S:D",
        elements=[("F", RangeValidator(value_type="int", minimum=0)),
                  ("S", RangeValidator(value_type="float", minimum=0)),
                  ("D", RangeValidator(value_type="float", minimum=0, exclusive_min=True))])


def _rot_weights():
    return CompoundValidator(
        format="P,Y,R", separator=",",
        elements=[(name, RangeValidator(value_type="float")) for name in ("P", "Y", "R")])


def _cut_threshold():
    return CompoundValidator(
        format="位置,角度", separator=",",
        elements=[(name, RangeValidator(value_type="float", minimum=0)) for name in ("位置", "角度")])


class TestOmittableEndpoints:
    def test_empty_text_is_none_when_allowed(self):
        validator = RangeValidator(value_type="int", minimum=0, allow_empty=True)
        assert validator("") is None

    def test_non_empty_text_is_still_validated_when_empty_is_allowed(self):
        validator = RangeValidator(value_type="int", minimum=0, allow_empty=True)
        assert validator("7") == 7
        assert "0 以上" in _reason(validator, "-1")
        assert "整数" in _reason(validator, "x")

    def test_empty_text_is_rejected_by_default(self):
        assert "整数" in _reason(RangeValidator(value_type="int", minimum=0), "")
        assert "数値" in _reason(RangeValidator(value_type="float"), "")

    def test_allowing_empty_does_not_change_the_published_constraint(self):
        plain = RangeValidator(value_type="int", minimum=0)
        omittable = RangeValidator(value_type="int", minimum=0, allow_empty=True)
        assert omittable.constraint == plain.constraint
        assert omittable.value_type == plain.value_type

    @pytest.mark.parametrize("text, expected", [
        ("3:9", (3, 9)), ("3:", (3, None)), (":9", (None, 9)), (":", (None, None)), ("5:5", (5, 5)),
    ])
    def test_range_with_omittable_ends_returns_none_for_omitted_ends(self, text, expected):
        assert _range_endpoints()(text) == expected

    def test_range_with_start_after_end_is_rejected_by_the_relation(self):
        assert "START<=END が必要" in _reason(_range_endpoints(), "9:3")

    def test_range_without_a_separator_is_rejected_showing_the_format(self):
        assert "START:END" in _reason(_range_endpoints(), "9")

    def test_range_with_a_negative_end_names_the_element(self):
        assert _reason(_range_endpoints(), "3:-1").startswith("END ")

    def test_range_constraint_is_the_hand_written_shape_of_the_existing_tools(self):
        assert _range_endpoints().constraint == {
            "format": "START:END",
            "fields": [_field("START", "int", 0, None, False), _field("END", "int", 0, None, False)]}


class TestFixedFormatsOfTheExistingTools:
    def test_impulse_accepts_int_frame_and_float_strength_and_positive_decay(self):
        assert _impulse()("12:0.5:0.25") == (12, 0.5, 0.25)

    @pytest.mark.parametrize("text, element", [
        ("-1:0.5:0.25", "F"), ("1.5:0.5:0.25", "F"), ("1:-0.5:0.25", "S"), ("1:0.5:0", "D"),
        ("1:0.5:nan", "D"), ("1:inf:1", "S"),
    ])
    def test_impulse_rejects_a_bad_element_naming_it(self, text, element):
        assert _reason(_impulse(), text).startswith(f"{element} ")

    def test_impulse_with_the_wrong_element_count_shows_the_format(self):
        assert "F:S:D" in _reason(_impulse(), "1:2")

    def test_impulse_constraint_is_the_hand_written_shape_of_the_existing_tool(self):
        assert _impulse().constraint == {
            "format": "F:S:D",
            "fields": [_field("F", "int", 0, None, False), _field("S", "float", 0, None, False),
                       _field("D", "float", 0, None, True)]}

    def test_rot_weights_accepts_any_finite_floats_with_a_comma_separator(self):
        assert _rot_weights()("1,-1,0.3") == (1.0, -1.0, 0.3)

    @pytest.mark.parametrize("text", ["1,2", "1,2,3,4", "1,x,3", "1,nan,3", "1,inf,3"])
    def test_rot_weights_rejects_wrong_count_and_non_finite_values(self, text):
        _reason(_rot_weights(), text)

    def test_rot_weights_constraint_has_unbounded_float_fields(self):
        assert _rot_weights().constraint == {
            "format": "P,Y,R",
            "fields": [_field(name, "float", None, None, False) for name in ("P", "Y", "R")]}

    def test_cut_threshold_rejects_negative_values(self):
        assert _cut_threshold()("5,20") == (5.0, 20.0)
        assert _reason(_cut_threshold(), "5,-1").startswith("角度 ")

    def test_cut_threshold_constraint_is_the_hand_written_shape_of_the_existing_tool(self):
        assert _cut_threshold().constraint == {
            "format": "位置,角度",
            "fields": [_field("位置", "float", 0, None, False), _field("角度", "float", 0, None, False)]}


class TestDescribeDerivedFromValidators:
    @staticmethod
    def _parser():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("input", nargs="?")
        parser.add_argument("--range", dest="ranges", action="append", type=_range_endpoints())
        parser.add_argument("--impulse", dest="impulses", action="append", type=_impulse())
        parser.add_argument("--rot-weights", dest="rot_weights", type=_rot_weights())
        parser.add_argument("--amp-rot", dest="amp_rot", type=RangeValidator(value_type="float", minimum=0))
        parser.add_argument("--freq", type=RangeValidator(value_type="float", minimum=0, exclusive_min=True))
        parser.add_argument("--seed", type=int, default=1)
        return parser

    _TABLE = {
        "input": ("str", None),
        "ranges": (None, None),
        "impulses": (None, None),
        "rot_weights": (None, None),
        "amp_rot": (None, None),
        "freq": (None, None),
        "seed": ("int", None),
    }

    def _described(self):
        return {o["name"]: o for o in describe_options(self._parser(), self._TABLE)}

    def test_compound_arguments_are_typed_compound_with_the_validator_constraint(self):
        described = self._described()
        for name, validator in (("--range", _range_endpoints()), ("--impulse", _impulse()),
                                ("--rot-weights", _rot_weights())):
            assert described[name]["type"] == "compound"
            assert described[name]["constraint"] == validator.constraint

    def test_repeatable_compound_arguments_are_marked_repeat(self):
        described = self._described()
        assert described["--range"]["repeat"] is True
        assert described["--impulse"]["repeat"] is True
        assert described["--rot-weights"]["repeat"] is False

    def test_numeric_arguments_take_type_and_constraint_from_their_validator(self):
        described = self._described()
        assert described["--amp-rot"]["type"] == "float"
        assert described["--amp-rot"]["constraint"] == {"min": 0, "max": None, "exclusive_min": False}
        assert described["--freq"]["constraint"] == {"min": 0, "max": None, "exclusive_min": True}

    def test_described_constraint_matches_what_the_parser_actually_accepts(self):
        parser = self._parser()
        assert parser.parse_args(["--freq", "0.001"]).freq == 0.001
        with pytest.raises(SystemExit):
            parser.parse_args(["--freq", "0"])
        assert self._described()["--freq"]["constraint"]["exclusive_min"] is True

    def test_validator_rejection_surfaces_as_a_usage_error_naming_the_argument(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            self._parser().parse_args(["--amp-rot", "-1"])
        assert excinfo.value.code == 2
        assert "argument --amp-rot" in capsys.readouterr().err
