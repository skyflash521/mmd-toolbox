import argparse

import pytest

from cli_options import CompoundValidator, RangeValidator


def _reason(validator, text):
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        validator(text)
    return str(excinfo.value)


def _range_endpoints():
    return CompoundValidator(
        format="START:END",
        elements=[(name, RangeValidator(value_type="int", minimum=0, allow_empty=True))
                  for name in ("START", "END")],
        relation=("START<=END が必要",
                  lambda values: values[0] is None or values[1] is None or values[0] <= values[1]))


def _rot_weights():
    return CompoundValidator(
        format="P,Y,R", separator=",",
        elements=[(name, RangeValidator(value_type="float")) for name in ("P", "Y", "R")])


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
        ("3:9", (3, 9)), ("3:", (3, None)),
    ])
    def test_range_with_omittable_ends_returns_none_for_omitted_ends(self, text, expected):
        assert _range_endpoints()(text) == expected


class TestFixedFormatsOfTheExistingTools:
    def test_rot_weights_rejects_more_elements_than_the_format(self):
        assert "P,Y,R" in _reason(_rot_weights(), "1,2,3,4")


class TestDescribeDerivedFromValidators:
    def test_validator_rejection_surfaces_as_a_usage_error_naming_the_argument(self, capsys):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--amp-rot", dest="amp_rot", type=RangeValidator(value_type="float", minimum=0))
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--amp-rot", "-1"])
        assert excinfo.value.code == 2
        assert "argument --amp-rot" in capsys.readouterr().err
