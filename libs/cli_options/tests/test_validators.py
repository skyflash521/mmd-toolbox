import argparse
import math

import pytest

from cli_options import CompoundValidator, RangeValidator


def _reason(validator, text):
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        validator(text)
    return str(excinfo.value)


def _rejects(validator, text):
    try:
        validator(text)
    except argparse.ArgumentTypeError:
        return True
    return False


def _at(value, *, is_int):
    return str(int(value)) if is_int else repr(float(value))


def _nearest_below(value, *, is_int):
    return str(int(value) - 1) if is_int else repr(math.nextafter(float(value), -math.inf))


def _nearest_above(value, *, is_int):
    return str(int(value) + 1) if is_int else repr(math.nextafter(float(value), math.inf))


def _assert_accepts_exactly_up_to_the_bounds(validator, constraint, *, is_int):
    minimum, maximum = constraint["min"], constraint["max"]
    if minimum is not None:
        if constraint["exclusive_min"]:
            assert _rejects(validator, _at(minimum, is_int=is_int))
            assert not _rejects(validator, _nearest_above(minimum, is_int=is_int))
        else:
            assert not _rejects(validator, _at(minimum, is_int=is_int))
        assert _rejects(validator, _nearest_below(minimum, is_int=is_int))
    if maximum is not None:
        assert not _rejects(validator, _at(maximum, is_int=is_int))
        assert _rejects(validator, _nearest_above(maximum, is_int=is_int))


def _assert_rejects_fractional_value_inside_the_range(validator, minimum):
    inside = 1.5 if minimum is None else float(minimum) + 0.5
    assert _rejects(validator, repr(inside))


def _assert_rejects_non_finite_values(validator):
    for text in ("nan", "inf", "-inf"):
        assert _rejects(validator, text)


def _on_off_validator():
    return CompoundValidator(
        format="ON:OFF",
        elements=[(name, RangeValidator(value_type="float", minimum=0, maximum=1))
                  for name in ("ON", "OFF")],
        relation=("ON<OFF が必要", lambda values: values[0] < values[1]))


class TestRangeAcceptance:
    def test_closed_range_accepts_both_bounds(self):
        validator = RangeValidator(value_type="float", minimum=0, maximum=1)
        assert validator("0") == 0.0
        assert validator("1") == 1.0
        assert validator("0.25") == 0.25

    def test_closed_range_rejects_outside(self):
        validator = RangeValidator(value_type="float", minimum=0, maximum=1)
        assert "0〜1" in _reason(validator, "-0.0001")
        assert "0〜1" in _reason(validator, "1.0001")

    def test_exclusive_minimum_rejects_the_bound_itself(self):
        validator = RangeValidator(value_type="float", minimum=0, exclusive_min=True)
        assert validator("0.001") == 0.001
        assert "0 より大きい" in _reason(validator, "0")

    def test_exclusive_minimum_with_a_maximum_states_both_ends_not_a_tilde_range(self):
        validator = RangeValidator(value_type="float", minimum=0, maximum=1, exclusive_min=True)
        assert "0 より大きく 1 以下" in _reason(validator, "0")

    def test_inclusive_minimum_accepts_the_bound(self):
        validator = RangeValidator(value_type="float", minimum=0)
        assert validator("0") == 0.0
        assert "0 以上" in _reason(validator, "-1")

    def test_maximum_only(self):
        validator = RangeValidator(value_type="int", minimum=None, maximum=10)
        assert validator("-100") == -100
        assert "10 以下" in _reason(validator, "11")

    def test_unbounded_accepts_any_finite_value(self):
        validator = RangeValidator(value_type="float")
        assert validator("-1e30") == -1e30
        assert validator("1e30") == 1e30

    def test_int_type_returns_int_and_rejects_fractional_text(self):
        validator = RangeValidator(value_type="int", minimum=0)
        value = validator("7")
        assert value == 7
        assert isinstance(value, int)
        assert "整数" in _reason(validator, "7.5")

    def test_float_type_accepts_python_float_spellings(self):
        validator = RangeValidator(value_type="float", minimum=0, maximum=1000)
        assert validator("1e2") == 100.0
        assert validator("+3.5") == 3.5
        assert validator(" 2.5 ") == 2.5

    def test_int_type_accepts_python_int_spellings(self):
        validator = RangeValidator(value_type="int")
        assert validator(" +7 ") == 7
        assert validator("1_000") == 1000

    def test_rejects_unparsable_text_with_the_expected_unit(self):
        assert "数値" in _reason(RangeValidator(value_type="float"), "abc")
        assert "整数" in _reason(RangeValidator(value_type="int"), "abc")

    @pytest.mark.parametrize("text", ["inf", "-inf", "nan"])
    def test_rejects_non_finite_float_as_not_finite_rather_than_out_of_range(self, text):
        validator = RangeValidator(value_type="float", minimum=0, maximum=1)
        assert "有限" in _reason(validator, text)

    def test_reason_does_not_contain_an_argument_name(self):
        validator = RangeValidator(value_type="float", minimum=0, maximum=1)
        for text in ("abc", "nan", "2"):
            assert "--" not in _reason(validator, text)

    @pytest.mark.parametrize("value_type, text", [
        pytest.param("int", "abc", id="unparsable"),
        pytest.param("int", "7.5", id="fractional-int"),
        pytest.param("float", "nan", id="non-finite"),
        pytest.param("float", "2", id="out-of-range"),
    ])
    def test_reason_is_a_single_line_showing_the_rejected_input(self, value_type, text):
        reason = _reason(RangeValidator(value_type=value_type, minimum=0, maximum=1), text)
        assert repr(text) in reason
        assert "\n" not in reason


class TestRangeConstraint:
    def test_constraint_always_has_three_keys(self):
        validator = RangeValidator(value_type="float", minimum=0, maximum=1)
        assert validator.constraint == {"min": 0, "max": 1, "exclusive_min": False}

    def test_unbounded_sides_are_none(self):
        validator = RangeValidator(value_type="int")
        assert validator.constraint == {"min": None, "max": None, "exclusive_min": False}

    def test_exclusive_minimum_is_published(self):
        validator = RangeValidator(value_type="float", minimum=0, exclusive_min=True)
        assert validator.constraint == {"min": 0, "max": None, "exclusive_min": True}

    @pytest.mark.parametrize("value_type", ["float", "int"])
    @pytest.mark.parametrize("minimum, maximum, exclusive_min", [
        (0, 1, False), (0, None, True), (-2, 2, False), (0, None, False),
        (None, 10, False), (0, 5, True),
    ])
    def test_accepts_exactly_the_published_range(self, value_type, minimum, maximum, exclusive_min):
        validator = RangeValidator(
            value_type=value_type, minimum=minimum, maximum=maximum, exclusive_min=exclusive_min)
        assert validator.constraint == {"min": minimum, "max": maximum, "exclusive_min": exclusive_min}
        _assert_accepts_exactly_up_to_the_bounds(
            validator, validator.constraint, is_int=value_type == "int")
        if value_type == "int":
            _assert_rejects_fractional_value_inside_the_range(validator, minimum)
        else:
            _assert_rejects_non_finite_values(validator)


class TestCompound:
    def _gain(self):
        return CompoundValidator(
            format="a:i:u",
            elements=[(name, RangeValidator(value_type="float", minimum=0))
                      for name in ("a", "i", "u")])

    def test_returns_element_values_in_format_order(self):
        assert self._gain()("1:2:3") == (1.0, 2.0, 3.0)

    def test_rejects_wrong_element_count_showing_the_format(self):
        reason = _reason(self._gain(), "1:2")
        assert "a:i:u" in reason
        assert "3" in reason

    def test_rejects_an_out_of_range_element_naming_it(self):
        reason = _reason(self._gain(), "1:-2:3")
        assert reason.startswith("i ")
        assert "0 以上" in reason

    def test_relation_violation_is_rejected_with_its_description(self):
        validator = _on_off_validator()
        assert validator("0.2:0.5") == (0.2, 0.5)
        assert "ON<OFF が必要" in _reason(validator, "0.5:0.2")

    def test_out_of_range_element_is_reported_before_the_relation(self):
        assert "OFF " in _reason(_on_off_validator(), "0.5:2")

    def test_separator_is_configurable(self):
        validator = CompoundValidator(
            format="beats/unit",
            separator="/",
            elements=[(name, RangeValidator(value_type="int", minimum=1))
                      for name in ("beats", "unit")])
        assert validator("3/4") == (3, 4)
        assert "beats/unit" in _reason(validator, "3:4")

    def test_constraint_flattens_each_element_range(self):
        assert self._gain().constraint == {
            "format": "a:i:u",
            "fields": [{"name": name, "type": "float", "min": 0, "max": None, "exclusive_min": False}
                       for name in ("a", "i", "u")],
        }

    def test_relation_is_not_published(self):
        assert set(_on_off_validator().constraint) == {"format", "fields"}

    def test_reason_does_not_contain_an_argument_name(self):
        for text in ("1:2", "1:-2:3", "x:1:2"):
            assert "--" not in _reason(self._gain(), text)
