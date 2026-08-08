"""範囲付き数値・複合トークンの検証子の単体テスト。"""

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


def _just_below(value, *, is_int):
    """範囲のすぐ外側(下)にある値の文字列。実数は表現可能な最も近い値を採る。"""
    return str(int(value) - 1) if is_int else repr(math.nextafter(float(value), -math.inf))


def _just_above(value, *, is_int):
    """範囲のすぐ外側(上)にある値の文字列。"""
    return str(int(value) + 1) if is_int else repr(math.nextafter(float(value), math.inf))


def _assert_accepts_exactly(validator, constraint, *, is_int):
    """検証子の受理範囲が constraint の境界と厳密に一致することを確かめる。

    境界のすぐ外側は隣接する値を採るので、境界そのものと1つでもずれれば落ちる。
    """
    minimum, maximum = constraint["min"], constraint["max"]
    if minimum is not None:
        if constraint["exclusive_min"]:
            assert _rejects(validator, _at(minimum, is_int=is_int))
            assert not _rejects(validator, _just_above(minimum, is_int=is_int))
        else:
            assert not _rejects(validator, _at(minimum, is_int=is_int))
        assert _rejects(validator, _just_below(minimum, is_int=is_int))
    if maximum is not None:
        assert not _rejects(validator, _at(maximum, is_int=is_int))
        assert _rejects(validator, _just_above(maximum, is_int=is_int))
    if is_int:
        # 整数の引数は整数以外の数値も弾く(切り捨てて受理すると指定と違う値で動く)。
        inside = 1.5 if minimum is None else float(minimum) + 0.5
        assert _rejects(validator, repr(inside))
    else:
        # 有限でない値は大小比較をすり抜けるので、範囲の判定とは別に弾かれることを確かめる。
        for text in ("nan", "inf", "-inf"):
            assert _rejects(validator, text)


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
        reason = _reason(validator, "0")
        assert "0 より大きい" in reason

    def test_exclusive_minimum_with_a_maximum_states_both_ends(self):
        # 下限を含まない範囲を「a〜b」と書くと、拒否された下限そのものを受理範囲に見せてしまう。
        validator = RangeValidator(value_type="float", minimum=0, maximum=1, exclusive_min=True)
        reason = _reason(validator, "0")
        assert "0 より大きく 1 以下" in reason

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

    def test_float_type_accepts_python_numeric_spellings(self):
        # 受理する字句は Python の float の変換規則そのもの(独自の字句規則を定義しない)。
        validator = RangeValidator(value_type="float", minimum=0, maximum=1000)
        assert validator("1e2") == 100.0
        assert validator("+3.5") == 3.5
        assert validator(" 2.5 ") == 2.5

    def test_rejects_unparsable_text_with_the_expected_unit(self):
        assert "数値" in _reason(RangeValidator(value_type="float"), "abc")
        assert "整数" in _reason(RangeValidator(value_type="int"), "abc")

    @pytest.mark.parametrize("text", ["inf", "-inf", "nan"])
    def test_rejects_non_finite_float_before_range_judgement(self, text):
        # 無限大・非数は大小比較をすり抜けるため、範囲判定より前に弾く。
        validator = RangeValidator(value_type="float", minimum=0, maximum=1)
        assert "有限" in _reason(validator, text)

    def test_reason_does_not_contain_an_argument_name(self):
        # argparse が理由の前へ引数名を付けるため、理由に引数名を入れると二重になる。
        validator = RangeValidator(value_type="float", minimum=0, maximum=1)
        for text in ("abc", "nan", "2"):
            assert "--" not in _reason(validator, text)

    def test_reason_shows_the_rejected_input(self):
        assert repr("abc") in _reason(RangeValidator(value_type="int"), "abc")


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
    def test_accepts_exactly_its_own_range(self, value_type, minimum, maximum, exclusive_min):
        # 検証子は与えられた範囲そのもので受理判定する(範囲の値と判定ロジックが別管理にならない)。
        validator = RangeValidator(
            value_type=value_type, minimum=minimum, maximum=maximum, exclusive_min=exclusive_min)
        assert validator.constraint == {"min": minimum, "max": maximum, "exclusive_min": exclusive_min}
        _assert_accepts_exactly(validator, validator.constraint, is_int=value_type == "int")


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

    def test_relation_is_checked_after_every_element(self):
        validator = CompoundValidator(
            format="ON:OFF",
            elements=[(name, RangeValidator(value_type="float", minimum=0, maximum=1))
                      for name in ("ON", "OFF")],
            relation=("ON<OFF が必要", lambda values: values[0] < values[1]))
        assert validator("0.2:0.5") == (0.2, 0.5)
        assert "ON<OFF が必要" in _reason(validator, "0.5:0.2")
        # 要素自体が範囲外なら、関係の判定より先に要素の理由で落ちる。
        assert "OFF " in _reason(validator, "0.5:2")

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
        validator = CompoundValidator(
            format="ON:OFF",
            elements=[(name, RangeValidator(value_type="float", minimum=0, maximum=1))
                      for name in ("ON", "OFF")],
            relation=("ON<OFF が必要", lambda values: values[0] < values[1]))
        assert set(validator.constraint) == {"format", "fields"}

    def test_reason_does_not_contain_an_argument_name(self):
        for text in ("1:2", "1:-2:3", "x:1:2"):
            assert "--" not in _reason(self._gain(), text)
