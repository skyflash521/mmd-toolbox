"""自己記述の options 配列の導出の単体テスト。"""

import argparse

from cli_options import CompoundValidator, RangeValidator, describe_options

_UNIT_FLOAT = RangeValidator(value_type="float", minimum=0, maximum=1)
_COUNT = CompoundValidator(
    format="w:h",
    elements=[(name, RangeValidator(value_type="int", minimum=1)) for name in ("w", "h")])


def _parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default=None, help="出力先")
    parser.add_argument("--amount", type=_UNIT_FLOAT, default=0.5, help="開き量")
    parser.add_argument("--size", type=_COUNT, default=(1, 1), help="大きさ")
    parser.add_argument("--preset", choices=("a", "b"), default="a", help="プリセット")
    parser.add_argument("--trim", action="store_true", default=True, help="端を落とす")
    parser.add_argument("--no-trim", dest="trim", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--machine", action="store_true", help="機械可読出力")
    return parser


_TYPE_TABLE = {
    "input": ("str", None),
    "output": ("str", None),
    "amount": (None, None),
    "size": (None, None),
    "preset": ("enum", {"choices": ["a", "b"]}),
    "trim": ("flag", None),
}


def _described():
    return describe_options(_parser(), _TYPE_TABLE)


def _by_name():
    return {option["name"]: option for option in _described()}


def test_order_follows_registration():
    assert [option["name"] for option in _described()] == [
        "input", "--output", "--amount", "--size", "--preset", "--trim"]


def test_arguments_absent_from_the_table_are_excluded():
    assert "--machine" not in _by_name()


def test_negated_form_alone_is_excluded():
    # 否定形は肯定形と同じ格納先で、肯定形の長形式が既に載る。
    assert "--no-trim" not in _by_name()
    assert _by_name()["--trim"]["help"] == "端を落とす"


def test_positional_uses_its_destination_name():
    assert _by_name()["input"]["name"] == "input"


def test_every_element_has_the_same_five_keys():
    for option in _described():
        assert set(option) == {"name", "type", "constraint", "default", "help"}


def test_table_entries_are_used_as_given():
    preset = _by_name()["--preset"]
    assert preset["type"] == "enum"
    assert preset["constraint"] == {"choices": ["a", "b"]}
    assert _by_name()["--output"]["constraint"] is None


def test_range_validator_supplies_type_and_constraint():
    amount = _by_name()["--amount"]
    assert amount["type"] == "float"
    assert amount["constraint"] == _UNIT_FLOAT.constraint


def test_compound_validator_is_typed_as_compound():
    size = _by_name()["--size"]
    assert size["type"] == "compound"
    assert size["constraint"] == _COUNT.constraint


def test_default_and_help_come_from_the_parser():
    assert _by_name()["--amount"]["default"] == 0.5
    assert _by_name()["--amount"]["help"] == "開き量"
    assert _by_name()["--output"]["default"] is None


def test_long_form_is_preferred_over_the_short_form():
    assert "--output" in _by_name()
    assert "-o" not in _by_name()
