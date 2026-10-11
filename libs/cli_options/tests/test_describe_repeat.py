import argparse

import pytest

from cli_options import RangeValidator, describe_options

_EXPECTED_KEYS = {"name", "type", "constraint", "default", "help", "repeat"}


def _parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input", nargs="?", help="入力")
    parser.add_argument("--amount", type=RangeValidator(value_type="float", minimum=0), default=0.5, help="量")
    parser.add_argument("--range", dest="ranges", action="append", type=str, help="範囲(反復可)")
    parser.add_argument("--bone", action="append", default=[], help="ボーン(反復可)")
    parser.add_argument("--keep-frame", dest="keep_frames", action="append",
                        type=RangeValidator(value_type="int", minimum=0), default=[], help="保持(反復可)")
    parser.add_argument("--tag", action="extend", nargs="+", default=[], help="タグ(反復可)")
    parser.add_argument("--smooth", default=True, action=argparse.BooleanOptionalAction, help="整理")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="検査のみ")
    return parser


_TYPE_TABLE = {
    "input": ("str", None),
    "amount": (None, None),
    "ranges": ("str", None),
    "bone": ("str", None),
    "keep_frames": (None, None),
    "tag": ("str", None),
    "smooth": ("flag", None),
    "dry_run": ("flag", None),
}


def _by_name(**kwargs):
    return {option["name"]: option for option in describe_options(_parser(), _TYPE_TABLE, **kwargs)}


def test_every_element_has_exactly_the_six_keys_including_repeat():
    for option in describe_options(_parser(), _TYPE_TABLE):
        assert set(option) == _EXPECTED_KEYS


def test_repeat_is_a_boolean_for_every_element():
    for option in describe_options(_parser(), _TYPE_TABLE):
        assert isinstance(option["repeat"], bool)


@pytest.mark.parametrize("name", ["--range", "--bone", "--keep-frame", "--tag"])
def test_repeatable_arguments_have_repeat_true(name):
    assert _by_name()[name]["repeat"] is True


@pytest.mark.parametrize("name", ["input", "--amount", "--smooth", "--dry-run"])
def test_non_repeatable_arguments_have_repeat_false(name):
    assert _by_name()[name]["repeat"] is False


@pytest.mark.parametrize("name", ["--range", "--bone", "--keep-frame", "--tag"])
def test_repeatable_arguments_publish_no_default_even_when_the_parser_default_is_an_empty_list(name):
    assert _by_name()[name]["default"] is None


def test_non_repeatable_defaults_still_come_from_the_parser():
    assert _by_name()["--amount"]["default"] == 0.5
    assert _by_name()["--smooth"]["default"] is True
    assert _by_name()["--dry-run"]["default"] is False


def test_repeatable_argument_still_gets_type_and_constraint_from_its_validator():
    keep = _by_name()["--keep-frame"]
    assert keep["type"] == "int"
    assert keep["constraint"] == {"min": 0, "max": None, "exclusive_min": False}


def test_defaults_override_replaces_the_default_of_the_listed_arguments_only():
    options = _by_name(defaults={"amount": 0.9, "dry_run": True})
    assert options["--amount"]["default"] == 0.9
    assert options["--dry-run"]["default"] is True
    assert options["--smooth"]["default"] is True


def test_defaults_override_is_keyed_by_destination_name_not_option_string():
    options = _by_name(defaults={"ranges": [1, 2], "--amount": 7})
    assert options["--range"]["default"] == [1, 2]
    assert options["--amount"]["default"] == 0.5


def test_defaults_override_does_not_change_the_parser():
    parser = _parser()
    describe_options(parser, _TYPE_TABLE, defaults={"amount": 0.9})
    assert parser.get_default("amount") == 0.5


def test_defaults_override_may_fill_an_argument_whose_parser_default_is_none():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--freq", type=RangeValidator(value_type="float", minimum=0, exclusive_min=True))
    (option,) = describe_options(parser, {"freq": (None, None)}, defaults={"freq": 1.2})
    assert option["default"] == 1.2


def test_tuple_default_override_is_published_as_given():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--weights", type=str)
    (option,) = describe_options(parser, {"weights": ("str", None)}, defaults={"weights": (1.0, 1.0, 0.3)})
    assert option["default"] == (1.0, 1.0, 0.3)


def test_absent_defaults_argument_leaves_everything_as_before():
    assert describe_options(_parser(), _TYPE_TABLE) == describe_options(_parser(), _TYPE_TABLE, defaults=None)
    assert describe_options(_parser(), _TYPE_TABLE) == describe_options(_parser(), _TYPE_TABLE, defaults={})
