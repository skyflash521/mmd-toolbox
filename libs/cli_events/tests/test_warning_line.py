import pytest

from cli_events import format_warning_line


def test_line_has_label_code_and_message():
    assert format_warning_line("selector_unmatched", "一致なし") == "warning: selector_unmatched: 一致なし"


def test_line_without_a_stable_code_omits_the_code_part():
    assert format_warning_line(None, "本文だけ") == "warning: 本文だけ"


def test_section_is_appended_to_the_message_in_parentheses():
    assert format_warning_line("bad_section", "壊れている", section="bone") == "warning: bad_section: 壊れている(bone)"


def test_several_sections_are_joined_by_commas():
    line = format_warning_line("passthrough", "透過した", section=["bone", "morph"])
    assert line == "warning: passthrough: 透過した(bone,morph)"


def test_tuple_of_sections_is_accepted():
    assert format_warning_line("passthrough", "透過した", section=("bone",)) == "warning: passthrough: 透過した(bone)"


@pytest.mark.parametrize("section", [None, "", [], ()])
def test_absent_or_empty_section_adds_no_parentheses(section):
    assert format_warning_line("c", "m", section=section) == "warning: c: m"


def test_result_is_a_single_line_without_trailing_newline():
    line = format_warning_line("c", "m", section="bone")
    assert "\n" not in line


def test_section_without_code_still_appends_to_the_message():
    assert format_warning_line(None, "m", section="bone") == "warning: m(bone)"
