import pytest

from vocal_analysis import phonemes


@pytest.mark.parametrize(
    "name,symbol",
    [
        ("is_moraic_nasal", "ɴ"),
        ("is_bilabial", "m"),
        ("is_bilabial", "mʲ"),
        ("is_bilabial", "b"),
        ("is_bilabial", "bʲ"),
        ("is_bilabial", "p"),
        ("is_bilabial", "pʲ"),
        ("is_rounded", "ɸ"),
        ("is_rounded", "w"),
        ("is_palatalized", "kʲ"),
    ],
)
def test_predicate_is_true_for_member_symbol(name, symbol):
    assert getattr(phonemes, name)(symbol) is True


@pytest.mark.parametrize(
    "name,symbol",
    [
        ("is_moraic_nasal", "n"),
        ("is_bilabial", "ɸ"),
        ("is_rounded", "ɯ"),
        ("is_palatalized", "ɕ"),
    ],
)
def test_predicate_is_false_for_nearest_non_member_symbol(name, symbol):
    assert getattr(phonemes, name)(symbol) is False


def test_is_palatalized_is_false_for_none():
    assert phonemes.is_palatalized(None) is False
