import pytest

from vocal_analysis import phonemes
from vocal_analysis.phonemes import _G2P_TO_VOCAB_SYMBOL

VOCAB = sorted(set(_G2P_TO_VOCAB_SYMBOL.values()))

MORAIC_NASAL = {"ɴ"}
BILABIAL = {"m", "mʲ", "b", "bʲ", "p", "pʲ"}
ROUNDED = {"ɸ", "w"}
PALATALIZED = {"kʲ", "ɡʲ", "tʲ", "dʲ", "bʲ", "pʲ", "mʲ"}

PREDICATES = [
    ("is_moraic_nasal", MORAIC_NASAL),
    ("is_bilabial", BILABIAL),
    ("is_rounded", ROUNDED),
    ("is_palatalized", PALATALIZED),
]

UNKNOWN_SYMBOLS = ["", "x", "ʔ", "ŋ", "pʰ", "ɴː", "A", "ɯː"]


def test_vocabulary_is_the_expected_inventory():
    assert set(VOCAB) == {
        "a", "i", "ɯ", "e̞", "o̞",
        "k", "kʲ", "ɡ", "ɡʲ", "s", "ɕ", "z", "dʑ", "t", "tʲ", "tɕ", "ts", "d", "dʲ",
        "n", "ɲ", "h", "ç", "ɸ", "b", "bʲ", "p", "pʲ", "m", "mʲ",
        "j", "ɾ", "w", "v", "ɴ",
    }


@pytest.mark.parametrize("name,expected_members", PREDICATES)
def test_predicates_are_public_functions(name, expected_members):
    assert callable(getattr(phonemes, name))
    assert expected_members <= set(VOCAB)


@pytest.mark.parametrize("symbol", VOCAB)
@pytest.mark.parametrize("name,expected_members", PREDICATES)
def test_predicate_matches_expected_set_for_every_vocabulary_symbol(name, expected_members, symbol):
    result = getattr(phonemes, name)(symbol)

    assert result is (symbol in expected_members)


@pytest.mark.parametrize("name,expected_members", PREDICATES)
def test_predicate_is_false_for_symbols_outside_the_vocabulary(name, expected_members):
    predicate = getattr(phonemes, name)

    for symbol in UNKNOWN_SYMBOLS:
        assert predicate(symbol) is False


@pytest.mark.parametrize("name,expected_members", PREDICATES)
def test_predicate_is_false_for_none(name, expected_members):
    assert getattr(phonemes, name)(None) is False


@pytest.mark.parametrize("name,expected_members", PREDICATES)
def test_predicate_is_false_for_every_vowel_symbol(name, expected_members):
    predicate = getattr(phonemes, name)

    for symbol in ("a", "i", "ɯ", "e̞", "o̞"):
        assert predicate(symbol) is False


def test_palatalized_is_exactly_the_vocabulary_symbols_ending_with_the_palatalization_mark():
    assert {s for s in VOCAB if s.endswith("ʲ")} == PALATALIZED
    assert {s for s in VOCAB if phonemes.is_palatalized(s)} == PALATALIZED
