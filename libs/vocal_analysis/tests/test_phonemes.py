import pytest


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("a", "a"),
        ("ɑ", "a"),
        ("ʌ", "a"),
        ("æ", "a"),
        ("i", "i"),
        ("ɪ", "i"),
        ("j", "i"),
        ("ɯ", "u"),
        ("u", "u"),
        ("ʊ", "u"),
        ("w", "u"),
        ("e", "e"),
        ("e̞", "e"),
        ("ɛ", "e"),
        ("o", "o"),
        ("o̞", "o"),
        ("ɔ", "o"),
    ],
)
def test_espeak_ipa_to_vowel_table_entries(symbol, expected):
    from vocal_analysis.phonemes import espeak_ipa_to_vowel

    assert espeak_ipa_to_vowel(symbol) == expected


@pytest.mark.parametrize("symbol", ["k", "s", "t", "n", "m", "", "z", "ʒ", "A", "I", "O"])
def test_espeak_ipa_to_vowel_symbol_outside_exact_match_table_is_none(symbol):
    from vocal_analysis.phonemes import espeak_ipa_to_vowel

    assert espeak_ipa_to_vowel(symbol) is None


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("a", "a"),
        ("i", "i"),
        ("i:", "i"),
        ("M", "u"),
        ("e", "e"),
        ("o", "o"),
    ],
)
def test_xsampa_vowel_letter_table_entries(symbol, expected):
    from vocal_analysis.phonemes import xsampa_vowel_letter

    assert xsampa_vowel_letter(symbol) == expected


@pytest.mark.parametrize("symbol", ["m", "b", "N\\", "-", "t", "p\\", "k"])
def test_xsampa_vowel_letter_non_vowel_is_none(symbol):
    from vocal_analysis.phonemes import xsampa_vowel_letter

    assert xsampa_vowel_letter(symbol) is None


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("a", "a"),
        ("i:", "i"),
        ("i::", "i"),
        ("N\\:", "N\\"),
        (":", ""),
        ("", ""),
        pytest.param(":a", ":a", id="leading-mark-kept"),
        pytest.param("a:b", "a:b", id="inner-mark-kept"),
    ],
)
def test_xsampa_base_symbol_strips_trailing_length_marks(symbol, expected):
    from vocal_analysis.phonemes import xsampa_base_symbol

    assert xsampa_base_symbol(symbol) == expected


@pytest.mark.parametrize(
    "symbol,expected", [("a:", "a"), ("M:", "u"), ("e:", "e"), ("o::", "o")]
)
def test_xsampa_vowel_letter_normalizes_length_marks(symbol, expected):
    from vocal_analysis.phonemes import xsampa_vowel_letter

    assert xsampa_vowel_letter(symbol) == expected


def test_g2p_to_phoneme_model_vocab_table_is_pinned():
    from vocal_analysis.phonemes import _BLANK_G2P_SYMBOLS, _G2P_TO_VOCAB_SYMBOL

    assert _G2P_TO_VOCAB_SYMBOL == {
        "a": "a", "i": "i", "u": "ɯ", "e": "e̞", "o": "o̞",
        "I": "i", "U": "ɯ",
        "k": "k", "ky": "kʲ", "g": "ɡ", "gy": "ɡʲ",
        "s": "s", "sh": "ɕ", "z": "z", "j": "dʑ",
        "t": "t", "ch": "tɕ", "ts": "ts", "d": "d",
        "n": "n", "ny": "ɲ", "h": "h", "hy": "ç", "f": "ɸ",
        "b": "b", "by": "bʲ", "p": "p", "py": "pʲ",
        "m": "m", "my": "mʲ", "y": "j", "r": "ɾ", "ry": "ɾ",
        "w": "w", "v": "v", "N": "ɴ",
    }
    assert _BLANK_G2P_SYMBOLS == {"pau", "cl"}


@pytest.mark.parametrize("symbol", ["k:", "N\\:", "-:", ":", ":a", "a:b"])
def test_xsampa_vowel_letter_non_vowel_base_stays_none(symbol):
    from vocal_analysis.phonemes import xsampa_vowel_letter

    assert xsampa_vowel_letter(symbol) is None
