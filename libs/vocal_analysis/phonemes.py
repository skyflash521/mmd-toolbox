import unicodedata
from typing import Literal


class RecognitionError(Exception):
    pass


_MIN_WORD_DURATION_SEC = 0.05


_ESPEAK_IPA_TO_VOWEL = {
    "a": "a",
    "ɑ": "a",
    "ʌ": "a",
    "æ": "a",
    "i": "i",
    "ɪ": "i",
    "j": "i",
    "ɯ": "u",
    "u": "u",
    "ʊ": "u",
    "w": "u",
    "e": "e",
    "e̞": "e",
    "ɛ": "e",
    "o": "o",
    "o̞": "o",
    "ɔ": "o",
}


def espeak_ipa_to_vowel(symbol: str) -> str | None:
    """戻り値は a/i/u/e/o のいずれか。表に無い記号は None。"""
    return _ESPEAK_IPA_TO_VOWEL.get(symbol)


_XSAMPA_VOWEL_LETTERS = {
    "a": "a",
    "i": "i",
    # X-SAMPA の M は非円唇後舌狭母音で、日本語の「う」の標準表記。
    "M": "u",
    "e": "e",
    "o": "o",
}

_XSAMPA_LENGTH_MARK = ":"


def xsampa_base_symbol(symbol: str) -> str:
    return symbol.rstrip(_XSAMPA_LENGTH_MARK)


def xsampa_vowel_letter(symbol: str) -> str | None:
    """戻り値は a/i/u/e/o のいずれか。母音でない記号は None。"""
    return _XSAMPA_VOWEL_LETTERS.get(xsampa_base_symbol(symbol))


_VOWEL_BASE_CHARACTERS = frozenset("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɚɝᵻ")


def _classify_symbol(symbol: str) -> Literal["vowel", "consonant"]:
    if not symbol:
        return "consonant"
    # NFD では長音記号や結合チルダなどの修飾記号が基底文字の後ろへ分離される。
    base = unicodedata.normalize("NFD", symbol)[0]
    return "vowel" if base in _VOWEL_BASE_CHARACTERS else "consonant"


_G2P_TO_VOCAB_SYMBOL: dict[str, str] = {
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
_BLANK_G2P_SYMBOLS = frozenset({"pau", "cl"})
