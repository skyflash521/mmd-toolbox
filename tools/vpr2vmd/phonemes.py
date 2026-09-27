from enum import Enum

from lipsync import ApertureClass, ConsonantClass, MouthShape
from vocal_analysis.phonemes import xsampa_base_symbol, xsampa_vowel_letter


class PhonemeCategory(Enum):
    VOWEL = "vowel"
    BILABIAL = "bilabial"
    MORAIC_NASAL = "moraic_nasal"
    GEMINATE_STOP = "geminate_stop"
    CONTINUATION = "continuation"
    OTHER = "other"


_VOWEL_LETTER_SHAPES = {
    "a": MouthShape.A,
    "i": MouthShape.I,
    "u": MouthShape.U,
    "e": MouthShape.E,
    "o": MouthShape.O,
}

# p\(X-SAMPA の無声両唇摩擦音 φ)は唇が閉じきらない。
_BILABIALS = {"m", "m'", "b", "b'", "p", "p'"}
_MORAIC_NASALS = {"N\\"}
_GEMINATE_STOPS = {"Q"}
_CONTINUATIONS = {"-"}

_ROUNDED_CONSONANTS = {"p\\", "w"}
_SPREAD_CONSONANTS = {"S", "dZ", "tS", "j"}

_FIRM_CLOSURE_CONSONANTS = {"t", "d", "n", "ts", "dz", "J"}
_NARROW_CHANNEL_CONSONANTS = {"s", "z", "S", "dZ", "tS", "j"}
_SLIGHT_CLOSURE_CONSONANTS = {"k", "k'", "g", "4"}


def vowel_shape(symbol: str) -> MouthShape | None:
    letter = xsampa_vowel_letter(symbol)
    if letter is None:
        return None
    return _VOWEL_LETTER_SHAPES[letter]


def categorize(symbol: str) -> PhonemeCategory:
    if xsampa_vowel_letter(symbol) is not None:
        return PhonemeCategory.VOWEL
    base = xsampa_base_symbol(symbol)
    if base in _BILABIALS:
        return PhonemeCategory.BILABIAL
    if base in _MORAIC_NASALS:
        return PhonemeCategory.MORAIC_NASAL
    if base in _GEMINATE_STOPS:
        return PhonemeCategory.GEMINATE_STOP
    if base in _CONTINUATIONS:
        return PhonemeCategory.CONTINUATION
    return PhonemeCategory.OTHER


def consonant_class(symbol: str) -> ConsonantClass:
    base = xsampa_base_symbol(symbol)
    if base in _ROUNDED_CONSONANTS:
        return ConsonantClass.ROUNDED
    if base in _SPREAD_CONSONANTS:
        return ConsonantClass.SPREAD
    return ConsonantClass.NEUTRAL


def aperture_class(symbol: str) -> ApertureClass:
    base = xsampa_base_symbol(symbol)
    if base in _FIRM_CLOSURE_CONSONANTS:
        return ApertureClass.FIRM_CLOSURE
    if base in _NARROW_CHANNEL_CONSONANTS:
        return ApertureClass.NARROW_CHANNEL
    if base in _SLIGHT_CLOSURE_CONSONANTS:
        return ApertureClass.SLIGHT_CLOSURE
    return ApertureClass.NONE
