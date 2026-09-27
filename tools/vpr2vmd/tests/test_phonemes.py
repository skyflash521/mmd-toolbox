from lipsync import ApertureClass, ConsonantClass, MouthShape
from vpr2vmd import phonemes

Cat = phonemes.PhonemeCategory


def test_vowel_shapes():
    assert phonemes.vowel_shape("a") is MouthShape.A
    assert phonemes.vowel_shape("i") is MouthShape.I
    assert phonemes.vowel_shape("i:") is MouthShape.I
    assert phonemes.vowel_shape("M") is MouthShape.U
    assert phonemes.vowel_shape("e") is MouthShape.E
    assert phonemes.vowel_shape("o") is MouthShape.O


def test_vowel_shape_is_none_for_non_vowel():
    for sym in ["m", "b", "N\\", "-", "t", "p\\", "k"]:
        assert phonemes.vowel_shape(sym) is None


def test_vowels_categorized_as_vowel():
    for sym in ["a", "i", "i:", "M", "e", "o"]:
        assert phonemes.categorize(sym) is Cat.VOWEL


def test_bilabials_categorized_as_bilabial():
    for sym in ["m", "m'", "b", "b'", "p", "p'"]:
        assert phonemes.categorize(sym) is Cat.BILABIAL


def test_bilabial_fricative_is_other_not_bilabial():
    assert phonemes.categorize("p\\") is Cat.OTHER


def test_moraic_nasal_categorized_as_moraic_nasal():
    assert phonemes.categorize("N\\") is Cat.MORAIC_NASAL


def test_continuation_categorized_as_continuation():
    assert phonemes.categorize("-") is Cat.CONTINUATION


def test_other_consonants_categorized_as_other():
    for sym in ["t", "d", "s", "k", "k'", "g", "n", "J", "w", "4", "S", "ts", "dZ", "h"]:
        assert phonemes.categorize(sym) is Cat.OTHER


def test_unknown_symbols_categorized_as_other():
    for sym in ["th", "gh", "zzz", ""]:
        assert phonemes.categorize(sym) is Cat.OTHER


def test_geminate_categorized_as_geminate_stop():
    assert phonemes.categorize("Q") is Cat.GEMINATE_STOP


def test_rounded_consonants_mapped_to_rounded():
    for sym in ["p\\", "w"]:
        assert phonemes.consonant_class(sym) is ConsonantClass.ROUNDED


def test_spread_consonants_mapped_to_spread():
    for sym in ["S", "dZ", "tS", "j"]:
        assert phonemes.consonant_class(sym) is ConsonantClass.SPREAD


def test_other_and_unknown_consonants_mapped_to_neutral():
    for sym in ["k", "k'", "g", "s", "z", "t", "d", "n", "J", "4", "ts", "dz", "h", "zzz"]:
        assert phonemes.consonant_class(sym) is ConsonantClass.NEUTRAL


def test_firm_closure_consonants_mapped_to_firm_closure():
    for sym in ["t", "d", "n", "ts", "dz", "J"]:
        assert phonemes.aperture_class(sym) is ApertureClass.FIRM_CLOSURE


def test_narrow_channel_consonants_mapped_to_narrow_channel():
    for sym in ["s", "z", "S", "dZ", "tS", "j"]:
        assert phonemes.aperture_class(sym) is ApertureClass.NARROW_CHANNEL


def test_slight_closure_consonants_mapped_to_slight_closure():
    for sym in ["k", "k'", "g", "4"]:
        assert phonemes.aperture_class(sym) is ApertureClass.SLIGHT_CLOSURE


def test_none_class_consonants_and_unknown_mapped_to_none_aperture():
    for sym in ["p\\", "w", "h", "zzz"]:
        assert phonemes.aperture_class(sym) is ApertureClass.NONE


def test_length_marked_symbols_classify_as_base_symbol():
    assert phonemes.categorize("N\\:") is Cat.MORAIC_NASAL
    assert phonemes.categorize("N\\::") is Cat.MORAIC_NASAL
    assert phonemes.categorize("m:") is Cat.BILABIAL
    assert phonemes.categorize("Q:") is Cat.GEMINATE_STOP
    assert phonemes.categorize("-:") is Cat.CONTINUATION
    assert phonemes.categorize("a:") is Cat.VOWEL
    assert phonemes.categorize("e:") is Cat.VOWEL
    assert phonemes.categorize("k:") is Cat.OTHER


def test_length_marked_vowel_shape_uses_base_symbol():
    assert phonemes.vowel_shape("a:") is MouthShape.A
    assert phonemes.vowel_shape("M:") is MouthShape.U
    assert phonemes.vowel_shape("e:") is MouthShape.E
    assert phonemes.vowel_shape("o::") is MouthShape.O
    assert phonemes.vowel_shape("k:") is None


def test_length_marked_consonant_and_aperture_use_base_symbol():
    assert phonemes.consonant_class("S:") is ConsonantClass.SPREAD
    assert phonemes.consonant_class("S::") is ConsonantClass.SPREAD
    assert phonemes.aperture_class("S:") is ApertureClass.NARROW_CHANNEL
    assert phonemes.aperture_class("S::") is ApertureClass.NARROW_CHANNEL
    assert phonemes.consonant_class("w:") is ConsonantClass.ROUNDED
    assert phonemes.aperture_class("t:") is ApertureClass.FIRM_CLOSURE
    assert phonemes.aperture_class("t::") is ApertureClass.FIRM_CLOSURE
    assert phonemes.aperture_class("k:") is ApertureClass.SLIGHT_CLOSURE


def test_entries_strip_length_marks_only_at_end():
    assert phonemes.categorize(":a") is Cat.OTHER
    assert phonemes.vowel_shape(":a") is None
    assert phonemes.categorize("a:b") is Cat.OTHER
    assert phonemes.vowel_shape("a:b") is None
    assert phonemes.consonant_class(":S") is ConsonantClass.NEUTRAL
    assert phonemes.aperture_class(":S") is ApertureClass.NONE
    assert phonemes.consonant_class("d:Z") is ConsonantClass.NEUTRAL
    assert phonemes.aperture_class("d:Z") is ApertureClass.NONE
