"""音素→5母音写像のテスト(vocal_analysis.md §7.1・§7.2)。

espeak IPA(wav2vec2 が返す母音セグメントの音素記号)を5母音 a/i/u/e/o へ完全一致テーブル参照で
バケット化する規則(§7.1)と、VOCALOID X-SAMPA の母音記号を5母音へ写像する規則(§7.2)を検証する。
いずれもテーブルに無い・判定不能な記号は None(母音でないことを示す)。
"""

import pytest


@pytest.mark.parametrize(
    "symbol,expected",
    [
        # §7.1 の完全一致テーブル: a/ɑ/ʌ/æ→あ、i/ɪ/j→い、ɯ/u/ʊ/w→う、e/ɛ→え、o/ɔ→お。
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
        ("ɛ", "e"),
        ("o", "o"),
        ("ɔ", "o"),
    ],
)
def test_espeak_ipa_to_vowel_table_entries(symbol, expected):
    from vocal_analysis.phonemes import espeak_ipa_to_vowel

    assert espeak_ipa_to_vowel(symbol) == expected


@pytest.mark.parametrize("symbol", ["k", "s", "t", "n", "m", "", "z", "ʒ", "A", "I", "O"])
def test_espeak_ipa_to_vowel_unmapped_symbol_is_gap(symbol):
    from vocal_analysis.phonemes import espeak_ipa_to_vowel

    # §7.1: テーブルに無い・判定不能なラベルはいずれも gap(None)とする(最近接等の曖昧な距離判定はしない)。
    # "A"/"I"/"O"(大文字)は完全一致テーブルの対象外(大文字小文字を正規化・畳み込みしない)。
    assert espeak_ipa_to_vowel(symbol) is None


@pytest.mark.parametrize(
    "symbol,expected",
    [
        # §7.2 の母音記号テーブル(VOCALOID日本語のX-SAMPA音素インベントリから確定)。
        # 長音記号":"付き(i:)は母音同一のまま扱う(長さはカテゴリでなく区間長の属性)。
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

    # §7.2: 母音記号テーブルに無ければ None(子音・継続記号等、母音でないことを示す)。
    assert xsampa_vowel_letter(symbol) is None
