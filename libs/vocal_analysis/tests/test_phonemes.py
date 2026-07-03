"""音素→5母音写像のテスト(vocal_analysis.md §7.1)。

espeak IPA(wav2vec2 が返す母音セグメントの音素記号)を5母音 a/i/u/e/o へ完全一致テーブル参照で
バケット化する規則を検証する。テーブルに無い・判定不能なラベルはいずれも gap(None)。
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
