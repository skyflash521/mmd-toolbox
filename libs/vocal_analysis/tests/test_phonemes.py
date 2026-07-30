"""音素→5母音写像のテスト。

espeak IPA(wav2vec2 が返す母音セグメントの音素記号)を5母音 a/i/u/e/o へ完全一致テーブル参照で
バケット化する規則と、VOCALOID X-SAMPA の母音記号を5母音へ写像する規則を検証する。
いずれもテーブルに無い・判定不能な記号は None(母音でないことを示す)。
"""

import pytest


@pytest.mark.parametrize(
    "symbol,expected",
    [
        # espeak IPA→5母音の完全一致テーブル: a/ɑ/ʌ/æ→あ、i/ɪ/j→い、ɯ/u/ʊ/w→う、e/e̞/ɛ→え、o/o̞/ɔ→お。
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
def test_espeak_ipa_to_vowel_unmapped_symbol_is_gap(symbol):
    from vocal_analysis.phonemes import espeak_ipa_to_vowel

    # テーブルに無い・判定不能なラベルはいずれも gap(None)とする(最近接等の曖昧な距離判定はしない)。
    # "A"/"I"/"O"(大文字)は完全一致テーブルの対象外(大文字小文字を正規化・畳み込みしない)。
    assert espeak_ipa_to_vowel(symbol) is None


@pytest.mark.parametrize(
    "symbol,expected",
    [
        # X-SAMPAの母音記号テーブル(VOCALOID日本語のX-SAMPA音素インベントリから確定)。
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

    # 母音記号テーブルに無ければ None(子音・継続記号等、母音でないことを示す)。
    assert xsampa_vowel_letter(symbol) is None


# --- X-SAMPA の長音記号の正規化 ---


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("a", "a"),
        ("i:", "i"),
        ("i::", "i"),
        # 母音以外の基底にも同じ規則が及ぶ(この関数は基底を返すだけで分類はしない)。
        ("N\\:", "N\\"),
        (":", ""),
        ("", ""),
        # 除去は末尾だけに効く(記号内の「:」を落とす実装だと分類が広がる)。
        (":a", ":a"),
        ("a:b", "a:b"),
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

    # 長音記号付きは基底の記号と同じ母音へ写す(個別登録に頼らない一般規則)。
    assert xsampa_vowel_letter(symbol) == expected


@pytest.mark.parametrize("symbol", ["k:", "N\\:", "-:", ":", ":a", "a:b"])
def test_xsampa_vowel_letter_non_vowel_base_stays_none(symbol):
    from vocal_analysis.phonemes import xsampa_vowel_letter

    # 正規化しても基底の記号が母音でなければ母音にはしない(長音記号の除去は分類を広げない)。
    # 末尾以外の「:」を落とす実装だと ":a"・"a:b" が母音になってしまうので、ここで弾く。
    assert xsampa_vowel_letter(symbol) is None
