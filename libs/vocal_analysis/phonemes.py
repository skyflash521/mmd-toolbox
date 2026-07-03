"""音素→5母音写像(vocal_analysis.md §7)。

認識器・vpr が返す音素記号を日本語の5母音 a/i/u/e/o へ写像する規則を提供する。系統の違う2つの入力
記号系(espeak IPA・VOCALOID X-SAMPA)を扱うため、写像も2つに分ける。
"""

# §7.1: espeak(wav2vec2)が返す母音セグメントの IPA を5母音へ完全一致のテーブル参照でバケット化する。
# テーブルは採用認識器の音素インベントリ(espeak)から S-1ゲート採点前に確定・固定する。
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
    "ɛ": "e",
    "o": "o",
    "ɔ": "o",
}


def espeak_ipa_to_vowel(symbol: str) -> str | None:
    """espeak IPA の母音記号を5母音(a/i/u/e/o)へ写像する(§7.1)。

    テーブルに無い・判定不能な記号はいずれも gap として None を返す(最近接等の曖昧な距離判定はしない)。
    """
    return _ESPEAK_IPA_TO_VOWEL.get(symbol)
