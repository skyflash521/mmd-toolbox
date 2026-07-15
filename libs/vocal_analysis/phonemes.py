"""音素→5母音写像・S2アダプタ共有の記号分類・共有例外。

認識器・vpr が返す音素記号を日本語の5母音 a/i/u/e/o へ写像する規則を提供する。系統の違う2つの入力
記号系(espeak IPA・VOCALOID X-SAMPA)を扱うため、写像も2つに分ける。加えて、S2の複数アダプタ
(wav2vec2 CTC経路・SOFA経路)が共有する「G2P記号→音素モデル語彙の写像表」「IPA記号の
母音/子音分類」「blank/gap記号集合」「認識失敗時の例外型」「単語タイムスタンプの最小長」もここに
置く(両アダプタから参照でき、アダプタ実装同士の循環importを避けるため)。
"""

import unicodedata
from typing import Literal


class RecognitionError(Exception):
    """S2 の認識失敗(transformers/SOFA 未導入、写像表に無い記号、強制アライメント失敗など)。

    wav2vec2 CTC経路・SOFA経路の両アダプタが共通して送出する(Recognizerアダプタ契約は
    forced_aligner の選択に関わらず不変)。
    """


_MIN_WORD_DURATION_SEC = 0.05  # 単語タイムスタンプ単調化の最小長。SOFA経路のcursorクランプも使う


# espeak(wav2vec2)が返す母音セグメントの IPA を5母音へ完全一致のテーブル参照でバケット化する。
# テーブルは採用認識器の音素インベントリ(espeak)から S-1測定の採点前に確定・固定する。
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
    "e̞": "e",  # lowered e(採用モデルの語彙における日本語「え」の表記)
    "ɛ": "e",
    "o": "o",
    "o̞": "o",  # lowered o(採用モデルの語彙における日本語「お」の表記)
    "ɔ": "o",
}


def espeak_ipa_to_vowel(symbol: str) -> str | None:
    """espeak IPA の母音記号を5母音(a/i/u/e/o)へ写像する。

    テーブルに無い・判定不能な記号はいずれも gap として None を返す(最近接等の曖昧な距離判定はしない)。
    """
    return _ESPEAK_IPA_TO_VOWEL.get(symbol)


# VOCALOID 日本語の X-SAMPA 音素インベントリから確定した母音記号テーブル。長音記号 ":" 付き
# (例 i:)は母音同一のまま扱う(長さはカテゴリでなく区間長の属性)。vpr を読む CLI(口形イベント確定)と
# S-1測定(vpr由来ラベル生成)が同一規則で使う(写像表の二重管理を避ける)。
_XSAMPA_VOWEL_LETTERS = {
    "a": "a",
    "i": "i",
    "i:": "i",
    "M": "u",  # X-SAMPA M は close back unrounded vowel で、日本語「う」の標準表記
    "e": "e",
    "o": "o",
}


def xsampa_vowel_letter(symbol: str) -> str | None:
    """VOCALOID X-SAMPA の母音記号を5母音(a/i/u/e/o)へ写像する。

    母音記号テーブルに無ければ None(子音・継続記号など、母音でないことを示す)。
    """
    return _XSAMPA_VOWEL_LETTERS.get(symbol)


# IPA母音チャートの基本母音28記号 + R音性母音2記号(ɚ・ɝ) + 拡張母音記号1(ᵻ)。
_VOWEL_BASE_CHARACTERS = frozenset("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɚɝᵻ")


def _classify_symbol(symbol: str) -> Literal["vowel", "consonant"]:
    """確定済みのIPA記号1つを母音/子音へ分類する(言語非依存・S2アダプタ共通)。

    NFD 正規化後の先頭の基底文字(長音記号・鼻音化の結合チルダ等の修飾記号は正規化により基底文字の
    後ろに分離される)が母音記号基準集合に含まれれば母音、そうでなければ子音とする。wav2vec2 CTC経路
    はCTC出力の各フレームが対応する1記号へ、SOFA経路はSOFAの出力セグメントが直接
    確定する1記号へ、この同じ分類規則を適用する。
    """
    if not symbol:
        return "consonant"
    base = unicodedata.normalize("NFD", symbol)[0]
    return "vowel" if base in _VOWEL_BASE_CHARACTERS else "consonant"


# G2P記号→音素モデル語彙の写像表(確定): pyopenjtalk-plus の音素記号(無声化母音 I/U を含む)を音素モデルの
# 語彙(espeak表記)へ対応付ける。pau・cl はここに含めず、blank/gap記号(呼び出し側が渡す
# blank_token_id、またはSOFA経路のgap判定)へ変換する。wav2vec2 CTC経路・SOFA経路の両方が使う
# (独自の写像表を新設しない)。
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
