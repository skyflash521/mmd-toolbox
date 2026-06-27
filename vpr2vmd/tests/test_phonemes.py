"""音素→カテゴリ写像のテスト(vpr2vmd.md §3、口形イベント確定の音素分類)。

VOCALOID 日本語の音素(X-SAMPA)を、口形イベント確定で使うカテゴリ
(母音 / 両唇閉鎖 / 撥音「ん」/ 継続 / その他子音)へ分類する。母音はさらに対応する
口形 MouthShape(A/I/U/E/O)を返す。インベントリは実 vpr(日本語初音ミク)と X-SAMPA の
標準で確定したもの。
"""

import pytest

phonemes = pytest.importorskip("vpr2vmd.phonemes", reason="impl pending: P-2 音素カテゴリ")

from lipsync import MouthShape  # noqa: E402

Cat = phonemes.PhonemeCategory


def test_vowel_shapes():
    assert phonemes.vowel_shape("a") is MouthShape.A
    assert phonemes.vowel_shape("i") is MouthShape.I
    assert phonemes.vowel_shape("i:") is MouthShape.I  # 長音記号付きも い
    assert phonemes.vowel_shape("M") is MouthShape.U   # X-SAMPA M = 日本語「う」
    assert phonemes.vowel_shape("e") is MouthShape.E
    assert phonemes.vowel_shape("o") is MouthShape.O


def test_vowel_shape_is_none_for_non_vowel():
    for sym in ["m", "b", "N\\", "-", "t", "p\\", "k"]:
        assert phonemes.vowel_shape(sym) is None


def test_vowels_categorized_as_vowel():
    for sym in ["a", "i", "i:", "M", "e", "o"]:
        assert phonemes.categorize(sym) is Cat.VOWEL


def test_bilabials_categorized_as_bilabial():
    # 語頭で唇が完全に閉じる m/b/p と口蓋化形(ま・ば・ぱ行)。
    for sym in ["m", "m'", "b", "b'", "p", "p'"]:
        assert phonemes.categorize(sym) is Cat.BILABIAL


def test_bilabial_fricative_is_other_not_bilabial():
    # p\(X-SAMPA φ=ふ の両唇摩擦音)は完全閉鎖でなく、後続母音「う」の口で見せるため
    # 両唇閉鎖でなくその他子音とする。
    assert phonemes.categorize("p\\") is Cat.OTHER


def test_moraic_nasal_categorized_as_moraic_nasal():
    # 撥音「ん」(後続母音を持たない単独の鼻音)。X-SAMPA では N\(uvular nasal)。
    assert phonemes.categorize("N\\") is Cat.MORAIC_NASAL


def test_continuation_categorized_as_continuation():
    # 継続/メリスマ(直前音の伸ばし)。
    assert phonemes.categorize("-") is Cat.CONTINUATION


def test_other_consonants_categorized_as_other():
    # 両唇音でない子音は自前イベントを作らず協調調音に委ねる。
    for sym in ["t", "d", "s", "k", "k'", "g", "n", "J", "w", "4", "S", "ts", "dZ", "h"]:
        assert phonemes.categorize(sym) is Cat.OTHER


def test_unknown_symbols_fall_back_to_other():
    # vpr_io インベントリで母音にも両唇音にも該当しない記号は、その他子音と同じく協調調音へ委ねる。
    for sym in ["th", "gh", "zzz", ""]:
        assert phonemes.categorize(sym) is Cat.OTHER


def test_geminate_is_other():
    # 促音「っ」(Q)は無音/閉鎖の準備区間で、口形は後続子音との協調調音に委ねる=その他子音。
    assert phonemes.categorize("Q") is Cat.OTHER
