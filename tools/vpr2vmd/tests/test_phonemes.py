"""音素→カテゴリ写像のテスト(vpr2vmd.md §3、口形イベント確定の音素分類)。

VOCALOID 日本語の音素(X-SAMPA)を、口形イベント確定で使うカテゴリ
(母音 / 両唇閉鎖 / 撥音「ん」/ 促音「っ」/ 継続 / その他子音)へ分類する。母音はさらに対応する
口形 MouthShape(A/I/U/E/O)を返す。インベントリは実 vpr(日本語初音ミク)と X-SAMPA の
標準で確定したもの。
"""

from lipsync import ApertureClass, ConsonantClass, MouthShape

from vpr2vmd import phonemes

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


def test_unknown_symbols_categorized_as_other():
    # 未知記号(標準 X-SAMPA 外)は写像上はその他子音と同じく OTHER に併合する(自前イベントを
    # 作らない)。未知音素の診断記録は別途(is_known)で行う。
    for sym in ["th", "gh", "zzz", ""]:
        assert phonemes.categorize(sym) is Cat.OTHER


def test_geminate_categorized_as_geminate_stop():
    # 促音「っ」(Q)は閉鎖・詰まりで、口形イベントは無音(閉口)にするため専用カテゴリにする。
    assert phonemes.categorize("Q") is Cat.GEMINATE_STOP


def test_rounded_consonants_mapped_to_rounded():
    # 唇を丸める子音 ふ(p\)・わ(w)は ConsonantClass.ROUNDED。
    for sym in ["p\\", "w"]:
        assert phonemes.consonant_class(sym) is ConsonantClass.ROUNDED


def test_spread_consonants_mapped_to_spread():
    # い 方向へ寄せる子音 し(S)・じ(dZ)・ち(tS)・拗音のわたり(j)は ConsonantClass.SPREAD。
    for sym in ["S", "dZ", "tS", "j"]:
        assert phonemes.consonant_class(sym) is ConsonantClass.SPREAD


def test_other_and_unknown_consonants_mapped_to_neutral():
    # 唇を動かさない子音(軟口蓋/歯茎/声門 等)と未知記号は ConsonantClass.NEUTRAL(純母音扱い)。
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
    # p\・w・h は判定表の NONE(残り)行に列挙され、ApertureClass.NONE(開口減衰なし)。未知記号も同様。
    for sym in ["p\\", "w", "h", "zzz"]:
        assert phonemes.aperture_class(sym) is ApertureClass.NONE
