"""音素→カテゴリ写像(口形イベント確定の音素分類)。

VOCALOID 日本語の音素(X-SAMPA 表記)を、口形イベント確定で使うカテゴリへ分類する。写像規則は、
母音→母音イベント、語頭の両唇音→両唇閉鎖、両唇閉鎖以外の子音は自前イベントを作らず、唇の方向
(ConsonantClass)と顎の開口減衰(ApertureClass)という独立な2軸で母音合成を変調する、というもの。
各記号がどのカテゴリに属するか(およびどの母音か)は、日本語初音ミクの実 vpr と
X-SAMPA・日本語音韻の標準で確定したインベントリに従い、実装者が独自判断しない。
"""

from enum import Enum

from lipsync import ApertureClass, ConsonantClass, MouthShape
from vocal_analysis.phonemes import xsampa_vowel_letter


class PhonemeCategory(Enum):
    """音素のカテゴリ(口形イベント確定で使う分類)。"""

    VOWEL = "vowel"  # 母音。対応する MouthShape は vowel_shape() で得る
    BILABIAL = "bilabial"  # 両唇閉鎖(ま・ば・ぱ行の語頭)。MouthShape.BILABIAL へ
    MORAIC_NASAL = "moraic_nasal"  # 撥音「ん」。MouthShape.N(--no-n-morph 時は無音)へ
    GEMINATE_STOP = "geminate_stop"  # 促音「っ」(Q)。無音(閉口)へ
    CONTINUATION = "continuation"  # 継続/メリスマ(直前音の伸ばし)
    OTHER = "other"  # その他子音・未知。自前イベントを作らず協調調音/直前口形継続へ委ねる


# 母音文字(a/i/u/e/o。母音記号テーブル自体は vocal_analysis が持つ。下記 vowel_shape 参照)→ MouthShape。
_VOWEL_LETTER_SHAPES = {
    "a": MouthShape.A,
    "i": MouthShape.I,
    "u": MouthShape.U,
    "e": MouthShape.E,
    "o": MouthShape.O,
}

# 両唇閉鎖を作る記号(語頭で唇が完全に閉じる ま・ば・ぱ行の子音と口蓋化形)。
# X-SAMPA: m=両唇鼻音、b=有声両唇破裂音、p=無声両唇破裂音。"'" は口蓋化を表す修飾で両唇性は保つ。
# p\(=φ、ふ の無声両唇摩擦音)は完全閉鎖でないため含めず、その他子音として後続母音「う」の口で見せる。
_BILABIALS = {"m", "m'", "b", "b'", "p", "p'"}

# 撥音「ん」(後続母音を持たない単独の鼻音)。X-SAMPA では N\(uvular nasal)。
_MORAIC_NASALS = {"N\\"}

# 促音「っ」。X-SAMPA では Q(日本語の特殊モーラ。閉鎖・詰まり)。
_GEMINATE_STOPS = {"Q"}

# 継続/メリスマ(直前音を伸ばす音符の記号)。
_CONTINUATIONS = {"-"}

# 唇に影響する子音(母音合成を変調する。lipsync の ConsonantClass へ写像。X-SAMPA)。
# ROUNDED=唇を丸める ふ(p\)・わ(w)。SPREAD=い 方向へ寄せる し(S)・じ(dZ)・ち(tS)・拗音のわたり(j)。
# これら以外の子音(唇を動かさない軟口蓋/歯茎/声門子音)と未知記号は NEUTRAL(純母音扱い)。
# 両唇音(ま/ば/ぱ行)は MouthShape.BILABIAL で表すため本写像の対象外。
_ROUNDED_CONSONANTS = {"p\\", "w"}
_SPREAD_CONSONANTS = {"S", "dZ", "tS", "j"}

# 舌位置が主体の子音(顎の開口量を部分的に減衰させる。lipsync の ApertureClass へ写像。X-SAMPA)。
_FIRM_CLOSURE_CONSONANTS = {"t", "d", "n", "ts", "dz", "J"}
_NARROW_CHANNEL_CONSONANTS = {"s", "z", "S", "dZ", "tS", "j"}
_SLIGHT_CLOSURE_CONSONANTS = {"k", "k'", "g", "4"}


def vowel_shape(symbol: str) -> MouthShape | None:
    """母音記号に対応する MouthShape(A/I/U/E/O)。母音でなければ None。

    母音記号テーブル自体は vocal_analysis(共有ドメイン層)が持つ(vpr を読む CLI と S-1認識測定が
    同一の写像表を使うため、二重管理を避ける)。
    """
    letter = xsampa_vowel_letter(symbol)
    if letter is None:
        return None
    return _VOWEL_LETTER_SHAPES[letter]


def categorize(symbol: str) -> PhonemeCategory:
    """音素記号をカテゴリへ分類する。

    母音・両唇音・撥音・促音・継続のいずれにも該当しない記号(既知のその他子音・未知記号)は、両唇閉鎖
    以外の子音と同じく自前イベントを作らず協調調音/直前口形継続へ委ねるため、まとめて OTHER とする。
    未知音素の診断記録(その他子音との区別)は写像でなく診断側で扱う。
    """
    if xsampa_vowel_letter(symbol) is not None:
        return PhonemeCategory.VOWEL
    if symbol in _BILABIALS:
        return PhonemeCategory.BILABIAL
    if symbol in _MORAIC_NASALS:
        return PhonemeCategory.MORAIC_NASAL
    if symbol in _GEMINATE_STOPS:
        return PhonemeCategory.GEMINATE_STOP
    if symbol in _CONTINUATIONS:
        return PhonemeCategory.CONTINUATION
    return PhonemeCategory.OTHER


def consonant_class(symbol: str) -> ConsonantClass:
    """子音記号を ConsonantClass へ写像する(唇への影響で分類)。

    唇を丸める子音は ROUNDED、い 方向へ寄せる子音は SPREAD、それ以外の子音(唇を動かさない子音)と
    未知記号は NEUTRAL。両唇音(ま/ば/ぱ行)は MouthShape.BILABIAL で表すので本写像の対象外で、
    呼び出し側が両唇音(BILABIAL カテゴリ)を除いた語頭子音を渡す。この関数が返す ConsonantClass は
    唇の方向だけを表し、開口減衰は別軸の aperture_class 関数が担う(独立に判定し、一方が他方に
    影響しない)。
    """
    if symbol in _ROUNDED_CONSONANTS:
        return ConsonantClass.ROUNDED
    if symbol in _SPREAD_CONSONANTS:
        return ConsonantClass.SPREAD
    return ConsonantClass.NEUTRAL


def aperture_class(symbol: str) -> ApertureClass:
    """子音記号を ApertureClass へ写像する(顎の開口減衰の強さで分類)。

    判定表に無い子音と未知記号は NONE。複数の子音から1つに絞る優先順(FIRM_CLOSURE >
    NARROW_CHANNEL > SLIGHT_CLOSURE > NONE)は呼び出し側の責務で、この関数自体は単一記号の
    分類のみを行う。
    """
    if symbol in _FIRM_CLOSURE_CONSONANTS:
        return ApertureClass.FIRM_CLOSURE
    if symbol in _NARROW_CHANNEL_CONSONANTS:
        return ApertureClass.NARROW_CHANNEL
    if symbol in _SLIGHT_CLOSURE_CONSONANTS:
        return ApertureClass.SLIGHT_CLOSURE
    return ApertureClass.NONE
