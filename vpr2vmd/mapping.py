"""音符内の音素→口形イベント写像(vpr2vmd.md §3)。

1つの採用音符のフレーム区間 [s, e) の音素列を口形イベント(MouthEvent)へ写像する。継続「-」は
直前母音に依存するため本モジュールでは扱わず、組み立て段が処理する。開き量(ベロシティ写像)は
後続ステップで、本段は open_amount=0.0。
"""

from lipsync import MouthEvent, MouthShape

from .phonemes import PhonemeCategory, categorize, vowel_shape

# 語頭両唇閉鎖の時間配分(vpr2vmd.md §3)。初期値で、実データ(視覚確認)で調整する。
_BILABIAL_NOMINAL_FRAMES = 3.0  # 公称長
_BILABIAL_SHARE_CAP = 0.5  # 音符長に対する取り分上限

_DEFAULT_VOWEL = MouthShape.A  # 母音が得られない音符の既定母音「あ」


def note_mouth_events(phonemes: list[str], start: float, end: float) -> list[MouthEvent]:
    """1採用音符 [start, end)(フレーム)の音素列を口形イベント列へ写像する(vpr2vmd.md §3)。

    - 撥音(後続母音を持たない単独の鼻音)は音符全体を「ん」(N)にする。
    - 語頭が両唇音なら音符先頭に閉鎖 [start, start+d_b)、d_b = min(公称長, (end-start)×取り分上限)。
    - 残り区間 [start+d_b, end) の母音 k 個を区間長で均等分割する(k=1 なら残り全体)。
    - 母音が得られない音符は残り区間へ既定母音「あ」を1つ置く(語頭両唇音があれば閉鎖は保持)。
    - 両唇閉鎖以外の子音は自前イベントを作らない(母音・協調調音へ委ねる)。
    """
    # 撥音(単独の鼻音)は音符全体を「ん」にする。仕様の「単独」に合わせ、全 phoneme が撥音記号の
    # ときだけ N とする(子音を伴う非単独の鼻音は撥音扱いせず、下の通常処理=既定母音/語頭両唇へ倒す)。
    if phonemes and all(categorize(p) is PhonemeCategory.MORAIC_NASAL for p in phonemes):
        return [MouthEvent(MouthShape.N, start, end)]

    vowels = [vowel_shape(p) for p in phonemes if categorize(p) is PhonemeCategory.VOWEL]
    events: list[MouthEvent] = []
    # 語頭両唇閉鎖の取り分(語中・語末の両唇音は対象外。語頭=phonemes[0] のみ判定)。
    d_b = 0.0
    if phonemes and categorize(phonemes[0]) is PhonemeCategory.BILABIAL:
        d_b = min(_BILABIAL_NOMINAL_FRAMES, (end - start) * _BILABIAL_SHARE_CAP)
        events.append(MouthEvent(MouthShape.BILABIAL, start, start + d_b))

    # 母音が得られなければ既定母音「あ」を1つ置く。
    if not vowels:
        vowels = [_DEFAULT_VOWEL]

    # 残り区間を母音 k 個で均等分割する。末尾は丸め誤差を避けるため end をそのまま使う。
    region_start = start + d_b
    width = (end - region_start) / len(vowels)
    for i, shape in enumerate(vowels):
        seg_start = region_start + i * width
        seg_end = end if i == len(vowels) - 1 else region_start + (i + 1) * width
        events.append(MouthEvent(shape, seg_start, seg_end))
    return events
