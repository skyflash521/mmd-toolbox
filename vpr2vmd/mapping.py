"""音符内の音素→口形イベント写像(vpr2vmd.md §3)。

1つの採用音符のフレーム区間 [s, e) の音素列を、文脈なしで定まる口形イベント(MouthEvent)へ写像する。
母音を持たず撥音/促音でもない音符(継続「-」・その他子音のみ・未知のみ・空)は直前の口形に依存するため
None を返し、組み立て段が直前の口形を継続する(既定母音「あ」へのフォールバックはしない)。本段は
開き量を付与せず(open_amount=0.0)、組み立て段の build_mouth_events が母音的口形へ開き量を刻印する。
"""

from lipsync import MouthEvent, MouthShape

from .phonemes import PhonemeCategory, categorize, vowel_shape

# 語頭両唇閉鎖の時間配分(vpr2vmd.md §3)。初期値で、実データ(視覚確認)で調整する。
_BILABIAL_NOMINAL_FRAMES = 3.0  # 公称長
_BILABIAL_SHARE_CAP = 0.5  # 音符長に対する取り分上限


def note_mouth_events(
    phonemes: list[str], start: float, end: float, use_n_morph: bool = True
) -> list[MouthEvent] | None:
    """1採用音符 [start, end)(フレーム)の音素列を口形イベント列へ写像する(vpr2vmd.md §3)。

    文脈なしで定まる口形だけを返す。優先順は次のとおり(MMD の見た目で口を不自然に開けない):

    - 母音を含む音符: 語頭が両唇音なら音符先頭に閉鎖 [start, start+d_b)、d_b = min(公称長,
      (end-start)×取り分上限)。残り区間 [start+d_b, end) の母音 k 個を均等分割する(両唇閉鎖以外の
      子音は自前イベントを作らない)。
    - 母音を持たない音符: 促音(`Q`)を含むなら無音(閉口)。撥音(全 phoneme が単独の鼻音)なら
      `use_n_morph` で「ん」(N)/無効時は無音(閉口)。
    - それ以外で母音を持たない音符(継続・その他子音のみ・未知のみ・空)は `None` を返し、組み立て段で
      直前の口形を継続する(既定母音へ倒さない)。
    """
    vowels = [vowel_shape(p) for p in phonemes if categorize(p) is PhonemeCategory.VOWEL]
    if vowels:
        return _vowel_events(phonemes, start, end, vowels)

    # 以下は母音を持たない音符。閉鎖・撥音を先に確定し、それ以外は直前口形継続(None)に委ねる。
    if any(categorize(p) is PhonemeCategory.GEMINATE_STOP for p in phonemes):
        return [MouthEvent(MouthShape.SILENCE, start, end)]
    if phonemes and all(categorize(p) is PhonemeCategory.MORAIC_NASAL for p in phonemes):
        shape = MouthShape.N if use_n_morph else MouthShape.SILENCE
        return [MouthEvent(shape, start, end)]
    return None


def _vowel_events(
    phonemes: list[str], start: float, end: float, vowels: list[MouthShape]
) -> list[MouthEvent]:
    """母音を含む音符の口形イベント(語頭両唇閉鎖 + 母音の等分)。"""
    events: list[MouthEvent] = []
    # 語頭両唇閉鎖の取り分(語中・語末の両唇音は対象外。語頭=phonemes[0] のみ判定)。
    d_b = 0.0
    if categorize(phonemes[0]) is PhonemeCategory.BILABIAL:
        d_b = min(_BILABIAL_NOMINAL_FRAMES, (end - start) * _BILABIAL_SHARE_CAP)
        events.append(MouthEvent(MouthShape.BILABIAL, start, start + d_b))

    # 残り区間を母音 k 個で均等分割する。末尾は丸め誤差を避けるため end をそのまま使う。
    region_start = start + d_b
    width = (end - region_start) / len(vowels)
    for i, shape in enumerate(vowels):
        seg_start = region_start + i * width
        seg_end = end if i == len(vowels) - 1 else region_start + (i + 1) * width
        events.append(MouthEvent(shape, seg_start, seg_end))
    return events
