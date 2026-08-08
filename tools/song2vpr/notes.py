"""song2vpr の音符分割。

時刻ごとの音高と音素セグメントから、音符の区間と音高を切る。表示歌詞・音素列・強弱の付与は
この後の段が、秒から tick への量子化はさらにその後が持つので、ここは区間と音高だけを決める。

区間はフレームの添字で組み立て、秒へ写すのは最後の1回だけにする。隣接の判定を時刻の一致で行うと、
同じ実数を別の順序で計算した値どうしの比較になり、丸めの向き次第で休符を挟んでいない音符が
隣接と見なされないことがあるため。
"""

from dataclasses import dataclass, field

import numpy as np

# これより短い音符は独立させず、隣接する音符へ吸収する(人が1音として歌っていない長さのため)。
_MIN_DURATION_SEC = 0.08

# 撥音の音素記号。母音を伴わずに1つの音節をなすので、音節の核として母音と同じに扱う。
_MORAIC_NASAL = "ɴ"


@dataclass
class Diagnostics:
    """分割の過程で数えた件数。利用者向けの診断へ出す。"""

    short_notes: int = 0


@dataclass(frozen=True)
class SplitResult:
    """音符列と、その分割で数えた件数。"""

    notes: "list[Note]" = field(default_factory=list)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


@dataclass(frozen=True)
class Note:
    """音符の区間と音高。区間は半開区間 [start_sec, end_sec) として扱う。"""

    start_sec: float
    end_sec: float
    midi: int


def _is_syllable_nucleus(segment):
    """音節の核か。母音と、母音を伴わない撥音がこれに当たる。"""
    return segment.type == "vowel" or (segment.type == "consonant"
                                       and segment.phoneme == _MORAIC_NASAL)


def _syllable_index_per_frame(times_sec, segments):
    """各フレームに、それが属する音節の通し番号を割り当てる。

    子音フレームは**後続**の音節の番号を持つ。音節の頭の子音をその音符へ含めるためで、直前の
    音節へ寄せると音符の音素列から頭の子音が落ちる。gap フレームは直前の音節を引き継ぐ
    (gap は認識器が音素を割り当てなかった区間で、発声の継続中にも出るため切れ目にしない)。
    """
    index = np.full(len(times_sec), -1)
    is_consonant = np.zeros(len(times_sec), dtype=bool)

    nuclei = [s for s in segments if _is_syllable_nucleus(s)]
    for number, segment in enumerate(nuclei):
        covered = (times_sec >= segment.start_sec) & (times_sec < segment.end_sec)
        index[covered] = number
    for segment in segments:
        if segment.type == "consonant" and not _is_syllable_nucleus(segment):
            is_consonant |= (times_sec >= segment.start_sec) & (times_sec < segment.end_sec)

    # 子音は後ろ向きに、それ以外(gap)は前向きに埋める。
    for i in range(len(index) - 2, -1, -1):
        if index[i] < 0 and is_consonant[i]:
            index[i] = index[i + 1]
    for i in range(1, len(index)):
        if index[i] < 0:
            index[i] = index[i - 1]
    return index, is_consonant


def _voiced_spans(track, syllable_index):
    """有声フレームを、丸めた音高と音節の切り替わりで区切る。"""
    voiced = np.asarray(track.voiced)
    rounded = np.where(voiced, np.rint(np.nan_to_num(track.midi, nan=0.0)), np.nan)

    spans = []
    start = None
    for i in range(len(voiced)):
        if not voiced[i]:
            if start is not None:
                spans.append([start, i, int(rounded[start])])
                start = None
            continue
        same_run = (start is not None
                    and rounded[i] == rounded[i - 1]
                    and syllable_index[i] == syllable_index[i - 1])
        if not same_run:
            if start is not None:
                spans.append([start, i, int(rounded[start])])
            start = i
    if start is not None:
        spans.append([start, len(voiced), int(rounded[start])])
    return spans


def _extend_to_leading_consonants(spans, syllable_index, is_consonant):
    """各音符の開始を、同じ音節の頭にある子音フレームまで前へ伸ばす。

    無声の子音でも音符へ含める(音符の音素列から音節の頭が落ちないようにするため)。前の音符の
    終わりは越えない。
    """
    for position, span in enumerate(spans):
        lo = span[0]
        # 直前の音符の終わりを越えない。音節の番号が先に変わるので通常はここへ届かないが、
        # 音符どうしを重ねないことをこの段だけで保証する。
        limit = spans[position - 1][1] if position > 0 else 0
        while lo > limit and is_consonant[lo - 1] and syllable_index[lo - 1] == syllable_index[span[0]]:
            lo -= 1
        span[0] = lo
    return spans


def _absorb_short_spans(spans, frame_sec):
    """最小長に満たない音符を、休符を挟まず連続する隣接音符へ吸収する。

    同じ音高の隣接があれば併合し、無ければ音高が最も近い隣接へ寄せる(距離が同じなら直前側)。
    吸収では吸収先の音高を保ち、短音符の区間を足して延長する。吸収先が無い(休符で隔てられた)
    短音符は、そのまま単独の音符として残す。
    """
    result = [list(span) for span in spans]
    changed = True
    while changed:
        changed = False
        for i, (lo, hi, midi) in enumerate(result):
            if (hi - lo) * frame_sec >= _MIN_DURATION_SEC:
                continue
            # 隣接はフレームの添字で決める(境界が接していれば休符を挟んでいない)。
            before = i - 1 if i > 0 and result[i - 1][1] == lo else None
            after = i + 1 if i + 1 < len(result) and result[i + 1][0] == hi else None
            if before is None and after is None:
                continue

            if before is not None and result[before][2] == midi:
                target = before
            elif after is not None and result[after][2] == midi:
                target = after
            elif before is None:
                target = after
            elif after is None:
                target = before
            else:
                distance_before = abs(result[before][2] - midi)
                distance_after = abs(result[after][2] - midi)
                target = before if distance_before <= distance_after else after

            if target == before:
                result[target][1] = hi
            else:
                result[target][0] = lo
            del result[i]
            changed = True
            break
    return result


def split(track, segments, *, frame_sec=None) -> SplitResult:
    """時刻ごとの音高と音素セグメントから音符の区間と音高を決める。

    frame_sec は時刻の刻み(省略時は track の時刻列から求める)。音符の区間は互いに重ならず、
    長さは正になる。まとめる先が無く最小長に満たないまま残った音符は、件数を診断に出す
    (最小長は内部の値で、呼び出し側からは数えられないため)。
    """
    times = np.asarray(track.times_sec)
    if len(times) < 2:
        return SplitResult()  # 刻みを決められない長さの入力からは音符を作らない
    if frame_sec is None:
        frame_sec = float(times[1] - times[0])

    syllable_index, is_consonant = _syllable_index_per_frame(times, segments)
    spans = _voiced_spans(track, syllable_index)
    spans = _extend_to_leading_consonants(spans, syllable_index, is_consonant)
    spans = _absorb_short_spans(spans, frame_sec)

    # 終端は次のフレームの時刻をそのまま使う(切れ目なく続く音符が同じ値を共有し、後段が
    # 「前の終わり == 次の始まり」で連続を判定できる)。末尾だけは次のフレームが無いので刻みを足す。
    result = [Note(start_sec=float(times[lo]),
                   end_sec=float(times[hi]) if hi < len(times)
                   else float(times[hi - 1]) + frame_sec,
                   midi=midi)
              for lo, hi, midi in spans]
    short = sum(1 for lo, hi, _midi in spans if (hi - lo) * frame_sec < _MIN_DURATION_SEC)
    return SplitResult(notes=result, diagnostics=Diagnostics(short_notes=short))
