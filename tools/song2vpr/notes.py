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
# 付与の段も音節の核を見分けるのに使うので、同じ判定を2か所で持たないよう公開する。
MORAIC_NASAL = "ɴ"

# 核へ届く音符を含まない連続成分は、先頭がその音節の核の終端からこれを超えて離れていれば落とす。
_MAX_START_AFTER_NUCLEUS_SEC = 2.0


@dataclass
class Diagnostics:
    """分割の過程で数えた件数。利用者向けの診断へ出す。"""

    short_notes: int = 0
    suppressed_notes: int = 0


@dataclass(frozen=True)
class SplitResult:
    """音符列と、音節ごとのセグメント帰属と、その分割で数えた件数。

    syllable_segments は音節番号を添字とし、その音節に属するセグメントを時間順に持つ。付与の段は
    これを使って音素列と表示歌詞を決める(音符区間との時間の重なりで決め直さない)。
    """

    notes: "list[Note]" = field(default_factory=list)
    syllable_segments: "list[list]" = field(default_factory=list)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


@dataclass(frozen=True)
class Note:
    """音符の区間と音高と、自分が属する音節。区間は半開区間 [start_sec, end_sec) として扱う。"""

    start_sec: float
    end_sec: float
    midi: int
    syllable: int


def _is_syllable_nucleus(segment):
    """音節の核か。母音と、母音を伴わない撥音がこれに当たる。"""
    return segment.type == "vowel" or (segment.type == "consonant"
                                       and segment.phoneme == MORAIC_NASAL)


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

    # 各フレームにとっての後続の核(終端がそのフレームより後にある最初の核)の番号。核の区間から
    # 直に決める。フレームの被覆から数えると、刻みより短い核はどのフレームも覆わないため番号が
    # 現れず、その核を頭とする子音が直前の音節へ倒れてセグメント側の帰属と食い違う。
    ends = np.array([segment.end_sec for segment in nuclei], dtype=float)
    following = np.searchsorted(ends, times_sec, side="right")
    following = np.where(following < len(nuclei), following, -1)
    # 各フレームにとっての直前の核(そのフレームまでに始まっている最後の核)の番号。
    starts = np.array([segment.start_sec for segment in nuclei], dtype=float)
    preceding = np.searchsorted(starts, times_sec, side="right") - 1

    index = np.where((index < 0) & is_consonant, following, index)
    # 残り(gap)は直前の音節を引き継ぐ。引き継ぎだけだと、フレームを覆わない核をまたいでも番号が
    # 進まないので、区間の上で直前にある核の番号と、前のフレームから引き継いだ番号の大きい方を採る
    # (頭の子音が先に後続の音節を指しているときは、その番号を保つ)。
    for i in range(len(index)):
        if index[i] < 0:
            index[i] = max(int(preceding[i]), int(index[i - 1]) if i > 0 else -1)
    return index, is_consonant


def _segments_per_syllable(segments, nuclei):
    """各音節に属するセグメントを時間順に並べる。

    フレーム単位の帰属と同じ規則にする: 核は自分の音節、頭の子音は後続の音節、後続の核が無い
    子音は直前の音節。音素を割り当てなかった区間(gap)はどの音節にも入れない(音素列に載せない
    ものを帰属させても使い道が無い)。
    """
    result = [[] for _ in nuclei]
    number = 0  # 次に現れる核の番号
    pending = []  # 後続の核へ付ける子音
    for segment in segments:
        if number < len(nuclei) and segment is nuclei[number]:
            result[number] = [*pending, segment]
            pending = []
            number += 1
        elif segment.type == "consonant":
            pending.append(segment)
    if pending and result:
        result[-1].extend(pending)
    return result


def _voiced_spans(track, syllable_index):
    """有声フレームを、丸めた音高と音節の切り替わりで区切る。"""
    voiced = np.asarray(track.voiced)
    rounded = np.where(voiced, np.rint(np.nan_to_num(track.midi, nan=0.0)), np.nan)

    spans = []
    start = None
    for i in range(len(voiced)):
        if not voiced[i]:
            if start is not None:
                spans.append([start, i, int(rounded[start]), int(syllable_index[start])])
                start = None
            continue
        same_run = (start is not None
                    and rounded[i] == rounded[i - 1]
                    and syllable_index[i] == syllable_index[i - 1])
        if not same_run:
            if start is not None:
                spans.append([start, i, int(rounded[start]), int(syllable_index[start])])
            start = i
    if start is not None:
        spans.append([start, len(voiced), int(rounded[start]), int(syllable_index[start])])
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
    """最小長に満たない音符を、休符を挟まず連続する**同じ音節**の隣接音符へ吸収する。

    同じ音高の隣接があれば併合し、無ければ音高が最も近い隣接へ寄せる(距離が同じなら直前側)。
    吸収では吸収先の音高を保ち、短音符の区間を足して延長する。吸収先が無い(休符または別の音節で
    隔てられた)短音符は、そのまま単独の音符として残す。音節をまたいでまとめると音節=モーラが
    消えるので、別の音節しか隣に無い場合は残す方を採る。
    """
    result = [list(span) for span in spans]
    changed = True
    while changed:
        changed = False
        for i, (lo, hi, midi, syllable) in enumerate(result):
            if (hi - lo) * frame_sec >= _MIN_DURATION_SEC:
                continue
            # 隣接はフレームの添字で決める(境界が接していれば休符を挟んでいない)。
            before = (i - 1 if i > 0 and result[i - 1][1] == lo and result[i - 1][3] == syllable
                      else None)
            after = (i + 1 if i + 1 < len(result) and result[i + 1][0] == hi
                     and result[i + 1][3] == syllable else None)
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


def _components(spans):
    """同じ音節に属し切れ目なく続く音符の並び(連続成分)へ区切る。"""
    result = []
    for span in spans:
        if result and result[-1][-1][1] == span[0] and result[-1][-1][3] == span[3]:
            result[-1].append(span)
        else:
            result.append([span])
    return result


def _suppress_unattached(spans, times, nuclei):
    """音節と結びつかない音符を落とす。落とした数を第2の戻り値で返す。

    落とすのは、どの音節にも属さない有声区間と、核へ届く音符を含まない連続成分のうち先頭が
    核の終端から離れて始まるもの(成分の全件)。核へ届く音符(核と重なる音符・頭子音を含むため
    核の開始より前から始まる音符)を含む成分は、その音符の開始が核の終端以前なので、成分の先頭は
    さらにそれ以前になる。つまり離れの判定だけで必ず残り、届くかどうかを別に見る必要はない。
    """
    kept = []
    for component in _components(spans):
        syllable = component[0][3]
        start_sec = float(times[component[0][0]])
        if syllable < 0:
            continue
        if start_sec - nuclei[syllable].end_sec > _MAX_START_AFTER_NUCLEUS_SEC:
            continue
        kept += component
    return kept, len(spans) - len(kept)


def split(track, segments, *, frame_sec=None) -> SplitResult:
    """時刻ごとの音高と音素セグメントから音符の区間と音高を決める。

    frame_sec は時刻の刻み(省略時は track の時刻列から求める)。音符の区間は互いに重ならず、
    長さは正になる。まとめる先が無く最小長に満たないまま残った音符と、音節と結びつかず落とした
    音符は、件数を診断に出す(どちらも内部の規則によるもので、呼び出し側からは数えられないため)。
    """
    times = np.asarray(track.times_sec)
    if len(times) < 2:
        return SplitResult()  # 刻みを決められない長さの入力からは音符を作らない
    if frame_sec is None:
        frame_sec = float(times[1] - times[0])

    nuclei = [s for s in segments if _is_syllable_nucleus(s)]
    syllable_index, is_consonant = _syllable_index_per_frame(times, segments)
    spans = _voiced_spans(track, syllable_index)
    spans = _extend_to_leading_consonants(spans, syllable_index, is_consonant)
    spans = _absorb_short_spans(spans, frame_sec)
    # 抑制は吸収の後に行う。短いまま残った音符の診断は、抑制を通って出力される音符だけを数える。
    spans, suppressed = _suppress_unattached(spans, times, nuclei)

    # 終端は次のフレームの時刻をそのまま使う(切れ目なく続く音符が同じ値を共有し、後段が
    # 「前の終わり == 次の始まり」で連続を判定できる)。末尾だけは次のフレームが無いので刻みを足す。
    result = [Note(start_sec=float(times[lo]),
                   end_sec=float(times[hi]) if hi < len(times)
                   else float(times[hi - 1]) + frame_sec,
                   midi=midi, syllable=syllable)
              for lo, hi, midi, syllable in spans]
    short = sum(1 for lo, hi, _midi, _syllable in spans
                if (hi - lo) * frame_sec < _MIN_DURATION_SEC)
    return SplitResult(notes=result,
                       syllable_segments=_segments_per_syllable(segments, nuclei),
                       diagnostics=Diagnostics(short_notes=short, suppressed_notes=suppressed))
