"""S-1 認識ゲートの参照ラベル処理。

参照ラベルの音素記号を採点用カテゴリ(母音 a/i/u/e/o・子音 c・無音/息 sil)へ写像する。想定外記号は
黙って捨てず `UnknownReferenceSymbolError` で停止する。モノフォンラベル形式(秒単位・HTK 100ns単位)の
パーサも提供する。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vpr.rests import rest_intervals
from vpr.types import Part, TempoEvent

from .phonemes import xsampa_vowel_letter

# モノフォンラベルのHTK 100ns単位を秒へ変換する係数。
_HTK_100NS_UNITS_PER_SECOND = 1e7

_VOWEL_SYMBOLS = frozenset({"a", "i", "u", "e", "o"})
_SILENCE_SYMBOLS = frozenset({"pau", "br"})
# cl(促音の閉鎖)・N(撥音)・その他の全子音。sy/ty/zy に相当する音は sh/ch/j で表記されるため
# 別記号として含めない。
_CONSONANT_SYMBOLS = frozenset({
    "b", "by", "ch", "cl", "d", "f", "g", "gy", "h", "hy", "j", "k", "ky",
    "m", "my", "n", "N", "ny", "p", "py", "r", "ry", "s", "sh", "t", "ts",
    "v", "w", "y", "z",
})
_EXCLUDED_SYMBOLS = frozenset({"xx"})


class UnknownReferenceSymbolError(Exception):
    """参照ラベルの写像表に無い想定外の音素記号(写像表を補う必要がある)。"""


class MonophoneLabelFormatError(Exception):
    """モノフォンラベルファイルの行が「開始 終了 音素」の3列形式でない(原因特定用にファイルパスと
    行内容を含む)。"""


@dataclass
class CategorySegment:
    """参照ラベルの1区間(採点用カテゴリ)。

    `category` は `reference_symbol_to_category` の戻り値(母音 a/i/u/e/o・子音 c・無音/息 sil、
    または xx による除外印の None)。
    """

    category: str | None
    start_sec: float
    end_sec: float


def reference_symbol_to_category(symbol: str) -> str | None:
    """参照ラベルの音素記号を採点用カテゴリへ写像する。

    母音は a/i/u/e/o、pau・br は sil、子音は c を返す。xx(未定義区間)は採点から除外する
    印として None を返す。写像表に無い記号は `UnknownReferenceSymbolError` で停止する。
    """
    if symbol in _VOWEL_SYMBOLS:
        return symbol
    if symbol in _SILENCE_SYMBOLS:
        return "sil"
    if symbol in _CONSONANT_SYMBOLS:
        return "c"
    if symbol in _EXCLUDED_SYMBOLS:
        return None
    raise UnknownReferenceSymbolError(
        f"参照ラベルの写像表に無い音素記号です: {symbol!r}(写像表を補ってください)"
    )


def _parse_monophone_label_lines(
    path: Path, lines: list[str], time_scale: float
) -> list[CategorySegment]:
    """「開始 終了 音素」の3列形式(空行は無視)を、時刻を time_scale で除して秒へ変換しつつ
    カテゴリ区間列へ変換する。記号→カテゴリの写像は reference_symbol_to_category に
    委譲し、ここでは独自実装しない。"""
    segments = []
    for line in lines:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise MonophoneLabelFormatError(
                f"{path}: 「開始 終了 音素」の3列形式ではない行です: {line!r}"
            )
        start_str, end_str, symbol = fields
        segments.append(
            CategorySegment(
                category=reference_symbol_to_category(symbol),
                start_sec=float(start_str) / time_scale,
                end_sec=float(end_str) / time_scale,
            )
        )
    return segments


def parse_seconds_monophone_label(path: str | Path) -> list[CategorySegment]:
    """モノフォンラベル(開始 終了 音素。時刻は秒)を読み、カテゴリ区間列へ変換する。"""
    path = Path(path)
    return _parse_monophone_label_lines(path, path.read_text(encoding="utf-8").splitlines(), time_scale=1.0)


def parse_htk100ns_monophone_label(path: str | Path) -> list[CategorySegment]:
    """モノフォンラベル(開始 終了 音素。時刻はHTK 100ns単位の整数)を読み、カテゴリ区間列へ
    変換する。"""
    path = Path(path)
    return _parse_monophone_label_lines(
        path, path.read_text(encoding="utf-8").splitlines(), time_scale=_HTK_100NS_UNITS_PER_SECOND
    )


def clip_segments_to_audio_duration(
    segments: list[CategorySegment], audio_duration_sec: float
) -> list[CategorySegment]:
    """採点の時間軸を [0, audio_duration_sec] に限定する。範囲外の区間は除外し、範囲をまたぐ
    区間は範囲内に収まるようクリップする。範囲内で参照ラベルが被覆しない区間(末尾欠落等)を
    埋める合成区間は追加しない。"""
    result = []
    for seg in segments:
        start = max(seg.start_sec, 0.0)
        end = min(seg.end_sec, audio_duration_sec)
        if end > start:
            result.append(CategorySegment(category=seg.category, start_sec=start, end_sec=end))
    return result


def remove_invalid_time_segments(segments: list[CategorySegment]) -> list[CategorySegment]:
    """ゼロ長・時刻逆転の区間を除去し、2つ以上の区間が重複する時間範囲(隣接する区間どうしに
    限らず、一方が他方を包含する場合や離れた区間と重なる場合も含む)を、どの区間からも除外する
    (採点対象に残さず隙間にする)。

    区間境界(開始・終了時刻)で時間軸を分割した各微小区間ごとに、それを覆う元区間の数を数える
    (掃引法)。覆う元区間がちょうど1つの微小区間だけを採用し、同一の元区間に由来する隣接微小
    区間は1つの区間へ結合する。覆う元区間が0または2つ以上の微小区間は捨てる。
    """
    valid = [seg for seg in segments if seg.end_sec > seg.start_sec]
    if not valid:
        return []

    boundaries = sorted({seg.start_sec for seg in valid} | {seg.end_sec for seg in valid})
    result: list[CategorySegment] = []
    owner_of_last: int | None = None
    for lo, hi in zip(boundaries, boundaries[1:]):
        if hi <= lo:
            continue
        mid = (lo + hi) / 2
        covering = [i for i, seg in enumerate(valid) if seg.start_sec <= mid < seg.end_sec]
        if len(covering) != 1:
            owner_of_last = None
            continue
        owner = covering[0]
        if result and owner_of_last == owner and result[-1].end_sec == lo:
            result[-1] = CategorySegment(category=valid[owner].category, start_sec=result[-1].start_sec, end_sec=hi)
        else:
            result.append(CategorySegment(category=valid[owner].category, start_sec=lo, end_sec=hi))
        owner_of_last = owner
    return result


# MIDI粗整合の食い違いとみなす連続時間の閾値。
_MIDI_MISMATCH_THRESHOLD_SEC = 0.3


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """区間列(開始, 終了)を開始時刻順に整列し、重なる/接する区間を結合する。"""
    ordered = sorted((iv for iv in intervals if iv[1] > iv[0]), key=lambda iv: iv[0])
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(
    base: tuple[float, float], subtract: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """base区間から subtract 区間群(結合済み・開始時刻順)を差し引いた残り区間列を返す。"""
    start, end = base
    remaining: list[tuple[float, float]] = []
    cursor = start
    for sub_start, sub_end in subtract:
        if sub_end <= cursor or sub_start >= end:
            continue
        if sub_start > cursor:
            remaining.append((cursor, min(sub_start, end)))
        cursor = max(cursor, sub_end)
        if cursor >= end:
            break
    if cursor < end:
        remaining.append((cursor, end))
    return remaining


def _intersect_intervals(
    base: tuple[float, float], others: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """base区間と others 区間群(結合済み・開始時刻順)との重なり区間列を返す。"""
    start, end = base
    result: list[tuple[float, float]] = []
    for other_start, other_end in others:
        lo = max(start, other_start)
        hi = min(end, other_end)
        if hi > lo:
            result.append((lo, hi))
    return result


def find_midi_mismatch_ranges(
    segments: list[CategorySegment], midi_notes: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """参照ラベルとMIDIノート(発音区間)を粗く突き合わせ、長時間の食い違いを検出する。

    母音区間のうちMIDIノートに重ならない部分、および sil 区間のうちMIDIノートに連続して
    覆われる部分について、`_MIDI_MISMATCH_THRESHOLD_SEC` 以上続くものを食い違い区間として
    返す。子音区間は対象外。
    """
    merged_notes = _merge_intervals(midi_notes)
    mismatches: list[tuple[float, float]] = []
    for seg in segments:
        if seg.category in _VOWEL_SYMBOLS:
            candidates = _subtract_intervals((seg.start_sec, seg.end_sec), merged_notes)
        elif seg.category == "sil":
            candidates = _intersect_intervals((seg.start_sec, seg.end_sec), merged_notes)
        else:
            continue
        for start, end in candidates:
            if end - start >= _MIDI_MISMATCH_THRESHOLD_SEC:
                mismatches.append((start, end))
    return mismatches


def exclude_ranges_from_segments(
    segments: list[CategorySegment], ranges_to_exclude: list[tuple[float, float]]
) -> list[CategorySegment]:
    """segments から、ranges_to_exclude の各区間と重なる時間範囲を除外した区間列を返す
    (重なる範囲だけを取り除き、区間を分割・全部除外・無変更のいずれかにする)。"""
    merged_exclusions = _merge_intervals(ranges_to_exclude)
    result: list[CategorySegment] = []
    for seg in segments:
        for start, end in _subtract_intervals((seg.start_sec, seg.end_sec), merged_exclusions):
            result.append(CategorySegment(category=seg.category, start_sec=start, end_sec=end))
    return result


def ticks_to_seconds(tick: int, tempos: list[TempoEvent], resolution: int) -> float:
    """vpr の tick 値を、テンポマップ(`tempos`)と分解能(`resolution`。tick/四分音符)に基づいて
    秒へ変換する(`vpr` は tick⇔秒変換を呼び出し側の責務とする)。

    `tempos` はテンポ変化イベントの列(tick の順序は問わない)。各テンポ区間ごとに
    経過秒数(`(区間のtick長 / resolution) * (60 / bpm)`)を積算する。
    """
    ordered = sorted(tempos, key=lambda t: t.tick)
    seconds = 0.0
    for i, tempo in enumerate(ordered):
        next_tick = ordered[i + 1].tick if i + 1 < len(ordered) else None
        segment_end = tick if next_tick is None or tick <= next_tick else next_tick
        ticks_in_segment = segment_end - tempo.tick
        seconds += (ticks_in_segment / resolution) * (60.0 / tempo.bpm)
        if next_tick is None or tick <= next_tick:
            break
    return seconds


def generate_vpr_reference_segments(
    part: Part, tempos: list[TempoEvent], resolution: int
) -> list[CategorySegment]:
    """vpr の1パート(歌唱区間)から参照ラベルのセグメント列を生成する。

    各音符の代表音素(音素列の末尾)を X-SAMPA母音写像で母音(a/i/u/e/o)または子音(c)へ分類する。
    音符全体が継続記号「-」単独、または音素列が空の音符は、直前の音符の代表音素を継承する
    (継続は音符全体の状態であり、他の音素と混在する「-」は想定しない。継承元が無い場合は
    その音符の区間を生成しない)。音符間の隙間は休符として sil 区間にする。同一カテゴリで
    時間的に連続する区間は1つに結合する。
    """
    segments: list[CategorySegment] = []
    previous_phoneme: str | None = None
    for note in part.notes:
        resolved = previous_phoneme if not note.phonemes or note.phonemes == ["-"] else note.phonemes[-1]
        if resolved is not None:
            previous_phoneme = resolved
        if resolved is None:
            continue
        vowel = xsampa_vowel_letter(resolved)
        category = vowel if vowel is not None else "c"
        segments.append(
            CategorySegment(
                category=category,
                start_sec=ticks_to_seconds(note.start_tick, tempos, resolution),
                end_sec=ticks_to_seconds(note.start_tick + note.duration_tick, tempos, resolution),
            )
        )

    end_tick = max((note.start_tick + note.duration_tick for note in part.notes), default=0)
    for rest_start_tick, rest_end_tick in rest_intervals(part.notes, end_tick):
        segments.append(
            CategorySegment(
                category="sil",
                start_sec=ticks_to_seconds(rest_start_tick, tempos, resolution),
                end_sec=ticks_to_seconds(rest_end_tick, tempos, resolution),
            )
        )

    segments.sort(key=lambda seg: seg.start_sec)
    merged: list[CategorySegment] = []
    for seg in segments:
        if merged and merged[-1].category == seg.category and merged[-1].end_sec == seg.start_sec:
            merged[-1] = CategorySegment(category=seg.category, start_sec=merged[-1].start_sec, end_sec=seg.end_sec)
        else:
            merged.append(seg)
    return merged


# 採点指標のフレーム展開幅。
_FRAME_SEC = 0.01
_NON_VOWEL_SCORED_CATEGORIES = frozenset({"c", "sil"})


def _frame_categories(segments: list[CategorySegment], duration_sec: float) -> list[str | None]:
    """[0, duration_sec) を `_FRAME_SEC` 刻みのフレームへ展開し、各フレーム代表時刻(フレーム中央)を
    覆う区間のカテゴリ列を返す。どの区間にも覆われないフレームは None。segments は開始時刻順である
    必要は無い(内部で並べ替える)。区間どうしは重複しない前提(`remove_invalid_time_segments` 適用後)。
    """
    ordered = sorted(segments, key=lambda seg: seg.start_sec)
    num_frames = round(duration_sec / _FRAME_SEC)
    categories: list[str | None] = []
    idx = 0
    for i in range(num_frames):
        t = (i + 0.5) * _FRAME_SEC
        while idx < len(ordered) and ordered[idx].end_sec <= t:
            idx += 1
        if idx < len(ordered) and ordered[idx].start_sec <= t:
            categories.append(ordered[idx].category)
        else:
            categories.append(None)
    return categories


def compute_vowel_accuracy(
    predicted: list[CategorySegment], reference: list[CategorySegment], duration_sec: float
) -> float:
    """母音正解率(§9.4): 基準が母音(a/i/u/e/o)のフレームのうち、予測の母音種別が一致した割合。
    未検出(予測がそのフレームを覆わない)は不一致として数える。"""
    ref_frames = _frame_categories(reference, duration_sec)
    pred_frames = _frame_categories(predicted, duration_sec)
    vowel_indices = [i for i, category in enumerate(ref_frames) if category in _VOWEL_SYMBOLS]
    matches = sum(1 for i in vowel_indices if pred_frames[i] == ref_frames[i])
    return matches / len(vowel_indices)


def compute_over_opening_rate(
    predicted: list[CategorySegment], reference: list[CategorySegment], duration_sec: float
) -> float:
    """過開口率(§9.4): 基準が c/sil のフレームのうち、予測が母音になったフレームの割合。"""
    ref_frames = _frame_categories(reference, duration_sec)
    pred_frames = _frame_categories(predicted, duration_sec)
    non_vowel_indices = [i for i, category in enumerate(ref_frames) if category in _NON_VOWEL_SCORED_CATEGORIES]
    bled = sum(1 for i in non_vowel_indices if pred_frames[i] in _VOWEL_SYMBOLS)
    return bled / len(non_vowel_indices)


def _overlap_sec(a: CategorySegment, b: CategorySegment) -> float:
    return max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))


def _better_assignment(
    a: tuple[float, tuple[tuple[int, int], ...]], b: tuple[float, tuple[tuple[int, int], ...]]
) -> tuple[float, tuple[tuple[int, int], ...]]:
    """総重なり時間の最大化を第一基準、対応ペア列(基準番号, 予測番号)の辞書式最小を第二基準として
    2つの候補のうち優先する方を返す。"""
    if a[0] != b[0]:
        return a if a[0] > b[0] else b
    return a if a[1] <= b[1] else b


def match_segments(
    predicted: list[CategorySegment], reference: list[CategorySegment]
) -> tuple[list[tuple[CategorySegment, CategorySegment]], list[CategorySegment], list[CategorySegment]]:
    """区間の対応付け(§9.4)。

    母音カテゴリ(a/i/u/e/o)の基準区間と予測区間のうち、母音種別が一致し時間重なりが正(0より大)の
    組に限り、総重なり時間を最大化する全体最適割当で1対1対応させる。子音・無音カテゴリの区間は
    母音種別を持たないため対応付けの対象にしない(未検出・余剰にも数えない)。最適割当が複数ある
    ときは、基準区間・予測区間をそれぞれ開始時刻昇順(同時刻なら終了時刻昇順)で番号付けし、割当を
    (基準番号, 予測番号)の組の昇順リストとして辞書式比較した最小の割当を採る。

    基準区間列・予測区間列はそれぞれ時間的に重複しない(remove_invalid_time_segments 適用後)前提
    とする。この前提の下では、基準番号・予測番号を跨いだ対応付けが交差する(番号の大きい基準区間が
    番号の小さい予測区間に、番号の小さい基準区間が番号の大きい予測区間に、同時に正の重なりを持つ)
    ことはあり得ないため、2つの整列済み列を先頭から同時に走査する動的計画法で全体最適割当を厳密に
    求められる。

    戻り値: (対応した(基準区間, 予測区間)の組のリスト, 対応の無い基準区間=未検出のリスト,
    対応の無い予測区間=余剰のリスト)。
    """
    ref_vowels = sorted(
        (seg for seg in reference if seg.category in _VOWEL_SYMBOLS),
        key=lambda seg: (seg.start_sec, seg.end_sec),
    )
    pred_vowels = sorted(
        (seg for seg in predicted if seg.category in _VOWEL_SYMBOLS),
        key=lambda seg: (seg.start_sec, seg.end_sec),
    )
    n, m = len(ref_vowels), len(pred_vowels)

    dp: list[list[tuple[float, tuple[tuple[int, int], ...]]]] = [[(0.0, ())] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0]
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = _better_assignment(dp[i - 1][j], dp[i][j - 1])
            ref_seg, pred_seg = ref_vowels[i - 1], pred_vowels[j - 1]
            overlap = _overlap_sec(ref_seg, pred_seg)
            if ref_seg.category == pred_seg.category and overlap > 0:
                prev_total, prev_pairs = dp[i - 1][j - 1]
                candidate = (prev_total + overlap, prev_pairs + ((i - 1, j - 1),))
                best = _better_assignment(best, candidate)
            dp[i][j] = best

    _, pairs = dp[n][m]
    matched_ref_indices = {ref_idx for ref_idx, _ in pairs}
    matched_pred_indices = {pred_idx for _, pred_idx in pairs}
    matched = [(ref_vowels[ref_idx], pred_vowels[pred_idx]) for ref_idx, pred_idx in pairs]
    undetected = [seg for idx, seg in enumerate(ref_vowels) if idx not in matched_ref_indices]
    excess = [seg for idx, seg in enumerate(pred_vowels) if idx not in matched_pred_indices]
    return matched, undetected, excess


def compute_boundary_deviation(
    matched_pairs: list[tuple[CategorySegment, CategorySegment]],
) -> tuple[float, float] | None:
    """境界時刻ずれ(§9.4): match_segments が返す対応済み(基準区間, 予測区間)の組ごとに開始時刻差
    |予測-基準|(ミリ秒)を求め、その中央値と95パーセンタイル(線形補間)を返す。対応区間が無い場合は
    未定義として None を返す(マクロ平均からの除外・ゲート不合格判定は集計処理=呼び出し側の責務)。
    """
    if not matched_pairs:
        return None
    deviations_ms = [
        abs(predicted.start_sec - reference.start_sec) * 1000.0 for reference, predicted in matched_pairs
    ]
    median, p95 = np.percentile(deviations_ms, [50, 95], method="linear")
    return float(median), float(p95)
