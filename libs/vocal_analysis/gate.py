"""S-1 認識ゲートの参照ラベル処理。

参照ラベルの音素記号を採点用カテゴリ(母音 a/i/u/e/o・子音 c・無音/息 sil)へ写像する。想定外記号は
黙って捨てず `UnknownReferenceSymbolError` で停止する。モノフォンラベル形式(秒単位・HTK 100ns単位)の
パーサも提供する。
"""

from dataclasses import dataclass
from pathlib import Path

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
