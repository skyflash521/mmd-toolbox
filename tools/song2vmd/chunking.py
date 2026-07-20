"""song2vmd 長尺分割の境界検出・セグメント結合。

処理資源対策のための自動分割(`--max-duration`)を担う純粋ロジック。実際の音声ファイル分割・
vocal_analysis(S0〜S2)呼び出し・RMSの曲全体基準での算出は、実音声を要する
統合の関心事であり本モジュールの対象外。
"""

from vocal_analysis import Segment


def find_chunk_boundaries(duration_sec, rms_times_sec, rms_values, *, max_duration_sec,
                           search_window_sec=5.0, silence_threshold=0.06):
    """長尺の分割境界を決める。

    最初の目標境界は先頭から max_duration_sec 秒後、以降の目標境界は直前に確定した実境界
    (無音点または強制分割点)から max_duration_sec 秒後とする。各目標境界の前後
    search_window_sec 秒の探索窓内で最も静かな点(RMSが最小の点)を選び、その値が
    silence_threshold 以下なら採用する。窓内の最小値が silence_threshold を超える場合は
    無音が無いとみなし、目標境界そのもので強制分割する。

    候補は前の実境界より後ろ(t > previous_boundary)に限ることで、無音点が繰り返し見つかっても
    境界が前進し続けることを保証する(max_duration_sec が search_window_sec 以下でも停止しない)。

    戻り値は (boundary_sec, forced) のタプル列。forced は探索窓内に無音が無く目標境界そのもので
    強制分割したかどうかを境界ごとに示す(song2vmd.md 12.1 の forced_split 警告の送出判定に使う)。
    """
    if max_duration_sec <= 0 or duration_sec <= max_duration_sec:
        return []

    boundaries = []
    previous_boundary = 0.0
    target = max_duration_sec
    while target < duration_sec:
        lo, hi = target - search_window_sec, target + search_window_sec
        candidates = [
            (t, v) for t, v in zip(rms_times_sec, rms_values) if lo <= t <= hi and t > previous_boundary
        ]
        if candidates:
            quietest_time, quietest_value = min(candidates, key=lambda tv: tv[1])
            forced = quietest_value > silence_threshold
            boundary = target if forced else quietest_time
        else:
            forced = True
            boundary = target
        boundaries.append((boundary, forced))
        previous_boundary = boundary
        target = boundary + max_duration_sec
    return boundaries


def merge_chunk_segments(chunk_segments, chunk_offsets_sec, boundaries_sec):
    """チャンクごとのセグメント列を全曲セグメント列へ統合する。

    各チャンクのセグメント(チャンクローカル時刻)を chunk_offsets_sec でグローバル時刻へ変換し、
    隣接チャンクとのオーバーラップ区間は境界(boundaries_sec)で切り詰めて重複を除く。境界を
    またぐ連続母音区間(type が "vowel" で同一 phoneme)だけを1区間へ結合する(同一チャンク内で
    元々隣接する同一母音の結合は本関数の対象外)。

    前提: len(chunk_offsets_sec) == len(chunk_segments)、
    len(boundaries_sec) == len(chunk_segments) - 1(チャンク数nに対し境界はn-1個)。
    呼び出し側(チャンク計画を作る側)がこの対応関係を保証する。
    """
    n = len(chunk_segments)
    boundary_set = set(boundaries_sec)
    global_segments = []
    for i in range(n):
        offset = chunk_offsets_sec[i]
        range_start = boundaries_sec[i - 1] if i > 0 else float("-inf")
        range_end = boundaries_sec[i] if i < n - 1 else float("inf")
        for s in chunk_segments[i]:
            g_start, g_end = s.start_sec + offset, s.end_sec + offset
            clipped_start, clipped_end = max(g_start, range_start), min(g_end, range_end)
            if clipped_end <= clipped_start:
                continue
            global_segments.append(Segment(
                type=s.type, start_sec=clipped_start, end_sec=clipped_end,
                phoneme=s.phoneme, confidence=s.confidence,
            ))

    merged = []
    for s in global_segments:
        at_boundary = s.start_sec in boundary_set
        if (merged and at_boundary and merged[-1].type == "vowel" and s.type == "vowel"
                and merged[-1].phoneme == s.phoneme and merged[-1].end_sec == s.start_sec):
            prev = merged[-1]
            merged[-1] = Segment(type=prev.type, start_sec=prev.start_sec, end_sec=s.end_sec,
                                  phoneme=prev.phoneme, confidence=None)
        else:
            merged.append(s)
    return merged
