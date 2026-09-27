from .types import Segment


def find_chunk_boundaries(duration_sec, rms_times_sec, rms_values, *, max_duration_sec,
                           search_window_sec=5.0, silence_threshold=0.06):
    """戻り値は (境界の秒, 無音が見つからず目標境界で強制分割したか) の列。"""
    if max_duration_sec <= 0 or duration_sec <= max_duration_sec:
        return []

    boundaries = []
    previous_boundary = 0.0
    target = max_duration_sec
    while target < duration_sec:
        lo, hi = target - search_window_sec, target + search_window_sec
        candidates = [
            (t, v) for t, v in zip(rms_times_sec, rms_values, strict=True) if lo <= t <= hi and t > previous_boundary
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
    """chunk_offsets_sec はチャンクと同数、boundaries_sec はチャンク数より1つ少なく渡す。"""
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
