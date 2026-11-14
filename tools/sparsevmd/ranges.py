class RangeError(ValueError):
    pass


def expand_and_normalize(parsed_ranges, global_min, global_max):
    """parsed_ranges は (start, end) のリスト(省略された側は None)。戻り値は昇順の (start, end) のリスト。"""
    expanded = []
    for start, end in parsed_ranges:
        s = global_min if start is None else start
        e = global_max if end is None else end
        if s > e:
            raise RangeError(f"展開後の範囲が START>END: ({s}, {e})")
        expanded.append((s, e))

    expanded.sort()
    for prev, cur in zip(expanded, expanded[1:], strict=False):
        if cur[0] <= prev[1]:
            raise RangeError(f"範囲が重複しています: {prev} と {cur}")
    return expanded


def intersect(global_ranges, first, last):
    result = []
    for s, e in global_ranges:
        lo = max(s, first)
        hi = min(e, last)
        if lo <= hi:
            result.append((lo, hi))
    return result
