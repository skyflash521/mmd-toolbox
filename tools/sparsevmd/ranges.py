class RangeError(ValueError):
    pass


def parse_range(text):
    """戻り値は (start, end)。省略された側は None。"""
    if text.count(":") != 1:
        raise RangeError(f"範囲は START:END 形式: {text!r}")
    start_str, end_str = text.split(":")
    start = _parse_endpoint(start_str, text)
    end = _parse_endpoint(end_str, text)
    if start is None and end is None:
        raise RangeError(f"範囲は START または END の少なくとも一方が必要: {text!r}")
    if start is not None and end is not None and start > end:
        raise RangeError(f"範囲は START<=END: {text!r}")
    return (start, end)


def _parse_endpoint(part, text):
    if part == "":
        return None
    if not (part.isascii() and part.isdigit()):
        raise RangeError(f"範囲端は0以上の整数: {text!r}")
    return int(part)


def expand_and_normalize(parsed_ranges, global_min, global_max):
    """parsed_ranges は parse_range の戻り値のリスト。戻り値は昇順の (start, end) のリスト。"""
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
