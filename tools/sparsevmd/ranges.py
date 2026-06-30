"""削減範囲(--range)の解析・展開・積集合(sparsevmd.md §2.2 の RANGE)。

- parse_range: 1個の RANGE 文字列(START:END / START: / :END)を (start, end) に解析。
  省略側は None。両端は0以上の10進整数(両端含む)。両方指定で START>END はエラー。
  両端省略(:)は仕様外でエラー。
- expand_and_normalize: 省略端を対象トラック全体の最小/最大に1回だけ展開し、
  昇順に正規化、端の接触を含む重複でエラー、展開後 START>END でエラー。
- intersect: グローバル範囲と当該トラックの先頭/末尾フレームの積集合。空なら空リスト。
"""


class RangeError(ValueError):
    """範囲指定のエラー(§2.2、CLI で終了コード2)。"""


def parse_range(text):
    """1個の RANGE 文字列を (start, end) に解析する。省略側は None。"""
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
    """範囲端を解析する。空文字は None、それ以外は0以上の10進整数。"""
    if part == "":
        return None
    if not (part.isascii() and part.isdigit()):
        raise RangeError(f"範囲端は0以上の整数: {text!r}")
    return int(part)


def expand_and_normalize(parsed_ranges, global_min, global_max):
    """省略端を展開し、昇順正規化と重複検査を行う(§2.2)。

    parsed_ranges は parse_range の戻り値((start|None, end|None))のリスト。
    省略端は global_min / global_max に1回だけ展開する。
    """
    expanded = []
    for start, end in parsed_ranges:
        s = global_min if start is None else start
        e = global_max if end is None else end
        if s > e:
            raise RangeError(f"展開後の範囲が START>END: ({s}, {e})")
        expanded.append((s, e))

    expanded.sort()
    for prev, cur in zip(expanded, expanded[1:]):
        # 端の接触(cur.start == prev.end)も1フレーム重複としてエラー(§2.2)。
        if cur[0] <= prev[1]:
            raise RangeError(f"範囲が重複しています: {prev} と {cur}")
    return expanded


def intersect(global_ranges, first, last):
    """グローバル範囲(正規化済み)とトラック範囲 [first, last] の積集合(§2.2)。"""
    result = []
    for s, e in global_ranges:
        lo = max(s, first)
        hi = min(e, last)
        if lo <= hi:
            result.append((lo, hi))
    return result
