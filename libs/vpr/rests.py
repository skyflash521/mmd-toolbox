"""休符導出(vpr.md §2.1)。

休符は同一トラック内の発音区間 [start_tick, start_tick + duration_tick) の和集合の補集合として
観測する。隣接差分でなく和集合の補集合とすることで、区間が重なっても偽の休符を作らない。
"""

from .types import Note


def rest_intervals(notes: list[Note], end_tick: int) -> list[tuple[int, int]]:
    """[0, end_tick) のうち発音区間の和集合の補集合(休符)を、(開始, 終了)の昇順で返す。"""
    sounds = []
    for note in notes:
        start = max(note.start_tick, 0)
        end = min(note.start_tick + note.duration_tick, end_tick)
        if start < end:
            sounds.append((start, end))
    sounds.sort()

    rests: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sounds:
        if start > cursor:
            rests.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < end_tick:
        rests.append((cursor, end_tick))
    return rests
