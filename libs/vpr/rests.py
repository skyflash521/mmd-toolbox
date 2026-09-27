from .types import Note


def rest_intervals(notes: list[Note], end_tick: int) -> list[tuple[int, int]]:
    """戻り値は [0, end_tick) 内の休符区間 (開始 tick, 終了 tick) を開始の昇順に並べたもの。"""
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
