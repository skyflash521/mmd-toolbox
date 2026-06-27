"""口形イベント確定(vpr2vmd.md §3)。

collect_notes が整列した音符列を単音前提の採用音符列へ非重複化し、解析範囲・休符の再導出、
音素→口形イベント写像と時間配分を経て lipsync の口形イベント列(MouthEvent)を作る。
"""

from dataclasses import replace

from vpr_io import Note


def resolve_overlaps(notes: list[Note]) -> list[Note]:
    """重なり音符を単音前提の採用音符列へ非重複化する(vpr2vmd.md §3)。

    入力は collect_notes が整列した音符列(start_tick 昇順・duration_tick 降順・パート出現順・
    索引昇順)。2段階で処理する:

    1. 同一 start_tick の重複除外: 整列順で同一 start は連続するので先頭(=最長、次いでパート
       出現順)だけ残す。これで残った音符は start_tick が相異なる。
    2. 後続開始への切り詰め: 残った音符の終端を次音符の開始へ切り詰め(end := min(end, 次の start))、
       長さ 0 以下になる音符は除外する。
    """
    deduped: list[Note] = []
    for note in notes:
        if deduped and deduped[-1].start_tick == note.start_tick:
            continue
        deduped.append(note)

    adopted: list[Note] = []
    for i, note in enumerate(deduped):
        end = note.start_tick + note.duration_tick
        if i + 1 < len(deduped):
            end = min(end, deduped[i + 1].start_tick)
        if end <= note.start_tick:
            continue
        adopted.append(replace(note, duration_tick=end - note.start_tick))
    return adopted
