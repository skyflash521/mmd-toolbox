"""口形イベント確定(vpr2vmd.md §3)。

collect_notes が整列した音符列を単音前提の採用音符列へ非重複化し、解析範囲・休符の再導出、
音素→口形イベント写像と時間配分を経て lipsync の口形イベント列(MouthEvent)を作る。
"""

from dataclasses import replace

from lipsync import MouthEvent, MouthShape
from vpr_io import Note, TempoEvent

from .mapping import note_mouth_events
from .timing import tick_to_frame


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


def build_mouth_events(
    adopted_notes: list[Note],
    tempos: list[TempoEvent],
    resolution: int,
    use_n_morph: bool = True,
) -> list[MouthEvent]:
    """採用音符列から lipsync の口形イベント列を組み立てる(vpr2vmd.md §3)。

    各採用音符を tick→フレーム変換し、note_mouth_events で文脈なしに定まる口形を得る。母音を持たず
    撥音/促音でもない音符(note_mouth_events が None)は、直前の確定口形を継続する(直前が無ければ無音)。
    音符間の隙間(休符=採用音符列の発音区間の補集合)は無音(SILENCE)で埋め、先頭〜最初の音符も無音に
    する。結果は時間順・隙間なく連続・非重複で、フレーム 0〜採用音符列の最後の終端までを被覆する
    (lipsync の入力契約)。採用音符列が空なら空列。
    """
    result: list[MouthEvent] = []
    prev_held: MouthShape | None = None  # 直前の確定口形(母音/撥音「ん」/閉口)。継続が引き継ぐ。
    cursor = 0.0
    for note in adopted_notes:
        start = tick_to_frame(note.start_tick, tempos, resolution)
        end = tick_to_frame(note.start_tick + note.duration_tick, tempos, resolution)
        if start > cursor:
            result.append(MouthEvent(MouthShape.SILENCE, cursor, start))  # 休符=無音
            prev_held = MouthShape.SILENCE  # 休符(閉口)が直前の確定口形になる
        note_events = note_mouth_events(note.phonemes, start, end, use_n_morph)
        if note_events is None:
            # 母音なし非撥音非促音 → 直前口形を継続(直前が無ければ閉口)。
            shape = prev_held if prev_held is not None else MouthShape.SILENCE
            result.append(MouthEvent(shape, start, end))
        else:
            result.extend(note_events)
            prev_held = note_events[-1].shape  # 直前口形を更新(母音/N/SILENCE)
        cursor = end
    return result
