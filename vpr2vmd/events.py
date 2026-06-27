"""口形イベント確定(vpr2vmd.md §3)。

collect_notes が整列した音符列を単音前提の採用音符列へ非重複化し、解析範囲・休符の再導出、
音素→口形イベント写像と時間配分を経て lipsync の口形イベント列(MouthEvent)を作る。
"""

from dataclasses import replace

from lipsync import MouthEvent, MouthShape
from vpr_io import Note, TempoEvent

from .mapping import note_mouth_events
from .timing import tick_to_frame

# 母音的口形(合成プロファイルを持ち開き量で保持値が決まる)。開き量はこれらのイベントのみ有意で、
# 両唇閉鎖・無音は閉口なので開き量を持たない(0)([lipsync 仕様](../lipsync/lipsync.md) §2.2)。
_VOWEL_LIKE = frozenset(
    {MouthShape.A, MouthShape.I, MouthShape.U, MouthShape.E, MouthShape.O, MouthShape.N}
)


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
    open_by_note: list[float] | None = None,
) -> list[MouthEvent]:
    """採用音符列から lipsync の口形イベント列を組み立てる(vpr2vmd.md §3)。

    各採用音符を tick→フレーム変換し、note_mouth_events で文脈なしに定まる口形を得る。母音を持たず
    撥音/促音でもない音符(note_mouth_events が None)は、直前の確定口形を継続する(直前が無ければ無音)。
    音符間の隙間(休符=採用音符列の発音区間の補集合)は無音(SILENCE)で埋め、先頭〜最初の音符も無音に
    する。結果は時間順・隙間なく連続・非重複で、フレーム 0〜採用音符列の最後の終端までを被覆する
    (lipsync の入力契約)。採用音符列が空なら空列。

    `open_by_note`(採用音符に整列した開き量列。`adopted_notes` と同長)を渡すと、その音符が生む
    母音的口形イベント(母音・撥音「ん」)へ該当音符の開き量を刻印する。両唇閉鎖・無音(休符・促音・
    閉口継続)は閉口なので開き量を持たない(0)。継続(母音なし)の保持イベントも、保持口形が母音的なら
    その継続音符自身の開き量を刻印する。`open_by_note` が None なら全イベントの開き量は 0。
    """
    result: list[MouthEvent] = []
    prev_held: MouthShape | None = None  # 直前の確定口形(母音/撥音「ん」/閉口)。継続が引き継ぐ。
    cursor = 0.0
    for i, note in enumerate(adopted_notes):
        note_open = open_by_note[i] if open_by_note is not None else 0.0
        start = tick_to_frame(note.start_tick, tempos, resolution)
        end = tick_to_frame(note.start_tick + note.duration_tick, tempos, resolution)
        if start > cursor:
            result.append(MouthEvent(MouthShape.SILENCE, cursor, start))  # 休符=無音(開き量 0)
            prev_held = MouthShape.SILENCE  # 休符(閉口)が直前の確定口形になる
        note_events = note_mouth_events(note.phonemes, start, end, use_n_morph)
        if note_events is None:
            # 母音なし非撥音非促音 → 直前口形を継続(直前が無ければ閉口)。
            shape = prev_held if prev_held is not None else MouthShape.SILENCE
            open_amount = note_open if shape in _VOWEL_LIKE else 0.0
            result.append(MouthEvent(shape, start, end, open_amount))
        else:
            for event in note_events:
                open_amount = note_open if event.shape in _VOWEL_LIKE else 0.0
                result.append(replace(event, open_amount=open_amount))
            prev_held = note_events[-1].shape  # 直前口形を更新(母音/N/SILENCE)
        cursor = end
    return result
