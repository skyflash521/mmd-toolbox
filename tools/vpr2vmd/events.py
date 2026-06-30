"""口形イベント確定(vpr2vmd.md §3)。

collect_notes が整列した音符列を単音前提の採用音符列へ非重複化し、解析範囲・休符の再導出、
音素→口形イベント写像と時間配分を経て lipsync の口形イベント列(MouthEvent)を作る。
"""

from collections import Counter
from dataclasses import dataclass, replace

from lipsync import MouthEvent, MouthShape
from vpr import Note, TempoEvent

from .mapping import note_mouth_events
from .phonemes import PhonemeCategory, categorize
from .timing import tick_to_frame

# 母音的口形(合成プロファイルを持ち開き量で保持値が決まる)。開き量はこれらのイベントのみ有意で、
# 両唇閉鎖・無音は閉口なので開き量を持たない(0)([lipsync 仕様](../lipsync/lipsync.md) §2.2)。
_VOWEL_LIKE = frozenset(
    {MouthShape.A, MouthShape.I, MouthShape.U, MouthShape.E, MouthShape.O, MouthShape.N}
)

# レガート間隙と判定する間隙長の上限(フレーム)。初期値は 8分音符相当の目安で、視覚で詰める
# (vpr2vmd.md §3)。実テンポへの適応は別途プリセット/テンポ補正が担う。
_LEGATO_MAX_FRAMES = 8.0


def _classify_gap(
    left_shape: MouthShape | None,
    right_shape: MouthShape,
    gap_len: float,
    legato_max_frames: float,
) -> MouthShape:
    """母音間の短い間隙をレガート間隙(LEGATO_GAP)/休符(SILENCE)へ分類する(vpr2vmd.md §3)。

    前後の実効口形がともに母音的(母音・撥音「ん」)で、間隙が `legato_max_frames` 以下のときだけ
    `LEGATO_GAP`(谷で繋ぐ)。非母音的な隣接(両唇閉鎖・促音閉口・直前が閉口の継続など)・長い間隙・
    曲頭(直前口形なし `left_shape is None`)は `SILENCE`(完全閉口)。判定は確定済みの口形だけに依り、
    `lipsync` 側はこの分類結果を入力として受ける([lipsync 仕様](../lipsync/lipsync.md) §6・§4.12)。
    """
    if left_shape not in _VOWEL_LIKE or right_shape not in _VOWEL_LIKE:
        return MouthShape.SILENCE
    if gap_len > legato_max_frames:
        return MouthShape.SILENCE
    return MouthShape.LEGATO_GAP


@dataclass(frozen=True)
class OverlapDiagnostics:
    """重なり解決で生じた除外・切り詰めの件数(vpr2vmd.md §4.4)。"""

    excluded: int  # 同一 start の重複・切り詰めで長さ0以下になり除外した音符数
    truncated: int  # 後続開始へ終端を切り詰めて採用した音符数


@dataclass(frozen=True)
class EventDiagnostics:
    """口形イベント確定の診断(vpr2vmd.md §4.4)。"""

    vowel_undetermined: int  # 母音が得られず直前口形を継続した音符数
    non_event_symbols: dict[str, int]  # 自前イベントを作らない記号(その他子音・未知)→件数


def resolve_overlaps(notes: list[Note]) -> tuple[list[Note], OverlapDiagnostics]:
    """重なり音符を単音前提の採用音符列へ非重複化し、採用列と診断を返す(vpr2vmd.md §3・§4.4)。

    入力は collect_notes が整列した音符列(start_tick 昇順・duration_tick 降順・パート出現順・
    索引昇順)。2段階で処理する:

    1. 同一 start_tick の重複除外: 整列順で同一 start は連続するので先頭(=最長、次いでパート
       出現順)だけ残す。これで残った音符は start_tick が相異なる。
    2. 後続開始への切り詰め: 残った音符の終端を次音符の開始へ切り詰め(end := min(end, 次の start))、
       長さ 0 以下になる音符は除外する。

    診断(`OverlapDiagnostics`)は、同一 start の重複除外と切り詰めで長さ0以下になり除外した件数を
    `excluded`、後続開始へ切り詰めて採用した件数を `truncated` として返す。
    """
    deduped: list[Note] = []
    excluded = 0
    for note in notes:
        if deduped and deduped[-1].start_tick == note.start_tick:
            excluded += 1  # 同一 start の重複除外
            continue
        deduped.append(note)

    adopted: list[Note] = []
    truncated = 0
    for i, note in enumerate(deduped):
        end = note.start_tick + note.duration_tick
        original_end = end
        if i + 1 < len(deduped):
            end = min(end, deduped[i + 1].start_tick)
        if end <= note.start_tick:
            excluded += 1  # 切り詰めで長さ0以下 → 除外
            continue
        if end < original_end:
            truncated += 1  # 後続開始へ切り詰めて採用
        adopted.append(replace(note, duration_tick=end - note.start_tick))
    return adopted, OverlapDiagnostics(excluded, truncated)


def build_mouth_events(
    adopted_notes: list[Note],
    tempos: list[TempoEvent],
    resolution: int,
    use_n_morph: bool = True,
    open_by_note: list[float] | None = None,
    legato_max_frames: float = _LEGATO_MAX_FRAMES,
) -> tuple[list[MouthEvent], EventDiagnostics]:
    """採用音符列から lipsync の口形イベント列を組み立てる(vpr2vmd.md §3)。

    各採用音符を tick→フレーム変換し、note_mouth_events で文脈なしに定まる口形を得る。母音を持たず
    撥音/促音でもない音符(note_mouth_events が None)は、直前の確定口形を継続する(直前が無ければ無音)。
    音符間の隙間(採用音符列の発音区間の補集合)は、前後の実効口形と間隙長から `_classify_gap` で
    レガート間隙(LEGATO_GAP、谷で繋ぐ)か休符(SILENCE、完全閉口)へ分類して埋める。先頭〜最初の音符は
    直前口形が無いので無音にする。結果は時間順・隙間なく連続・非重複で、フレーム 0〜採用音符列の最後の
    終端までを被覆する(lipsync の入力契約)。採用音符列が空なら空列。
    `legato_max_frames` はレガート間隙と判定する間隙長の上限(視覚で詰める)。

    `open_by_note`(採用音符に整列した開き量列。`adopted_notes` と同長)を渡すと、その音符が生む
    母音的口形イベント(母音・撥音「ん」)へ該当音符の開き量を刻印する。両唇閉鎖・無音(休符・促音・
    閉口継続)は閉口なので開き量を持たない(0)。継続(母音なし)の保持イベントも、保持口形が母音的なら
    その継続音符自身の開き量を刻印する。`open_by_note` が None なら全イベントの開き量は 0。

    診断(`EventDiagnostics`)として、母音が得られず直前口形を継続した音符数(`vowel_undetermined`)と、
    自前イベントを作らない記号(その他子音・未知記号 = `PhonemeCategory.OTHER`)の記号→件数
    (`non_event_symbols`)を併せて返す(vpr2vmd.md §4.4)。
    """
    result: list[MouthEvent] = []
    prev_held: MouthShape | None = None  # 直前の確定口形(母音/撥音「ん」/閉口)。継続が引き継ぐ。
    cursor = 0.0
    vowel_undetermined = 0
    non_event_symbols: Counter[str] = Counter()
    for i, note in enumerate(adopted_notes):
        for phoneme in note.phonemes:
            if categorize(phoneme) is PhonemeCategory.OTHER:
                non_event_symbols[phoneme] += 1  # 自前イベントを作らない記号(診断)
        note_open = open_by_note[i] if open_by_note is not None else 0.0
        start = tick_to_frame(note.start_tick, tempos, resolution)
        end = tick_to_frame(note.start_tick + note.duration_tick, tempos, resolution)
        note_events = note_mouth_events(note.phonemes, start, end, use_n_morph)
        if start > cursor:
            # 間隙を分類する。次音符の実効STARTING口形は、継続(None)なら直前の確定口形へ解決した口形。
            right_shape = (
                note_events[0].shape
                if note_events is not None
                else (prev_held if prev_held is not None else MouthShape.SILENCE)
            )
            gap_shape = _classify_gap(prev_held, right_shape, start - cursor, legato_max_frames)
            result.append(MouthEvent(gap_shape, cursor, start))  # 非発音区間(開き量 0)
            if gap_shape is MouthShape.SILENCE:
                prev_held = MouthShape.SILENCE  # 休符(閉口)が直前の確定口形になる
            # LEGATO_GAP は谷で繋ぐ非発音区間。直前の母音的口形を保ち、継続解決へ引き継ぐ。
        if note_events is None:
            # 母音なし非撥音非促音 → 直前口形を継続(直前が無ければ閉口)。
            vowel_undetermined += 1
            shape = prev_held if prev_held is not None else MouthShape.SILENCE
            open_amount = note_open if shape in _VOWEL_LIKE else 0.0
            result.append(MouthEvent(shape, start, end, open_amount))
        else:
            for event in note_events:
                open_amount = note_open if event.shape in _VOWEL_LIKE else 0.0
                result.append(replace(event, open_amount=open_amount))
            prev_held = note_events[-1].shape  # 直前口形を更新(母音/N/SILENCE)
        cursor = end
    return result, EventDiagnostics(vowel_undetermined, dict(non_event_symbols))
