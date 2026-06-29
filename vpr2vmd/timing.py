"""tick→秒→フレーム変換(vpr2vmd.md §3・§6)。

vpr は時刻を tick(整数、resolution=tick/四分音符)で、テンポを `TempoEvent{tick, bpm}` の列で
渡す。秒への変換はテンポマップの区分積分で行い、30fps の float フレームへ写す。整数フレームへの
量子化は lipsync 側が行うため、本モジュールは float フレームのまま返す。拍子は絶対時間変換に使わない。
"""

from vpr_io import Note, TempoEvent


def _seconds_per_tick(bpm: float, resolution: int) -> float:
    """1 tick の秒数。bpm 一定の区間で `60 / (bpm × resolution)`。"""
    return 60.0 / (bpm * resolution)


def tick_to_seconds(tick: int, tempos: list[TempoEvent], resolution: int) -> float:
    """tick の絶対秒をテンポマップの区分積分で求める。

    各テンポ区間 `[tick_i, tick_{i+1})` は `bpm_i` 一定とする。最初の `TempoEvent.tick` が 0 で
    なくても、その bpm を tick 0 まで遡って適用する(先頭区間は 0 始まり)。区間は半開なので、
    テンポ境界 tick ちょうどの累積秒は先行区間までの積分で、境界以降は新区間の bpm が効く。
    """
    events = sorted(tempos, key=lambda e: e.tick)
    seconds = 0.0
    cursor = 0  # 先頭区間は tick 0 から(最初のテンポを遡及適用)
    for i, event in enumerate(events):
        spt = _seconds_per_tick(event.bpm, resolution)
        next_tick = events[i + 1].tick if i + 1 < len(events) else None
        if next_tick is None or tick <= next_tick:
            return seconds + (tick - cursor) * spt
        seconds += (next_tick - cursor) * spt
        cursor = next_tick
    return seconds


def tick_to_frame(tick: int, tempos: list[TempoEvent], resolution: int) -> float:
    """tick を 30fps の float フレームへ変換する(秒 × 30、量子化しない)。"""
    return tick_to_seconds(tick, tempos, resolution) * 30.0


def note_effective_bpm(
    note: Note, tempos: list[TempoEvent], resolution: int
) -> float | None:
    """採用音符1つの有効BPM(その発音長を一定テンポで表したときのBPM)。

    可変テンポで音符がテンポ境界をまたいでも、`tick_to_seconds` の区分積分で得た実発音秒へ
    畳んでから、拍数(`duration_tick / resolution`)と実秒から有効BPM=`60 × 拍数 / 秒`を求める
    (区間分割せず音符全体を1サンプルとする)。発音長・実秒が 0 以下(切り詰めで消えるなど)なら None。
    """
    if note.duration_tick <= 0:
        return None
    start = note.start_tick
    end = note.start_tick + note.duration_tick
    seconds = tick_to_seconds(end, tempos, resolution) - tick_to_seconds(
        start, tempos, resolution
    )
    if seconds <= 0.0:
        return None
    beats = note.duration_tick / resolution
    return 60.0 * beats / seconds


def representative_bpm(
    adopted_notes: list[Note],
    tempos: list[TempoEvent],
    resolution: int,
    *,
    default_bpm: float = 120.0,
) -> float:
    """採用音符列の代表BPM(発音秒で重み付けした有効BPMの重み付き中央値)。

    各音符の有効BPM(`note_effective_bpm`)を、その発音秒に比例する重み(発音秒×30=見た目の
    フレーム長)で重み付けし、重み付き中央値を取る。長く発音される音符ほど代表BPMへ強く効く。
    有効BPMが得られない音符(発音長・実秒 0)は除外する。全除外(発音なし等)なら `default_bpm`。
    重み付き中央値は、BPM昇順で累積重みが総重みの半分以上に最初に達するBPMとする(同点は小さい側)。
    """
    samples: list[tuple[float, float]] = []  # (有効BPM, 重み)
    for note in adopted_notes:
        bpm = note_effective_bpm(note, tempos, resolution)
        if bpm is None or bpm <= 0.0:
            continue
        seconds = tick_to_seconds(
            note.start_tick + note.duration_tick, tempos, resolution
        ) - tick_to_seconds(note.start_tick, tempos, resolution)
        weight = seconds * 30.0
        if weight <= 0.0:
            continue
        samples.append((bpm, weight))
    if not samples:
        return default_bpm
    samples.sort(key=lambda s: s[0])
    half = sum(weight for _, weight in samples) / 2.0
    cumulative = 0.0
    for bpm, weight in samples:
        cumulative += weight
        if cumulative >= half:
            return bpm
    return samples[-1][0]
