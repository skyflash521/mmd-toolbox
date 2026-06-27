"""tick→秒→フレーム変換(vpr2vmd.md §3・§6)。

vpr は時刻を tick(整数、resolution=tick/四分音符)で、テンポを `TempoEvent{tick, bpm}` の列で
渡す。秒への変換はテンポマップの区分積分で行い、30fps の float フレームへ写す。整数フレームへの
量子化は lipsync 側が行うため、本モジュールは float フレームのまま返す。拍子は絶対時間変換に使わない。
"""

from vpr_io import TempoEvent


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
