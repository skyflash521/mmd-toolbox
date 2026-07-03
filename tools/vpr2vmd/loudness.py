"""声量(ダイナミクス)コントローラ→開き量の写像(vpr2vmd.md §3 入口処理)。

vpr が公開する連続コントローラ曲線のうち声量に相当するもの(`dynamics`/`s5Expression`)を選び、
各採用音符の発音区間で階段平均し、コントローラ値域で 0〜1 へ正規化して開き量へ写す。声量コントローラが
無ければ `None` を返し、呼び出し側が velocity 由来の写像(openness)へフォールバックする。

声量コントローラ名・値域は形式仕様(docs/specs/vpr/VPR_file_format.md)で確定したインベントリに従い、
実装者が独自に名前を増やさない(未知名は採用しない)。曲線が音符区間を被覆しない箇所は、その曲線が
持つ最近の値(始端より前は始端値、終端より後は終端値)を保持する。
"""

from vpr import Part

# 声量コントローラのインベントリ(優先順・値域。中立値・被覆の扱いが確定している名前のみ採用する)。
# 値域はコントローラ生値の最小〜最大で、正規化はこの範囲で行う(視覚基準は lo/hi/gamma 側で吸収)。
# 注: `s5Expression`(VOCALOID5系/互換ボイスバンク由来)は声量に使える実測名だが、中立値が断定できず
# 自動化が部分区間だけのことがあり被覆外の解釈が定まらないため、初期実装では採用しない(将来、形式仕様で
# 中立値・被覆の扱いを確定してから追加する)。未採用名の存在は診断で報告する余地を残す。
_LOUDNESS_CONTROLLERS = {
    "dynamics": {"min": 0, "max": 127, "priority": 1},
}


def choose_loudness_controller(parts: list[Part]) -> str | None:
    """パート群から声量コントローラ名を1つ選ぶ(優先順 `dynamics` > `s5Expression`)。

    確定名のうち1点以上のイベントを持つものを集め、最高優先の名前を返す。声量曲線が無ければ `None`。
    複数パートが別の声量名を使う場合も最高優先の1名へ統一する(平均・合成はしない)。
    """
    present = {
        controller.name
        for part in parts
        for controller in part.controllers
        if controller.name in _LOUDNESS_CONTROLLERS and controller.events
    }
    if not present:
        return None
    return min(present, key=lambda name: _LOUDNESS_CONTROLLERS[name]["priority"])


def _merged_events(parts: list[Part], name: str) -> list[tuple[int, int]]:
    """指定名の声量曲線を全パートから集めて (tick, value) の tick 昇順列にする。"""
    events = [
        (event.tick, event.value)
        for part in parts
        for controller in part.controllers
        if controller.name == name
        for event in controller.events
    ]
    # tick のみをキーに安定ソートする(同一 tick は入力順=収集順を保持し、値の大小で後勝ちにしない)。
    events.sort(key=lambda event: event[0])
    return events


def _value_at(events: list[tuple[int, int]], t: int, default: int) -> int:
    """時刻 t 以前の最後の値(階段状曲線の t での値)。t より前に点が無ければ default。"""
    result = default
    for tick, value in events:
        if tick <= t:
            result = value
        else:
            break
    return result


def _step_average(events: list[tuple[int, int]], start: int, end: int, default: int) -> float:
    """階段状(次イベントまで値を保持)曲線を区間 [start, end) で時間平均する。end<=start は始端値。"""
    if end <= start:
        return float(_value_at(events, start, default))
    value = _value_at(events, start, default)
    cursor = start
    acc = 0.0
    for tick, next_value in events:
        if tick <= start:
            continue
        if tick >= end:
            break
        acc += value * (tick - cursor)
        value = next_value
        cursor = tick
    acc += value * (end - cursor)
    return acc / (end - start)


def open_amounts_from_loudness(
    parts: list[Part],
    notes,
    *,
    lo: float,
    hi: float,
    open_max: float,
    gamma: float,
) -> list[float] | None:
    """声量コントローラがあれば各採用音符の開き量列を返す。無ければ `None`(velocity へフォールバック)。

    各音符の発音区間 [start, start+duration) で声量曲線を階段平均し、コントローラ値域で 0〜1 へ正規化、
    velocity と同じ写像 `lo + (hi - lo) * n**gamma`(レンジ [lo,hi] クランプ・`open_max` 上限)で開き量にする。
    `notes` は採用音符列(`start_tick`/`duration_tick` を持つ)。返り値は `notes` と同長。
    """
    name = choose_loudness_controller(parts)
    if name is None:
        return None
    events = _merged_events(parts, name)
    if not events:
        return None
    spec = _LOUDNESS_CONTROLLERS[name]
    value_min, value_max = spec["min"], spec["max"]
    default_value = events[0][1]  # 曲線始端より前は始端値を保持
    amounts: list[float] = []
    for note in notes:
        average = _step_average(
            events, note.start_tick, note.start_tick + note.duration_tick, default_value
        )
        n = (average - value_min) / (value_max - value_min) if value_max > value_min else 0.0
        n = min(max(n, 0.0), 1.0)
        value = lo + (hi - lo) * (n ** gamma)
        value = min(max(value, lo), hi)
        amounts.append(min(value, open_max))
    return amounts
