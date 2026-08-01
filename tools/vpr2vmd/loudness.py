"""声量(ダイナミクス)コントローラ→開き量の写像(入口処理)。

vpr が公開する連続コントローラ曲線のうち声量に相当するもの(`dynamics`)を選び、
各採用音符の発音区間で階段平均し、コントローラ値域で 0〜1 へ正規化して開き量へ写す。声量コントローラが
無ければ `None` を返し、呼び出し側が velocity 由来の写像(openness)へフォールバックする。

声量コントローラ名・値域は、実ファイルの観測で中立値・値域が確定した名前だけを採用し、
実装者が独自に名前を増やさない(未知名は採用しない)。曲線が音符区間を被覆しない箇所は、その曲線が
持つ最近の値(始端より前は始端値、終端より後は終端値)を保持する。
"""

import bisect

from vpr import Part

# 声量コントローラのインベントリ(優先順・値域。中立値・被覆の扱いが確定している名前のみ採用する)。
# 値域はコントローラ生値の最小〜最大で、正規化はこの範囲で行う(視覚基準は lo/hi/gamma 側で吸収)。
# 注: `s5Expression`(VOCALOID5系/互換ボイスバンク由来)は声量に使えると観測された名前だが、中立値が断定できず
# 制御点が部分区間にしか無いことがあり被覆外の解釈が定まらないため採用しない。これだけを持つ vpr は声量曲線
# 無しとして velocity 由来の写像へ回る。
_LOUDNESS_CONTROLLERS = {
    "dynamics": {"min": 0, "max": 127, "priority": 1},
}


def choose_loudness_controller(parts: list[Part]) -> str | None:
    """パート群から声量コントローラ名を1つ選ぶ。

    採用インベントリのうち1点以上のイベントを持つものを集め、最高優先の名前を返す。声量曲線が
    無ければ `None`。複数パートが別の声量名を使う場合も最高優先の1つへ統一する(平均・合成はしない)。
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


class _StepCurve:
    """階段状(次の制御点まで値を保持)のコントローラ曲線。区間平均を累積積分の差分で求める。

    音符ごとに曲線を走り直すと採用音符数×制御点数に比例した走査になり、制御点の密な曲線で効く。
    制御点までの積分を一度だけ作れば、各音符は二分探索と引き算で済む。tick も値も整数なので積分も
    整数で厳密に持て、走査で足し込む場合と同じ値になる。
    """

    def __init__(self, events: list[tuple[int, int]], default: int) -> None:
        """events は tick 昇順の (tick, 値) 列(空でないこと)。default は曲線の始端より前の値。

        同一 tick に複数の点があるときは、後に現れた点の値を採る(events の並び順が決める)。
        """
        self._ticks = [tick for tick, _ in events]
        self._values = [value for _, value in events]
        self._default = default
        integral = [0]
        for i in range(len(events) - 1):
            integral.append(integral[i] + self._values[i] * (self._ticks[i + 1] - self._ticks[i]))
        self._integral = integral  # 始端 ticks[0] から各制御点までの積分

    def _index_at(self, t: int) -> int:
        """時刻 t 以前(t ちょうどを含む)の最後の制御点の索引。無ければ -1。"""
        return bisect.bisect_right(self._ticks, t) - 1

    def value_at(self, t: int) -> int:
        """時刻 t での値。t 以前に制御点が無ければ default。"""
        i = self._index_at(t)
        return self._default if i < 0 else self._values[i]

    def _integral_to(self, t: int) -> int:
        """始端 ticks[0] から t までの積分(t が始端より前なら負になる)。"""
        i = self._index_at(t)
        if i < 0:
            return self._default * (t - self._ticks[0])
        return self._integral[i] + self._values[i] * (t - self._ticks[i])

    def average(self, start: int, end: int) -> float:
        """区間 [start, end) の時間平均。end<=start は start 時点の値。"""
        if end <= start:
            return float(self.value_at(start))
        return (self._integral_to(end) - self._integral_to(start)) / (end - start)


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
    events = _merged_events(parts, name) if name is not None else []
    if not events:
        return None
    spec = _LOUDNESS_CONTROLLERS[name]
    value_min, value_max = spec["min"], spec["max"]
    curve = _StepCurve(events, default=events[0][1])  # 曲線始端より前は始端値を保持
    amounts: list[float] = []
    for note in notes:
        average = curve.average(note.start_tick, note.start_tick + note.duration_tick)
        n = (average - value_min) / (value_max - value_min) if value_max > value_min else 0.0
        n = min(max(n, 0.0), 1.0)
        value = lo + (hi - lo) * (n ** gamma)
        value = min(max(value, lo), hi)
        amounts.append(min(value, open_max))
    return amounts
