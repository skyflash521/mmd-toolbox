import bisect
from dataclasses import dataclass

from vpr import Note, Part


@dataclass(frozen=True)
class _LoudnessController:
    value_min: int
    value_max: int
    priority: int


_LOUDNESS_CONTROLLERS = {
    "dynamics": _LoudnessController(value_min=0, value_max=127, priority=1),
}


def choose_loudness_controller(parts: list[Part]) -> str | None:
    present = {
        controller.name
        for part in parts
        for controller in part.controllers
        if controller.name in _LOUDNESS_CONTROLLERS and controller.events
    }
    if not present:
        return None
    return min(present, key=lambda name: _LOUDNESS_CONTROLLERS[name].priority)


def _merged_events(parts: list[Part], name: str) -> list[tuple[int, int]]:
    events = [
        (event.tick, event.value)
        for part in parts
        for controller in part.controllers
        if controller.name == name
        for event in controller.events
    ]
    events.sort(key=lambda event: event[0])
    return events


class _StepCurve:
    def __init__(self, events: list[tuple[int, int]], value_before_first: int) -> None:
        """events は tick 昇順で空でないこと。"""
        self._ticks = [tick for tick, _ in events]
        self._values = [value for _, value in events]
        self._value_before_first = value_before_first
        integral_to_point = [0]
        for i in range(len(events) - 1):
            integral_to_point.append(
                integral_to_point[i] + self._values[i] * (self._ticks[i + 1] - self._ticks[i])
            )
        self._integral_to_point = integral_to_point

    def _last_point_at_or_before(self, t: int) -> int:
        return bisect.bisect_right(self._ticks, t) - 1

    def value_at(self, t: int) -> int:
        i = self._last_point_at_or_before(t)
        return self._value_before_first if i < 0 else self._values[i]

    def _integral_from_first_point(self, t: int) -> int:
        i = self._last_point_at_or_before(t)
        if i < 0:
            return self._value_before_first * (t - self._ticks[0])
        return self._integral_to_point[i] + self._values[i] * (t - self._ticks[i])

    def average(self, start: int, end: int) -> float:
        if end <= start:
            return float(self.value_at(start))
        return (self._integral_from_first_point(end) - self._integral_from_first_point(start)) / (
            end - start
        )


def open_amounts_from_loudness(
    parts: list[Part],
    notes: list[Note],
    *,
    lo: float,
    hi: float,
    open_max: float,
    gamma: float,
) -> list[float] | None:
    """声量コントローラが無ければ None。あれば notes と同じ並びの開き量。"""
    name = choose_loudness_controller(parts)
    events = _merged_events(parts, name) if name is not None else []
    if not events:
        return None
    controller = _LOUDNESS_CONTROLLERS[name]
    value_min, value_max = controller.value_min, controller.value_max
    first_value = events[0][1]
    curve = _StepCurve(events, value_before_first=first_value)
    amounts: list[float] = []
    for note in notes:
        average = curve.average(note.start_tick, note.start_tick + note.duration_tick)
        n = (average - value_min) / (value_max - value_min) if value_max > value_min else 0.0
        n = min(max(n, 0.0), 1.0)
        value = lo + (hi - lo) * (n ** gamma)
        value = min(max(value, lo), hi)
        amounts.append(min(value, open_max))
    return amounts
