from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

from vmd import MorphKey

from .types import ApertureClass, ConsonantClass, GenerationParams, MouthEvent, MouthShape

_MAIN_MORPH = {
    MouthShape.A: "あ",
    MouthShape.I: "い",
    MouthShape.U: "う",
    MouthShape.E: "え",
    MouthShape.O: "お",
    MouthShape.N: "ん",
}

_PURE_VOWEL_PROFILES: dict[MouthShape, dict[str, float]] = {
    MouthShape.A: {"あ": 1.0},
    MouthShape.I: {"い": 1.0},
    MouthShape.U: {"う": 1.0},
    MouthShape.E: {"え": 1.0},
    MouthShape.O: {"お": 1.0},
    MouthShape.N: {"ん": 1.0},
}

_ROUNDED_U_GAIN = 0.3
_ROUNDED_O_GAIN = 0.0
_SPREAD_I_GAIN = 0.3

_APERTURE_SCALE: dict[ApertureClass, float] = {
    ApertureClass.NONE: 1.0,
    ApertureClass.FIRM_CLOSURE: 0.75,
    ApertureClass.NARROW_CHANNEL: 0.85,
    ApertureClass.SLIGHT_CLOSURE: 0.92,
}

_VOWEL_SCALE_INDEX = {
    MouthShape.A: 0,
    MouthShape.I: 1,
    MouthShape.U: 2,
    MouthShape.E: 3,
    MouthShape.O: 4,
    MouthShape.N: 5,
}

_STANDARD_MORPHS = ("あ", "い", "う", "え", "お", "ん")

_MORPH_NAME_ENCODING = "cp932"
_MORPH_NAME_BYTES = 15

_MIN_TRIANGLE_SLACK = 2
_MIN_MORA_VALLEY_HALF_WIDTH = 2
_VIBRATO_SUPPRESSION_EXTRA_FRAMES = 2


class _Target(NamedTuple):
    morph: str
    frame: float
    weight: float


class _ValleyWeights(NamedTuple):
    left_shoulder: float
    center: float
    right_shoulder: float


def _consonant_profile(shape: MouthShape, consonant_class: ConsonantClass) -> dict[str, float]:
    profile = dict(_PURE_VOWEL_PROFILES[shape])
    main = _MAIN_MORPH[shape]
    if consonant_class is ConsonantClass.ROUNDED:
        aux = {"う": _ROUNDED_U_GAIN, "お": _ROUNDED_O_GAIN}
    elif consonant_class is ConsonantClass.SPREAD:
        aux = {"い": _SPREAD_I_GAIN}
    else:
        aux = {}
    for morph, gain in aux.items():
        if gain > 0.0 and morph != main:
            profile[morph] = profile.get(morph, 0.0) + gain
    return profile


def _morph_key(name: str, frame: int, weight: float) -> MorphKey:
    name_raw = name.encode(_MORPH_NAME_ENCODING).ljust(_MORPH_NAME_BYTES, b"\x00")
    return MorphKey(name_raw, frame, weight)


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _effective_profile(
    shape: MouthShape, consonant_class: ConsonantClass, params: GenerationParams
) -> dict[str, float]:
    main = _MAIN_MORPH[shape]
    return {
        morph: weight if morph == main else weight * params.exaggeration
        for morph, weight in _consonant_profile(shape, consonant_class).items()
    }


def _hold_value(shape: MouthShape, open_amount: float, params: GenerationParams) -> float:
    hold = open_amount * params.vowel_scale[_VOWEL_SCALE_INDEX[shape]]
    return min(max(hold, 0.0), params.open_cap)


def _weights_from_hold(
    shape: MouthShape, consonant_class: ConsonantClass, hold: float, params: GenerationParams
) -> dict[str, float]:
    weights = {
        morph: weight * hold
        for morph, weight in _effective_profile(shape, consonant_class, params).items()
    }
    total = sum(weights.values())
    if total > params.open_cap:
        factor = params.open_cap / total
        weights = {morph: weight * factor for morph, weight in weights.items()}
    return weights


def _final_weights(event: MouthEvent, params: GenerationParams) -> dict[str, float]:
    weights = _weights_from_hold(
        event.shape,
        event.consonant_class,
        _hold_value(event.shape, event.open_amount, params),
        params,
    )
    scale = _APERTURE_SCALE[event.aperture_class]
    return {morph: weight * scale for morph, weight in weights.items()}


def _shape_diff(
    shape_a: MouthShape,
    class_a: ConsonantClass,
    shape_b: MouthShape,
    class_b: ConsonantClass,
    params: GenerationParams,
) -> float:
    def _unit(shape: MouthShape, consonant_class: ConsonantClass) -> list[float]:
        eff = _effective_profile(shape, consonant_class, params)
        vec = [eff.get(morph, 0.0) for morph in _STANDARD_MORPHS]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0.0 else vec

    ua, ub = _unit(shape_a, class_a), _unit(shape_b, class_b)
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(ua, ub, strict=True)))
    return dist / math.sqrt(2.0)


def _transition_frames(diff: float, shorter_len: float, params: GenerationParams) -> int:
    cap = shorter_len / 2.0
    return _round_half_up(max(1.0, min(float(params.coartic_overlap_max), cap)))


def _legato_valley_depth(gap_len: float, params: GenerationParams) -> float:
    d = params.legato_valley_shallow - params.legato_valley_slope * gap_len
    return min(max(d, params.legato_valley_deep), params.legato_valley_shallow)


def _is_legato_gap_span(events: Sequence[MouthEvent], gap_start: float, gap_end: float) -> bool:
    if gap_end <= gap_start:
        return False
    span = [ev for ev in events if ev.start >= gap_start and ev.end <= gap_end and ev.start < ev.end]
    if not span or span[0].start != gap_start or span[-1].end != gap_end:
        return False
    return all(ev.shape is MouthShape.LEGATO_GAP for ev in span)


def _same_shape_runs(events: Sequence[MouthEvent]) -> list[list[MouthEvent]]:
    groups: list[list[MouthEvent]] = []
    current: list[MouthEvent] = []
    for ev in events:
        if ev.shape not in _PURE_VOWEL_PROFILES:
            if current:
                groups.append(current)
                current = []
            continue
        if current and ev.shape == current[-1].shape:
            current.append(ev)
        else:
            if current:
                groups.append(current)
            current = [ev]
    if current:
        groups.append(current)
    return groups


@dataclass
class _Group:
    events: list[MouthEvent]
    start: float
    end: float
    shape: MouthShape
    needs_absorption: bool = False
    triangle: bool = False
    attack: float = 0.0
    release: float = 0.0


def _classify_by_length(group: _Group, params: GenerationParams) -> None:
    length = group.end - group.start
    group.needs_absorption = length < params.triangle_min_frames
    group.triangle = (not group.needs_absorption) and max(
        0.0, length - params.min_hold_frames
    ) < _MIN_TRIANGLE_SLACK


def _peak_final_weight(event: MouthEvent, params: GenerationParams) -> float:
    return max(_final_weights(event, params).values())


def _effective_attack_release(
    length: float, params: GenerationParams
) -> tuple[float, float]:
    a, r = float(params.attack_frames), float(params.release_frames)
    available = max(0.0, length - params.min_hold_frames)
    if available >= a + r:
        return a, r
    scale = available / (a + r)
    a_eff = max(1.0, a * scale)
    r_eff = available - a_eff
    if r_eff < 1.0:
        r_eff, a_eff = 1.0, available - 1.0
    return a_eff, r_eff


def _absorb_winner(
    prev: _Group | None, nxt: _Group | None, params: GenerationParams
) -> _Group | None:
    if prev is None or nxt is None:
        return prev or nxt
    po, no = _peak_final_weight(prev.events[-1], params), _peak_final_weight(nxt.events[0], params)
    if po != no:
        return prev if po > no else nxt
    return nxt if (nxt.end - nxt.start) > (prev.end - prev.start) else prev


def _absorb_short_runs(groups: list[_Group], params: GenerationParams) -> list[_Group]:
    survivors: list[_Group] = []
    i, n = 0, len(groups)
    while i < n:
        if not groups[i].needs_absorption:
            survivors.append(groups[i])
            i += 1
            continue
        j = i
        while (
            j < n
            and groups[j].needs_absorption
            and (j == i or groups[j - 1].end == groups[j].start)
        ):
            j += 1
        run_start, run_end = groups[i].start, groups[j - 1].end
        prev_anchor = survivors[-1] if survivors and survivors[-1].end == run_start else None
        nxt_anchor = (
            groups[j]
            if j < n and not groups[j].needs_absorption and groups[j].start == run_end
            else None
        )
        winner = _absorb_winner(prev_anchor, nxt_anchor, params)
        if winner is prev_anchor and prev_anchor is not None:
            prev_anchor.events[-1] = replace(prev_anchor.events[-1], end=run_end)
            prev_anchor.end = run_end
        elif winner is nxt_anchor and nxt_anchor is not None:
            nxt_anchor.events[0] = replace(nxt_anchor.events[0], start=run_start)
            nxt_anchor.start = run_start
        i = j
    return survivors


def _merge_touching_same_shape(groups: list[_Group]) -> list[_Group]:
    merged: list[_Group] = []
    for g in groups:
        if merged and merged[-1].shape == g.shape and merged[-1].end == g.start:
            merged[-1].events = merged[-1].events + g.events
            merged[-1].end = g.end
        else:
            merged.append(g)
    return merged


def _normalize_groups(
    events: Sequence[MouthEvent], params: GenerationParams
) -> list[_Group]:
    groups = [_Group(g, g[0].start, g[-1].end, g[0].shape) for g in _same_shape_runs(events)]
    for g in groups:
        _classify_by_length(g, params)
    merged = _merge_touching_same_shape(_absorb_short_runs(groups, params))
    for g in merged:
        _classify_by_length(g, params)
        if not g.triangle:
            g.attack, g.release = _effective_attack_release(g.end - g.start, params)
    return merged


def _lerp_morph(
    w_prev: dict[str, float], w_next: dict[str, float], t_prev: float, t_next: float, t: float, morph: str
) -> float:
    v_prev, v_next = w_prev.get(morph, 0.0), w_next.get(morph, 0.0)
    if t_next == t_prev:
        return v_prev
    return v_prev + (v_next - v_prev) * (t - t_prev) / (t_next - t_prev)


@dataclass
class _Valley:
    b: float
    hw: int
    aperture_scale: float
    disp: float
    weights: dict[str, _ValleyWeights]

    @property
    def left(self) -> float:
        return self.b - self.hw

    @property
    def right(self) -> float:
        return self.b + self.hw


def _mora_valley_candidates(
    group: _Group, gw: list[dict[str, float]], group_morphs: Sequence[str], params: GenerationParams
) -> list[_Valley]:
    events = group.events
    if len(events) < 2:
        return []
    mids = [(ev.start + ev.end) / 2.0 for ev in events]
    candidates: list[_Valley] = []
    for j in range(len(events) - 1):
        b = events[j].end
        aperture_class = events[j + 1].aperture_class
        if aperture_class is ApertureClass.NONE:
            continue
        scale = _APERTURE_SCALE[aperture_class]
        hw = math.floor(min(params.mora_valley_frames, b - mids[j], mids[j + 1] - b))
        if hw < _MIN_MORA_VALLEY_HALF_WIDTH:
            continue
        d = scale / 2.0
        weights: dict[str, _ValleyWeights] = {}
        disp = 0.0
        for morph in group_morphs:
            left_val = _lerp_morph(gw[j], gw[j + 1], mids[j], mids[j + 1], b - hw, morph)
            right_val = _lerp_morph(gw[j], gw[j + 1], mids[j], mids[j + 1], b + hw, morph)
            weights[morph] = _ValleyWeights(left_val, d * (left_val + right_val), right_val)
            disp += (1.0 - scale) * _lerp_morph(gw[j], gw[j + 1], mids[j], mids[j + 1], b, morph)
        candidates.append(_Valley(b, hw, scale, disp, weights))
    return candidates


def _select_valleys(candidates: Sequence[_Valley], params: GenerationParams) -> list[_Valley]:
    ordered = sorted(candidates, key=lambda c: c.right)
    n = len(ordered)
    q_left = [_round_half_up(c.left) for c in ordered]
    q_right = [_round_half_up(c.right) for c in ordered]
    gap = params.mora_valley_min_gap_frames
    compatible_prefix_len: list[int] = []
    for i in range(n):
        best = 0
        for j in range(i):
            if q_left[i] - q_right[j] - 1 >= gap:
                best = j + 1
        compatible_prefix_len.append(best)
    opt = [0.0] * (n + 1)
    take_choice = [False] * (n + 1)
    for i in range(1, n + 1):
        take = ordered[i - 1].disp + opt[compatible_prefix_len[i - 1]]
        skip = opt[i - 1]
        if take >= skip:
            opt[i], take_choice[i] = take, True
        else:
            opt[i], take_choice[i] = skip, False
    selected: list[_Valley] = []
    i = n
    while i > 0:
        if take_choice[i]:
            selected.append(ordered[i - 1])
            i = compatible_prefix_len[i - 1]
        else:
            i -= 1
    return selected


def _preceding_event(events: Sequence[MouthEvent], start: float) -> MouthEvent | None:
    for ev in events:
        if ev.end == start:
            return ev
    return None


def _following_event(events: Sequence[MouthEvent], end: float) -> MouthEvent | None:
    for ev in events:
        if ev.start == end:
            return ev
    return None


def _lead_lag_frames(neighbor: MouthEvent | None, opening: float, params: GenerationParams) -> int:
    if neighbor is None or neighbor.shape not in (MouthShape.SILENCE, MouthShape.BILABIAL):
        return 0
    ratio = opening / params.open_cap if params.open_cap > 0.0 else 0.0
    want = params.anticipation_frames * min(max(ratio, 0.0), 1.0)
    return min(_round_half_up(want), math.floor((neighbor.end - neighbor.start) / 2))


def _quantize_targets(targets: Sequence[_Target]) -> list[MorphKey]:
    """targets は生成順に並べて渡す。"""
    best: dict[tuple[str, int], tuple[float, int, float]] = {}
    for seq, (morph, frame_f, weight) in enumerate(targets):
        cell = (morph, _round_half_up(frame_f))
        current = best.get(cell)
        if current is None or (frame_f, seq) > (current[0], current[1]):
            best[cell] = (frame_f, seq, weight)
    by_morph: dict[str, list[tuple[int, float, int, float]]] = {}
    for (morph, frame), (frame_f, seq, weight) in best.items():
        by_morph.setdefault(morph, []).append((frame, frame_f, seq, weight))
    keys: list[MorphKey] = []
    for morph, items in by_morph.items():
        items.sort()
        prev: int | None = None
        for frame, _frame_f, _seq, weight in items:
            if prev is not None and frame <= prev:
                frame = prev + 1
            prev = frame
            keys.append(_morph_key(morph, frame, weight))
    keys.sort(key=lambda k: k.frame)
    return keys


def _interp_points(points: Sequence[tuple[float, float]], t: float) -> float:
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    for (f0, v0), (f1, v1) in zip(points, points[1:], strict=False):
        if f0 <= t <= f1:
            return v0 if f1 == f0 else v0 + (v1 - v0) * (t - f0) / (f1 - f0)
    return points[-1][1]


def _consonant_at(events: Sequence[MouthEvent], t: float) -> ConsonantClass:
    for ev in events:
        if ev.start <= t < ev.end:
            return ev.consonant_class
    nearest = min(events, key=lambda ev: min(abs(t - ev.start), abs(t - ev.end)))
    return nearest.consonant_class


def _vibrato_extremum_suppressed(
    t: float, sign: float, period: float, valleys: Sequence[_Valley]
) -> bool:
    return any(
        abs(t - v.b)
        <= max(v.hw + _VIBRATO_SUPPRESSION_EXTRA_FRAMES, period if sign < 0.0 else period / 2.0)
        for v in valleys
    )


def _vibrato_targets(
    group: _Group,
    plateau_start: float,
    plateau_end: float,
    params: GenerationParams,
    valleys: Sequence[_Valley] = (),
) -> list[_Target]:
    holds = [_hold_value(group.shape, ev.open_amount, params) for ev in group.events]
    scales = [_APERTURE_SCALE[ev.aperture_class] for ev in group.events]
    hold_points: list[tuple[float, float]] = [(plateau_start, holds[0])]
    aperture_points: list[tuple[float, float]] = [(plateau_start, scales[0])]
    if len(group.events) >= 2:
        for ev, hold, scale in zip(group.events, holds, scales, strict=True):
            mid = (ev.start + ev.end) / 2.0
            if plateau_start < mid < plateau_end:
                hold_points.append((mid, hold))
                aperture_points.append((mid, scale))
    hold_points.append((plateau_end, holds[-1]))
    aperture_points.append((plateau_end, scales[-1]))
    hold_points.sort()
    aperture_points.sort()
    period = params.vibrato_period
    nodes: list[_Target] = []
    k = 0
    while True:
        t = plateau_start + period * (0.25 + 0.5 * k)
        if t >= plateau_end:
            break
        k += 1
        base = _interp_points(hold_points, t)
        if base <= 0.0:
            continue
        sign = math.sin(2.0 * math.pi * (t - plateau_start) / period)
        if valleys and _vibrato_extremum_suppressed(t, sign, period, valleys):
            continue
        amp_eff = min(params.vibrato_amp, base)
        open_v = min(max(base + amp_eff * sign, 0.0), params.open_cap)
        consonant_class = _consonant_at(group.events, t)
        aperture_v = _interp_points(aperture_points, t)
        for morph, weight in _weights_from_hold(group.shape, consonant_class, open_v, params).items():
            nodes.append(_Target(morph, t, weight * aperture_v))
    return nodes


def _triangle_peak_weights(group: _Group, gw: list[dict[str, float]]) -> dict[str, float]:
    if len(group.events) < 2:
        return gw[0]
    lens = [ev.end - ev.start for ev in group.events]
    total_len = sum(lens)
    peak_morphs = sorted({morph for w in gw for morph in w})
    return {
        morph: sum(length * w.get(morph, 0.0) for length, w in zip(lens, gw, strict=True)) / total_len
        for morph in peak_morphs
    }


def generate_morph_keys(
    events: Sequence[MouthEvent], params: GenerationParams
) -> list[MorphKey]:
    """events は時間順で、各イベントの end が次のイベントの start と一致し、全時間軸を隙間なく覆うこと。

    戻り値はフレーム昇順。
    """
    groups = _normalize_groups(events, params)
    weights = [[_final_weights(ev, params) for ev in g.events] for g in groups]
    n = len(groups)
    coart_half: dict[int, float] = {}
    for i in range(n - 1):
        if groups[i].end != groups[i + 1].start:
            continue
        left, right = groups[i].events[-1], groups[i + 1].events[0]
        diff = _shape_diff(
            left.shape, left.consonant_class, right.shape, right.consonant_class, params
        )
        shorter = min(groups[i].end - groups[i].start, groups[i + 1].end - groups[i + 1].start)
        coart_half[i] = _transition_frames(diff, shorter, params) / 2.0
    legato_at: dict[int, tuple[float, float]] = {}
    for i in range(n - 1):
        gs, ge = groups[i].end, groups[i + 1].start
        if _is_legato_gap_span(events, gs, ge):
            legato_at[i] = (gs, ge)
    targets: list[_Target] = []
    plateaus: list[tuple[float, float]] = []
    valleys_by_group: dict[int, list[_Valley]] = {}
    for i, g in enumerate(groups):
        gw = weights[i]
        if g.triangle:
            mid = (g.start + g.end) / 2.0
            connect_in = (i - 1) in coart_half or (i - 1) in legato_at
            connect_out = i in coart_half or i in legato_at
            for morph, weight in _triangle_peak_weights(g, gw).items():
                if not connect_in:
                    targets.append(_Target(morph, g.start, 0.0))
                targets.append(_Target(morph, mid, weight))
                if not connect_out:
                    targets.append(_Target(morph, g.end, 0.0))
            plateaus.append((mid, mid))
            continue
        group_morphs = sorted({morph for w in gw for morph in w})
        if (i - 1) in coart_half:
            plateau_start = groups[i - 1].end + coart_half[i - 1]
        elif (i - 1) in legato_at:
            plateau_start = g.start
        else:
            antic = _lead_lag_frames(_preceding_event(events, g.start), max(gw[0].values()), params)
            if antic > 0:
                f_start, f_attack = g.start - antic, g.start
            else:
                f_start, f_attack = g.start, g.start + g.attack
            for morph in group_morphs:
                targets.append(_Target(morph, f_start, 0.0))
                targets.append(_Target(morph, f_attack, gw[0].get(morph, 0.0)))
            plateau_start = f_attack
        if i in coart_half:
            plateau_end = g.end - coart_half[i]
        elif i in legato_at:
            plateau_end = g.end
        else:
            lag = _lead_lag_frames(_following_event(events, g.end), max(gw[-1].values()), params)
            if lag > 0:
                f_hold_end, f_end = g.end, g.end + lag
            else:
                f_hold_end, f_end = g.end - g.release, g.end
            for morph in group_morphs:
                targets.append(_Target(morph, f_hold_end, gw[-1].get(morph, 0.0)))
                targets.append(_Target(morph, f_end, 0.0))
            plateau_end = f_hold_end
        if len(g.events) >= 2:
            for ev, w in zip(g.events, gw, strict=True):
                f_mid = (ev.start + ev.end) / 2.0
                for morph in group_morphs:
                    targets.append(_Target(morph, f_mid, w.get(morph, 0.0)))
            selected = _select_valleys(_mora_valley_candidates(g, gw, group_morphs, params), params)
            if selected:
                valleys_by_group[i] = selected
                for valley in selected:
                    for morph in group_morphs:
                        left_val, center_val, right_val = valley.weights[morph]
                        targets.append(_Target(morph, valley.left, left_val))
                        targets.append(_Target(morph, valley.b, center_val))
                        targets.append(_Target(morph, valley.right, right_val))
        plateaus.append((plateau_start, plateau_end))
    for i in range(n - 1):
        if i not in coart_half:
            continue
        wa, wb = weights[i][-1], weights[i + 1][0]
        boundary = groups[i].end
        half = coart_half[i]
        for morph in _STANDARD_MORPHS:
            a, b = wa.get(morph, 0.0), wb.get(morph, 0.0)
            if a == 0.0 and b == 0.0:
                continue
            targets.append(_Target(morph, boundary - half, a))
            targets.append(_Target(morph, boundary, (a + b) / 2.0))
            targets.append(_Target(morph, boundary + half, b))
    for i, (gs, ge) in legato_at.items():
        wa, wb = weights[i][-1], weights[i + 1][0]
        gm = (gs + ge) / 2.0
        depth = _legato_valley_depth(ge - gs, params)
        for morph in _STANDARD_MORPHS:
            a, b = wa.get(morph, 0.0), wb.get(morph, 0.0)
            if a == 0.0 and b == 0.0:
                continue
            targets.append(_Target(morph, gs, a))
            targets.append(_Target(morph, gm, depth * (a + b)))
            targets.append(_Target(morph, ge, b))
    if params.vibrato_amp > 0.0 and params.vibrato_period > 0:
        for i, g in enumerate(groups):
            plateau_start, plateau_end = plateaus[i]
            if plateau_end - plateau_start > params.vibrato_threshold:
                targets.extend(
                    _vibrato_targets(
                        g, plateau_start, plateau_end, params, valleys_by_group.get(i, [])
                    )
                )
    return _quantize_targets(targets)
