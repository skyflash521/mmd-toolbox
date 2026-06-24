"""lipsync コアのモーフキー生成(lipsync.md §4, implementation-plan.md §4.1/§4.7)。

入力(開き量を同梱した口形イベント列＋生成パラメータ)から、標準口モーフ
(あ・い・う・え・お)のモーフキー列を決定論的に生成する。VMD への組み立て・
書き出しは呼び出し側が `mmd_toolbox.vmd` 経由で行う。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from mmd_toolbox.vmd import MorphKey

from .types import GenerationParams, MouthEvent, MouthShape

# 各母音の主モーフ(目的母音と同名の標準口モーフ。implementation-plan.md §4.1)。
_MAIN_MORPH = {
    MouthShape.A: "あ",
    MouthShape.I: "い",
    MouthShape.U: "う",
    MouthShape.E: "え",
    MouthShape.O: "お",
}

# 母音合成プロファイル(保持値1.0時の相対重み。implementation-plan.md §4.1)。主モーフ以外の
# 非ゼロ重みが補助モーフ。表中 0.0 のモーフは持たない。`GenerationParams` とは別の lipsync 既定。
_PROFILES: dict[MouthShape, dict[str, float]] = {
    MouthShape.A: {"あ": 1.0},
    MouthShape.I: {"あ": 0.1, "い": 1.0},
    MouthShape.U: {"う": 1.0, "お": 0.2},
    MouthShape.E: {"あ": 0.2, "い": 0.2, "え": 1.0},
    MouthShape.O: {"う": 0.2, "お": 1.0},
}

# vowel_scale=(a, i, u, e, o) の添字(implementation-plan.md §4.8)。
_VOWEL_INDEX = {
    MouthShape.A: 0,
    MouthShape.I: 1,
    MouthShape.U: 2,
    MouthShape.E: 3,
    MouthShape.O: 4,
}

# 標準口モーフの固定列(口形差ベクトルの軸・同フレーム出力の決定論的順序)。
_VOWEL_ORDER = ("あ", "い", "う", "え", "お")


def _morph_key(name: str, frame: int, weight: float) -> MorphKey:
    """標準口モーフ名を cp932・15バイト固定の name_raw に符号化したキーを作る。"""
    return MorphKey(name.encode("cp932").ljust(15, b"\x00"), frame, weight)


def _compose(shape: MouthShape, open_amount: float, params: GenerationParams) -> dict[str, float]:
    """母音の合成プロファイルから各口モーフの重みを求める(implementation-plan.md §4.1)。

    手順: (1)プロファイル選択 →(2)補助重みに誇張係数を乗算 →(3)保持値 hold を上限クランプ →
    (4)各モーフ重み = 有効プロファイル × hold →(5)合成後総量が open_cap 超過時のみ比例縮小。
    """
    profile = _PROFILES[shape]
    main = _MAIN_MORPH[shape]
    effective = {
        morph: weight if morph == main else weight * params.exaggeration
        for morph, weight in profile.items()
    }
    hold = open_amount * params.vowel_scale[_VOWEL_INDEX[shape]]
    hold = min(max(hold, 0.0), params.open_cap)
    weights = {morph: weight * hold for morph, weight in effective.items()}
    total = sum(weights.values())
    if total > params.open_cap:
        factor = params.open_cap / total
        weights = {morph: weight * factor for morph, weight in weights.items()}
    return weights


def _shape_diff(shape_a: MouthShape, shape_b: MouthShape, params: GenerationParams) -> float:
    """両母音の口形差(0〜1。implementation-plan.md §4.3)。

    各母音の有効プロファイル(誇張適用後・保持値非依存の相対重み)を標準口モーフ5次元ベクトルとし、
    L2 正規化したうえでユークリッド距離を取り sqrt(2) で割る。同一口形は 0.0、互いに重ならない口形
    方向は 1.0。
    """
    def _unit(shape: MouthShape) -> list[float]:
        main = _MAIN_MORPH[shape]
        eff = {
            morph: weight if morph == main else weight * params.exaggeration
            for morph, weight in _PROFILES[shape].items()
        }
        vec = [eff.get(morph, 0.0) for morph in _VOWEL_ORDER]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0.0 else vec

    ua, ub = _unit(shape_a), _unit(shape_b)
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(ua, ub)))
    return dist / math.sqrt(2.0)


def _transition_frames(diff: float, shorter_len: float, params: GenerationParams) -> int:
    """協調調音の遷移長(implementation-plan.md §4.3)。

    基準長(=重なり上限)× (1 − 0.5・口形差) を、下限1・上限 min(重なり上限, 短い側区間長/2) で
    クランプして整数フレーム化する。
    """
    base = params.coartic_overlap_max
    value = base * (1.0 - 0.5 * diff)
    cap = min(float(base), shorter_len / 2.0)
    return round(max(1.0, min(value, cap)))


def _vowel_groups(events: Sequence[MouthEvent]) -> list[list[MouthEvent]]:
    """連続する同一母音イベントを極大グループへ束ねる(implementation-plan.md §4.2)。

    プロファイル対象外(両唇閉鎖・無音)はグループ境界として扱い、ここでは出力しない。閉口は隣接母音の
    リリース/アタックの 0.0 キーとキー不在(MMD 上 0.0)で表す(専用の閉口キーは設けない。§4.11)。
    """
    groups: list[list[MouthEvent]] = []
    current: list[MouthEvent] = []
    for ev in events:
        if ev.shape not in _PROFILES:
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
    """正規化対象の母音グループ(implementation-plan.md §4.4)。

    `start`/`end` は吸収で延長されうる実効区間、`attack`/`release` は競合短縮後の実効値(浮動小数)。
    """

    events: list[MouthEvent]
    start: float
    end: float
    shape: MouthShape
    short: bool = False
    attack: float = 0.0
    release: float = 0.0


def _opening(event: MouthEvent, params: GenerationParams) -> float:
    """境界イベントの開き量(吸収先タイブレーク用。implementation-plan.md §4.4)。"""
    scale = params.vowel_scale[_VOWEL_INDEX[event.shape]]
    return min(max(event.open_amount * scale, 0.0), params.open_cap)


def _effective_attack_release(
    length: float, params: GenerationParams
) -> tuple[float, float]:
    """競合短縮後の実効アタック/リリース(implementation-plan.md §4.4)。

    保持を最優先で確保した残り `available = max(0, length − min_hold_frames)` に収まるよう、アタック+
    リリース全量が入らない区間で比例縮小する(各最小1フレーム)。`available ≥ a+r` なら縮小しない。
    """
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
    """短区間の吸収先を選ぶ(開き量大 → 長い側 → 前側。implementation-plan.md §4.4)。"""
    if prev is None or nxt is None:
        return prev or nxt
    po, no = _opening(prev.events[-1], params), _opening(nxt.events[0], params)
    if po != no:
        return prev if po > no else nxt
    return nxt if (nxt.end - nxt.start) > (prev.end - prev.start) else prev


def _normalize_groups(
    events: Sequence[MouthEvent], params: GenerationParams
) -> list[_Group]:
    """§4.2 の母音グループを §4.4 で正規化する: 短区間の吸収/除去・吸収後の同母音連結・競合短縮。

    出力は実効スパンと実効アタック/リリースを持つ生き残りグループ列(フレーム浮動小数。量子化は §4.5)。
    """
    groups = [_Group(g, g[0].start, g[-1].end, g[0].shape) for g in _vowel_groups(events)]
    for g in groups:
        g.short = max(0.0, (g.end - g.start) - params.min_hold_frames) < 2
    # 短区間 run を隣接母音アンカーへ吸収/除去する。
    survivors: list[_Group] = []
    i, n = 0, len(groups)
    while i < n:
        if not groups[i].short:
            survivors.append(groups[i])
            i += 1
            continue
        j = i
        while j < n and groups[j].short and (j == i or groups[j - 1].end == groups[j].start):
            j += 1
        run_start, run_end = groups[i].start, groups[j - 1].end
        prev_anchor = survivors[-1] if survivors and survivors[-1].end == run_start else None
        nxt_anchor = (
            groups[j] if j < n and not groups[j].short and groups[j].start == run_end else None
        )
        winner = _absorb_winner(prev_anchor, nxt_anchor, params)
        if winner is prev_anchor and prev_anchor is not None:
            prev_anchor.end = run_end
        elif winner is nxt_anchor and nxt_anchor is not None:
            nxt_anchor.start = run_start
        i = j
    # 吸収後に直接隣接した同一母音グループを §4.2 の連結へ統合する。
    merged: list[_Group] = []
    for g in survivors:
        if merged and merged[-1].shape == g.shape and merged[-1].end == g.start:
            merged[-1].events = merged[-1].events + g.events
            merged[-1].end = g.end
        else:
            merged.append(g)
    for g in merged:
        g.attack, g.release = _effective_attack_release(g.end - g.start, params)
    return merged


def _preceding_event(events: Sequence[MouthEvent], start: float) -> MouthEvent | None:
    """終端フレームが start に一致する直前イベント(連続契約により一意。無ければ None)。"""
    for ev in events:
        if ev.end == start:
            return ev
    return None


def _anticipation_frames(prev: MouthEvent | None, params: GenerationParams) -> int:
    """先行準備の前倒し量 A_eff(implementation-plan.md §4.10)。

    直前が無音区間のときのみ、先行フレーム数を直前区間長の 1/2 で自動短縮した値。直前が無い・母音・
    両唇閉鎖のときは 0(先行しない)。前区間長の 1/2 上限により前区間を侵食せず負フレームにも出ない。
    """
    if prev is None or prev.shape is not MouthShape.SILENCE:
        return 0
    return min(params.anticipation_frames, math.floor((prev.end - prev.start) / 2))


def generate_morph_keys(
    events: Sequence[MouthEvent], params: GenerationParams
) -> list[MorphKey]:
    """口形イベント列からモーフキー列を生成する(implementation-plan.md §4.7/§4.9/§4.2/§4.3)。

    連続する同一母音イベントを1グループへ連結し(§4.2)、§4.4 で最小保持未満の短区間を隣接母音へ吸収/
    除去し競合短縮で実効アタック/リリースを求めたうえで、グループごとに §4.9 のエンベロープを置く:
    先頭にのみアタック(開始0.0・保持値)、末尾にのみリリース(保持値・終了0.0)、各小区間の中央に開き量
    の強弱節点を置いて節点間を線形に変える。直接隣接する異母音グループの境界では閉口を挟まず、§4.3 の
    協調調音(境界 b を中心とした幅 T の窓で前母音の保持値から次母音の保持値へ線形クロスフェードし、
    境界に中間口形を置く)へ置き換える。無音直後の母音は §4.10 の先行準備でアタックを前倒す。両唇閉鎖・無音は
    隣接母音の 0.0 キーとキー不在(MMD 上 0.0)で閉口を表し、専用の閉口キーは置かない(§4.11)。量子化は
    後続ステップ(§4.5/L-9)で行う。返すキーは時間順(§4.7)。
    """
    groups = _normalize_groups(events, params)
    weights = [[_compose(ev.shape, ev.open_amount, params) for ev in g.events] for g in groups]
    keys: list[MorphKey] = []
    # 各グループの保持区間(アタック/リリースは協調調音しない端のみ。中央に強弱節点)。実効スパン・実効 a'/r'。
    for i, g in enumerate(groups):
        gw = weights[i]
        coart_in = i > 0 and groups[i - 1].end == g.start
        coart_out = i < len(groups) - 1 and groups[i + 1].start == g.end
        if not coart_in:
            # §4.10 先行準備: 直前が無音なら口形の立ち上がりを A_eff だけ前倒す(実効アタック長は不変)。
            antic = _anticipation_frames(_preceding_event(events, g.start), params)
            f_start = round(g.start - antic)
            f_attack = round(g.start - antic + g.attack)
            for morph, weight in gw[0].items():
                keys.append(_morph_key(morph, f_start, 0.0))
                keys.append(_morph_key(morph, f_attack, weight))
        if not coart_out:
            f_hold_end = round(g.end - g.release)
            f_end = round(g.end)
            for morph, weight in gw[-1].items():
                keys.append(_morph_key(morph, f_hold_end, weight))
                keys.append(_morph_key(morph, f_end, 0.0))
        if len(g.events) >= 2:
            for ev, w in zip(g.events, gw):
                f_mid = round((ev.start + ev.end) / 2)
                for morph, weight in w.items():
                    keys.append(_morph_key(morph, f_mid, weight))
    # 隣接する異母音グループ境界の協調調音(§4.3)。閉口を挟まず中間口形へ線形遷移する。
    for i in range(len(groups) - 1):
        if groups[i].end != groups[i + 1].start:
            continue
        wa, wb = weights[i][-1], weights[i + 1][0]
        boundary = groups[i].end
        diff = _shape_diff(groups[i].shape, groups[i + 1].shape, params)
        shorter = min(groups[i].end - groups[i].start, groups[i + 1].end - groups[i + 1].start)
        half = _transition_frames(diff, shorter, params) / 2.0
        f_s, f_b, f_e = round(boundary - half), round(boundary), round(boundary + half)
        for morph in _VOWEL_ORDER:
            a, b = wa.get(morph, 0.0), wb.get(morph, 0.0)
            if a == 0.0 and b == 0.0:
                continue
            keys.append(_morph_key(morph, f_s, a))
            keys.append(_morph_key(morph, f_b, (a + b) / 2.0))
            keys.append(_morph_key(morph, f_e, b))
    keys.sort(key=lambda k: k.frame)
    return keys
