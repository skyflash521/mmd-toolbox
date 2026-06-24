"""lipsync コアのモーフキー生成(lipsync.md §4, implementation-plan.md §4.1/§4.7)。

入力(開き量を同梱した口形イベント列＋生成パラメータ)から、標準口モーフ
(あ・い・う・え・お)のモーフキー列を決定論的に生成する。VMD への組み立て・
書き出しは呼び出し側が `mmd_toolbox.vmd` 経由で行う。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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

    プロファイル対象外(両唇閉鎖・無音)はグループ境界として扱い、ここでは出力しない
    (閉口キーは L-8 で置く)。
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


def _adjacent(prev: list[MouthEvent], nxt: list[MouthEvent]) -> bool:
    """2グループが両唇閉鎖・無音を挟まず直接隣接するか(前グループ終端=次グループ始端)。"""
    return prev[-1].end == nxt[0].start


def _span_length(group: list[MouthEvent]) -> float:
    """グループ(連結後の母音区間)の総フレーム長。"""
    return group[-1].end - group[0].start


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

    連続する同一母音イベントを1グループへ連結し(§4.2)、グループごとに §4.9 のエンベロープを置く:
    先頭にのみアタック(開始0.0・保持値)、末尾にのみリリース(保持値・終了0.0)、各小区間の中央に開き量
    の強弱節点を置いて節点間を線形に変える。直接隣接する異母音グループの境界では閉口を挟まず、§4.3 の
    協調調音(境界 b を中心とした幅 T の窓で前母音の保持値から次母音の保持値へ線形クロスフェードし、
    境界に中間口形を置く)へ置き換える。きびきび遷移・量子化などは後続ステップで加える。両唇閉鎖・無音の
    閉口キーは後続ステップ(L-8)で置く。返すキーは時間順(§4.7)。
    """
    groups = _vowel_groups(events)
    weights = [[_compose(ev.shape, ev.open_amount, params) for ev in g] for g in groups]
    keys: list[MorphKey] = []
    # 各グループの保持区間(アタック/リリースは協調調音しない端のみ。中央に強弱節点)。
    for i, group in enumerate(groups):
        gw = weights[i]
        coart_in = i > 0 and _adjacent(groups[i - 1], group)
        coart_out = i < len(groups) - 1 and _adjacent(group, groups[i + 1])
        if not coart_in:
            # §4.10 先行準備: 直前が無音なら口形の立ち上がりを A_eff だけ前倒す(アタック長は不変)。
            antic = _anticipation_frames(_preceding_event(events, group[0].start), params)
            f_start = round(group[0].start - antic)
            f_attack = round(group[0].start - antic + params.attack_frames)
            for morph, weight in gw[0].items():
                keys.append(_morph_key(morph, f_start, 0.0))
                keys.append(_morph_key(morph, f_attack, weight))
        if not coart_out:
            f_hold_end = round(group[-1].end - params.release_frames)
            f_end = round(group[-1].end)
            for morph, weight in gw[-1].items():
                keys.append(_morph_key(morph, f_hold_end, weight))
                keys.append(_morph_key(morph, f_end, 0.0))
        if len(group) >= 2:
            for ev, w in zip(group, gw):
                f_mid = round((ev.start + ev.end) / 2)
                for morph, weight in w.items():
                    keys.append(_morph_key(morph, f_mid, weight))
    # 隣接する異母音グループ境界の協調調音(§4.3)。閉口を挟まず中間口形へ線形遷移する。
    for i in range(len(groups) - 1):
        if not _adjacent(groups[i], groups[i + 1]):
            continue
        wa, wb = weights[i][-1], weights[i + 1][0]
        boundary = groups[i][-1].end
        diff = _shape_diff(groups[i][-1].shape, groups[i + 1][0].shape, params)
        shorter = min(_span_length(groups[i]), _span_length(groups[i + 1]))
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
