"""lipsync コアのモーフキー生成(要求仕様 lipsync.md §4)。

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

# 各母音の主モーフ(目的母音と同名の標準口モーフ)。
_MAIN_MORPH = {
    MouthShape.A: "あ",
    MouthShape.I: "い",
    MouthShape.U: "う",
    MouthShape.E: "え",
    MouthShape.O: "お",
}

# 母音合成プロファイル(保持値1.0時の相対重み)。主モーフ以外の
# 非ゼロ重みが補助モーフ。表中 0.0 のモーフは持たない。`GenerationParams` とは別の lipsync 既定。
_PROFILES: dict[MouthShape, dict[str, float]] = {
    MouthShape.A: {"あ": 1.0},
    MouthShape.I: {"あ": 0.1, "い": 1.0},
    MouthShape.U: {"う": 1.0, "お": 0.2},
    MouthShape.E: {"あ": 0.2, "い": 0.2, "え": 1.0},
    MouthShape.O: {"う": 0.2, "お": 1.0},
}

# vowel_scale=(a, i, u, e, o) の添字。
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


def _half_up(value: float) -> int:
    """四捨五入(0.5 は切り上げ)。Python の round() の銀行家丸めを避ける。

    フレーム番号の偶奇に依存せず `.5` 目標(協調調音で遷移長 T が奇数のときの b ± T/2 等)を確定する。
    """
    return math.floor(value + 0.5)


def _effective_profile(shape: MouthShape, params: GenerationParams) -> dict[str, float]:
    """有効プロファイル(補助重みに誇張係数を乗算した相対重み)。"""
    main = _MAIN_MORPH[shape]
    return {
        morph: weight if morph == main else weight * params.exaggeration
        for morph, weight in _PROFILES[shape].items()
    }


def _hold_value(shape: MouthShape, open_amount: float, params: GenerationParams) -> float:
    """保持値 hold(母音別倍率を掛け open_cap で上限クランプした開き量)。"""
    hold = open_amount * params.vowel_scale[_VOWEL_INDEX[shape]]
    return min(max(hold, 0.0), params.open_cap)


def _weights_from_hold(shape: MouthShape, hold: float, params: GenerationParams) -> dict[str, float]:
    """保持値 hold から各口モーフ重みを求める(有効プロファイル × hold、合成総量の比例縮小)。"""
    weights = {morph: weight * hold for morph, weight in _effective_profile(shape, params).items()}
    total = sum(weights.values())
    if total > params.open_cap:
        factor = params.open_cap / total
        weights = {morph: weight * factor for morph, weight in weights.items()}
    return weights


def _compose(shape: MouthShape, open_amount: float, params: GenerationParams) -> dict[str, float]:
    """母音の合成プロファイルから各口モーフの重みを求める。

    手順: (1)プロファイル選択 →(2)補助重みに誇張係数を乗算 →(3)保持値 hold を上限クランプ →
    (4)各モーフ重み = 有効プロファイル × hold →(5)合成後総量が open_cap 超過時のみ比例縮小。
    """
    return _weights_from_hold(shape, _hold_value(shape, open_amount, params), params)


def _shape_diff(shape_a: MouthShape, shape_b: MouthShape, params: GenerationParams) -> float:
    """両母音の口形差(0〜1)。

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
    """協調調音の遷移長。

    基準長(=重なり上限)× (1 − 0.5・口形差) を、下限1・上限 min(重なり上限, 短い側区間長/2) で
    クランプして整数フレーム化する。
    """
    base = params.coartic_overlap_max
    value = base * (1.0 - 0.5 * diff)
    cap = min(float(base), shorter_len / 2.0)
    return _half_up(max(1.0, min(value, cap)))


def _vowel_groups(events: Sequence[MouthEvent]) -> list[list[MouthEvent]]:
    """連続する同一母音イベントを極大グループへ束ねる。

    プロファイル対象外(両唇閉鎖・無音)はグループ境界として扱い、ここでは出力しない。閉口は隣接母音の
    リリース/アタックの 0.0 キーとキー不在(MMD 上 0.0)で表す(専用の閉口キーは設けない)。
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
    """正規化対象の母音グループ。

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
    """境界イベントの開き量(吸収先タイブレーク用)。"""
    scale = params.vowel_scale[_VOWEL_INDEX[event.shape]]
    return min(max(event.open_amount * scale, 0.0), params.open_cap)


def _effective_attack_release(
    length: float, params: GenerationParams
) -> tuple[float, float]:
    """競合短縮後の実効アタック/リリース。

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
    """短区間の吸収先を選ぶ(開き量大 → 長い側 → 前側)。"""
    if prev is None or nxt is None:
        return prev or nxt
    po, no = _opening(prev.events[-1], params), _opening(nxt.events[0], params)
    if po != no:
        return prev if po > no else nxt
    return nxt if (nxt.end - nxt.start) > (prev.end - prev.start) else prev


def _normalize_groups(
    events: Sequence[MouthEvent], params: GenerationParams
) -> list[_Group]:
    """母音グループを正規化する: 短区間の吸収/除去・吸収後の同母音連結・競合短縮。

    出力は実効スパンと実効アタック/リリースを持つ生き残りグループ列(フレーム浮動小数。量子化は後段)。
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
    # 吸収後に直接隣接した同一母音グループを連結へ統合する。
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
    """先行準備の前倒し量 A_eff。

    直前が無音区間のときのみ、先行フレーム数を直前区間長の 1/2 で自動短縮した値。直前が無い・母音・
    両唇閉鎖のときは 0(先行しない)。前区間長の 1/2 上限により前区間を侵食せず負フレームにも出ない。
    """
    if prev is None or prev.shape is not MouthShape.SILENCE:
        return 0
    return min(params.anticipation_frames, math.floor((prev.end - prev.start) / 2))


def _quantize_targets(targets: Sequence[tuple[str, float, float]]) -> list[MorphKey]:
    """float 目標位置の (モーフ名, フレーム, 重み) 列を整数フレームへ量子化する。

    `targets` は生成順(=列の添字が生成順)。手順:
    (1) 各目標を `_half_up`(四捨五入)で整数フレーム化する。
    (2) 同一モーフ・同一整数フレームへ潰れた目標は、量子化前フレームが最も後ろ(タイは生成順が後)の
        目標値へ統合する(協調調音 T=1 の b ± T/2 が同フレームへ潰れる衝突もここで解消)。
    (3) モーフごとに (整数フレーム, 量子化前フレーム, 生成順) 昇順で走査し、直前に確定した同一モーフの
        キーと同じフレームかそれ以前へ来るキーを `直前 + 1` へずらして厳密昇順化する(1フレーム以上離れた
        キーは動かさない)。

    返すのは時間順の MorphKey 列。
    """
    # (1)+(2) 同一(モーフ,整数フレーム)へ統合。量子化前フレーム最後尾、タイは生成順が後を残す。
    best: dict[tuple[str, int], tuple[float, int, float]] = {}
    for seq, (morph, frame_f, weight) in enumerate(targets):
        cell = (morph, _half_up(frame_f))
        current = best.get(cell)
        if current is None or (frame_f, seq) > (current[0], current[1]):
            best[cell] = (frame_f, seq, weight)
    # (3) モーフごとに昇順走査して単調化する。
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


def _interp_open(points: Sequence[tuple[float, float]], t: float) -> float:
    """制御点列(フレーム昇順の (フレーム, 開き量))を時刻 t で線形補間する。端の外側は端値で一定。"""
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    for (f0, v0), (f1, v1) in zip(points, points[1:]):
        if f0 <= t <= f1:
            return v0 if f1 == f0 else v0 + (v1 - v0) * (t - f0) / (f1 - f0)
    return points[-1][1]


def _vibrato_targets(
    group: _Group, plateau_start: float, plateau_end: float, params: GenerationParams
) -> list[tuple[str, float, float]]:
    """保持プラトーに伸び表現の揺らぎ節点を生成する。

    公称開き量 `base_open(t)` を保持・強弱節点の線形補間で求め、正弦波で変調した実効開き量 `open_v` から
    合成手順で各モーフ重みを出す。節点は正弦波の極値(`t_k = plateau_start + P·(1/4 + k/2)`)の
    厳密内側のみ。返すのは float 目標 `(モーフ名, フレーム, 重み)` 列(量子化は後段)。
    """
    holds = [_hold_value(group.shape, ev.open_amount, params) for ev in group.events]
    # 公称開き量の制御点: プラトー始端(先頭 hold)・プラトー内の各イベント中央(その hold)・終端(末尾 hold)。
    points: list[tuple[float, float]] = [(plateau_start, holds[0])]
    if len(group.events) >= 2:
        for ev, hold in zip(group.events, holds):
            mid = (ev.start + ev.end) / 2.0
            if plateau_start < mid < plateau_end:
                points.append((mid, hold))
    points.append((plateau_end, holds[-1]))
    points.sort()
    period = params.vibrato_period
    nodes: list[tuple[str, float, float]] = []
    k = 0
    while True:
        t = plateau_start + period * (0.25 + 0.5 * k)
        if t >= plateau_end:
            break
        k += 1
        base = _interp_open(points, t)
        if base <= 0.0:
            continue
        amp_eff = min(params.vibrato_amp, base)
        offset = amp_eff * math.sin(2.0 * math.pi * (t - plateau_start) / period)
        open_v = min(max(base + offset, 0.0), params.open_cap)
        for morph, weight in _weights_from_hold(group.shape, open_v, params).items():
            nodes.append((morph, t, weight))
    return nodes


def generate_morph_keys(
    events: Sequence[MouthEvent], params: GenerationParams
) -> list[MorphKey]:
    """口形イベント列からモーフキー列を生成する(要求仕様 lipsync.md §4)。

    連続する同一母音イベントを1グループへ連結し、最小保持未満の短区間を隣接母音へ吸収/除去し競合短縮で
    実効アタック/リリースを求めたうえで、グループごとにエンベロープを置く: 先頭にのみアタック(開始0.0・
    保持値)、末尾にのみリリース(保持値・終了0.0)、各小区間の中央に開き量の強弱節点を置いて節点間を
    線形に変える。直接隣接する異母音グループの境界では閉口を挟まず、協調調音(境界 b を中心とした幅 T の
    窓で前母音の保持値から次母音の保持値へ線形クロスフェードし、境界に中間口形を置く)へ置き換える。
    無音直後の母音は先行準備でアタックを前倒す。長く伸ばす母音の保持プラトーには伸び表現で揺らぎ節点を
    任意に加える。両唇閉鎖・無音は隣接母音の 0.0 キーとキー不在(MMD 上 0.0)で閉口を表し、専用の閉口
    キーは置かない。整数フレームへの量子化は最後に一括して行う。返すキーは時間順。
    """
    groups = _normalize_groups(events, params)
    weights = [[_compose(ev.shape, ev.open_amount, params) for ev in g.events] for g in groups]
    n = len(groups)
    # 直接隣接する異母音グループ境界の協調調音の半幅 T/2(境界 i と i+1 の間)。保持プラトー端の算出にも使う。
    coart_half: dict[int, float] = {}
    for i in range(n - 1):
        if groups[i].end != groups[i + 1].start:
            continue
        diff = _shape_diff(groups[i].shape, groups[i + 1].shape, params)
        shorter = min(groups[i].end - groups[i].start, groups[i + 1].end - groups[i + 1].start)
        coart_half[i] = _transition_frames(diff, shorter, params) / 2.0
    # 要所キーは (モーフ名, 目標フレーム(float), 重み) の目標値として生成順に集め、最後に一括量子化する。
    targets: list[tuple[str, float, float]] = []
    plateaus: list[tuple[float, float]] = []  # グループごとの保持プラトー [始端, 終端](伸び表現の対象)。
    # 各グループの保持区間(アタック/リリースは協調調音しない端のみ。中央に強弱節点)。実効スパン・実効 a'/r'。
    for i, g in enumerate(groups):
        gw = weights[i]
        coart_in = i > 0 and groups[i - 1].end == g.start
        coart_out = i < n - 1 and groups[i + 1].start == g.end
        if not coart_in:
            # 先行準備: 直前が無音なら口形の立ち上がりを A_eff だけ前倒す(実効アタック長は不変)。
            antic = _anticipation_frames(_preceding_event(events, g.start), params)
            f_start = g.start - antic
            f_attack = g.start - antic + g.attack
            for morph, weight in gw[0].items():
                targets.append((morph, f_start, 0.0))
                targets.append((morph, f_attack, weight))
            plateau_start = f_attack
        else:
            plateau_start = groups[i - 1].end + coart_half[i - 1]
        if not coart_out:
            f_hold_end = g.end - g.release
            f_end = g.end
            for morph, weight in gw[-1].items():
                targets.append((morph, f_hold_end, weight))
                targets.append((morph, f_end, 0.0))
            plateau_end = f_hold_end
        else:
            plateau_end = g.end - coart_half[i]
        if len(g.events) >= 2:
            for ev, w in zip(g.events, gw):
                f_mid = (ev.start + ev.end) / 2.0
                for morph, weight in w.items():
                    targets.append((morph, f_mid, weight))
        plateaus.append((plateau_start, plateau_end))
    # 隣接する異母音グループ境界の協調調音。閉口を挟まず中間口形へ線形遷移する。
    for i in range(n - 1):
        if i not in coart_half:
            continue
        wa, wb = weights[i][-1], weights[i + 1][0]
        boundary = groups[i].end
        half = coart_half[i]
        f_s, f_b, f_e = boundary - half, boundary, boundary + half
        for morph in _VOWEL_ORDER:
            a, b = wa.get(morph, 0.0), wb.get(morph, 0.0)
            if a == 0.0 and b == 0.0:
                continue
            targets.append((morph, f_s, a))
            targets.append((morph, f_b, (a + b) / 2.0))
            targets.append((morph, f_e, b))
    # 伸び表現: 公称エンベロープ・強弱・協調調音の後に、長い保持プラトーへ揺らぎ節点を加える(任意)。
    if params.vibrato_amp > 0.0 and params.vibrato_period > 0:
        for i, g in enumerate(groups):
            plateau_start, plateau_end = plateaus[i]
            if plateau_end - plateau_start > params.vibrato_threshold:
                targets.extend(_vibrato_targets(g, plateau_start, plateau_end, params))
    return _quantize_targets(targets)
