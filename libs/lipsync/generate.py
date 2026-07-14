"""lipsync コアのモーフキー生成(要求仕様 lipsync.md §4)。

入力(開き量を同梱した口形イベント列＋生成パラメータ)から、標準口モーフ
(あ・い・う・え・お・ん)のモーフキー列を決定論的に生成する。VMD への組み立て・
書き出しは呼び出し側が `vmd` 経由で行う。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from vmd import MorphKey

from .types import ApertureClass, ConsonantClass, GenerationParams, MouthEvent, MouthShape

# 各母音的口形の主モーフ(目的口形と同名の標準口モーフ)。
_MAIN_MORPH = {
    MouthShape.A: "あ",
    MouthShape.I: "い",
    MouthShape.U: "う",
    MouthShape.E: "え",
    MouthShape.O: "お",
    MouthShape.N: "ん",
}

# 純母音プロファイル(保持値1.0時の主モーフ単独。子音変調は _preprofile が補助を足す)。
# 純母音を主モーフ単独にすることで母音どうしの口形差(_shape_diff)が最大化する。`GenerationParams`
# とは別の lipsync 既定。表中に無いモーフは寄与なし(0.0)。
_PROFILES: dict[MouthShape, dict[str, float]] = {
    MouthShape.A: {"あ": 1.0},
    MouthShape.I: {"い": 1.0},
    MouthShape.U: {"う": 1.0},
    MouthShape.E: {"え": 1.0},
    MouthShape.O: {"お": 1.0},
    MouthShape.N: {"ん": 1.0},
}

# 子音変調の補助重み(保持値1.0時の相対量。視覚で詰める初期値)。ROUNDED は丸め(う・お方向)、
# SPREAD は横引き(い方向)の補助を主モーフへ足す。主モーフと同名のモーフへは足さない(二重計上回避)。
# NONE/NEUTRAL は変調なし(純母音)。CLI 公開はせず lipsync 内の定数として持つ。
_ROUNDED_U_GAIN = 0.3  # ROUNDED → う 方向の丸め
_ROUNDED_O_GAIN = 0.0  # ROUNDED → お 方向の丸め(初期は う のみ。視覚で調整)
_SPREAD_I_GAIN = 0.3  # SPREAD → い 方向の横引き

# 開口減衰(子音がもたらす顎の狭め具合)の係数。合成後総量の比例縮小の後に全モーフへ一律に掛ける
# (比例縮小より前に掛けると縮小の分子・分母から相殺され効果が消えるため)。視覚で詰める初期値。
# CLI 公開はせず lipsync 内の定数として持つ。
_APERTURE_SCALE: dict[ApertureClass, float] = {
    ApertureClass.NONE: 1.0,
    ApertureClass.FIRM_CLOSURE: 0.75,
    ApertureClass.NARROW_CHANNEL: 0.85,
    ApertureClass.SLIGHT_CLOSURE: 0.92,
}


def _preprofile(shape: MouthShape, consonant_class: ConsonantClass) -> dict[str, float]:
    """母音口形と先頭子音種別から、子音変調を反映した相対プロファイル(保持値1.0時の重み)を作る。

    純母音(NONE/NEUTRAL)は主モーフ単独。ROUNDED は丸め(う・お)、SPREAD は横引き(い)の補助重みを
    主モーフへ足す。主モーフと同名のモーフへは足さない(母音が既にその方向のとき二重計上しない)。
    """
    profile = dict(_PROFILES[shape])
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

# vowel_scale=(a, i, u, e, o, n) の添字。
_VOWEL_INDEX = {
    MouthShape.A: 0,
    MouthShape.I: 1,
    MouthShape.U: 2,
    MouthShape.E: 3,
    MouthShape.O: 4,
    MouthShape.N: 5,
}

# 標準口モーフの固定列(口形差ベクトルの軸・同フレーム出力の決定論的順序)。
_VOWEL_ORDER = ("あ", "い", "う", "え", "お", "ん")


def _morph_key(name: str, frame: int, weight: float) -> MorphKey:
    """標準口モーフ名を cp932・15バイト固定の name_raw に符号化したキーを作る。"""
    return MorphKey(name.encode("cp932").ljust(15, b"\x00"), frame, weight)


def _half_up(value: float) -> int:
    """四捨五入(0.5 は切り上げ)。Python の round() の銀行家丸めを避ける。

    フレーム番号の偶奇に依存せず `.5` 目標(協調調音で遷移長 T が奇数のときの b ± T/2 等)を確定する。
    """
    return math.floor(value + 0.5)


def _effective_profile(
    shape: MouthShape, consonant_class: ConsonantClass, params: GenerationParams
) -> dict[str, float]:
    """有効プロファイル(子音変調込みの相対重みで、主モーフ以外に誇張係数を乗算)。"""
    main = _MAIN_MORPH[shape]
    return {
        morph: weight if morph == main else weight * params.exaggeration
        for morph, weight in _preprofile(shape, consonant_class).items()
    }


def _hold_value(shape: MouthShape, open_amount: float, params: GenerationParams) -> float:
    """保持値 hold(母音別倍率を掛け open_cap で上限クランプした開き量)。"""
    hold = open_amount * params.vowel_scale[_VOWEL_INDEX[shape]]
    return min(max(hold, 0.0), params.open_cap)


def _weights_from_hold(
    shape: MouthShape, consonant_class: ConsonantClass, hold: float, params: GenerationParams
) -> dict[str, float]:
    """保持値 hold から各口モーフ重みを求める(有効プロファイル × hold、合成総量の比例縮小)。"""
    weights = {
        morph: weight * hold
        for morph, weight in _effective_profile(shape, consonant_class, params).items()
    }
    total = sum(weights.values())
    if total > params.open_cap:
        factor = params.open_cap / total
        weights = {morph: weight * factor for morph, weight in weights.items()}
    return weights


def _compose(
    shape: MouthShape,
    consonant_class: ConsonantClass,
    aperture_class: ApertureClass,
    open_amount: float,
    params: GenerationParams,
) -> dict[str, float]:
    """母音の合成プロファイルから各口モーフの重みを求める。

    手順: (1)母音＋子音種別のプロファイル選択 →(2)補助重みに誇張係数を乗算 →(3)保持値 hold を上限
    クランプ →(4)各モーフ重み = 有効プロファイル × hold →(5)合成後総量が open_cap 超過時のみ比例縮小
    →(6)縮小後の全モーフへ一律に開口減衰係数を掛ける。開口減衰は手順5より前(hold や暫定重みへの
    乗算)ではなく必ず手順5の後に適用する: 手順5より前に掛けると、全モーフへ一律に掛かる係数のため
    比例縮小の分子・分母から相殺され、縮小が発動する組み合わせで開口減衰の効果が完全に消える。
    """
    weights = _weights_from_hold(
        shape, consonant_class, _hold_value(shape, open_amount, params), params
    )
    scale = _APERTURE_SCALE[aperture_class]
    return {morph: weight * scale for morph, weight in weights.items()}


def _shape_diff(
    shape_a: MouthShape,
    class_a: ConsonantClass,
    shape_b: MouthShape,
    class_b: ConsonantClass,
    params: GenerationParams,
) -> float:
    """両母音的口形(子音変調込み)の口形差(0〜1)。

    各口形の有効プロファイル(子音変調・誇張適用後・保持値非依存の相対重み)を標準口モーフ6次元ベクトル
    (あいうえおん)とし、L2 正規化したうえでユークリッド距離を取り sqrt(2) で割る。同一口形は 0.0、
    互いに重ならない口形方向は 1.0。比較対象は実際に見える口形なので子音変調を含める。
    """
    def _unit(shape: MouthShape, consonant_class: ConsonantClass) -> list[float]:
        eff = _effective_profile(shape, consonant_class, params)
        vec = [eff.get(morph, 0.0) for morph in _VOWEL_ORDER]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0.0 else vec

    ua, ub = _unit(shape_a, class_a), _unit(shape_b, class_b)
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(ua, ub)))
    return dist / math.sqrt(2.0)


def _transition_frames(diff: float, shorter_len: float, params: GenerationParams) -> int:
    """協調調音の遷移長。

    基準長 `coartic_overlap_max` を、下限1・上限 短い側区間長/2 でクランプして整数フレーム化する。
    口形差での短縮はしない(隣接母音の自然な移行には広い遷移窓が要るため、差に依らず基準長まで広く取り、
    短い区間だけ区間長 1/2 で頭打ちする)。`diff` は将来の調整余地として受けるが現状は使わない。
    """
    cap = shorter_len / 2.0
    return _half_up(max(1.0, min(float(params.coartic_overlap_max), cap)))


def _valley_depth(gap_len: float, params: GenerationParams) -> float:
    """レガート間隙長から谷係数 d を線形で求める(間隙が長いほど小さく=深い)。

    d = clamp(shallow − slope·gap_len, deep, shallow)。数値は GenerationParams の初期目安で、
    視覚チューニングで詰める。
    """
    d = params.legato_valley_shallow - params.legato_valley_slope * gap_len
    return min(max(d, params.legato_valley_deep), params.legato_valley_shallow)


def _legato_bridge(events: Sequence[MouthEvent], gap_start: float, gap_end: float) -> bool:
    """区間 [gap_start, gap_end) がレガート間隙(LEGATO_GAP のみで連続被覆)なら True。

    無音・両唇閉鎖を1つでも挟む間隙は閉口優先で谷を作らない(False)。間隙分類は呼び出し側が
    LEGATO_GAP/SILENCE で確定済み(lipsync.md §6)で、本判定はその確定入力を読むだけ。
    """
    if gap_end <= gap_start:
        return False
    span = [ev for ev in events if ev.start >= gap_start and ev.end <= gap_end and ev.start < ev.end]
    if not span or span[0].start != gap_start or span[-1].end != gap_end:
        return False
    return all(ev.shape is MouthShape.LEGATO_GAP for ev in span)


def _vowel_groups(events: Sequence[MouthEvent]) -> list[list[MouthEvent]]:
    """連続する同一母音(shape)のイベントを極大グループへ束ねる。

    同じ母音が続く区間は1つの連続した保持にする(子音種別が途中で変わっても分けない)。子音変調で生じる
    補助モーフ(ROUNDED の う/お、SPREAD の い)はグループ内のイベントごとに異なりうるが、補助はグループ端
    および補助が消えるイベントの中央で 0 へフェードさせる(§4.2 のエンベロープ)ので、同母音の連続を子音種別で
    分断して短いモーラを別々に立て閉口を挟む不自然さ(同じ母音なのに途中で口が閉じる)を避ける。プロファイル
    対象外(両唇閉鎖・無音)はグループ境界として扱い、ここでは出力しない。閉口は隣接母音のリリース/アタックの
    0.0 キーとキー不在(MMD 上 0.0)で表す(専用の閉口キーは設けない)。
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
    short: bool = False  # L<triangle_min: 吸収対象
    triangle: bool = False  # triangle_min≤L<min_hold+2: 三角形ピークで残す
    attack: float = 0.0
    release: float = 0.0


def _opening(event: MouthEvent, params: GenerationParams) -> float:
    """境界イベントの開き量(吸収先タイブレーク用)。

    唇形変調(ConsonantClass)と開口減衰(ApertureClass)の両方を適用した最終重みの最大値を返す。
    コードベース全体で「その口形イベントがどれだけ開くか」を表す量を `_compose` の出力に一本化し、
    同じ性質を表す複数の異なる式を持たない。
    """
    return max(
        _compose(event.shape, event.consonant_class, event.aperture_class, event.open_amount, params).values()
    )


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
    # 長さで3分類: L<triangle_min は吸収(short)、triangle_min≤L<min_hold+2 は三角形(triangle、生存)、
    # それ以上は通常形状。三角形は極短母音を1点ピークで残し、発声中の閉口を防ぐ(§4.9・§4.4)。
    for g in groups:
        L = g.end - g.start
        g.short = L < params.triangle_min_frames
        g.triangle = (not g.short) and max(0.0, L - params.min_hold_frames) < 2
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
            # 境界イベント自身の実効 end も広げる(§4.4)。元のイベントは変更せず複製で差し替える。
            prev_anchor.events[-1] = replace(prev_anchor.events[-1], end=run_end)
            prev_anchor.end = run_end
        elif winner is nxt_anchor and nxt_anchor is not None:
            nxt_anchor.events[0] = replace(nxt_anchor.events[0], start=run_start)
            nxt_anchor.start = run_start
        i = j
    # 吸収後に直接隣接した同一母音(shape)グループを連結へ統合する(子音種別が違っても同母音は連続)。
    merged: list[_Group] = []
    for g in survivors:
        if merged and merged[-1].shape == g.shape and merged[-1].end == g.start:
            merged[-1].events = merged[-1].events + g.events
            merged[-1].end = g.end
        else:
            merged.append(g)
    # 吸収で span が伸びた/連結で結合した最終長で短区間分類をやり直す。連結で通常長になったグループが
    # 三角形フラグを引きずって三角形経路(先頭イベントのみ)へ落ち、後続イベントのプロファイル・補助フェードを
    # 取りこぼすのを防ぐ。
    for g in merged:
        L = g.end - g.start
        g.short = L < params.triangle_min_frames
        g.triangle = (not g.short) and max(0.0, L - params.min_hold_frames) < 2
        if not g.triangle:
            g.attack, g.release = _effective_attack_release(L, params)
    return merged


def _lerp_morph(
    w_prev: dict[str, float], w_next: dict[str, float], t_prev: float, t_next: float, t: float, morph: str
) -> float:
    """2つの小区間の最終重み(モーフ別)を結ぶ直線を時刻 t で評価する(§4.2 の `lerp_i`)。"""
    v_prev, v_next = w_prev.get(morph, 0.0), w_next.get(morph, 0.0)
    if t_next == t_prev:
        return v_prev
    return v_prev + (v_next - v_prev) * (t - t_prev) / (t_next - t_prev)


@dataclass
class _Valley:
    """モーラ境界の谷の候補(§4.2)。`values` はモーフ名→(左肩, 中央, 右肩) の最終重み。"""

    b: float
    hw: int
    aperture_scale: float
    disp: float
    values: dict[str, tuple[float, float, float]]

    @property
    def left(self) -> float:
        return self.b - self.hw

    @property
    def right(self) -> float:
        return self.b + self.hw


def _mora_valley_candidates(
    group: _Group, gw: list[dict[str, float]], group_morphs: Sequence[str], params: GenerationParams
) -> list[_Valley]:
    """通常長グループの内部境界ごとに、谷の候補(ApertureClass が NONE でなく `hw>=2`)を作る。"""
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
        if hw < 2:
            continue
        d = scale / 2.0
        values: dict[str, tuple[float, float, float]] = {}
        disp = 0.0
        for morph in group_morphs:
            left_val = _lerp_morph(gw[j], gw[j + 1], mids[j], mids[j + 1], b - hw, morph)
            right_val = _lerp_morph(gw[j], gw[j + 1], mids[j], mids[j + 1], b + hw, morph)
            center_val = d * (left_val + right_val)
            values[morph] = (left_val, center_val, right_val)
            disp += (1.0 - scale) * _lerp_morph(gw[j], gw[j + 1], mids[j], mids[j + 1], b, morph)
        candidates.append(_Valley(b, hw, scale, disp, values))
    return candidates


def _select_valleys(candidates: Sequence[_Valley], params: GenerationParams) -> list[_Valley]:
    """密集回避: 視覚的な変位量の総和を最大化する部分集合を動的計画法で選ぶ(§4.2)。

    互換性判定は整数フレーム化した両端点で行う(`round(L_j) - round(R_i) - 1 ≥
    mora_valley_min_gap_frames`)。同値の場合は必ず採用する側を選ぶ(候補を右端の昇順に処理する
    決定論的な単一パスなので、複数の最適解が並ぶ曖昧さは生じない)。
    """
    ordered = sorted(candidates, key=lambda c: c.right)
    n = len(ordered)
    q_left = [_half_up(c.left) for c in ordered]
    q_right = [_half_up(c.right) for c in ordered]
    gap = params.mora_valley_min_gap_frames
    pred: list[int] = []  # 1-indexed 互換先(0 は「候補なし」)
    for i in range(n):
        best = 0
        for j in range(i):
            if q_left[i] - q_right[j] - 1 >= gap:
                best = j + 1
        pred.append(best)
    opt = [0.0] * (n + 1)
    take_choice = [False] * (n + 1)
    for i in range(1, n + 1):
        take = ordered[i - 1].disp + opt[pred[i - 1]]
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
            i = pred[i - 1]
        else:
            i -= 1
    return selected


def _preceding_event(events: Sequence[MouthEvent], start: float) -> MouthEvent | None:
    """終端フレームが start に一致する直前イベント(連続契約により一意。無ければ None)。"""
    for ev in events:
        if ev.end == start:
            return ev
    return None


def _following_event(events: Sequence[MouthEvent], end: float) -> MouthEvent | None:
    """始端フレームが end に一致する直後イベント(連続契約により一意。無ければ None)。"""
    for ev in events:
        if ev.start == end:
            return ev
    return None


def _lead_lag_eff(neighbor: MouthEvent | None, opening: float, params: GenerationParams) -> int:
    """先行準備(立ち上がり前倒し)/後行残し(閉じ後ろずらし)の実効量。

    隣接が無音区間のときのみ働く(直前/直後が無い・母音・両唇閉鎖は 0)。基準フレーム `anticipation_frames`
    を開き量比 `opening/open_cap` でスケールするため、機械的な固定値にならず開きの大小で量が変動する
    (大きく開くほど長い余韻)。隣接無音長の 1/2 上限で前/後区間を侵食しすぎない。先行準備と後行残しで対称。
    """
    if neighbor is None or neighbor.shape is not MouthShape.SILENCE:
        return 0
    ratio = opening / params.open_cap if params.open_cap > 0.0 else 0.0
    want = params.anticipation_frames * min(max(ratio, 0.0), 1.0)
    return min(_half_up(want), math.floor((neighbor.end - neighbor.start) / 2))


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


def _consonant_at(events: Sequence[MouthEvent], t: float) -> ConsonantClass:
    """時刻 t を含むイベントの先頭子音種別(無ければ最近傍端のもの)。

    同母音グループ内で子音種別が変わりうる(子音付きモーラ＋継続の伸ばし等)ため、伸び表現の各節点は
    その時刻のイベントの子音種別で変調する。t が(吸収で延長された)端の外側なら最近傍端の値を返す。
    """
    for ev in events:
        if ev.start <= t < ev.end:
            return ev.consonant_class
    # t がイベント区間外(吸収で延長された端の外側・吸収で除去された内部ギャップ)なら最近傍イベント。
    nearest = min(events, key=lambda ev: min(abs(t - ev.start), abs(t - ev.end)))
    return nearest.consonant_class


def _suppressed_by_valley(t: float, sign: float, period: float, valleys: Sequence[_Valley]) -> bool:
    """揺らぎ極値(時刻 t・符号 sign)が、いずれかの谷(常に負方向)との近接で間引かれるべきか。

    谷と逆方向(正)の極値は `|t-b| ≤ max(hw+2, vibrato_period/2)` で、谷と同じ方向(負)の極値は
    `|t-b| ≤ max(hw+2, vibrato_period)` で間引く(§4.2)。マージンは実際にクランプ済みの半幅
    `hw` を使う(パラメータ `mora_valley_frames` そのものは使わない)。
    """
    return any(
        abs(t - v.b) <= max(v.hw + 2, period if sign < 0.0 else period / 2.0) for v in valleys
    )


def _vibrato_targets(
    group: _Group,
    plateau_start: float,
    plateau_end: float,
    params: GenerationParams,
    valleys: Sequence[_Valley] = (),
) -> list[tuple[str, float, float]]:
    """保持プラトーに伸び表現の揺らぎ節点を生成する。

    公称開き量 `base_open(t)` を保持・強弱節点の線形補間で求め、正弦波で変調した実効開き量 `open_v` から
    合成手順(手順1〜5相当)で各モーフ重みを出し、最後に開口減衰 `aperture_v(t)`(同じ制御点で
    `aperture_scale` を線形補間したもの)を一律に掛ける。開口減衰を `open_v` の計算に混ぜず最後に
    別途掛けるのは、合成後総量の比例縮小と相殺させないため(`_compose` と同じ理由)。節点は正弦波の
    極値(`t_k = plateau_start + P·(1/4 + k/2)`)の厳密内側のみ。プラトー内にモーラ境界の谷(§4.2)が
    ある場合、谷と逆方向(正)の極値は半周期、谷と同じ方向(負)の極値は1周期の間引き幅で除外する
    (同符号の二重ディップ・谷直後の近接ディップを防ぐ)。返すのは float 目標
    `(モーフ名, フレーム, 重み)` 列(量子化は後段)。
    """
    holds = [_hold_value(group.shape, ev.open_amount, params) for ev in group.events]
    scales = [_APERTURE_SCALE[ev.aperture_class] for ev in group.events]
    # 公称開き量の制御点: プラトー始端(先頭 hold)・プラトー内の各イベント中央(その hold)・終端(末尾 hold)。
    # 開口減衰の制御点も同じ位置で aperture_scale を使って作る(hold とは独立に補間する)。
    points: list[tuple[float, float]] = [(plateau_start, holds[0])]
    aperture_points: list[tuple[float, float]] = [(plateau_start, scales[0])]
    if len(group.events) >= 2:
        for ev, hold, scale in zip(group.events, holds, scales):
            mid = (ev.start + ev.end) / 2.0
            if plateau_start < mid < plateau_end:
                points.append((mid, hold))
                aperture_points.append((mid, scale))
    points.append((plateau_end, holds[-1]))
    aperture_points.append((plateau_end, scales[-1]))
    points.sort()
    aperture_points.sort()
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
        sign = math.sin(2.0 * math.pi * (t - plateau_start) / period)
        if valleys and _suppressed_by_valley(t, sign, period, valleys):
            continue
        amp_eff = min(params.vibrato_amp, base)
        offset = amp_eff * sign
        open_v = min(max(base + offset, 0.0), params.open_cap)
        # 同母音グループ内で子音種別が変わりうるので、その時刻のイベントの子音種別で変調する。
        cc = _consonant_at(group.events, t)
        aperture_v = _interp_open(aperture_points, t)
        for morph, weight in _weights_from_hold(group.shape, cc, open_v, params).items():
            nodes.append((morph, t, weight * aperture_v))
    return nodes


def generate_morph_keys(
    events: Sequence[MouthEvent], params: GenerationParams
) -> list[MorphKey]:
    """口形イベント列からモーフキー列を生成する(要求仕様 lipsync.md §4)。

    連続する同一母音イベントを1グループへ連結し、長さで3分類(極短は吸収/除去・短いものは三角形ピークで残す・
    通常は競合短縮で実効アタック/リリースを求める)したうえで、グループごとにエンベロープを置く: 通常グループは
    先頭にのみアタック(開始0.0・保持値)、末尾にのみリリース(保持値・終了0.0)、各小区間の中央に開き量の強弱節点を
    置いて節点間を線形に変え、三角形グループは中央に保持値ピーク1点を置く。直接隣接する異母音グループの境界では閉口を挟まず、協調調音(境界 b を中心とした幅 T の
    窓で前母音の保持値から次母音の保持値へ線形クロスフェードし、境界に中間口形を置く)へ置き換える。
    無音に隣接する母音は、先行準備で立ち上がりを無音側へ伸ばして緩やかに開き(音符開始で保持値へ達する)、
    後行残しで閉じを無音側へ伸ばして緩やかに閉じる(開き量比例)。長く伸ばす母音の保持プラトーには伸び表現で
    揺らぎ節点を任意に加える。両唇閉鎖・無音は隣接母音の 0.0 キーとキー不在(MMD 上 0.0)で閉口を表し、専用の
    閉口キーは置かない。整数フレームへの量子化は最後に一括して行う。返すキーは時間順。
    """
    groups = _normalize_groups(events, params)
    weights = [
        [
            _compose(ev.shape, ev.consonant_class, ev.aperture_class, ev.open_amount, params)
            for ev in g.events
        ]
        for g in groups
    ]
    n = len(groups)
    # 直接隣接する異母音グループ境界の協調調音の半幅 T/2(境界 i と i+1 の間)。保持プラトー端の算出にも使う。
    # 三角形短区間も含める(三角形は境界保持値=ピーク相当の合成重みで協調調音に参加し、有声側で閉口しない)。
    coart_half: dict[int, float] = {}
    for i in range(n - 1):
        if groups[i].end != groups[i + 1].start:
            continue
        # 境界のクロスフェードは group i の末尾イベントと group i+1 の先頭イベントを繋ぐので、
        # 口形差もその2イベント(子音変調込み)で測る。
        left, right = groups[i].events[-1], groups[i + 1].events[0]
        diff = _shape_diff(
            left.shape, left.consonant_class, right.shape, right.consonant_class, params
        )
        shorter = min(groups[i].end - groups[i].start, groups[i + 1].end - groups[i + 1].start)
        coart_half[i] = _transition_frames(diff, shorter, params) / 2.0
    # 母音グループ i と i+1 の間がレガート間隙なら、その span [gs, ge] を谷で橋渡しする(閉口しない)。
    # 三角形短区間も含める(三角形のピーク相当の合成重みを境界保持値として谷に参加させる)。
    legato_at: dict[int, tuple[float, float]] = {}
    for i in range(n - 1):
        gs, ge = groups[i].end, groups[i + 1].start
        if _legato_bridge(events, gs, ge):
            legato_at[i] = (gs, ge)
    # 要所キーは (モーフ名, 目標フレーム(float), 重み) の目標値として生成順に集め、最後に一括量子化する。
    targets: list[tuple[str, float, float]] = []
    plateaus: list[tuple[float, float]] = []  # グループごとの保持プラトー [始端, 終端](伸び表現の対象)。
    valleys_by_group: dict[int, list[_Valley]] = {}  # モーラ境界の谷(§4.2)。伸び表現の間引きが参照する。
    # 各グループの保持区間(アタック/リリースは協調調音しない端のみ。中央に強弱節点)。実効スパン・実効 a'/r'。
    for i, g in enumerate(groups):
        gw = weights[i]
        if g.triangle:
            # 三角形短区間: 中央に保持値ピーク1点。極短母音を吸収せず開いて見せる。有声(協調調音・レガート
            # 間隙)に接する側は端点 0 を置かず境界キー(クロスフェード/谷)へ繋ぎ、閉口/曲端に接する側だけ
            # 0 へ閉じる(有声が続く区間内で口を閉じてフリッカーにしない)。内部小区間数によらず単一ピーク
            # へ平滑化し、モーラ境界の谷は適用しない(時間解像度が無いため)。ピーク重みは n≥2 なら各小区間の
            # 長さによる長さ加重平均、n=1 ならその1小区間の最終重み。
            mid = (g.start + g.end) / 2.0
            if len(g.events) >= 2:
                lens = [ev.end - ev.start for ev in g.events]
                total_len = sum(lens)
                peak_morphs = sorted({morph for w in gw for morph in w})
                peak_weight = {
                    morph: sum(length * w.get(morph, 0.0) for length, w in zip(lens, gw)) / total_len
                    for morph in peak_morphs
                }
            else:
                peak_weight = gw[0]
            connect_in = (i - 1) in coart_half or (i - 1) in legato_at
            connect_out = i in coart_half or i in legato_at
            for morph, weight in peak_weight.items():
                if not connect_in:
                    targets.append((morph, g.start, 0.0))
                targets.append((morph, mid, weight))
                if not connect_out:
                    targets.append((morph, g.end, 0.0))
            plateaus.append((mid, mid))
            continue
        # グループ内の全モーフ(子音変調でイベントごとに補助モーフが変わりうる)。各モーフはグループ端で
        # 0 へアンカーし、補助が無いイベントの中央では 0 を置くことで、同母音の連続を保ったまま補助モーフ
        # (ROUNDED の う/お・SPREAD の い)だけを滑らかに増減させる(残留・途中閉口を防ぐ)。
        group_morphs = sorted({morph for w in gw for morph in w})
        # 協調調音する境界(三角形隣接も含む)は coart_half の有無で判定する。
        coart_in = (i - 1) in coart_half
        coart_out = i in coart_half
        legato_in = (i - 1) in legato_at  # 直前グループとの間がレガート間隙(谷が立ち上がりを担う)
        legato_out = i in legato_at  # 次グループとの間がレガート間隙(谷が立ち下がりを担う)
        if coart_in:
            plateau_start = groups[i - 1].end + coart_half[i - 1]
        elif legato_in:
            # 谷が onset を担うため先頭アタックの 0.0/到達キーは置かない。
            plateau_start = g.start
        else:
            # 先行準備: 直前が無音なら口形を A_eff フレーム手前から緩やかに立ち上げ、音符開始で保持値へ達する
            # (開き量比例)。先行が無ければ通常アタック。
            antic = _lead_lag_eff(_preceding_event(events, g.start), max(gw[0].values()), params)
            if antic > 0:
                f_start, f_attack = g.start - antic, g.start
            else:
                f_start, f_attack = g.start, g.start + g.attack
            for morph in group_morphs:
                targets.append((morph, f_start, 0.0))
                targets.append((morph, f_attack, gw[0].get(morph, 0.0)))
            plateau_start = f_attack
        if coart_out:
            plateau_end = g.end - coart_half[i]
        elif legato_out:
            # 谷が offset を担うため末尾リリースの保持/0.0 キーは置かない。
            plateau_end = g.end
        else:
            # 後行残し: 直後が無音なら音符終了まで保持し、その後 R_eff フレームかけて緩やかに閉じる
            # (先行準備と対称・開き量比例)。直後が無音でなければ通常リリース。
            lag = _lead_lag_eff(_following_event(events, g.end), max(gw[-1].values()), params)
            if lag > 0:
                f_hold_end, f_end = g.end, g.end + lag
            else:
                f_hold_end, f_end = g.end - g.release, g.end
            for morph in group_morphs:
                targets.append((morph, f_hold_end, gw[-1].get(morph, 0.0)))
                targets.append((morph, f_end, 0.0))
            plateau_end = f_hold_end
        if len(g.events) >= 2:
            for ev, w in zip(g.events, gw):
                f_mid = (ev.start + ev.end) / 2.0
                for morph in group_morphs:
                    targets.append((morph, f_mid, w.get(morph, 0.0)))
            # モーラ境界の谷(§4.2): ApertureClass が NONE でなく半幅が2フレーム以上の内部境界を候補にし、
            # 密集回避(動的計画法)で採用された谷だけを3点キーとして追加する。
            candidates = _mora_valley_candidates(g, gw, group_morphs, params)
            selected = _select_valleys(candidates, params)
            if selected:
                valleys_by_group[i] = selected
                for valley in selected:
                    for morph in group_morphs:
                        left_val, center_val, right_val = valley.values[morph]
                        targets.append((morph, valley.left, left_val))
                        targets.append((morph, valley.b, center_val))
                        targets.append((morph, valley.right, right_val))
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
    # レガート間隙(あ→閉じかけ→う)の谷橋渡し: 完全閉口でなく前後母音の口形を中央で重ねる(オーバーラップ)。
    # 前母音の境界保持値 w_a から、谷値 d·(w_a+w_b)(両母音を加算で重ねた値。d で部分的な閉じ=閉じかけへ抑制)を
    # 経て、次母音の境界保持値 w_b へ線形に繋ぐ。隣接の自然な移行(§4.3 のクロスフェード)と違い、ここでは加算で
    # 重ねるが、d<1 が総開き量を抑えるので開きすぎない。前後母音の 0.0 リリース/アタックキーは上で抑制済み。
    for i, (gs, ge) in legato_at.items():
        wa, wb = weights[i][-1], weights[i + 1][0]
        gm = (gs + ge) / 2.0
        depth = _valley_depth(ge - gs, params)
        for morph in _VOWEL_ORDER:
            a, b = wa.get(morph, 0.0), wb.get(morph, 0.0)
            if a == 0.0 and b == 0.0:
                continue
            targets.append((morph, gs, a))
            targets.append((morph, gm, depth * (a + b)))
            targets.append((morph, ge, b))
    # 伸び表現: 公称エンベロープ・強弱・協調調音の後に、長い保持プラトーへ揺らぎ節点を加える(任意)。
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
