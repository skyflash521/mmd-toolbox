"""lipsync コアのモーフキー生成(lipsync.md §4, implementation-plan.md §4.1/§4.7)。

入力(開き量を同梱した口形イベント列＋生成パラメータ)から、標準口モーフ
(あ・い・う・え・お)のモーフキー列を決定論的に生成する。VMD への組み立て・
書き出しは呼び出し側が `mmd_toolbox.vmd` 経由で行う。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def generate_morph_keys(
    events: Sequence[MouthEvent], params: GenerationParams
) -> list[MorphKey]:
    """口形イベント列からモーフキー列を生成する(implementation-plan.md §4.7/§4.9)。

    各母音イベントを §4.1 の合成プロファイルで複数の口モーフへ展開し、§4.9 の形状
    エンベロープ(各モーフに 開始0.0・アタックで保持値・リリース直前まで保持値・終了0.0 の
    4点)で配置する。協調調音・同母音連結・きびきび遷移・量子化などは後続ステップで段階的に
    加える。両唇閉鎖・無音の閉口キーは後続ステップ(L-8)で置く。返すキーは時間順(§4.7)。
    """
    keys: list[MorphKey] = []
    for ev in events:
        if ev.shape not in _PROFILES:
            continue
        weights: Mapping[str, float] = _compose(ev.shape, ev.open_amount, params)
        # §4.9 のエンベロープ目標位置(整数量子化は L-0/L-1 と同様 round。厳密化は §4.5/L-9)。
        f_start = round(ev.start)
        f_attack = round(ev.start + params.attack_frames)
        f_release = round(ev.end - params.release_frames)
        f_end = round(ev.end)
        for morph, weight in weights.items():
            keys.append(_morph_key(morph, f_start, 0.0))
            keys.append(_morph_key(morph, f_attack, weight))
            keys.append(_morph_key(morph, f_release, weight))
            keys.append(_morph_key(morph, f_end, 0.0))
    keys.sort(key=lambda k: k.frame)
    return keys
