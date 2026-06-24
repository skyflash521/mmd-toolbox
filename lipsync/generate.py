"""lipsync コアのモーフキー生成(lipsync.md §4, implementation-plan.md §4.7)。

入力(開き量を同梱した口形イベント列＋生成パラメータ)から、標準口モーフ
(あ・い・う・え・お)のモーフキー列を決定論的に生成する。VMD への組み立て・
書き出しは呼び出し側が `mmd_toolbox.vmd` 経由で行う。
"""

from __future__ import annotations

from collections.abc import Sequence

from mmd_toolbox.vmd import MorphKey

from .types import GenerationParams, MouthEvent, MouthShape

# 母音種別 → 標準口モーフ名(lipsync.md §3, implementation-plan.md §4.1)。
_VOWEL_MORPH = {
    MouthShape.A: "あ",
    MouthShape.I: "い",
    MouthShape.U: "う",
    MouthShape.E: "え",
    MouthShape.O: "お",
}


def _morph_key(name: str, frame: int, weight: float) -> MorphKey:
    """標準口モーフ名を cp932・15バイト固定の name_raw に符号化したキーを作る。"""
    return MorphKey(name.encode("cp932").ljust(15, b"\x00"), frame, weight)


def generate_morph_keys(
    events: Sequence[MouthEvent], params: GenerationParams
) -> list[MorphKey]:
    """口形イベント列からモーフキー列を生成する(implementation-plan.md §4.7)。

    現状は L-0 の足場で、各母音イベントに対しその母音の口モーフへ1キーを置く。
    合成プロファイル・形状・協調調音・量子化などは後続ステップで段階的に加える。
    `params` は後続ステップで参照する。
    """
    keys: list[MorphKey] = []
    for ev in events:
        morph = _VOWEL_MORPH.get(ev.shape)
        if morph is None:
            # 両唇閉鎖・無音の閉口キーは後続ステップで置く。
            continue
        keys.append(_morph_key(morph, round(ev.start), ev.open_amount))
    return keys
