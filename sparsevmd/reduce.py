"""区間分割・キー削減の全体制御(sparsevmd.md §5.1, §5.5, §2.5)。

必須境界(範囲端・不連続・keep-frame。cuts.py 由来)の間を区間化し、各区間を全
チャンネルが許容誤差以内で表現できるか検査する。超過時は最大正規化誤差フレームで
再帰分割する。max-segment-frames で事前分割し、min-segment-frames を下回る tol 探索
分割はしない。非strictでは min-segment まで分割しても満たせない区間を1フレームまで
密に保持し(§1.2 の破綻回避)、strictでは StrictError(終了コード4)を送出する。

チャンネルは normalized(a, b) -> (正規化誤差, 最大誤差フレーム|None) を持つダックタイプ。
"""

import math


class StrictError(Exception):
    """--strict 指定時に許容誤差を満たせない(§2.5、終了コード4)。"""


def reduce_track(boundaries, channels, min_seg, max_seg, strict):
    """必須境界とチャンネル群から、出力キーのフレーム列(昇順)を返す(§5.1)。"""
    bounds = sorted(set(boundaries))
    presplit = _presplit(bounds, max_seg)

    keys = set(presplit)
    for a, b in zip(presplit, presplit[1:]):
        _process_segment(a, b, channels, min_seg, strict, keys)
    return sorted(keys)


def _presplit(bounds, max_seg):
    """必須境界間を max_seg 以下に事前分割する(§5.1 step3)。"""
    out = [bounds[0]]
    for a, b in zip(bounds, bounds[1:]):
        span = b - a
        if span > max_seg:
            pieces = math.ceil(span / max_seg)
            for i in range(1, pieces):
                out.append(a + round(i * span / pieces))
        out.append(b)
    # 丸めの衝突を除去(昇順は構成上保たれる)。
    dedup = []
    for x in out:
        if not dedup or dedup[-1] != x:
            dedup.append(x)
    return dedup


def _worst_channel(a, b, channels):
    """区間 [a,b] の (採否用の最大正規化誤差, 分割フレーム) を返す。

    採否は全チャンネルの最大正規化誤差(分割フレームを持たないチャンネルも含む)で
    判定する。分割フレームは、フレームを報告するチャンネルのうち正規化誤差が最大の
    ものを採る。最大誤差が許容超過でも分割フレームが得られない場合は None を返し、
    呼び出し側で atomic 扱いにする。
    """
    max_norm = 0.0
    split_norm = 0.0
    split_frame = None
    for ch in channels:
        ne, frame = ch.normalized(a, b)
        if ne > max_norm:
            max_norm = ne
        if frame is not None and ne > split_norm:
            split_norm = ne
            split_frame = frame
    return max_norm, split_frame


def _process_segment(a, b, channels, min_seg, strict, keys):
    """区間 [a,b] を再帰的に処理し、必要なキーを keys に加える。"""
    stack = [(a, b)]
    while stack:
        a, b = stack.pop()
        if b - a <= 1:
            continue  # 隣接区間は内部点が無く受理(両端は既にキー)

        max_norm, split_frame = _worst_channel(a, b, channels)
        if max_norm <= 1.0:
            continue  # 全チャンネル許容内

        # min_seg を下回る、または両側を min_seg で割れない区間はこれ以上 tol 分割不可。
        # 分割フレームが得られない場合も同様に atomic 扱い。
        if b - a < 2 * min_seg or split_frame is None:
            _atomic_fail(a, b, strict, keys)
            continue

        # 分割候補フレームを [a+min_seg, b-min_seg] に収める(min_seg を下回る区間を作らない)。
        w = min(max(split_frame, a + min_seg), b - min_seg)
        keys.add(w)
        stack.append((a, w))
        stack.append((w, b))


def _atomic_fail(a, b, strict, keys):
    """min_seg 下限に達しても許容を満たせない区間の処理(§2.5)。"""
    if strict:
        raise StrictError(f"許容誤差を満たせない区間: [{a}, {b}]")
    # 非strict: 下限を無視し1フレームまで密に保持(§1.2 の破綻回避)。
    for f in range(a + 1, b):
        keys.add(f)
