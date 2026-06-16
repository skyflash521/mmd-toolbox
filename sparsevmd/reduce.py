"""区間分割・キー削減の全体制御(sparsevmd.md §5.1, §5.5, §2.5)。

必須境界(範囲端・不連続・keep-frame。cuts.py 由来)の間を区間化し、各区間を全
チャンネルが許容誤差以内で表現できるか検査する。超過時は最大正規化誤差フレームで
再帰分割する。max-segment-frames で事前分割し、min-segment-frames を下回る tol 探索
分割はしない。非strictでは min-segment まで分割しても満たせない区間を1フレームまで
密に保持し(§1.2 の破綻回避)、strictでは StrictError(終了コード4)を送出する。

チャンネルは normalized(a, b) -> (正規化誤差, 最大誤差フレーム|None) を持つダックタイプ。
"""

import math

from mmd_toolbox.vmd import interp
from mmd_toolbox.vmd.types import BoneKey, CameraKey

from .fit import _round_half_up
from .sample import perspective_series

# linear mode の補間ブロック(真の線形: 各チャンネル x1==y1, x2==y2)。
CAMERA_LINEAR_INTERP = bytes([20, 107, 20, 107]) * 6


def bone_interp_bytes(x_cp, y_cp, z_cp, r_cp):
    """4チャンネルの制御点 (x1,y1,x2,y2) からボーン補間64バイトを組み立てる。

    docs/specs/vmd/VMD_file_format.md のレイアウトに従い、先頭16バイト
    (Byte[0]〜Byte[15])に本体を置き、Byte[16]以降は先頭シーケンスを1バイトずつ
    左シフトしたコピーを格納する。これにより Byte[2]/[3] が物理フラグで上書きされても
    シフトコピー側から制御点を復元できる(vmd-io.md §2.2)。詰めパッドは現行MMDに倣い0。
    """
    first = [
        x_cp[0], y_cp[0], z_cp[0], r_cp[0],  # x1
        x_cp[1], y_cp[1], z_cp[1], r_cp[1],  # y1
        x_cp[2], y_cp[2], z_cp[2], r_cp[2],  # x2
        x_cp[3], y_cp[3], z_cp[3], r_cp[3],  # y2
    ]
    b = bytearray(64)
    b[0:16] = bytes(first)
    b[16:31] = bytes(first[1:16])
    b[32:46] = bytes(first[2:16])
    b[48:61] = bytes(first[3:16])
    # 詰めパッド Byte[31]/[46]/[47]/[61]/[62]/[63] は0のまま(版依存・非検証)。
    return bytes(b)


_LINEAR_CP = (20, 20, 107, 107)
BONE_LINEAR_INTERP = bone_interp_bytes(_LINEAR_CP, _LINEAR_CP, _LINEAR_CP, _LINEAR_CP)


def build_camera_keys(source_keys, frames):
    """削減後フレーム列からカメラ出力キーを生成する(§3.2, §4.2)。

    各フレームで位置・回転(Euler)・距離・視野角(整数度へ四捨五入)・perspective
    (直近ホールド)をソースキーからサンプリングし、線形補間ブロックを付与する。
    """
    keys = []
    for f in sorted(frames):
        keys.append(
            CameraKey(
                frame=f,
                distance=interp.sample(source_keys, "distance", f),
                position=(
                    interp.sample(source_keys, "pos_x", f),
                    interp.sample(source_keys, "pos_y", f),
                    interp.sample(source_keys, "pos_z", f),
                ),
                rotation=interp.sample(source_keys, "rot", f),
                interpolation=CAMERA_LINEAR_INTERP,
                fov=_round_half_up(interp.sample(source_keys, "fov", f)),
                perspective=perspective_series(source_keys, f, f)[0],
            )
        )
    return keys


def build_bone_keys(source_keys, frames):
    """削減後フレーム列からボーン出力キーを生成する(§3.2)。

    name_raw はソースの生バイトを保持する。各フレームで位置・回転(quaternion)を
    サンプリングし、線形補間ブロックを付与する。
    """
    name_raw = source_keys[0].name_raw
    keys = []
    for f in sorted(frames):
        keys.append(
            BoneKey(
                name_raw=name_raw,
                frame=f,
                position=(
                    interp.sample(source_keys, "pos_x", f),
                    interp.sample(source_keys, "pos_y", f),
                    interp.sample(source_keys, "pos_z", f),
                ),
                rotation=interp.sample(source_keys, "rot", f),
                interpolation=BONE_LINEAR_INTERP,
            )
        )
    return keys


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
