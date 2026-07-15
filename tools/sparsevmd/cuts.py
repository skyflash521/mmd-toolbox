"""カット閾値パース + 不連続検出の後方互換 re-export。

CLI 層の閾値文字列パース(`POS,ROT,DIST` / `POS,ROT`)は sparsevmd 固有として本モジュールに
残す。不連続検出・必須境界の本体(detect_cuts_* / perspective_cut_frames / assemble_boundaries
と角度ヘルパ)は共通ライブラリ vmd.cuts にあるので、旧 import パス
(`sparsevmd.cuts`)維持のため同名で再公開する。
"""

import math

from vmd.cuts import (  # noqa: F401
    _axis_angle_deg,
    _quat_angle_deg,
    assemble_boundaries,
    detect_cuts_bone,
    detect_cuts_camera,
    perspective_cut_frames,
)


def parse_cut_threshold_camera(text):
    """`POS,ROT,DIST` を (pos, rot, dist) に解析する。"""
    return _parse_thresholds(text, 3)


def parse_cut_threshold_bone(text):
    """`POS,ROT` を (pos, rot) に解析する。"""
    return _parse_thresholds(text, 2)


def _parse_thresholds(text, n):
    parts = text.split(",")
    if len(parts) != n:
        raise ValueError(f"閾値は {n} 個のカンマ区切り: {text!r}")
    vals = []
    for p in parts:
        if p == "" or p != p.strip():
            raise ValueError(f"閾値に空要素・空白は不可: {text!r}")
        # 数値変換は float() を用いる(shakevmd の _finite_float と同じ規約)。
        # 符号付き・指数表記等の float() が受理する形式は許容し、非有限・負値のみ弾く。
        # より厳格な書式制限はプロジェクト全体の数値CLIパース方針として別途扱う。
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"閾値は数値: {text!r}")
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"閾値は0以上の有限値: {text!r}")
        vals.append(v)
    return tuple(vals)
