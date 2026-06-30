"""範囲端の継ぎ目処理のテスト(sparsevmd.md §6.3)。

--range で一部だけを処理し範囲外キーを逐語保持する場合、範囲端の到達側補間曲線が
範囲外区間に影響する。方針(§6.3): 範囲外キーは一切変更しない(逐語保持)。

- 下側継ぎ目: 範囲開始キー f0(範囲内)の到達側曲線は、手前の範囲外キー B から f0 までの
  区間 [B, f0] を支配する。f0 は範囲内キーなので、その曲線を元サンプルから再フィットして
  [B, f0] の元の動きを忠実に保つ(bezier)。範囲外キーは変更しない。
- 上側継ぎ目: 範囲終了 f1 の直後の範囲外キー C の到達側曲線は [f1, C] を支配するが、C は
  範囲外キーなので変更不可。C は逐語保持し、書き換えない([f1, C] の不整合は許容)。

範囲全体がトラックを覆う(範囲外キーなし)場合と linear モードでは継ぎ目の書き換えをしない。
"""

import pytest

from vmd import interp
from vmd.types import CameraKey
from vmd.reduce import (
    Tolerances,
    camera_interp_bytes,
    reduce_camera_track,
)

# balanced プリセット相当の許容(sparsevmd.presets の balanced 値)。
TOLS = Tolerances(
    bone_pos=0.01,
    bone_rot=0.10,
    camera_pos=0.02,
    camera_rot=0.05,
    camera_distance=0.02,
    camera_fov=0.50,
)
CAM_CUT = (5.0, 20.0, 5.0)
EASE = (96, 0, 96, 30)
LIN = (20, 20, 107, 107)
CAM_LINEAR = bytes([20, 107, 20, 107]) * 6


def _blk(eased_posx):
    # pos_x 軸だけ eased / それ以外は線形のカメラ補間ブロック。
    return camera_interp_bytes(EASE if eased_posx else LIN, LIN, LIN, LIN, LIN, LIN)


def cam(frame, x, eased=False):
    return CameraKey(frame, -30.0, (x, 0.0, 0.0), (0.0, 0.0, 0.0), _blk(eased), 30, 0)


def camera_track(source, ranges, **kw):
    opts = dict(
        cut_thresholds=CAM_CUT,
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
    )
    opts.update(kw)
    return reduce_camera_track(source, ranges, TOLS, **opts)


def _max_seam_dev(out, src, lo, hi):
    return max(
        abs(interp.sample(out, "pos_x", f) - interp.sample(src, "pos_x", f))
        for f in range(lo, hi + 1)
    )


def test_lower_seam_curve_preserved():
    # key10 の pos_x は [0,10] が eased。range[10,20] を処理 → frame0 は範囲外で保持。
    # 継ぎ目 [0,10] は出力 key10 の到達曲線が支配する。§6.3 で曲線を再フィットすれば忠実。
    src = [cam(0, 0.0), cam(10, 100.0, eased=True), cam(20, 100.0)]
    out = camera_track(src, [(10, 20)])
    assert _max_seam_dev(out, src, 0, 10) <= TOLS.camera_pos
    # 出力 key10 の pos_x 到達曲線は線形ブロックでなく書き換えられている。
    k10 = next(k for k in out if k.frame == 10)
    assert k10.interpolation[0:4] != bytes([20, 107, 20, 107])


def test_upper_seam_out_of_range_key_unchanged():
    # 範囲外キー(frame30)は逐語保持し、補間ブロックを含め一切変更しない(レンジ外変更不可)。
    # 上側継ぎ目 [20,30] の到達曲線は範囲外キー key30 に乗るため、これを書き換えると範囲外
    # キーの変更になる。よって書き換えず key30 はソースと完全一致させる(継ぎ目の不整合は許容)。
    src = [cam(0, 0.0), cam(10, 0.0), cam(30, 100.0, eased=True)]
    out = camera_track(src, [(0, 20)])
    k30 = next(k for k in out if k.frame == 30)
    assert k30 == src[2]


def test_no_seam_when_range_covers_track():
    # 範囲がトラック全体を覆う(範囲外キーなし) → 継ぎ目処理は発生しない。
    # 先頭キーは線形のまま(到達側に範囲外区間がない)。
    src = [cam(0, 0.0), cam(10, 100.0, eased=True), cam(20, 100.0)]
    out = camera_track(src, [(0, 20)])
    k0 = next(k for k in out if k.frame == 0)
    assert k0.interpolation == CAM_LINEAR


def test_linear_mode_no_seam_rewrite():
    # linear モードでは継ぎ目曲線の書き換えをしない(線形ブロック固定)。
    src = [cam(0, 0.0), cam(10, 100.0, eased=True), cam(20, 100.0)]
    out = camera_track(src, [(10, 20)], curve_mode="linear")
    k10 = next(k for k in out if k.frame == 10)
    assert k10.interpolation == CAM_LINEAR
