"""force_bezier: 共有ファストパスのオプトアウト。

線形ファストパス/cheap accept は採否・キー数を変えないが、許容内に収まる区間の出力曲線を
線形へ寄せる。曲線形状の忠実度が要る呼び出し側向けに、各カメラチャンネルと
reduce_camera_track へ force_bezier フラグを設け、True でファストパスを切り least_squares
由来の実ベジェ曲線を強制する。既定 False は従来挙動(ファストパス有効)。

「線形でも許容内だが実際は曲がっている」= eased データ + 大きい許容、で対比する。各テストは
まず既定挙動(ファストパスで線形)を確認したのち、force_bezier=True で非線形化することを見る。
"""

import math

from vmd import interp
from vmd.types import CameraKey
from vmd.fit import (
    CameraRotationChannel,
    EuclideanVectorChannel,
    FovChannel,
    LinearScalarChannel,
    read_fit_counters,
    reset_fit_counters,
)
from vmd.reduce import (
    CAMERA_LINEAR_INTERP,
    Tolerances,
    camera_interp_bytes,
    reduce_camera_track,
)

EASE = (96, 0, 96, 30)
LIN = (20, 20, 107, 107)
CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
CAM_CUT = (5.0, 20.0, 5.0)
_BIG = 1e6  # 線形が必ず許容内に収まる大許容(既定ファストパス発火を保証)
BIG_TOLS = Tolerances(
    bone_pos=_BIG, bone_rot=_BIG, camera_pos=_BIG, camera_rot=_BIG,
    camera_distance=_BIG, camera_fov=_BIG,
)


def _eased(v0, v1, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*EASE, f / span) for f in range(n)]


def _coeff(n=11):
    span = n - 1
    return [interp._solve_factor(*EASE, f / span) for f in range(n)]


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), interp_block=None):
    return CameraKey(frame, dist, center, rot, interp_block or CAM_LINEAR, 30, 0)


def camera_track(source, ranges, tols=BIG_TOLS, **kw):
    opts = dict(
        cut_thresholds=CAM_CUT, keep_frames=[], no_cut_detect=True,
        min_seg=1, max_seg=180, strict=False, curve_mode="bezier",
    )
    opts.update(kw)
    return reduce_camera_track(source, ranges, tols, **opts)


def _assert_real_fit(counts):
    # 両ファストパスを切り least_squares 由来の実フィットを回したことを直接確認する。
    # !=LIN だけでは「線形は切るが cheap accept は残す」実装(最初の cheap CP は LIN でない)を
    # 取り逃がすため、fastpath_linear と cheap_accept がともに不発で lsq_calls>0 であることを見る。
    assert counts["lsq_calls"] > 0
    assert counts["fastpath_linear"] == 0
    assert counts["cheap_accept"] == 0


# --- 各カメラチャンネルの force_bezier(curve が非線形 cp を返す) ----------------


def test_scalar_channel_force_bezier_opts_out_fastpath():
    vals = _eased(0.0, 100.0)
    assert LinearScalarChannel(0, vals, tol=_BIG, mode="bezier").curve(0, 10) == LIN
    reset_fit_counters()
    ch = LinearScalarChannel(0, vals, tol=_BIG, mode="bezier", force_bezier=True)
    assert ch.curve(0, 10) != LIN
    _assert_real_fit(read_fit_counters())


def test_fov_channel_force_bezier_opts_out_fastpath():
    vals = _eased(30.0, 80.0)
    assert FovChannel(0, vals, tol=_BIG, mode="bezier").curve(0, 10) == LIN
    reset_fit_counters()
    ch = FovChannel(0, vals, tol=_BIG, mode="bezier", force_bezier=True)
    assert ch.curve(0, 10) != LIN
    _assert_real_fit(read_fit_counters())


def test_euclidean_channel_force_bezier_opts_out_fastpath():
    ys = _eased(0.0, 100.0)
    vecs = [(0.0, y, 0.0) for y in ys]  # Y 軸だけ eased
    assert EuclideanVectorChannel(0, vecs, tol=_BIG, mode="bezier").curve(0, 10)[1] == LIN
    reset_fit_counters()
    ch = EuclideanVectorChannel(0, vecs, tol=_BIG, mode="bezier", force_bezier=True)
    assert ch.curve(0, 10)[1] != LIN
    _assert_real_fit(read_fit_counters())


def test_camera_rotation_channel_force_bezier_opts_out_fastpath():
    # 回転は _fit_coeff_curve 経由(位置/距離/FOV と別経路)なので個別に検証。
    c = _coeff()
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    eulers = [(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f]) for f in range(11)]
    assert CameraRotationChannel(0, eulers, tol=_BIG, mode="bezier").curve(0, 10) == LIN
    reset_fit_counters()
    ch = CameraRotationChannel(0, eulers, tol=_BIG, mode="bezier", force_bezier=True)
    assert ch.curve(0, 10) != LIN
    _assert_real_fit(read_fit_counters())


# --- reduce_camera_track の force_bezier 伝播 ---------------------------------


def test_reduce_camera_force_bezier_non_constant_not_linear_interp():
    # eased な距離+回転、大許容 → 既定はファストパスで線形 interp に退化。force_bezier で非線形化。
    c = _coeff()
    dist = _eased(-10.0, -110.0)
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    source = [
        cam(f, dist=dist[f], rot=(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f]))
        for f in range(11)
    ]
    base = camera_track(source, [(0, 10)])
    assert all(k.interpolation == CAMERA_LINEAR_INTERP for k in base)  # 既定=ファストパス退化
    reset_fit_counters()
    fb = camera_track(source, [(0, 10)], force_bezier=True)
    assert fb[-1].interpolation != CAMERA_LINEAR_INTERP
    assert fb[-1].interpolation[12:16] != bytes([20, 107, 20, 107])  # 回転
    assert fb[-1].interpolation[16:20] != bytes([20, 107, 20, 107])  # 距離
    _assert_real_fit(read_fit_counters())  # reduce 全体でファストパスが一切発火しないこと


def test_reduce_camera_force_bezier_false_keeps_fastpath_degradation():
    # force_bezier=False(既定)は従来どおりファストパスで線形 interp に退化したまま(回帰ガード)。
    dist = _eased(-10.0, -110.0)
    source = [cam(f, dist=dist[f]) for f in range(11)]
    out = camera_track(source, [(0, 10)])  # 既定(force_bezier 未指定)
    assert all(k.interpolation == CAMERA_LINEAR_INTERP for k in out)
    out_false = camera_track(source, [(0, 10)], force_bezier=False)
    assert [k.interpolation for k in out_false] == [k.interpolation for k in out]


# --- 範囲端の継ぎ目再フィット(_camera_seam_interp)の force_bezier ---------------


def _blk_eased_posx():
    # pos_x 軸だけ eased / それ以外は線形のカメラ補間ブロック。
    return camera_interp_bytes(EASE, LIN, LIN, LIN, LIN, LIN)


def test_seam_refit_force_bezier_opts_out_fastpath():
    # 範囲開始キーの継ぎ目曲線(_camera_seam_interp)も force_bezier で非線形化する。
    # range[10,20] を処理 → frame0 は範囲外で保持。継ぎ目 [0,10] は出力 key10 の到達曲線が支配。
    # source key10 の pos_x が eased なので [0,10] のサンプルは曲がっている。
    src = [
        cam(0, center=(0.0, 0.0, 0.0)),
        cam(10, center=(100.0, 0.0, 0.0), interp_block=_blk_eased_posx()),
        cam(20, center=(100.0, 0.0, 0.0)),
    ]
    base = camera_track(src, [(10, 20)])  # 大許容 → 継ぎ目はファストパスで線形に退化
    k10 = next(k for k in base if k.frame == 10)
    assert k10.interpolation[0:4] == bytes([20, 107, 20, 107])  # pos_x 継ぎ目が線形(退化)
    reset_fit_counters()
    fb = camera_track(src, [(10, 20)], force_bezier=True)
    k10fb = next(k for k in fb if k.frame == 10)
    assert k10fb.interpolation[0:4] != bytes([20, 107, 20, 107])  # 継ぎ目が非線形化
    _assert_real_fit(read_fit_counters())  # 継ぎ目再フィットもファストパスを切る
