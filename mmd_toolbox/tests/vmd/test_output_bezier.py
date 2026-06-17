"""出力のベジェ再構築テスト(sparsevmd.md §3.2, §4.2, §5.3)。

curve_mode="bezier" のとき、各出力区間 [前キー, 当キー] についてチャンネル別の
ベジェ曲線を再フィットし、到達側(後側)キーの補間バイトに制御点を格納する。
カメラ補間は 6 軸 24 バイト(各軸 ax,bx,ay,by = x1,x2,y1,y2)、ボーンは
bone_interp_bytes のシフトコピー 64 バイト。回転はカメラが 3 軸共通の 1 曲線、
ボーンが slerp 係数の 1 曲線。

出力キーを mmd_toolbox.vmd.interp.sample で再サンプルすると、格納した曲線で元の
イージング動作が許容内に復元される(評価器は採否時と同じ _solve_factor を使う)。
"""

import math

import pytest

from mmd_toolbox.vmd import interp
from mmd_toolbox.vmd.types import BoneKey, CameraKey
from mmd_toolbox.vmd.fit import (
    BoneRotationChannel,
    CameraRotationChannel,
    EuclideanVectorChannel,
    FovChannel,
    LinearScalarChannel,
)
from mmd_toolbox.vmd.reduce import (
    CAMERA_LINEAR_INTERP,
    Tolerances,
    camera_interp_bytes,
    reduce_bone_track,
    reduce_camera_track,
)

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
EASE = (96, 0, 96, 30)
_LINEAR_CP = (20, 20, 107, 107)


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()
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
BONE_CUT = (1.0, 30.0)


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, persp)


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, rot, BL)


def _eased(v0, v1, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*EASE, f / span) for f in range(n)]


def _coeff(n=11):
    span = n - 1
    return [interp._solve_factor(*EASE, f / span) for f in range(n)]


def _z_quat(deg):
    h = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(h), math.cos(h))


def frames(keys):
    return [k.frame for k in keys]


def camera_track(source, ranges, tols=TOLS, **kw):
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
    return reduce_camera_track(source, ranges, tols, **opts)


def bone_track(source, ranges, **kw):
    opts = dict(
        cut_thresholds=BONE_CUT,
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
    )
    opts.update(kw)
    return reduce_bone_track(source, ranges, TOLS, **opts)


# --- camera_interp_bytes レイアウト -----------------------------------------


def test_camera_interp_bytes_layout():
    # 6 軸それぞれ cp=(x1,y1,x2,y2) を ax,bx,ay,by=(x1,x2,y1,y2) 順で 4 バイト。
    b = camera_interp_bytes(
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
        (21, 22, 23, 24),
    )
    assert len(b) == 24
    assert list(b[0:4]) == [1, 3, 2, 4]      # X位置: ax=x1,bx=x2,ay=y1,by=y2
    assert list(b[4:8]) == [5, 7, 6, 8]      # Y位置
    assert list(b[8:12]) == [9, 11, 10, 12]  # Z位置
    assert list(b[12:16]) == [13, 15, 14, 16]  # 回転
    assert list(b[16:20]) == [17, 19, 18, 20]  # 距離
    assert list(b[20:24]) == [21, 23, 22, 24]  # 視野角


def test_camera_interp_bytes_linear_matches_constant():
    b = camera_interp_bytes(*([_LINEAR_CP] * 6))
    assert b == CAMERA_LINEAR_INTERP


# --- channel.curve ----------------------------------------------------------


def test_scalar_curve_linear_mode_returns_linear_cp():
    ch = LinearScalarChannel(0, _eased(0.0, 100.0), tol=1.0, mode="linear")
    assert ch.curve(0, 10) == _LINEAR_CP


def test_scalar_curve_bezier_recovers_ease():
    ch = LinearScalarChannel(0, _eased(0.0, 100.0), tol=1.0, mode="bezier")
    assert ch.curve(0, 10) == EASE


def test_scalar_curve_bezier_flat_segment_is_linear():
    ch = LinearScalarChannel(0, [0.0, 1.0, -1.0, 0.0, 0.0], tol=1.0, mode="bezier")
    # 端点同値(0→0)は正規化不能なので線形扱い。
    assert ch.curve(0, 4) == _LINEAR_CP


def test_fov_curve_bezier_recovers_ease():
    ch = FovChannel(0, _eased(30.0, 80.0), tol=1.0, mode="bezier")
    assert ch.curve(0, 10) == EASE


def test_euclidean_curve_returns_three_axis_cps():
    ys = _eased(0.0, 100.0)
    vecs = [(0.0, y, 0.0) for y in ys]
    ch = EuclideanVectorChannel(0, vecs, tol=1.0, mode="bezier")
    cps = ch.curve(0, 10)
    assert len(cps) == 3
    assert cps[0] == _LINEAR_CP   # X 軸は不動 → 線形
    assert cps[1] == EASE         # Y 軸は ease
    assert cps[2] == _LINEAR_CP   # Z 軸は不動 → 線形


def test_camera_rotation_curve_shared_single_cp():
    c = _coeff()
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    eulers = [(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f]) for f in range(11)]
    ch = CameraRotationChannel(0, eulers, tol=0.1, mode="bezier")
    assert ch.curve(0, 10) == EASE


def test_bone_rotation_curve_slerp_coeff_cp():
    c = _coeff()
    quats = [_z_quat(90.0 * c[f]) for f in range(11)]
    ch = BoneRotationChannel(0, quats, tol=0.1, mode="bezier")
    assert ch.curve(0, 10) == EASE


# --- 統合: reduce_*_track の curve_mode=bezier -------------------------------


def test_camera_distance_bezier_reduces_to_endpoints_and_reconstructs():
    dist = _eased(-10.0, -110.0)  # 距離を ease で動かす(他チャンネルは不動)
    source = [cam(f, dist=dist[f]) for f in range(11)]
    bez = camera_track(source, [(0, 10)])
    lin = camera_track(source, [(0, 10)], curve_mode="linear")
    # 1 本の曲線で表せるので bezier は両端の 2 キーへ、linear は多数に分割。
    assert frames(bez) == [0, 10]
    assert len(frames(lin)) > 2
    # 到達キー(frame 10)に非線形の距離曲線が入る。
    assert bez[-1].interpolation[16:20] != bytes([20, 107, 20, 107])
    # 出力を再サンプルすると元の距離が許容内に復元される。
    for f in range(11):
        assert abs(interp.sample(bez, "distance", f) - dist[f]) <= TOLS.camera_distance


def test_camera_position_bezier_reconstructs():
    ys = _eased(0.0, 50.0)
    source = [cam(f, center=(0.0, ys[f], 0.0)) for f in range(11)]
    bez = camera_track(source, [(0, 10)])
    assert frames(bez) == [0, 10]
    for f in range(11):
        assert abs(interp.sample(bez, "pos_y", f) - ys[f]) <= TOLS.camera_pos


def test_camera_rotation_bezier_shared_curve_reconstructs():
    c = _coeff()
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    source = [cam(f, rot=(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f])) for f in range(11)]
    bez = camera_track(source, [(0, 10)])
    assert frames(bez) == [0, 10]
    for f in range(11):
        got = interp.sample(bez, "rot", f)
        for i in range(3):
            assert abs(math.degrees(got[i] - e1[i] * c[f])) <= TOLS.camera_rot


def test_bone_position_and_rotation_bezier_reconstructs():
    c = _coeff()
    pos = _eased(0.0, 30.0)
    quats = [_z_quat(90.0 * c[f]) for f in range(11)]
    source = [bone("センター", f, pos=(pos[f], 0.0, 0.0), rot=quats[f]) for f in range(11)]
    bez = bone_track(source, [(0, 10)])
    assert frames(bez) == [0, 10]
    for f in range(11):
        assert abs(interp.sample(bez, "pos_x", f) - pos[f]) <= TOLS.bone_pos
        got = interp.sample(bez, "rot", f)
        ang = math.degrees(2.0 * math.acos(min(1.0, abs(sum(a * b for a, b in zip(got, quats[f]))))))
        assert ang <= TOLS.bone_rot


def test_camera_default_mode_linear_keeps_linear_interp():
    dist = _eased(-10.0, -110.0)
    source = [cam(f, dist=dist[f]) for f in range(11)]
    keys = reduce_camera_track(
        source, [(0, 10)], TOLS,
        cut_thresholds=CAM_CUT, keep_frames=[], no_cut_detect=True,
        min_seg=1, max_seg=180, strict=False,
    )
    assert all(k.interpolation == CAMERA_LINEAR_INTERP for k in keys)
