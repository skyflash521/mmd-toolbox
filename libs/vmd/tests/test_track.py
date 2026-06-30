"""トラック統合(per-track 削減パイプライン)のテスト(vmd-reduce.md §4, §9)。

reduce_camera_track / reduce_bone_track は、ソースキー列と処理範囲から、各範囲を
サンプリング→不連続検出→チャンネル構築→区間削減→出力キー生成し、範囲外の
元キーはそのまま保持して、出力キー列(フレーム昇順)を返す。linear mode。
"""

import pytest

from vmd.types import BoneKey, CameraKey
from vmd.reduce import (
    CAMERA_LINEAR_INTERP,
    Tolerances,
    reduce_bone_track,
    reduce_camera_track,
)

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()
# balanced プリセット相当の許容値。
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


def camera_track(source, ranges, **kw):
    opts = dict(
        cut_thresholds=CAM_CUT,
        keep_frames=[],
        no_cut_detect=False,
        min_seg=1,
        max_seg=180,
        strict=False,
    )
    opts.update(kw)
    return reduce_camera_track(source, ranges, TOLS, **opts)


def frames(keys):
    return [k.frame for k in keys]


# --- camera -----------------------------------------------------------------


def test_camera_linear_reduces_to_endpoints():
    # 連続フレームの線形移動 → 先頭/末尾の2キーへ。
    source = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(31)]
    keys = camera_track(source, [(0, 30)])
    assert frames(keys) == [0, 30]
    assert all(k.interpolation == CAMERA_LINEAR_INTERP for k in keys)


def test_camera_peak_kept():
    # 0→15で上昇、15→30で下降(山)。極値15が残る。
    source = []
    for f in range(31):
        y = float(f) if f <= 15 else float(30 - f)
        source.append(cam(f, center=(0.0, y, 0.0)))
    keys = camera_track(source, [(0, 30)])
    assert frames(keys) == [0, 15, 30]


def test_camera_cut_detected_preserved_as_adjacent_jump():
    # frame10 で位置が大ジャンプ(不連続)。境界として保持され、ジャンプは
    # 隣接フレーム(9→10)としてなまらず残る(vmd-reduce.md §7.2)。区間をまたいだ補間をしない。
    source = []
    for f in range(21):
        x = 0.0 if f < 10 else 50.0
        source.append(cam(f, center=(x, 0.0, 0.0)))
    keys = camera_track(source, [(0, 20)])
    fr = frames(keys)
    assert 9 in fr and 10 in fr  # ジャンプ直前直後が隣接キーとして残る
    k9 = next(k for k in keys if k.frame == 9)
    k10 = next(k for k in keys if k.frame == 10)
    assert k9.position[0] == pytest.approx(0.0)
    assert k10.position[0] == pytest.approx(50.0)


def test_camera_fov_peak_kept():
    # 位置は静止、FOV が 30→45→30 の山。極値15が残る(FovChannel が駆動)。
    source = []
    for f in range(31):
        fov = 30 + (15 - abs(f - 15))
        source.append(cam(f, fov=fov))
    keys = camera_track(source, [(0, 30)])
    assert 15 in frames(keys)


def test_camera_rotation_peak_kept():
    import math

    # 位置・FOV 静止、Y回転が 0→大→0 の山。極値15が残る(CameraRotationChannel が駆動)。
    source = []
    for f in range(31):
        ry = math.radians(15 - abs(f - 15))  # 度→ラジアン、ピーク15度
        source.append(cam(f, rot=(0.0, ry, 0.0)))
    keys = camera_track(source, [(0, 30)])
    assert 15 in frames(keys)


def test_camera_keep_frame_forced():
    source = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(31)]
    keys = camera_track(source, [(0, 30)], keep_frames=[12])
    assert 12 in frames(keys)


def test_camera_range_outside_keys_preserved_verbatim():
    # 範囲[0,10]のみ削減。範囲外の元キー(20,30)は元のフィールド値・補間ブロックを
    # そのまま逐語保持する(vmd-reduce.md §9)。
    # (削減区間に隣接する範囲外キーの到着側補間曲線書き換え=vmd-reduce.md §7.3 は本MVPでは行わず、
    #  範囲外キーは逐語保持する方針。将来 --snap-range / 境界キー注入で対応。)
    source = [cam(f, center=(float(f), 0.0, 0.0), fov=30 + f) for f in (0, 5, 10, 20, 30)]
    keys = camera_track(source, [(0, 10)])
    fr = frames(keys)
    assert 20 in fr and 30 in fr  # 範囲外は保持
    assert fr == sorted(fr)
    k20 = next(k for k in keys if k.frame == 20)
    assert k20.position[0] == pytest.approx(20.0)  # 元の値を保持
    assert k20.fov == 50  # 元の値(30+20)
    assert k20.interpolation == CAM_LINEAR  # 元の補間ブロックを保持


def test_camera_multiple_ranges():
    # 2つの非重複範囲。各範囲を独立に削減し、範囲間(ギャップ)の元キーは保持。
    source = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(61)]
    keys = camera_track(source, [(0, 20), (40, 60)])
    fr = frames(keys)
    # 各範囲は線形なので端点に削減。範囲端 0,20,40,60 は残る。
    assert 0 in fr and 20 in fr and 40 in fr and 60 in fr
    # 範囲間(21..39)の中間は、線形削減対象外なので元キーが残る(少なくとも一部)。
    assert any(21 <= f <= 39 for f in fr)
    assert fr == sorted(fr)


# --- bone -------------------------------------------------------------------


def bone_track(source, ranges, **kw):
    opts = dict(
        cut_thresholds=BONE_CUT,
        keep_frames=[],
        no_cut_detect=False,
        min_seg=1,
        max_seg=180,
        strict=False,
    )
    opts.update(kw)
    return reduce_bone_track(source, ranges, TOLS, **opts)


def test_bone_linear_reduces_to_endpoints():
    source = [bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
    keys = bone_track(source, [(0, 30)])
    assert frames(keys) == [0, 30]
    assert all(k.name == "センター" for k in keys)


def test_bone_peak_kept():
    source = []
    for f in range(31):
        y = float(f) if f <= 15 else float(30 - f)
        source.append(bone("頭", f, pos=(0.0, y, 0.0)))
    keys = bone_track(source, [(0, 30)])
    assert frames(keys) == [0, 15, 30]


def test_bone_cut_detected_preserved_as_adjacent_jump():
    # frame10 で位置が大ジャンプ。境界として隣接フレームで保持(vmd-reduce.md §7.2)。
    source = [bone("センター", f, pos=(0.0, 0.0 if f < 10 else 5.0, 0.0)) for f in range(21)]
    keys = bone_track(source, [(0, 20)])
    fr = frames(keys)
    assert 9 in fr and 10 in fr


def test_bone_keep_frame_forced():
    source = [bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
    keys = bone_track(source, [(0, 30)], keep_frames=[12])
    assert 12 in frames(keys)
