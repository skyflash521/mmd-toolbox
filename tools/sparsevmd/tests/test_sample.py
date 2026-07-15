"""サンプリング層のテスト。

sample.build_tracks は対象セクション(camera/bone)を内部作業ビューで正規化
(フレームソート・同一キー後勝ち)し、camera を1トラック、bone をボーン名ごとの
トラックに分割する。サンプリングは vmd.interp に委譲する:
スカラーは numpy 配列、回転は値のリスト、perspective は離散ホールド。
"""

import numpy as np
import pytest

from vmd.types import BoneKey, CameraKey, VmdDocument
from sparsevmd import sample
from sparsevmd.sample import build_tracks, sample_rotation, sample_scalar


# 真の線形補間になる制御点(各チャンネル x1==y1, x2==y2 → y=x)。
CAM_LINEAR = bytes([20, 107, 20, 107]) * 6  # 24バイト


def _bone_linear():
    """全チャンネル (20,20,107,107) の線形ボーン補間64バイト。"""
    b = bytearray(64)
    for i in (0, 1, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BONE_LINEAR = _bone_linear()


# pos_x チャンネルだけ非線形(ease)なカメラ補間。残りは線形。
# pos_x bytes=[ax=x1, bx=x2, ay=y1, by=y2]=(40,90,10,118)。
CAM_POSX_EASE = bytes([40, 90, 10, 118]) + bytes([20, 107, 20, 107]) * 5


def _bone_ease_R():
    """R チャンネルだけ非線形(40,10,90,118)、X/Y/Z は線形のボーン補間64バイト。"""
    b = bytearray(_bone_linear())
    b[18] = 40  # R_x1(シフトコピー側)
    b[7] = 10   # R_y1
    b[11] = 90  # R_x2
    b[15] = 118  # R_y2
    return bytes(b)


BONE_EASE_R = _bone_ease_R()


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, persp)


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, rot, BONE_LINEAR)


# --- build_tracks -----------------------------------------------------------


def test_camera_single_track():
    doc = VmdDocument(camera=[cam(0), cam(30), cam(60)])
    tracks = build_tracks(doc, "camera")
    assert len(tracks) == 1
    t = tracks[0]
    assert t.kind == "camera"
    assert t.first == 0 and t.last == 60


def test_bone_split_by_name():
    doc = VmdDocument(
        bone=[
            bone("センター", 0),
            bone("頭", 0),
            bone("センター", 30),
            bone("頭", 60),
        ]
    )
    tracks = build_tracks(doc, "bone")
    # ボーン名ごとにちょうど1トラック。
    assert len(tracks) == 2
    by_name = {t.name: t for t in tracks}
    assert set(by_name) == {"センター", "頭"}
    assert by_name["センター"].first == 0 and by_name["センター"].last == 30
    assert by_name["頭"].first == 0 and by_name["頭"].last == 60


def test_normalization_sort_and_dedup_last_wins():
    # 順不同 + 同一フレーム重複(後勝ち)。
    doc = VmdDocument(camera=[cam(60), cam(0), cam(30, dist=-10.0), cam(30, dist=-99.0)])
    t = build_tracks(doc, "camera")[0]
    frames = [k.frame for k in t.keys]
    assert frames == [0, 30, 60]  # ソート済み・重複は1つ
    # フレーム30は後勝ち(-99.0)。
    k30 = [k for k in t.keys if k.frame == 30][0]
    assert k30.distance == pytest.approx(-99.0)


def test_bone_normalization_sort_and_dedup_per_track():
    # bone も camera 同様に正規化(同一ボーン同一フレームは後勝ち、ソート)。
    doc = VmdDocument(
        bone=[
            bone("センター", 30, pos=(0.0, 0.0, 0.0)),
            bone("センター", 0),
            bone("センター", 30, pos=(9.0, 0.0, 0.0)),
        ]
    )
    t = build_tracks(doc, "bone")[0]
    frames = [k.frame for k in t.keys]
    assert frames == [0, 30]
    k30 = [k for k in t.keys if k.frame == 30][0]
    assert k30.position[0] == pytest.approx(9.0)  # 後勝ち


def test_target_all_includes_both():
    doc = VmdDocument(camera=[cam(0), cam(30)], bone=[bone("センター", 0), bone("センター", 30)])
    kinds = {t.kind for t in build_tracks(doc, "all")}
    assert kinds == {"camera", "bone"}


def test_target_camera_excludes_bone():
    doc = VmdDocument(camera=[cam(0), cam(30)], bone=[bone("センター", 0), bone("センター", 30)])
    tracks = build_tracks(doc, "camera")
    # camera1トラックのみ・bone は含まない(空リストでの空振りを避け厳密に確認)。
    assert [t.kind for t in tracks] == ["camera"]


def test_target_bone_excludes_camera():
    doc = VmdDocument(camera=[cam(0), cam(30)], bone=[bone("センター", 0), bone("センター", 30)])
    tracks = build_tracks(doc, "bone")
    assert [t.kind for t in tracks] == ["bone"]


def test_single_key_track_first_equals_last():
    doc = VmdDocument(bone=[bone("センター", 42)])
    t = build_tracks(doc, "bone")[0]
    assert t.first == 42 and t.last == 42


# --- scalar sampling --------------------------------------------------------


def test_sample_scalar_linear_camera_pos():
    # 2キーの線形移動 pos_x: 0→10 over frames 0..10。
    keys = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(10.0, 0.0, 0.0))]
    arr = sample_scalar(keys, "pos_x", 0, 10)
    assert isinstance(arr, np.ndarray)
    assert len(arr) == 11
    np.testing.assert_allclose(arr, np.linspace(0.0, 10.0, 11), atol=1e-6)


def test_sample_scalar_linear_bone_pos():
    keys = [bone("センター", 0, pos=(0.0, 0.0, 0.0)), bone("センター", 10, pos=(0.0, 5.0, 0.0))]
    arr = sample_scalar(keys, "pos_y", 0, 10)
    np.testing.assert_allclose(arr, np.linspace(0.0, 5.0, 11), atol=1e-6)


def test_sample_scalar_subrange():
    keys = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(10.0, 0.0, 0.0))]
    arr = sample_scalar(keys, "pos_x", 2, 5)
    assert len(arr) == 4
    np.testing.assert_allclose(arr, [2.0, 3.0, 4.0, 5.0], atol=1e-6)


def test_sample_scalar_honors_interpolation_curve():
    # 非線形(ease)カーブの入力では、フレーム比例の単純線形ではなく VMD 補間曲線を
    # 評価した値になること(補間評価は vmd.interp へ委譲)。
    # pos_x: 0→10 over 0..10、ease カーブ。frame3 はカーブ評価で約 2.37523(線形なら3.0)。
    keys = [
        CameraKey(0, -30.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), CAM_POSX_EASE, 30, 0),
        CameraKey(10, -30.0, (10.0, 0.0, 0.0), (0.0, 0.0, 0.0), CAM_POSX_EASE, 30, 0),
    ]
    arr = sample_scalar(keys, "pos_x", 0, 10)
    assert arr[3] == pytest.approx(2.3752295941041206, abs=1e-6)
    assert arr[3] < 2.9  # 線形(3.0)では決して出ない値


# --- rotation sampling ------------------------------------------------------


def test_sample_rotation_camera_euler():
    # camera回転は軸別線形補間。0→(0,1,0) の中点は (0,0.5,0)。
    keys = [cam(0, rot=(0.0, 0.0, 0.0)), cam(10, rot=(0.0, 1.0, 0.0))]
    rots = sample_rotation(keys, 0, 10)
    assert len(rots) == 11
    assert all(len(r) == 3 for r in rots)  # Euler 3要素
    assert rots[0] == pytest.approx((0.0, 0.0, 0.0))
    assert rots[10] == pytest.approx((0.0, 1.0, 0.0))
    assert rots[5] == pytest.approx((0.0, 0.5, 0.0))


def test_sample_rotation_bone_quaternion_slerp():
    import math

    # identity → Z軸90度回転。slerp 系列が単位quaternionで端点が一致すること。
    q1 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    keys = [
        bone("センター", 0, rot=(0.0, 0.0, 0.0, 1.0)),
        bone("センター", 10, rot=q1),
    ]
    rots = sample_rotation(keys, 0, 10)
    assert len(rots) == 11
    assert all(len(r) == 4 for r in rots)  # quaternion 4要素
    assert rots[0] == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert rots[10] == pytest.approx(q1)
    # 全サンプルが単位quaternion。
    for r in rots:
        assert sum(c * c for c in r) == pytest.approx(1.0, abs=1e-6)
    # 中点は q0→q1 の slerp、すなわち Z軸45度 = (0,0,sin(pi/8),cos(pi/8))。
    # 「内部を全て q1 にする」等の誤実装を弾くため厳密値で固定する。
    mid = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    assert rots[5] == pytest.approx(mid)


def test_sample_rotation_bone_honors_interpolation_curve():
    import math

    # R チャンネルが非線形(ease)。補間係数は frame 比でなく曲線評価になり、
    # その係数で slerp する。frame3 は約 (0,0,0.185470,0.982650)。
    # raw frame 比(0.3)での slerp(z≈0.2334)や normalized lerp とは異なる値で、
    # 「曲線係数を使う」「slerpである」の両方を弾く。
    q1 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    keys = [
        BoneKey(b"c".ljust(15, b"\x00"), 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_EASE_R),
        BoneKey(b"c".ljust(15, b"\x00"), 10, (0.0, 0.0, 0.0), q1, BONE_EASE_R),
    ]
    rots = sample_rotation(keys, 0, 10)
    assert rots[3] == pytest.approx(
        (0.0, 0.0, 0.18546995755930634, 0.9826499350444946), abs=1e-6
    )
