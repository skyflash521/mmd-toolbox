"""出力キー再構築のテスト(sparsevmd.md §3.2, §4.2)。

build_camera_keys / build_bone_keys は、削減後のキーフレーム列とソースキー列から、
各フレームでサンプリングした値を持つ出力 CameraKey / BoneKey を生成する。
linear mode では補間ブロックは線形固定。perspective は直近ホールド、視野角は整数度。
"""

import pytest

pytest.importorskip("mmd_toolbox.vmd.reduce", reason="impl pending: Step 1c reduce extraction")

from mmd_toolbox.vmd.types import BoneKey, CameraKey, VmdDocument  # noqa: E402,F401
from mmd_toolbox.vmd import reduce as reducer  # noqa: E402,F401
from mmd_toolbox.vmd.reduce import (  # noqa: E402
    BONE_LINEAR_INTERP,
    CAMERA_LINEAR_INTERP,
    bone_interp_bytes,
    build_bone_keys,
    build_camera_keys,
)

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
# pos_x チャンネルだけ非線形(ease)なカメラ補間(到達側カーブ評価の検証用)。
CAM_POSX_EASE = bytes([40, 90, 10, 118]) + bytes([20, 107, 20, 107]) * 5


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, persp)


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, rot, BL)


# --- 線形補間ブロック定数 ---------------------------------------------------


def test_camera_linear_interp_is_24_bytes_linear():
    assert len(CAMERA_LINEAR_INTERP) == 24
    # 真の線形(各チャンネル x1==y1, x2==y2)。
    assert CAMERA_LINEAR_INTERP == bytes([20, 107, 20, 107]) * 6


def test_bone_linear_interp_is_64_bytes_and_canonical_and_shiftcopy():
    assert len(BONE_LINEAR_INTERP) == 64
    # control_points() が全チャンネル (20,20,107,107) を返す(シフトコピー側から復元)。
    k = BoneKey(b"c".ljust(15, b"\x00"), 0, (0, 0, 0), (0, 0, 0, 1), BONE_LINEAR_INTERP)
    cps = k.control_points()
    for ch in ("X", "Y", "Z", "R"):
        assert cps[ch] == (20, 20, 107, 107)
    # 物理フラグ上書き対策として Byte[2]/Byte[3] にも canonical 値が書かれている
    # (vmd-io.md §2.2: Byte[2]/[3] は物理フラグで上書きされうるためシフトコピー側から復元)。
    assert BONE_LINEAR_INTERP[2] == 20 and BONE_LINEAR_INTERP[3] == 20


def test_bone_interp_bytes_full_shiftcopy_layout():
    # 識別可能な値でレイアウト(docs/specs/vmd/VMD_file_format.md)を厳密に検証する。
    b = bone_interp_bytes((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16))
    assert len(b) == 64
    first = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
    # 先頭16バイト(本体)。
    assert list(b[0:16]) == first
    # シフトコピー(1/2/3バイト左シフト)を全範囲で検証する。
    assert list(b[16:31]) == first[1:16]  # 15バイト
    assert list(b[32:46]) == first[2:16]  # 14バイト
    assert list(b[48:61]) == first[3:16]  # 13バイト
    # 詰めパッドは0。
    assert [b[i] for i in (31, 46, 47, 61, 62, 63)] == [0, 0, 0, 0, 0, 0]
    # control_points で各チャンネルの制御点が復元できる。
    k = BoneKey(b"c".ljust(15, b"\x00"), 0, (0, 0, 0), (0, 0, 0, 1), b)
    cps = k.control_points()
    assert cps["X"] == (1, 2, 3, 4)
    assert cps["Y"] == (5, 6, 7, 8)
    assert cps["Z"] == (9, 10, 11, 12)
    assert cps["R"] == (13, 14, 15, 16)


# --- build_camera_keys ------------------------------------------------------


def test_build_camera_keys_samples_values():
    source = [
        cam(0, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0),
        cam(10, dist=-20.0, center=(10.0, 0.0, 0.0), rot=(0.0, 0.5, 0.0), fov=40, persp=1),
    ]
    keys = build_camera_keys(source, [0, 5, 10])
    assert [k.frame for k in keys] == [0, 5, 10]
    mid = keys[1]
    assert mid.frame == 5
    assert mid.distance == pytest.approx(-25.0)
    assert mid.position[0] == pytest.approx(5.0)
    assert mid.rotation[1] == pytest.approx(0.25)
    assert mid.fov == 35  # 線形 30→40 の中点、整数
    assert isinstance(mid.fov, int)  # CameraKey.fov は int 契約
    assert mid.perspective == 0  # frame10(persp1)より前は直近(frame0=0)
    # 補間ブロックは線形固定。
    assert all(k.interpolation == CAMERA_LINEAR_INTERP for k in keys)


def test_build_camera_keys_samples_arriving_side_curve():
    # ソースキーが非線形(ease)補間を持つ場合、build は到達側カーブを評価した値を
    # サンプリングする(単純なフレーム比例ではない。§3.2 到達側キー格納)。
    # 出発側(frame0)は線形、到達側(frame10)のみ ease。補間は到達側キーに格納されるため、
    # 到達側カーブを読む正しい実装は frame3≈2.37523、出発側を読む誤実装は3.0になる。
    source = [
        CameraKey(0, -30.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), CAM_LINEAR, 30, 0),
        CameraKey(10, -30.0, (10.0, 0.0, 0.0), (0.0, 0.0, 0.0), CAM_POSX_EASE, 30, 0),
    ]
    keys = build_camera_keys(source, [0, 3, 10])
    assert keys[1].position[0] == pytest.approx(2.3752295941041206, abs=1e-6)


def test_build_camera_keys_fov_rounded_half_up():
    source = [cam(0, fov=30), cam(2, fov=31)]
    keys = build_camera_keys(source, [0, 1, 2])
    # frame1 は 30.5 → 四捨五入(切り上げ)で 31。
    assert keys[1].fov == 31


def test_build_camera_keys_perspective_switch_held():
    source = [cam(0, persp=0), cam(10, persp=1)]
    keys = build_camera_keys(source, [0, 10])
    assert keys[0].perspective == 0
    assert keys[1].perspective == 1


# --- build_bone_keys --------------------------------------------------------


def test_build_bone_keys_samples_values_and_name():
    source = [
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 10, pos=(0.0, 10.0, 0.0)),
    ]
    keys = build_bone_keys(source, [0, 5, 10])
    assert [k.frame for k in keys] == [0, 5, 10]
    assert keys[1].position[1] == pytest.approx(5.0)
    assert all(k.name == "センター" for k in keys)
    # name_raw はソースの15バイト生バイト(null終端含む)をそのまま保持する(§3.2 ソート規約)。
    assert all(k.name_raw == source[0].name_raw for k in keys)
    assert len(keys[0].name_raw) == 15
    assert all(k.interpolation == BONE_LINEAR_INTERP for k in keys)


def test_build_bone_keys_quaternion_sampled():
    import math

    q1 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    source = [bone("頭", 0, rot=(0.0, 0.0, 0.0, 1.0)), bone("頭", 10, rot=q1)]
    keys = build_bone_keys(source, [0, 5, 10])
    # 中点は slerp 45度 = (0,0,sin(pi/8),cos(pi/8))。
    assert keys[1].rotation == pytest.approx((0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)))
    # 単位 quaternion。
    assert sum(c * c for c in keys[1].rotation) == pytest.approx(1.0, abs=1e-6)
