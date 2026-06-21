"""mocapvmd テスト用の共通ヘルパ(VMDキー構築・ドキュメント書き出し)。"""

from mmd_toolbox.vmd import io
from mmd_toolbox.vmd.reduce import (
    BONE_LINEAR_INTERP,
    bone_interp_bytes,
    camera_interp_bytes,
)
from mmd_toolbox.vmd.types import (
    BoneKey,
    CameraKey,
    IkBone,
    IkPropertyKey,
    LightKey,
    MorphKey,
    SelfShadowKey,
    VmdDocument,
)

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6

# 逐語透過の検証用に、線形でない補間バイト列。再構築・線形化されれば値が変わる。
BONE_NONLINEAR = bone_interp_bytes((30, 20, 90, 100), (25, 15, 80, 110), (35, 45, 70, 95), (40, 50, 60, 85))
CAM_NONLINEAR = camera_interp_bytes(
    (30, 40, 80, 90), (25, 35, 75, 95), (20, 30, 70, 100),
    (15, 45, 65, 105), (10, 50, 60, 110), (5, 55, 55, 115),
)


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), interp=BONE_LINEAR_INTERP):
    """ボーンキー。name は cp932 で15バイト固定にエンコードする。"""
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, rot, interp)


def morph(name, frame, weight=0.0):
    return MorphKey(name.encode("cp932").ljust(15, b"\x00"), frame, weight)


def cam(frame, center=(0.0, 0.0, 0.0), interp=CAM_LINEAR):
    return CameraKey(frame, -30.0, center, (0.0, 0.0, 0.0), interp, 30, 0)


def light(frame, color=(1.0, 1.0, 1.0), pos=(0.5, -0.5, 0.5)):
    return LightKey(frame, color, pos)


def self_shadow(frame, mode=1, distance=0.0):
    return SelfShadowKey(frame, mode, distance)


def ik_property(frame, names, display=1):
    """IKプロパティキー。names は (ボーン名, enable) のタプル列。名前は20バイト固定。"""
    ik_bones = [IkBone(n.encode("cp932").ljust(20, b"\x00"), enable) for n, enable in names]
    return IkPropertyKey(frame, display, ik_bones)


def write_vmd(path, **sections):
    io.write_file(VmdDocument(**sections), str(path))
