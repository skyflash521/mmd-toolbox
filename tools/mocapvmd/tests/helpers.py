"""mocapvmd テスト用の共通ヘルパ(VMDキー構築・PMX構築・ドキュメント書き出し)。"""

import struct

from vmd import io
from vmd.reduce import (
    BONE_LINEAR_INTERP,
    bone_interp_bytes,
    camera_interp_bytes,
)
from vmd.types import (
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


def _pmx_textbuf(s):
    b = s.encode("utf-16-le")
    return struct.pack("<i", len(b)) + b


def build_standard_pmx(bone_names):
    """指定ボーン名だけを持つ最小PMX(平坦階層・PMX2.0/UTF16)のバイト列。

    頂点・面・テクスチャ・材質・モーフ以降は個数0。各ボーンは親なし・
    回転/移動可・接続先オフセット指定。read_pmx でボーン名解決の検証に使う。
    """
    out = bytearray()
    out += b"PMX "
    out += struct.pack("<f", 2.0)
    out += struct.pack("<B", 8)
    out += bytes([0, 0, 1, 1, 1, 1, 1, 1])  # utf16, 追加UV0, 各indexサイズ1
    for _ in range(4):
        out += _pmx_textbuf("")  # モデル情報
    out += struct.pack("<i", 0)  # 頂点
    out += struct.pack("<i", 0)  # 面
    out += struct.pack("<i", 0)  # テクスチャ
    out += struct.pack("<i", 0)  # 材質
    out += struct.pack("<i", len(bone_names))
    for i, name in enumerate(bone_names):
        out += _pmx_textbuf(name)
        out += _pmx_textbuf("")
        out += struct.pack("<3f", 0.0, float(i), 0.0)  # 位置
        out += struct.pack("<b", -1)  # 親(なし)
        out += struct.pack("<i", 0)  # 変形階層
        out += struct.pack("<H", 0x0002 | 0x0004)  # 回転+移動可
        out += struct.pack("<3f", 0.0, 0.0, 0.0)  # 接続先オフセット
    out += struct.pack("<i", 0) * 4  # モーフ/表示枠/剛体/Joint
    return bytes(out)
