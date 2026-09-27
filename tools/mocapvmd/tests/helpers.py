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

BONE_NONLINEAR = bone_interp_bytes((30, 20, 90, 100), (25, 15, 80, 110), (35, 45, 70, 95), (40, 50, 60, 85))
CAM_NONLINEAR = camera_interp_bytes(
    (30, 40, 80, 90), (25, 35, 75, 95), (20, 30, 70, 100),
    (15, 45, 65, 105), (10, 50, 60, 110), (5, 55, 55, 115),
)

_VMD_BONE_NAME_BYTES = 15
_VMD_IK_NAME_BYTES = 20


def _vmd_name(name, size):
    return name.encode("cp932").ljust(size, b"\x00")


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), interp=BONE_LINEAR_INTERP):
    return BoneKey(_vmd_name(name, _VMD_BONE_NAME_BYTES), frame, pos, rot, interp)


def morph(name, frame, weight=0.0):
    return MorphKey(_vmd_name(name, _VMD_BONE_NAME_BYTES), frame, weight)


def cam(frame, center=(0.0, 0.0, 0.0), interp=CAM_LINEAR):
    return CameraKey(frame, -30.0, center, (0.0, 0.0, 0.0), interp, 30, 0)


def light(frame, color=(1.0, 1.0, 1.0), pos=(0.5, -0.5, 0.5)):
    return LightKey(frame, color, pos)


def self_shadow(frame, mode=1, distance=0.0):
    return SelfShadowKey(frame, mode, distance)


def ik_property(frame, bone_name_enable_pairs, display=1):
    ik_bones = [IkBone(_vmd_name(n, _VMD_IK_NAME_BYTES), enable) for n, enable in bone_name_enable_pairs]
    return IkPropertyKey(frame, display, ik_bones)


def write_vmd(path, **sections):
    io.write_file(VmdDocument(**sections), str(path))


_PMX_ENCODING_UTF16 = 0
_PMX_BONE_ROTATABLE = 0x0002
_PMX_BONE_MOVABLE = 0x0004
_PMX_NO_PARENT = -1
_PMX_MODEL_INFO_TEXT_COUNT = 4
_PMX_SECTIONS_AFTER_BONES = 4


def _pmx_textbuf(s):
    b = s.encode("utf-16-le")
    return struct.pack("<i", len(b)) + b


def _pmx_count(n):
    return struct.pack("<i", n)


def _pmx_globals(*, encoding, add_uv, vertex_index, texture_index, material_index, bone_index,
                 morph_index, rigid_index):
    values = [encoding, add_uv, vertex_index, texture_index, material_index, bone_index,
              morph_index, rigid_index]
    return struct.pack("<B", len(values)) + bytes(values)


def build_standard_pmx(bone_names):
    out = bytearray()
    out += b"PMX "
    out += struct.pack("<f", 2.0)
    out += _pmx_globals(encoding=_PMX_ENCODING_UTF16, add_uv=0, vertex_index=1, texture_index=1,
                        material_index=1, bone_index=1, morph_index=1, rigid_index=1)
    for _ in range(_PMX_MODEL_INFO_TEXT_COUNT):
        out += _pmx_textbuf("")
    vertex_count = face_count = texture_count = material_count = 0
    for count in (vertex_count, face_count, texture_count, material_count):
        out += _pmx_count(count)
    out += _pmx_count(len(bone_names))
    for i, name in enumerate(bone_names):
        out += _pmx_textbuf(name)
        out += _pmx_textbuf("")
        out += struct.pack("<3f", 0.0, float(i), 0.0)
        out += struct.pack("<b", _PMX_NO_PARENT)
        deform_layer = 0
        out += struct.pack("<i", deform_layer)
        out += struct.pack("<H", _PMX_BONE_ROTATABLE | _PMX_BONE_MOVABLE)
        tail_offset = (0.0, 0.0, 0.0)
        out += struct.pack("<3f", *tail_offset)
    out += _pmx_count(0) * _PMX_SECTIONS_AFTER_BONES
    return bytes(out)
