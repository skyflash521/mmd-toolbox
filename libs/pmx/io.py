import struct
from pathlib import Path

from .types import PmxBone, PmxFormatError, PmxModel, PmxWarning

_MAGIC = b"PMX "
_SUPPORTED_VERSIONS = (2.0, 2.1)
_SOFTBODY_MIN_VERSION = 2.05
_INDEX_FMT = {1: "<b", 2: "<h", 4: "<i"}
_ENCODINGS = {0: "utf-16-le", 1: "utf-8"}
_MIN_GLOBALS_COUNT = 8
_MAX_ADD_UV = 4
_MODEL_INFO_TEXT_COUNT = 4

_BYTE = 1
_UINT16 = 2
_INT = 4
_FLOAT = 4
_VEC2 = 8
_VEC3 = 12
_VEC4 = 16

_FLAG_TAIL_IS_BONE = 0x0001
_FLAG_ROTATABLE = 0x0002
_FLAG_MOVABLE = 0x0004
_FLAG_IK = 0x0020
_FLAG_LOCAL_GRANT = 0x0080
_FLAG_ROT_GRANT = 0x0100
_FLAG_MOV_GRANT = 0x0200
_FLAG_FIXED_AXIS = 0x0400
_FLAG_LOCAL_AXIS = 0x0800
_FLAG_PHYSICS_AFTER = 0x1000
_FLAG_EXTERNAL_PARENT = 0x2000

_UNINTERPRETED_FLAGS = (
    _FLAG_IK
    | _FLAG_LOCAL_GRANT
    | _FLAG_ROT_GRANT
    | _FLAG_MOV_GRANT
    | _FLAG_PHYSICS_AFTER
    | _FLAG_EXTERNAL_PARENT
)

_WEIGHT_BDEF1 = 0
_WEIGHT_BDEF2 = 1
_WEIGHT_BDEF4 = 2
_WEIGHT_SDEF = 3
_WEIGHT_QDEF = 4

_MORPH_GROUP = 0
_MORPH_VERTEX = 1
_MORPH_BONE = 2
_MORPH_UV_KINDS = (3, 4, 5, 6, 7)
_MORPH_MATERIAL = 8
_MORPH_FLIP = 9
_MORPH_IMPULSE = 10

_FRAME_TARGET_BONE = 0
_FRAME_TARGET_MORPH = 1


def _size(**field_sizes: int) -> int:
    return sum(field_sizes.values())


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0:
            raise PmxFormatError(f"負の読取長: {n}")
        end = self.pos + n
        if end > len(self.data):
            raise PmxFormatError(
                f"セクション長不足: offset {self.pos} から {n} バイト読めない"
            )
        b = self.data[self.pos : end]
        self.pos = end
        return b

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def count(self) -> int:
        n = self.i32()
        if n < 0:
            raise PmxFormatError(f"要素数が負: {n}")
        return n

    def f32x3(self) -> tuple[float, float, float]:
        return struct.unpack("<3f", self.take(12))

    def index(self, size: int) -> int:
        return struct.unpack(_INDEX_FMT[size], self.take(size))[0]


def _skip_textbuf(r: _Reader) -> None:
    r.take(r.i32())


def _skip_names(r: _Reader) -> None:
    _skip_textbuf(r)
    _skip_textbuf(r)


def _read_textbuf(r: _Reader, encoding: str) -> tuple[bytes, str]:
    raw = r.take(r.i32())
    return raw, raw.decode(encoding, errors="replace")


def _skip_vertex(r: _Reader, add_uv: int, idx: dict) -> None:
    r.take(_size(position=_VEC3, normal=_VEC3, uv=_VEC2, add_uvs=add_uv * _VEC4))
    weight_type = r.u8()
    bone = idx["bone"]
    if weight_type == _WEIGHT_BDEF1:
        r.take(_size(bones=bone))
    elif weight_type == _WEIGHT_BDEF2:
        r.take(_size(bones=2 * bone, weight=_FLOAT))
    elif weight_type in (_WEIGHT_BDEF4, _WEIGHT_QDEF):
        r.take(_size(bones=4 * bone, weights=4 * _FLOAT))
    elif weight_type == _WEIGHT_SDEF:
        r.take(_size(bones=2 * bone, weight=_FLOAT, sdef_c=_VEC3, sdef_r0=_VEC3, sdef_r1=_VEC3))
    else:
        raise PmxFormatError(f"未対応のウェイト変形方式: {weight_type}")
    r.take(_size(edge_scale=_FLOAT))


def _skip_material(r: _Reader, idx: dict) -> None:
    texture = idx["texture"]
    _skip_names(r)
    r.take(
        _size(
            diffuse=_VEC4,
            specular=_VEC3,
            specular_power=_FLOAT,
            ambient=_VEC3,
            draw_flags=_BYTE,
            edge_color=_VEC4,
            edge_size=_FLOAT,
            texture=texture,
            sphere_texture=texture,
            sphere_mode=_BYTE,
        )
    )
    uses_shared_toon = r.u8() != 0
    if uses_shared_toon:
        r.take(_size(shared_toon_number=_BYTE))
    else:
        r.take(_size(toon_texture=texture))
    _skip_textbuf(r)
    r.take(_size(face_vertex_count=_INT))


def _morph_offset_size(morph_type: int, idx: dict) -> int:
    if morph_type in (_MORPH_GROUP, _MORPH_FLIP):
        return _size(morph=idx["morph"], ratio=_FLOAT)
    if morph_type == _MORPH_VERTEX:
        return _size(vertex=idx["vertex"], offset=_VEC3)
    if morph_type == _MORPH_BONE:
        return _size(bone=idx["bone"], translation=_VEC3, rotation=_VEC4)
    if morph_type in _MORPH_UV_KINDS:
        return _size(vertex=idx["vertex"], offset=_VEC4)
    if morph_type == _MORPH_MATERIAL:
        return _size(
            material=idx["material"],
            operation=_BYTE,
            diffuse=_VEC4,
            specular=_VEC3,
            specular_power=_FLOAT,
            ambient=_VEC3,
            edge_color=_VEC4,
            edge_size=_FLOAT,
            texture_coef=_VEC4,
            sphere_coef=_VEC4,
            toon_coef=_VEC4,
        )
    if morph_type == _MORPH_IMPULSE:
        return _size(rigid=idx["rigid"], is_local=_BYTE, velocity=_VEC3, torque=_VEC3)
    raise PmxFormatError(f"未対応のモーフ種類: {morph_type}")


def _skip_morph(r: _Reader, idx: dict) -> None:
    _skip_names(r)
    r.take(_size(panel=_BYTE))
    morph_type = r.u8()
    offset_count = r.count()
    r.take(offset_count * _morph_offset_size(morph_type, idx))


def _skip_display_frame(r: _Reader, idx: dict) -> None:
    _skip_names(r)
    r.take(_size(is_special=_BYTE))
    for _ in range(r.count()):
        target = r.u8()
        if target == _FRAME_TARGET_BONE:
            r.take(idx["bone"])
        elif target == _FRAME_TARGET_MORPH:
            r.take(idx["morph"])
        else:
            raise PmxFormatError(f"未対応の表示枠要素対象: {target}")


def _skip_rigidbody(r: _Reader, idx: dict) -> None:
    _skip_names(r)
    r.take(
        _size(
            bone=idx["bone"],
            group=_BYTE,
            non_collision_groups=_UINT16,
            shape=_BYTE,
            size=_VEC3,
            position=_VEC3,
            rotation=_VEC3,
            mass=_FLOAT,
            linear_damping=_FLOAT,
            angular_damping=_FLOAT,
            restitution=_FLOAT,
            friction=_FLOAT,
            physics_mode=_BYTE,
        )
    )


def _skip_joint(r: _Reader, idx: dict) -> None:
    _skip_names(r)
    r.take(
        _size(
            joint_type=_BYTE,
            rigid_a=idx["rigid"],
            rigid_b=idx["rigid"],
            position=_VEC3,
            rotation=_VEC3,
            translation_min=_VEC3,
            translation_max=_VEC3,
            rotation_min=_VEC3,
            rotation_max=_VEC3,
            spring_translation=_VEC3,
            spring_rotation=_VEC3,
        )
    )


def _skip_softbody(r: _Reader, idx: dict) -> None:
    _skip_names(r)
    r.take(
        _size(
            shape=_BYTE,
            material=idx["material"],
            group=_BYTE,
            non_collision_groups=_UINT16,
            flags=_BYTE,
            b_link_distance=_INT,
            cluster_count=_INT,
            total_mass=_FLOAT,
            collision_margin=_FLOAT,
            aero_model=_INT,
            config=12 * _FLOAT,
            cluster=6 * _FLOAT,
            iteration=4 * _INT,
            material_params=3 * _FLOAT,
        )
    )
    for _ in range(r.count()):
        r.take(_size(rigid=idx["rigid"], vertex=idx["vertex"], near_mode=_BYTE))
    pin_vertex_count = r.count()
    r.take(pin_vertex_count * idx["vertex"])


def _read_bone(r: _Reader, encoding: str, bone_index_size: int) -> PmxBone:
    name_raw, name = _read_textbuf(r, encoding)
    en_raw, en = _read_textbuf(r, encoding)
    position = r.f32x3()
    parent = r.index(bone_index_size)
    r.take(_size(deform_layer=_INT))
    flags = r.u16()
    if flags & _FLAG_TAIL_IS_BONE:
        r.take(_size(tail_bone=bone_index_size))
    else:
        r.take(_size(tail_offset=_VEC3))
    if flags & (_FLAG_ROT_GRANT | _FLAG_MOV_GRANT):
        r.take(_size(grant_parent=bone_index_size, grant_ratio=_FLOAT))
    if flags & _FLAG_FIXED_AXIS:
        r.take(_size(fixed_axis=_VEC3))
    if flags & _FLAG_LOCAL_AXIS:
        r.take(_size(local_x_axis=_VEC3, local_z_axis=_VEC3))
    if flags & _FLAG_EXTERNAL_PARENT:
        r.take(_size(external_parent_key=_INT))
    if flags & _FLAG_IK:
        r.take(_size(ik_target=bone_index_size, loop_count=_INT, limit_angle=_FLOAT))
        for _ in range(r.count()):
            r.take(_size(link_bone=bone_index_size))
            has_angle_limit = r.u8() != 0
            if has_angle_limit:
                r.take(_size(lower_limit=_VEC3, upper_limit=_VEC3))
    return PmxBone(
        name=name,
        name_raw=name_raw,
        english_name=en,
        english_name_raw=en_raw,
        parent=None if parent < 0 else parent,
        position=position,
        movable=bool(flags & _FLAG_MOVABLE),
        rotatable=bool(flags & _FLAG_ROTATABLE),
        flags=flags,
    )


def _read_header(r: _Reader) -> tuple[float, str, int, dict]:
    magic = r.take(4)
    if magic != _MAGIC:
        raise PmxFormatError(f"PMXマジック不正: {magic!r}")
    version = struct.unpack("<f", r.take(4))[0]
    if not any(abs(version - v) < 1e-4 for v in _SUPPORTED_VERSIONS):
        raise PmxFormatError(f"未対応のPMXバージョン: {version}")

    globals_count = r.u8()
    globals_bytes = r.take(globals_count)
    if globals_count < _MIN_GLOBALS_COUNT:
        raise PmxFormatError(f"globals数が不足: {globals_count}")
    encoding_id = globals_bytes[0]
    if encoding_id not in _ENCODINGS:
        raise PmxFormatError(f"未対応の文字コード: {encoding_id}")
    add_uv = globals_bytes[1]
    if not 0 <= add_uv <= _MAX_ADD_UV:
        raise PmxFormatError(f"追加UV数が範囲外: {add_uv}")
    idx = {
        "vertex": globals_bytes[2],
        "texture": globals_bytes[3],
        "material": globals_bytes[4],
        "bone": globals_bytes[5],
        "morph": globals_bytes[6],
        "rigid": globals_bytes[7],
    }
    for name, size in idx.items():
        if size not in _INDEX_FMT:
            raise PmxFormatError(f"{name} indexサイズ不正: {size}")
    return version, _ENCODINGS[encoding_id], add_uv, idx


def _uninterpreted_flag_warning(bone: PmxBone, bone_index: int) -> PmxWarning:
    return PmxWarning(
        code="uninterpreted_bone_flag",
        message=f"FK近似で解釈しないボーンフラグを含む(flags=0x{bone.flags:04x})",
        bone_index=bone_index,
    )


def read_pmx(source: str | Path | bytes) -> PmxModel:
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        raise TypeError(f"read_pmx: 未対応の入力型 {type(source)!r}")

    r = _Reader(data)
    version, encoding, add_uv, idx = _read_header(r)

    for _ in range(_MODEL_INFO_TEXT_COUNT):
        _skip_textbuf(r)
    for _ in range(r.count()):
        _skip_vertex(r, add_uv, idx)
    face_vertex_count = r.count()
    r.take(face_vertex_count * idx["vertex"])
    for _ in range(r.count()):
        _skip_textbuf(r)
    for _ in range(r.count()):
        _skip_material(r, idx)

    bones: list[PmxBone] = []
    warnings: list[PmxWarning] = []
    name_to_index: dict[str, int] = {}
    for i in range(r.count()):
        bone = _read_bone(r, encoding, idx["bone"])
        bones.append(bone)
        name_to_index.setdefault(bone.name, i)
        if bone.flags & _UNINTERPRETED_FLAGS:
            warnings.append(_uninterpreted_flag_warning(bone, i))

    for _ in range(r.count()):
        _skip_morph(r, idx)
    for _ in range(r.count()):
        _skip_display_frame(r, idx)
    for _ in range(r.count()):
        _skip_rigidbody(r, idx)
    for _ in range(r.count()):
        _skip_joint(r, idx)
    if version >= _SOFTBODY_MIN_VERSION:
        for _ in range(r.count()):
            _skip_softbody(r, idx)

    return PmxModel(
        bones=tuple(bones),
        name_to_index=name_to_index,
        warnings=tuple(warnings),
    )
