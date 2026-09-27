import struct

import pytest

from pmx.io import read_pmx
from pmx.types import PmxFormatError

_IDX_FMT = {1: "<b", 2: "<h", 4: "<i"}
_FLAG_ROTATABLE = 0x0002
_FLAG_MOVABLE = 0x0004
_FLAG_VISIBLE = 0x0008
_FLAG_IK = 0x0020
_FLAG_ROT_GRANT = 0x0100
_FLAG_MOV_GRANT = 0x0200
_FLAG_FIXED_AXIS = 0x0400
_FLAG_LOCAL_AXIS = 0x0800
_FLAG_PHYSICS_AFTER = 0x1000
_FLAG_EXTERNAL_PARENT = 0x2000
_DEFAULT_FLAGS = _FLAG_ROTATABLE | _FLAG_MOVABLE

_GLOBALS_OFFSET = len(b"PMX ") + 4 + 1
_ENCODING_OFFSET = _GLOBALS_OFFSET
_ADD_UV_OFFSET = _GLOBALS_OFFSET + 1
_INDEX_SIZE_OFFSETS = range(_GLOBALS_OFFSET + 2, _GLOBALS_OFFSET + 8)


def _fields(**parts: bytes) -> bytes:
    return b"".join(parts.values())


def _u8(v: int) -> bytes:
    return struct.pack("<B", v)


def _u16(v: int) -> bytes:
    return struct.pack("<H", v)


def _i32(v: int) -> bytes:
    return struct.pack("<i", v)


def _f32(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _textbuf_raw(raw: bytes) -> bytes:
    return _i32(len(raw)) + raw


def _textbuf(s: str, encoding: int) -> bytes:
    return _textbuf_raw(s.encode("utf-16-le" if encoding == 0 else "utf-8"))


def _idx(v, size: int) -> bytes:
    return struct.pack(_IDX_FMT[size], -1 if v is None else v)


def build_bone(bone: dict, *, encoding: int, bone_index_size: int) -> bytes:
    flags = bone.get("flags", _DEFAULT_FLAGS)
    name = bone.get("name_raw")
    out = bytearray(
        _fields(
            name=_textbuf_raw(name) if name is not None else _textbuf(bone["name"], encoding),
            english_name=_textbuf(bone.get("name_en", ""), encoding),
            position=_f32(*bone.get("position", (0.0, 0.0, 0.0))),
            parent=_idx(bone.get("parent"), bone_index_size),
            deform_layer=_i32(bone.get("layer", 0)),
            flags=_u16(flags),
            tail_offset=_f32(*bone.get("tail_offset", (0.0, 0.0, 0.0))),
        )
    )
    if flags & (_FLAG_ROT_GRANT | _FLAG_MOV_GRANT):
        out += _fields(
            grant_parent=_idx(bone.get("grant_parent", 0), bone_index_size),
            grant_ratio=_f32(bone.get("grant_rate", 1.0)),
        )
    if flags & _FLAG_FIXED_AXIS:
        out += _f32(*bone.get("fixed_axis", (1.0, 0.0, 0.0)))
    if flags & _FLAG_LOCAL_AXIS:
        out += _fields(
            local_x_axis=_f32(*bone.get("local_x", (1.0, 0.0, 0.0))),
            local_z_axis=_f32(*bone.get("local_z", (0.0, 0.0, 1.0))),
        )
    if flags & _FLAG_EXTERNAL_PARENT:
        out += _i32(bone.get("external_key", 0))
    if flags & _FLAG_IK:
        links = bone.get("ik_links", [])
        out += _fields(
            ik_target=_idx(bone.get("ik_target", 0), bone_index_size),
            loop_count=_i32(bone.get("ik_loop", 1)),
            limit_angle=_f32(bone.get("ik_limit", 0.1)),
            link_count=_i32(len(links)),
        )
        for link in links:
            limited = link.get("limited", False)
            out += _fields(
                link_bone=_idx(link.get("bone", 0), bone_index_size),
                has_angle_limit=_u8(1 if limited else 0),
            )
            if limited:
                out += _fields(
                    lower_limit=_f32(*link.get("lower", (0.0, 0.0, 0.0))),
                    upper_limit=_f32(*link.get("upper", (0.0, 0.0, 0.0))),
                )
    return bytes(out)


def build_vertex(*, bone_index_size: int) -> bytes:
    return _fields(
        position=_f32(0.0, 0.0, 0.0),
        normal=_f32(0.0, 1.0, 0.0),
        uv=_f32(0.0, 0.0),
        weight_type_bdef1=_u8(0),
        bone=_idx(0, bone_index_size),
        edge_scale=_f32(1.0),
    )


def build_material(*, encoding: int, tex_index_size: int) -> bytes:
    return _fields(
        name=_textbuf("mat", encoding),
        english_name=_textbuf("mat_en", encoding),
        diffuse=_f32(1.0, 1.0, 1.0, 1.0),
        specular=_f32(0.0, 0.0, 0.0),
        specular_power=_f32(5.0),
        ambient=_f32(0.5, 0.5, 0.5),
        draw_flags=_u8(0),
        edge_color=_f32(0.0, 0.0, 0.0, 1.0),
        edge_size=_f32(1.0),
        texture=_idx(-1, tex_index_size),
        sphere_texture=_idx(-1, tex_index_size),
        sphere_mode=_u8(0),
        uses_shared_toon=_u8(1),
        shared_toon_number=_u8(0),
        memo=_textbuf("", encoding),
        face_vertex_count=_i32(0),
    )


def pmx_pre_bone_section(
    *,
    encoding: int = 0,
    version: float = 2.0,
    bone_index_size: int = 1,
    vertex_index_size: int = 1,
    texture_index_size: int = 1,
    material_index_size: int = 1,
    morph_index_size: int = 1,
    rigid_index_size: int = 1,
    n_vertices: int = 0,
    n_materials: int = 0,
    magic: bytes = b"PMX ",
) -> bytes:
    globals_bytes = bytes(
        [
            encoding,
            0,
            vertex_index_size,
            texture_index_size,
            material_index_size,
            bone_index_size,
            morph_index_size,
            rigid_index_size,
        ]
    )
    out = bytearray(
        _fields(
            magic=magic,
            version=_f32(version),
            globals_count=_u8(len(globals_bytes)),
            globals=globals_bytes,
        )
    )
    for s in ("model", "model_en", "comment", "comment_en"):
        out += _textbuf(s, encoding)
    out += _i32(n_vertices)
    for _ in range(n_vertices):
        out += build_vertex(bone_index_size=bone_index_size)
    out += _fields(face_vertex_count=_i32(0), texture_count=_i32(0))
    out += _i32(n_materials)
    for _ in range(n_materials):
        out += build_material(encoding=encoding, tex_index_size=texture_index_size)
    return bytes(out)


def build_morph(*, encoding: int, bone_index_size: int) -> bytes:
    return _fields(
        name=_textbuf("morph", encoding),
        english_name=_textbuf("morph_en", encoding),
        panel=_u8(1),
        morph_type_bone=_u8(2),
        offset_count=_i32(1),
        bone=_idx(0, bone_index_size),
        translation=_f32(0.0, 0.0, 0.0),
        rotation=_f32(0.0, 0.0, 0.0, 1.0),
    )


def build_display_frame(*, encoding: int, bone_index_size: int) -> bytes:
    return _fields(
        name=_textbuf("frame", encoding),
        english_name=_textbuf("frame_en", encoding),
        is_special=_u8(0),
        element_count=_i32(1),
        target_bone=_u8(0),
        bone=_idx(0, bone_index_size),
    )


def build_rigidbody(*, encoding: int, bone_index_size: int) -> bytes:
    return _fields(
        name=_textbuf("rb", encoding),
        english_name=_textbuf("rb_en", encoding),
        bone=_idx(0, bone_index_size),
        group=_u8(0),
        non_collision_groups=_u16(0),
        shape=_u8(0),
        size=_f32(1.0, 1.0, 1.0),
        position=_f32(0.0, 0.0, 0.0),
        rotation=_f32(0.0, 0.0, 0.0),
        mass_damping_restitution_friction=_f32(1.0, 0.0, 0.0, 0.0, 0.0),
        physics_mode=_u8(0),
    )


def build_joint(*, encoding: int, rigid_index_size: int) -> bytes:
    return _fields(
        name=_textbuf("jt", encoding),
        english_name=_textbuf("jt_en", encoding),
        joint_type=_u8(0),
        rigid_a=_idx(-1, rigid_index_size),
        rigid_b=_idx(-1, rigid_index_size),
        position_rotation_limits_springs=_f32(*([0.0] * 24)),
    )


def build_softbody(
    *, encoding: int, material_index_size: int, rigid_index_size: int, vertex_index_size: int
) -> bytes:
    return _fields(
        name=_textbuf("sb", encoding),
        english_name=_textbuf("sb_en", encoding),
        shape=_u8(0),
        material=_idx(-1, material_index_size),
        group=_u8(0),
        non_collision_groups=_u16(0),
        flags=_u8(0),
        b_link_distance=_i32(0),
        cluster_count=_i32(0),
        total_mass=_f32(1.0),
        collision_margin=_f32(0.0),
        aero_model=_i32(0),
        config_cluster_iteration_material=_f32(*([0.0] * 25)),
        anchor_count=_i32(0),
        pin_vertex_count=_i32(0),
    )


def build_pmx(
    bones,
    *,
    morphs: int = 0,
    display_frames: int = 0,
    rigidbodies: int = 0,
    joints: int = 0,
    softbodies: int = 0,
    **kwargs,
) -> bytes:
    encoding = kwargs.get("encoding", 0)
    version = kwargs.get("version", 2.0)
    bone_index_size = kwargs.get("bone_index_size", 1)
    material_index_size = kwargs.get("material_index_size", 1)
    rigid_index_size = kwargs.get("rigid_index_size", 1)
    vertex_index_size = kwargs.get("vertex_index_size", 1)
    out = bytearray(pmx_pre_bone_section(**kwargs))
    out += _i32(len(bones))
    for b in bones:
        out += build_bone(b, encoding=encoding, bone_index_size=bone_index_size)
    out += _i32(morphs)
    for _ in range(morphs):
        out += build_morph(encoding=encoding, bone_index_size=bone_index_size)
    out += _i32(display_frames)
    for _ in range(display_frames):
        out += build_display_frame(encoding=encoding, bone_index_size=bone_index_size)
    out += _i32(rigidbodies)
    for _ in range(rigidbodies):
        out += build_rigidbody(encoding=encoding, bone_index_size=bone_index_size)
    out += _i32(joints)
    for _ in range(joints):
        out += build_joint(encoding=encoding, rigid_index_size=rigid_index_size)
    if version >= 2.05:
        out += _i32(softbodies)
        for _ in range(softbodies):
            out += build_softbody(
                encoding=encoding,
                material_index_size=material_index_size,
                rigid_index_size=rigid_index_size,
                vertex_index_size=vertex_index_size,
            )
    return bytes(out)


def test_read_minimal_single_bone():
    data = build_pmx(
        [{"name": "センター", "position": (1.0, 2.0, 3.0), "parent": None}]
    )
    model = read_pmx(data)
    assert len(model.bones) == 1
    b = model.bones[0]
    assert b.name == "センター"
    assert b.parent is None
    assert b.position == pytest.approx((1.0, 2.0, 3.0))
    assert b.movable is True
    assert b.rotatable is True


def test_read_japanese_and_english_names():
    data = build_pmx([{"name": "頭", "name_en": "head"}])
    b = read_pmx(data).bones[0]
    assert b.name == "頭"
    assert b.english_name == "head"
    assert b.name_raw == "頭".encode("utf-16-le")
    assert b.english_name_raw == "head".encode("utf-16-le")


def test_undecodable_name_is_replaced_and_raw_bytes_kept():
    raw = b"a\xffb"
    data = build_pmx([{"name_raw": raw}], encoding=1)
    b = read_pmx(data).bones[0]
    assert b.name == "a�b"
    assert b.name_raw == raw


def test_read_parent_index():
    data = build_pmx(
        [
            {"name": "親", "parent": None},
            {"name": "子", "parent": 0},
        ]
    )
    model = read_pmx(data)
    assert model.bones[0].parent is None
    assert model.bones[1].parent == 0


def test_name_to_index():
    data = build_pmx([{"name": "A"}, {"name": "B"}, {"name": "C"}])
    model = read_pmx(data)
    assert model.name_to_index == {"A": 0, "B": 1, "C": 2}


def test_duplicate_bone_name_maps_to_first_index():
    data = build_pmx([{"name": "A"}, {"name": "B"}, {"name": "A"}])
    model = read_pmx(data)
    assert model.name_to_index == {"A": 0, "B": 1}
    assert len(model.bones) == 3


@pytest.mark.parametrize(
    "flags,movable,rotatable",
    [
        pytest.param(_FLAG_ROTATABLE, False, True, id="rotatable_only"),
        pytest.param(_FLAG_MOVABLE, True, False, id="movable_only"),
        pytest.param(_FLAG_ROTATABLE | _FLAG_MOVABLE, True, True, id="both"),
        pytest.param(_FLAG_VISIBLE, False, False, id="neither"),
    ],
)
def test_flag_movable_rotatable(flags, movable, rotatable):
    data = build_pmx([{"name": "b", "flags": flags}])
    b = read_pmx(data).bones[0]
    assert b.movable is movable
    assert b.rotatable is rotatable
    assert b.flags == flags


@pytest.mark.parametrize("size", [1, 2, 4])
def test_index_size_variants(size):
    data = build_pmx(
        [
            {"name": "親", "parent": None},
            {"name": "子", "parent": 0},
        ],
        bone_index_size=size,
    )
    model = read_pmx(data)
    assert model.bones[1].parent == 0
    assert model.bones[0].parent is None


def test_encoding_utf8():
    data = build_pmx([{"name": "腕", "name_en": "arm"}], encoding=1)
    b = read_pmx(data).bones[0]
    assert b.name == "腕"
    assert b.name_raw == "腕".encode("utf-8")


def test_skip_vertices_and_materials():
    data = build_pmx(
        [{"name": "センター"}],
        n_vertices=3,
        n_materials=2,
    )
    model = read_pmx(data)
    assert len(model.bones) == 1
    assert model.bones[0].name == "センター"


def test_bad_magic_raises():
    data = build_pmx([{"name": "a"}], magic=b"Pmx ")
    with pytest.raises(PmxFormatError):
        read_pmx(data)


def test_unsupported_version_raises():
    data = build_pmx([{"name": "a"}], version=1.0)
    with pytest.raises(PmxFormatError):
        read_pmx(data)


def test_unsupported_encoding_raises():
    data = bytearray(build_pmx([{"name": "a"}]))
    data[_ENCODING_OFFSET] = 5
    with pytest.raises(PmxFormatError):
        read_pmx(bytes(data))


@pytest.mark.parametrize("offset", _INDEX_SIZE_OFFSETS)
def test_index_size_other_than_1_2_4_raises(offset):
    data = bytearray(build_pmx([{"name": "a"}]))
    data[offset] = 3
    with pytest.raises(PmxFormatError):
        read_pmx(bytes(data))


def test_add_uv_count_above_4_raises():
    data = bytearray(build_pmx([{"name": "a"}]))
    data[_ADD_UV_OFFSET] = 5
    with pytest.raises(PmxFormatError):
        read_pmx(bytes(data))


def test_declared_bone_missing_from_data_raises():
    truncated = pmx_pre_bone_section() + _i32(1)
    with pytest.raises(PmxFormatError):
        read_pmx(truncated)


def test_negative_bone_count_raises():
    data = pmx_pre_bone_section() + _i32(-1)
    with pytest.raises(PmxFormatError):
        read_pmx(data)


def test_declared_morph_missing_after_bones_raises():
    data = pmx_pre_bone_section() + _fields(bone_count=_i32(0), morph_count=_i32(1))
    with pytest.raises(PmxFormatError):
        read_pmx(data)


def test_reads_bones_past_full_trailing_sections():
    data = build_pmx(
        [{"name": "センター"}, {"name": "子", "parent": 0}],
        n_vertices=2,
        n_materials=1,
        morphs=2,
        display_frames=1,
        rigidbodies=1,
        joints=1,
    )
    model = read_pmx(data)
    assert [b.name for b in model.bones] == ["センター", "子"]


def test_reads_bones_past_softbody_pmx21():
    data = build_pmx(
        [{"name": "センター"}],
        version=2.1,
        morphs=1,
        rigidbodies=1,
        softbodies=1,
    )
    model = read_pmx(data)
    assert model.bones[0].name == "センター"


def test_physics_after_flag_warns_and_reading_continues():
    data = build_pmx([{"name": "物理", "flags": _DEFAULT_FLAGS | _FLAG_PHYSICS_AFTER}])
    model = read_pmx(data)
    assert len(model.bones) == 1
    assert any(w.bone_index == 0 for w in model.warnings)


def test_plain_bone_has_no_warning():
    data = build_pmx([{"name": "普通", "flags": _DEFAULT_FLAGS}])
    model = read_pmx(data)
    assert model.warnings == ()
