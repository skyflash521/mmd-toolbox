import os
import struct

import pytest

from vmd import (
    BoneKey,
    CameraKey,
    VmdDocument,
    VmdFormatError,
    normalize,
    read,
    write,
    write_file,
)

MAGIC_V2 = b"Vocaloid Motion Data 0002".ljust(30, b"\x00")
MAGIC_V1 = b"Vocaloid Motion Data file".ljust(30, b"\x00")
CAMERA_MODEL_NAME = "カメラ・照明".encode("cp932").ljust(20, b"\x00")


def _fields(**parts: bytes) -> bytes:
    return b"".join(parts.values())


def u8(n: int) -> bytes:
    return struct.pack("<B", n)


def u32(n: int) -> bytes:
    return struct.pack("<I", n)


def f32(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


CAMERA_INTERP = bytes(range(100, 124))
CAMERA_KEY = _fields(
    frame=u32(5),
    distance=f32(-45.5),
    center=f32(1.0, 2.0, 3.0),
    rotation_rad=f32(0.5, 0.25, -0.125),
    interpolation=CAMERA_INTERP,
    fov=u32(30),
    perspective_off=u8(1),
)
LIGHT_KEY = _fields(frame=u32(8), color=f32(1.0, 0.5, 0.25), position=f32(0.0, -1.0, 0.5))
SELF_SHADOW_KEY = _fields(frame=u32(9), mode=u8(2), distance=f32(0.0625))
IK_NAME = "左足ＩＫ".encode("cp932").ljust(20, b"\x00")
IK_KEY = _fields(frame=u32(12), display=u8(1), ik_bone_count=u32(1), name=IK_NAME, enable=u8(0))

CAMERA_FILE = _fields(
    magic=MAGIC_V2,
    model_name=CAMERA_MODEL_NAME,
    bone_count=u32(0),
    morph_count=u32(0),
    camera=u32(1) + CAMERA_KEY,
    light=u32(1) + LIGHT_KEY,
    self_shadow=u32(1) + SELF_SHADOW_KEY,
    ik=u32(1) + IK_KEY,
)

TRUNCATED_AT_SELF_SHADOW = _fields(
    magic=MAGIC_V2,
    model_name=CAMERA_MODEL_NAME,
    bone_count=u32(0),
    morph_count=u32(0),
    camera=u32(1) + CAMERA_KEY,
    light=u32(1) + LIGHT_KEY,
)
TRUNCATED_AT_IK = TRUNCATED_AT_SELF_SHADOW + u32(1) + SELF_SHADOW_KEY

TRUNCATED_MID_KEY = _fields(
    magic=MAGIC_V2,
    model_name=CAMERA_MODEL_NAME,
    bone_count=u32(0),
    morph_count=u32(0),
    camera=u32(1) + CAMERA_KEY[: len(CAMERA_KEY) // 2],
)

DEFAULT_CAMERA_INTERP = bytes([20, 107, 20, 107]) * 6


def cam_key_bytes(frame: int) -> bytes:
    return _fields(
        frame=u32(frame),
        distance=f32(-30.0),
        center=f32(0.0, 0.0, 0.0),
        rotation_rad=f32(0.0, 0.0, 0.0),
        interpolation=DEFAULT_CAMERA_INTERP,
        fov=u32(30),
        perspective_on=u8(0),
    )


UNSORTED_FILE = _fields(
    magic=MAGIC_V2,
    model_name=CAMERA_MODEL_NAME,
    bone_count=u32(0),
    morph_count=u32(0),
    camera=u32(2) + cam_key_bytes(20) + cam_key_bytes(10),
    light_count=u32(0),
    self_shadow_count=u32(0),
    ik_count=u32(0),
)

MODEL_MODEL_NAME = "テストモデル".encode("cp932").ljust(20, b"\x00")
BONE_NAME = "センター".encode("cp932").ljust(15, b"\x00")

CONTROL_POINT_BYTES = bytes([10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43])
BONE_INTERP = _fields(
    body=CONTROL_POINT_BYTES,
    shift1=CONTROL_POINT_BYTES[1:] + b"\x01",
    shift2=CONTROL_POINT_BYTES[2:] + b"\x01\x00",
    shift3=CONTROL_POINT_BYTES[3:] + b"\x01\x00\x00",
)
PHYSICS_FLAG_BYTES = bytes([99, 15])
PHYS_INTERP = BONE_INTERP[:2] + PHYSICS_FLAG_BYTES + BONE_INTERP[4:]

def _bone_key(name_raw: bytes) -> bytes:
    return _fields(
        name=name_raw,
        frame=u32(3),
        position=f32(0.0, 1.0, 0.0),
        rotation=f32(0.0, 0.0, 0.0, 1.0),
        interpolation=PHYS_INTERP,
    )


BONE_KEY = _bone_key(BONE_NAME)
MORPH_KEY = _fields(name="まばたき".encode("cp932").ljust(15, b"\x00"), frame=u32(7), weight=f32(1.0))


def _model_file(bone_key: bytes) -> bytes:
    return _fields(
        magic=MAGIC_V2,
        model_name=MODEL_MODEL_NAME,
        bone=u32(1) + bone_key,
        morph=u32(1) + MORPH_KEY,
        camera_count=u32(0),
        light_count=u32(0),
        self_shadow_count=u32(0),
        ik_count=u32(0),
    )


MODEL_FILE = _model_file(BONE_KEY)

assert len(MAGIC_V2) == 30 and len(CAMERA_MODEL_NAME) == 20
assert len(CAMERA_KEY) == 61
assert len(LIGHT_KEY) == 28
assert len(SELF_SHADOW_KEY) == 9
assert len(BONE_INTERP) == 64
assert len(BONE_KEY) == 111
assert len(MORPH_KEY) == 23


def _model_file_with_bone_name(name_raw: bytes) -> bytes:
    return _model_file(_bone_key(name_raw))


class TestFieldAssert:
    def test_header(self):
        doc, _ = read(CAMERA_FILE)
        assert doc.model_name_raw == CAMERA_MODEL_NAME
        assert doc.model_name == "カメラ・照明"

    def test_camera_key_fields(self):
        doc, _ = read(CAMERA_FILE)
        assert len(doc.camera) == 1
        k = doc.camera[0]
        assert k.frame == 5
        assert k.distance == -45.5
        assert k.position == (1.0, 2.0, 3.0)
        assert k.rotation == (0.5, 0.25, -0.125)
        assert k.interpolation == CAMERA_INTERP
        assert k.fov == 30
        assert k.perspective == 1

    def test_light_key_fields(self):
        doc, _ = read(CAMERA_FILE)
        assert len(doc.light) == 1
        k = doc.light[0]
        assert k.frame == 8
        assert k.color == (1.0, 0.5, 0.25)
        assert k.position == (0.0, -1.0, 0.5)

    def test_self_shadow_key_fields(self):
        doc, _ = read(CAMERA_FILE)
        assert len(doc.self_shadow) == 1
        k = doc.self_shadow[0]
        assert k.frame == 9
        assert k.mode == 2
        assert k.distance == 0.0625

    def test_ik_property_key_fields(self):
        doc, _ = read(CAMERA_FILE)
        assert len(doc.ik_property) == 1
        k = doc.ik_property[0]
        assert k.frame == 12
        assert k.display == 1
        assert len(k.ik_bones) == 1
        assert k.ik_bones[0].name_raw == IK_NAME
        assert k.ik_bones[0].name == "左足ＩＫ"
        assert k.ik_bones[0].enable == 0

    def test_bone_key_fields(self):
        doc, _ = read(MODEL_FILE)
        assert len(doc.bone) == 1
        k = doc.bone[0]
        assert k.name_raw == BONE_NAME
        assert k.name == "センター"
        assert k.frame == 3
        assert k.position == (0.0, 1.0, 0.0)
        assert k.rotation == (0.0, 0.0, 0.0, 1.0)
        assert k.interpolation == PHYS_INTERP

    def test_morph_key_fields(self):
        doc, _ = read(MODEL_FILE)
        assert len(doc.morph) == 1
        k = doc.morph[0]
        assert k.name == "まばたき"
        assert k.frame == 7
        assert k.weight == 1.0

    def test_write_reproduces_handwritten_bytes(self):
        doc, _ = read(CAMERA_FILE)
        assert write(doc) == CAMERA_FILE
        doc, _ = read(MODEL_FILE)
        assert write(doc) == MODEL_FILE


class TestRoundtrip:
    def test_camera_basic(self, camera_basic_bytes):
        doc, _ = read(camera_basic_bytes)
        assert write(doc) == camera_basic_bytes

    def test_model_motion(self, model_motion_bytes):
        doc, _ = read(model_motion_bytes)
        assert write(doc) == model_motion_bytes

    def test_truncated_at_self_shadow(self):
        doc, warnings = read(TRUNCATED_AT_SELF_SHADOW)
        assert doc.has_self_shadow_section is False
        assert doc.has_ik_section is False
        assert doc.self_shadow == []
        assert doc.ik_property == []
        assert any(w.code == "sections-missing" for w in warnings)
        assert write(doc) == TRUNCATED_AT_SELF_SHADOW

    def test_truncated_at_ik(self):
        doc, warnings = read(TRUNCATED_AT_IK)
        assert doc.has_self_shadow_section is True
        assert doc.has_ik_section is False
        assert len(doc.self_shadow) == 1
        assert any(w.code == "sections-missing" for w in warnings)
        assert write(doc) == TRUNCATED_AT_IK

    def test_physics_flag_raw_preserved(self):
        doc, _ = read(MODEL_FILE)
        assert write(doc) == MODEL_FILE

    def test_bytes_after_name_terminator_preserved(self):
        name_raw = "センター".encode("cp932") + b"\x00XYZ".ljust(15 - len("センター".encode("cp932")), b"\x00")
        data = _model_file_with_bone_name(name_raw)
        doc, _ = read(data)
        assert doc.bone[0].name_raw == name_raw
        assert doc.bone[0].name == "センター"
        assert write(doc) == data


class TestNameDecoding:
    def test_undecodable_name_warns_with_location_and_keeps_raw(self):
        name_raw = b"\x81".ljust(15, b"\x00")
        doc, warnings = read(_model_file_with_bone_name(name_raw))
        assert doc.bone[0].name_raw == name_raw
        assert doc.bone[0].name == "�"
        decode_warnings = [w for w in warnings if w.code == "decode-error"]
        assert len(decode_warnings) == 1
        w = decode_warnings[0]
        assert (w.section, w.key_index, w.frame) == ("bone", 0, 3)


class TestWriteFile:
    def test_writes_same_bytes_as_write(self, tmp_path):
        doc, _ = read(CAMERA_FILE)
        out = tmp_path / "out.vmd"
        write_file(doc, out)
        assert out.read_bytes() == write(doc) == CAMERA_FILE

    def test_overwrite_in_place_replaces_and_leaves_no_temp_file(self, tmp_path):
        target = tmp_path / "cam.vmd"
        target.write_bytes(MODEL_FILE)
        doc, _ = read(CAMERA_FILE)
        write_file(doc, target)
        assert target.read_bytes() == CAMERA_FILE
        assert list(tmp_path.glob("*.tmp")) == []

    def test_failed_write_keeps_existing_file_and_removes_temp_file(self, tmp_path, monkeypatch):
        target = tmp_path / "cam.vmd"
        target.write_bytes(MODEL_FILE)
        doc, _ = read(CAMERA_FILE)

        def fail_replace(src, dst):
            raise OSError("replace failed")

        monkeypatch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError):
            write_file(doc, target)
        assert target.read_bytes() == MODEL_FILE
        assert list(tmp_path.glob("*.tmp")) == []


class TestV1Reject:
    def test_v1_rejected_with_specific_message(self):
        with pytest.raises(VmdFormatError, match="v1"):
            read(MAGIC_V1)


class TestErrors:
    def test_bad_magic(self):
        with pytest.raises(VmdFormatError):
            read(b"NOT A VMD FILE".ljust(30, b"\x00") + b"\x00" * 20)

    def test_empty(self):
        with pytest.raises(VmdFormatError):
            read(b"")

    def test_truncated_mid_keyframe_reports_offset(self):
        with pytest.raises(VmdFormatError, match="offset"):
            read(TRUNCATED_MID_KEY)

    def test_trailing_bytes_after_ik_section(self):
        with pytest.raises(VmdFormatError):
            read(CAMERA_FILE + b"\x00")

    def test_missing_file_raises_os_error_not_format_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read(tmp_path / "missing.vmd")


def make_cam(frame: int, fov: int = 30) -> CameraKey:
    return CameraKey(
        frame=frame,
        distance=-30.0,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        interpolation=DEFAULT_CAMERA_INTERP,
        fov=fov,
        perspective=0,
    )


def make_bone(name: str, frame: int, x: float = 0.0) -> BoneKey:
    return BoneKey(
        name_raw=name.encode("cp932").ljust(15, b"\x00"),
        frame=frame,
        position=(x, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        interpolation=BONE_INTERP,
    )


class TestNormalize:
    def test_read_does_not_sort(self):
        doc, _ = read(UNSORTED_FILE)
        assert [k.frame for k in doc.camera] == [20, 10]

    def test_sorts_camera_by_frame(self):
        doc = VmdDocument(camera=[make_cam(30), make_cam(10), make_cam(0)])
        ndoc, warnings = normalize(doc)
        assert [k.frame for k in ndoc.camera] == [0, 10, 30]
        assert any(w.code == "normalize-sorted" for w in warnings)

    def test_duplicate_later_wins(self):
        doc = VmdDocument(camera=[make_cam(10, fov=1), make_cam(10, fov=2)])
        ndoc, warnings = normalize(doc)
        assert len(ndoc.camera) == 1
        assert ndoc.camera[0].fov == 2
        assert any(w.code == "normalize-duplicate" and w.frame == 10 for w in warnings)

    def test_bone_sorted_by_name_then_frame(self):
        doc = VmdDocument(
            bone=[make_bone("B", 3), make_bone("A", 5), make_bone("A", 0)]
        )
        ndoc, _ = normalize(doc)
        assert [(k.name, k.frame) for k in ndoc.bone] == [("A", 0), ("A", 5), ("B", 3)]

    def test_bone_duplicate_identity_is_name_and_frame(self):
        doc = VmdDocument(
            bone=[make_bone("A", 5, x=1.0), make_bone("B", 5), make_bone("A", 5, x=9.0)]
        )
        ndoc, _ = normalize(doc)
        assert len(ndoc.bone) == 2
        a = [k for k in ndoc.bone if k.name == "A"][0]
        assert a.position[0] == 9.0

    def test_names_equal_after_decoding_but_different_raw_bytes_are_distinct(self):
        plain = make_bone("A", 5)
        trailing = BoneKey(b"A\x00X".ljust(15, b"\x00"), 5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_INTERP)
        assert plain.name == trailing.name
        ndoc, warnings = normalize(VmdDocument(bone=[plain, trailing]))
        assert len(ndoc.bone) == 2
        assert not any(w.code == "normalize-duplicate" for w in warnings)

    def test_sections_limited_leaves_others_untouched(self):
        doc = VmdDocument(
            camera=[make_cam(30), make_cam(10)],
            bone=[make_bone("B", 3), make_bone("A", 5)],
        )
        ndoc, _ = normalize(doc, sections=["camera"])
        assert [k.frame for k in ndoc.camera] == [10, 30]
        assert [(k.name, k.frame) for k in ndoc.bone] == [("B", 3), ("A", 5)]

    def test_already_normalized_reports_no_warnings(self):
        doc = VmdDocument(camera=[make_cam(0), make_cam(10)])
        ndoc, warnings = normalize(doc)
        assert [k.frame for k in ndoc.camera] == [0, 10]
        assert warnings == []


class TestPhysicsFlagControlPoints:
    def test_control_points_restored_from_shifted_copy(self):
        doc, _ = read(MODEL_FILE)
        cp = doc.bone[0].control_points()
        assert cp["X"] == (10, 20, 30, 40)
        assert cp["Y"] == (11, 21, 31, 41)
        assert cp["Z"] == (12, 22, 32, 42)
        assert cp["R"] == (13, 23, 33, 43)
