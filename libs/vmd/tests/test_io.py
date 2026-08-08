"""vmd-io のテスト。

フィクスチャは宣言的リテラルのみ:
- io層: VMD形式のレイアウトから手書きしたバイト列
- 正規化: データモデルの直接構築
"""

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

# ---------------------------------------------------------------------------
# 手書きバイト列フィクスチャ
# 値はフィールド取り違えを検出できるよう、すべて異なる識別可能な値にしてある。
# 浮動小数点はf32で正確に表現できる値のみを使う。
# ---------------------------------------------------------------------------

MAGIC_V2 = b"Vocaloid Motion Data 0002".ljust(30, b"\x00")
MAGIC_V1 = b"Vocaloid Motion Data file".ljust(30, b"\x00")
CAMERA_MODEL_NAME = "カメラ・照明".encode("cp932").ljust(20, b"\x00")


def u32(n: int) -> bytes:
    return struct.pack("<I", n)


# --- カメラ系ファイル ---

CAMERA_INTERP = bytes(range(100, 124))  # 24バイト、全バイト識別可能
CAMERA_KEY = (
    u32(5)                                    # frame
    + struct.pack("<f", -45.5)                # distance
    + struct.pack("<3f", 1.0, 2.0, 3.0)       # カメラ中心
    + struct.pack("<3f", 0.5, 0.25, -0.125)   # 回転(ラジアン)
    + CAMERA_INTERP
    + u32(30)                                 # 視野角
    + struct.pack("<B", 1)                    # パースペクティブ(1=OFF)
)
LIGHT_KEY = struct.pack("<I3f3f", 8, 1.0, 0.5, 0.25, 0.0, -1.0, 0.5)
SELF_SHADOW_KEY = struct.pack("<IBf", 9, 2, 0.0625)
IK_NAME = "左足ＩＫ".encode("cp932").ljust(20, b"\x00")
IK_KEY = struct.pack("<IBI", 12, 1, 1) + IK_NAME + struct.pack("<B", 0)

CAMERA_FILE = (
    MAGIC_V2 + CAMERA_MODEL_NAME
    + u32(0)                    # ボーン
    + u32(0)                    # モーフ
    + u32(1) + CAMERA_KEY
    + u32(1) + LIGHT_KEY
    + u32(1) + SELF_SHADOW_KEY
    + u32(1) + IK_KEY
)

# セルフ影セクション境界で終わる旧版相当ファイル
TRUNCATED_AT_SELF_SHADOW = (
    MAGIC_V2 + CAMERA_MODEL_NAME
    + u32(0) + u32(0)
    + u32(1) + CAMERA_KEY
    + u32(1) + LIGHT_KEY
)
# IKセクション境界で終わる(セルフ影まで存在)
TRUNCATED_AT_IK = TRUNCATED_AT_SELF_SHADOW + u32(1) + SELF_SHADOW_KEY

# カメラキーの途中(61バイト中30バイト)で切れた不正ファイル
TRUNCATED_MID_KEY = (
    MAGIC_V2 + CAMERA_MODEL_NAME + u32(0) + u32(0) + u32(1) + CAMERA_KEY[:30]
)

# 順不同のカメラキー(frame 20 → 10 の順で格納)
DEFAULT_CAMERA_INTERP = bytes([20, 107, 20, 107]) * 6


def cam_key_bytes(frame: int) -> bytes:
    return (
        u32(frame)
        + struct.pack("<f", -30.0)
        + struct.pack("<3f", 0.0, 0.0, 0.0)
        + struct.pack("<3f", 0.0, 0.0, 0.0)
        + DEFAULT_CAMERA_INTERP
        + u32(30)
        + struct.pack("<B", 0)
    )


UNSORTED_FILE = (
    MAGIC_V2 + CAMERA_MODEL_NAME
    + u32(0) + u32(0)
    + u32(2) + cam_key_bytes(20) + cam_key_bytes(10)
    + u32(0) + u32(0) + u32(0)
)

# --- モデル系ファイル(物理フラグ付きボーンキー) ---

MODEL_MODEL_NAME = "テストモデル".encode("cp932").ljust(20, b"\x00")
BONE_NAME = "センター".encode("cp932").ljust(15, b"\x00")

# 補間ブロック64バイト: 先頭16バイトの並びは
# [X_x1 Y_x1 Z_x1 R_x1 / X_y1 Y_y1 Z_y1 R_y1 / X_x2 Y_x2 Z_x2 R_x2 / X_y2 Y_y2 Z_y2 R_y2]
SEQ = bytes([10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43])
# Byte[16]以降は1バイトずつ左シフトしたコピー+詰めパッド(旧仕様の01)
BONE_INTERP = SEQ + SEQ[1:] + b"\x01" + SEQ[2:] + b"\x01\x00" + SEQ[3:] + b"\x01\x00\x00"
# 物理フラグ: Byte[2], Byte[3](Z_x1, R_x1の位置)を (99, 15) で上書き
PHYS_INTERP = BONE_INTERP[:2] + bytes([99, 15]) + BONE_INTERP[4:]

BONE_KEY = (
    BONE_NAME
    + struct.pack("<I3f4f", 3, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    + PHYS_INTERP
)
MORPH_KEY = "まばたき".encode("cp932").ljust(15, b"\x00") + struct.pack("<If", 7, 1.0)

MODEL_FILE = (
    MAGIC_V2 + MODEL_MODEL_NAME
    + u32(1) + BONE_KEY
    + u32(1) + MORPH_KEY
    + u32(0) + u32(0) + u32(0) + u32(0)
)

# フィクスチャ自己検査(VMD形式のキーサイズと一致すること)
assert len(MAGIC_V2) == 30 and len(CAMERA_MODEL_NAME) == 20
assert len(CAMERA_KEY) == 61
assert len(LIGHT_KEY) == 28
assert len(SELF_SHADOW_KEY) == 9
assert len(BONE_INTERP) == 64
assert len(BONE_KEY) == 111
assert len(MORPH_KEY) == 23


# ---------------------------------------------------------------------------
# テスト6: フィールド単位assert(手書きバイト列との双方向比較)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# テスト1: ラウンドトリップ(バイト一致)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# write_file: 原子書き出し
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_writes_same_bytes_as_write(self, tmp_path):
        doc, _ = read(CAMERA_FILE)
        out = tmp_path / "out.vmd"
        write_file(doc, out)
        assert out.read_bytes() == write(doc) == CAMERA_FILE

    def test_overwrite_in_place_replaces_atomically(self, tmp_path):
        # 既存ファイル(=入力と同一パス)への上書きで内容が置き換わり、
        # 一時ファイルが残らないこと
        target = tmp_path / "cam.vmd"
        target.write_bytes(MODEL_FILE)
        doc, _ = read(CAMERA_FILE)
        write_file(doc, target)
        assert target.read_bytes() == CAMERA_FILE
        assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# テスト2: v1拒否
# ---------------------------------------------------------------------------


class TestV1Reject:
    def test_v1_rejected_with_specific_message(self):
        with pytest.raises(VmdFormatError, match="v1"):
            read(MAGIC_V1)


# ---------------------------------------------------------------------------
# テスト4: エラー
# ---------------------------------------------------------------------------


class TestErrors:
    def test_bad_magic(self):
        with pytest.raises(VmdFormatError):
            read(b"NOT A VMD FILE".ljust(30, b"\x00") + b"\x00" * 20)

    def test_empty(self):
        with pytest.raises(VmdFormatError):
            read(b"")

    def test_truncated_mid_keyframe(self):
        with pytest.raises(VmdFormatError):
            read(TRUNCATED_MID_KEY)


# ---------------------------------------------------------------------------
# テスト3: 正規化(データモデル直接構築)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# テスト5: 物理フラグ付きボーン補間の制御点復元
# ---------------------------------------------------------------------------


class TestPhysicsFlagControlPoints:
    def test_control_points_restored_from_shifted_copy(self):
        doc, _ = read(MODEL_FILE)
        cp = doc.bone[0].control_points()
        assert cp["X"] == (10, 20, 30, 40)
        assert cp["Y"] == (11, 21, 31, 41)
        # Z_x1, R_x1 は物理フラグで上書きされており、シフトコピー側から復元される
        assert cp["Z"] == (12, 22, 32, 42)
        assert cp["R"] == (13, 23, 33, 43)
