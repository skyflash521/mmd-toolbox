"""PMX読み取り read_pmx のテスト。

最小PMXバイト列をテスト内で組み立てて read_pmx を検証する。
ボーン配列までを読めればよいので、頂点・面・テクスチャ・材質は
0個または極小個で構成する。
"""

import struct

import pytest

from pmx.io import read_pmx
from pmx.types import PmxFormatError


# ---------------------------------------------------------------------------
# PMXバイト列ビルダー
# ---------------------------------------------------------------------------

_IDX_FMT = {1: "<b", 2: "<h", 4: "<i"}


def _textbuf(s: str, encoding: int) -> bytes:
    enc = "utf-16-le" if encoding == 0 else "utf-8"
    b = s.encode(enc)
    return struct.pack("<i", len(b)) + b


def _idx(v, size: int) -> bytes:
    return struct.pack(_IDX_FMT[size], -1 if v is None else v)


def build_bone(
    bone: dict, *, encoding: int, bone_index_size: int
) -> bytes:
    """1ボーン分のバイト列。接続先は常にオフセット指定(0x0001を立てない)。"""
    flags = bone.get("flags", 0x0002 | 0x0004)
    assert not (flags & 0x0001), "ビルダーは接続先オフセット指定のみ対応"
    out = bytearray()
    out += _textbuf(bone["name"], encoding)
    out += _textbuf(bone.get("name_en", ""), encoding)
    out += struct.pack("<3f", *bone.get("position", (0.0, 0.0, 0.0)))
    out += _idx(bone.get("parent"), bone_index_size)
    out += struct.pack("<i", bone.get("layer", 0))
    out += struct.pack("<H", flags)
    # 接続先:0 -> 座標オフセット float3
    out += struct.pack("<3f", *bone.get("tail_offset", (0.0, 0.0, 0.0)))
    # 付与
    if flags & (0x0100 | 0x0200):
        out += _idx(bone.get("grant_parent", 0), bone_index_size)
        out += struct.pack("<f", bone.get("grant_rate", 1.0))
    # 軸固定
    if flags & 0x0400:
        out += struct.pack("<3f", *bone.get("fixed_axis", (1.0, 0.0, 0.0)))
    # ローカル軸
    if flags & 0x0800:
        out += struct.pack("<3f", *bone.get("local_x", (1.0, 0.0, 0.0)))
        out += struct.pack("<3f", *bone.get("local_z", (0.0, 0.0, 1.0)))
    # 外部親変形
    if flags & 0x2000:
        out += struct.pack("<i", bone.get("external_key", 0))
    # IK
    if flags & 0x0020:
        out += _idx(bone.get("ik_target", 0), bone_index_size)
        out += struct.pack("<i", bone.get("ik_loop", 1))
        out += struct.pack("<f", bone.get("ik_limit", 0.1))
        links = bone.get("ik_links", [])
        out += struct.pack("<i", len(links))
        for link in links:
            out += _idx(link.get("bone", 0), bone_index_size)
            limited = link.get("limited", False)
            out += struct.pack("<b", 1 if limited else 0)
            if limited:
                out += struct.pack("<3f", *link.get("lower", (0.0, 0.0, 0.0)))
                out += struct.pack("<3f", *link.get("upper", (0.0, 0.0, 0.0)))
    return bytes(out)


def build_vertex(*, bone_index_size: int) -> bytes:
    """BDEF1・追加UV0個の最小頂点。"""
    out = bytearray()
    out += struct.pack("<3f", 0.0, 0.0, 0.0)  # 位置
    out += struct.pack("<3f", 0.0, 1.0, 0.0)  # 法線
    out += struct.pack("<2f", 0.0, 0.0)  # UV
    out += struct.pack("<b", 0)  # ウェイト方式 BDEF1
    out += _idx(0, bone_index_size)  # 単一ボーン
    out += struct.pack("<f", 1.0)  # エッジ倍率
    return bytes(out)


def build_material(*, encoding: int, tex_index_size: int) -> bytes:
    """共有Toon指定の最小材質(面数0)。"""
    out = bytearray()
    out += _textbuf("mat", encoding)
    out += _textbuf("mat_en", encoding)
    out += struct.pack("<4f", 1.0, 1.0, 1.0, 1.0)  # diffuse
    out += struct.pack("<3f", 0.0, 0.0, 0.0)  # specular
    out += struct.pack("<f", 5.0)  # specular係数
    out += struct.pack("<3f", 0.5, 0.5, 0.5)  # ambient
    out += struct.pack("<B", 0)  # 描画フラグ
    out += struct.pack("<4f", 0.0, 0.0, 0.0, 1.0)  # エッジ色
    out += struct.pack("<f", 1.0)  # エッジサイズ
    out += _idx(-1, tex_index_size)  # 通常テクスチャ
    out += _idx(-1, tex_index_size)  # スフィア
    out += struct.pack("<B", 0)  # スフィアモード
    out += struct.pack("<B", 1)  # 共有Toonフラグ:1
    out += struct.pack("<B", 0)  # 共有Toon番号
    out += _textbuf("", encoding)  # メモ
    out += struct.pack("<i", 0)  # 面(頂点)数
    return bytes(out)


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
    """ヘッダ〜材質セクションまで(ボーン数 int の直前まで)。"""
    out = bytearray()
    out += magic
    out += struct.pack("<f", version)
    out += struct.pack("<B", 8)  # globals数
    out += bytes(
        [
            encoding,
            0,  # 追加UV数
            vertex_index_size,
            texture_index_size,
            material_index_size,
            bone_index_size,
            morph_index_size,
            rigid_index_size,
        ]
    )
    # モデル情報(名前4種)
    for s in ("model", "model_en", "comment", "comment_en"):
        out += _textbuf(s, encoding)
    # 頂点
    out += struct.pack("<i", n_vertices)
    for _ in range(n_vertices):
        out += build_vertex(bone_index_size=bone_index_size)
    # 面
    out += struct.pack("<i", 0)
    # テクスチャ
    out += struct.pack("<i", 0)
    # 材質
    out += struct.pack("<i", n_materials)
    for _ in range(n_materials):
        out += build_material(encoding=encoding, tex_index_size=texture_index_size)
    return bytes(out)


def build_morph(*, encoding: int, bone_index_size: int) -> bytes:
    """ボーンモーフ1件(オフセット1個)。"""
    out = bytearray()
    out += _textbuf("morph", encoding)
    out += _textbuf("morph_en", encoding)
    out += struct.pack("<B", 1)  # 操作パネル
    out += struct.pack("<B", 2)  # モーフ種類: ボーン
    out += struct.pack("<i", 1)  # オフセット数
    out += _idx(0, bone_index_size)  # ボーンIndex
    out += struct.pack("<3f", 0.0, 0.0, 0.0)  # 移動量
    out += struct.pack("<4f", 0.0, 0.0, 0.0, 1.0)  # 回転量
    return bytes(out)


def build_display_frame(*, encoding: int, bone_index_size: int) -> bytes:
    """ボーン要素1個の表示枠。"""
    out = bytearray()
    out += _textbuf("frame", encoding)
    out += _textbuf("frame_en", encoding)
    out += struct.pack("<B", 0)  # 特殊枠フラグ
    out += struct.pack("<i", 1)  # 枠内要素数
    out += struct.pack("<B", 0)  # 要素対象: ボーン
    out += _idx(0, bone_index_size)
    return bytes(out)


def build_rigidbody(*, encoding: int, bone_index_size: int) -> bytes:
    out = bytearray()
    out += _textbuf("rb", encoding)
    out += _textbuf("rb_en", encoding)
    out += _idx(0, bone_index_size)  # 関連ボーン
    out += struct.pack("<B", 0)  # グループ
    out += struct.pack("<H", 0)  # 非衝突グループ
    out += struct.pack("<B", 0)  # 形状
    out += struct.pack("<3f", 1.0, 1.0, 1.0)  # サイズ
    out += struct.pack("<3f", 0.0, 0.0, 0.0)  # 位置
    out += struct.pack("<3f", 0.0, 0.0, 0.0)  # 回転
    out += struct.pack("<5f", 1.0, 0.0, 0.0, 0.0, 0.0)  # 質量・各減衰・反発・摩擦
    out += struct.pack("<B", 0)  # 物理演算
    return bytes(out)


def build_joint(*, encoding: int, rigid_index_size: int) -> bytes:
    out = bytearray()
    out += _textbuf("jt", encoding)
    out += _textbuf("jt_en", encoding)
    out += struct.pack("<B", 0)  # Joint種類
    out += _idx(-1, rigid_index_size)  # 関連剛体A
    out += _idx(-1, rigid_index_size)  # 関連剛体B
    for _ in range(8):  # 位置・回転・移動制限下上・回転制限下上・バネ移動・バネ回転
        out += struct.pack("<3f", 0.0, 0.0, 0.0)
    return bytes(out)


def build_softbody(
    *, encoding: int, material_index_size: int, rigid_index_size: int, vertex_index_size: int
) -> bytes:
    """アンカー・Pin 0個の最小SoftBody(PMX2.1)。"""
    out = bytearray()
    out += _textbuf("sb", encoding)
    out += _textbuf("sb_en", encoding)
    out += struct.pack("<B", 0)  # 形状
    out += _idx(-1, material_index_size)  # 関連材質
    out += struct.pack("<B", 0)  # グループ
    out += struct.pack("<H", 0)  # 非衝突グループ
    out += struct.pack("<B", 0)  # フラグ
    out += struct.pack("<i", 0)  # B-Link作成距離
    out += struct.pack("<i", 0)  # クラスタ数
    out += struct.pack("<f", 1.0)  # 総質量
    out += struct.pack("<f", 0.0)  # 衝突マージン
    out += struct.pack("<i", 0)  # AeroModel
    out += struct.pack("<25f", *([0.0] * 25))  # config12+cluster6+iteration4+material3
    out += struct.pack("<i", 0)  # アンカー剛体数
    out += struct.pack("<i", 0)  # Pin頂点数
    return bytes(out)


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
    """ボーン配列とそれ以降のセクションを含む構造的に完全なPMX。

    ボーン以降(モーフ・表示枠・剛体・Joint、PMX2.1ではSoftBody)も
    形式どおりの順序で個数付きで置く。
    """
    encoding = kwargs.get("encoding", 0)
    version = kwargs.get("version", 2.0)
    bone_index_size = kwargs.get("bone_index_size", 1)
    material_index_size = kwargs.get("material_index_size", 1)
    rigid_index_size = kwargs.get("rigid_index_size", 1)
    vertex_index_size = kwargs.get("vertex_index_size", 1)
    out = bytearray(pmx_pre_bone_section(**kwargs))
    out += struct.pack("<i", len(bones))
    for b in bones:
        out += build_bone(b, encoding=encoding, bone_index_size=bone_index_size)
    # モーフ
    out += struct.pack("<i", morphs)
    for _ in range(morphs):
        out += build_morph(encoding=encoding, bone_index_size=bone_index_size)
    # 表示枠
    out += struct.pack("<i", display_frames)
    for _ in range(display_frames):
        out += build_display_frame(encoding=encoding, bone_index_size=bone_index_size)
    # 剛体
    out += struct.pack("<i", rigidbodies)
    for _ in range(rigidbodies):
        out += build_rigidbody(encoding=encoding, bone_index_size=bone_index_size)
    # Joint
    out += struct.pack("<i", joints)
    for _ in range(joints):
        out += build_joint(encoding=encoding, rigid_index_size=rigid_index_size)
    # SoftBody(PMX2.1のみ)
    if version >= 2.05:
        out += struct.pack("<i", softbodies)
        for _ in range(softbodies):
            out += build_softbody(
                encoding=encoding,
                material_index_size=material_index_size,
                rigid_index_size=rigid_index_size,
                vertex_index_size=vertex_index_size,
            )
    return bytes(out)


# ---------------------------------------------------------------------------
# ボーン読み取り
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("flags,movable,rotatable", [
    (0x0002, False, True),   # 回転のみ
    (0x0004, True, False),   # 移動のみ
    (0x0002 | 0x0004, True, True),
    (0x0008, False, False),  # 表示のみ(変形不可)
])
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
    """頂点・材質があってもボーンまで正しく読み飛ばす。"""
    data = build_pmx(
        [{"name": "センター"}],
        n_vertices=3,
        n_materials=2,
    )
    model = read_pmx(data)
    assert len(model.bones) == 1
    assert model.bones[0].name == "センター"


# ---------------------------------------------------------------------------
# エラー
# ---------------------------------------------------------------------------


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
    # globals先頭(マジック8 + count1 = offset 9)がエンコード方式
    data[9] = 5
    with pytest.raises(PmxFormatError):
        read_pmx(bytes(data))


# globals は offset 9 から [encoding, addUV, vertex, texture, material, bone, morph, rigid]
@pytest.mark.parametrize(
    "offset",
    [11, 12, 13, 14, 15, 16],  # vertex/texture/material/bone/morph/rigid のIndexサイズ
)
def test_bad_index_size_raises(offset):
    data = bytearray(build_pmx([{"name": "a"}]))
    data[offset] = 3  # 1/2/4 以外
    with pytest.raises(PmxFormatError):
        read_pmx(bytes(data))


def test_bad_add_uv_raises():
    data = bytearray(build_pmx([{"name": "a"}]))
    data[10] = 5  # 追加UV数(0..4 範囲外)
    with pytest.raises(PmxFormatError):
        read_pmx(bytes(data))


def test_truncated_section_raises():
    """ボーン数=1 を宣言した直後でデータを打ち切る(セクション長不足)。"""
    truncated = pmx_pre_bone_section() + struct.pack("<i", 1)
    with pytest.raises(PmxFormatError):
        read_pmx(truncated)


def test_negative_bone_count_raises():
    data = pmx_pre_bone_section() + struct.pack("<i", -1)
    with pytest.raises(PmxFormatError):
        read_pmx(data)


def test_truncated_morph_section_raises():
    """ボーン0・モーフ1を宣言するがモーフ本体が無い(ボーン以降も走査する)。"""
    data = pmx_pre_bone_section() + struct.pack("<i", 0) + struct.pack("<i", 1)
    with pytest.raises(PmxFormatError):
        read_pmx(data)


# ---------------------------------------------------------------------------
# ボーン以降のセクションのスキップ
# ---------------------------------------------------------------------------


def test_reads_bones_past_full_trailing_sections():
    """非空のモーフ・表示枠・剛体・Joint があってもボーンを正しく読む。"""
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
    """PMX2.1 の SoftBody セクションを読み飛ばしてボーンを読む。"""
    data = build_pmx(
        [{"name": "センター"}],
        version=2.1,
        morphs=1,
        rigidbodies=1,
        softbodies=1,
    )
    model = read_pmx(data)
    assert model.bones[0].name == "センター"


# ---------------------------------------------------------------------------
# 未解釈フラグの警告
# ---------------------------------------------------------------------------


def test_uninterpreted_flag_warns():
    """物理後変形(0x1000)は未解釈フラグとして警告に残るが読み取りは継続。"""
    data = build_pmx(
        [{"name": "物理", "flags": 0x0002 | 0x0004 | 0x1000}]
    )
    model = read_pmx(data)
    assert len(model.bones) == 1
    assert any(w.bone_index == 0 for w in model.warnings)


def test_plain_bone_has_no_warning():
    data = build_pmx([{"name": "普通", "flags": 0x0002 | 0x0004}])
    model = read_pmx(data)
    assert model.warnings == ()
