"""PMX読み取りのテスト(pmx-read-fk-plan.md §6.1)。

最小PMXバイト列をテスト内で組み立てて read_pmx を検証する。
ボーン配列までを読めればよいので、頂点・面・テクスチャ・材質は
0個または極小個で構成する。
"""

import struct

import pytest

try:
    from mmd_toolbox.pmx.io import read_pmx
    from mmd_toolbox.pmx.types import PmxFormatError

    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False

# impl pending: Step1 PMX読み取り
pytestmark = (
    [] if _IMPORT_OK else pytest.mark.skip(reason="impl pending: Step1 PMX読み取り")
)


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


def build_pmx(bones, **kwargs) -> bytes:
    """ボーン配列までを含む構造的に完全な最小PMX。

    ボーンの後に続くモーフ・表示枠・剛体・Joint の各セクションは個数0で置く
    (PMX仕様の構造概要に従い、ボーン以降のセクションも形式上存在させる)。
    """
    encoding = kwargs.get("encoding", 0)
    bone_index_size = kwargs.get("bone_index_size", 1)
    out = bytearray(pmx_pre_bone_section(**kwargs))
    out += struct.pack("<i", len(bones))
    for b in bones:
        out += build_bone(b, encoding=encoding, bone_index_size=bone_index_size)
    # モーフ・表示枠・剛体・Joint の各個数(すべて0)
    out += struct.pack("<i", 0) * 4
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


def test_bad_index_size_raises():
    data = bytearray(build_pmx([{"name": "a"}]))
    # globals[5] ボーンIndexサイズ = offset 9 + 5 = 14
    data[14] = 3
    with pytest.raises(PmxFormatError):
        read_pmx(bytes(data))


def test_truncated_section_raises():
    """ボーン数=1 を宣言した直後でデータを打ち切る(セクション長不足)。"""
    truncated = pmx_pre_bone_section() + struct.pack("<i", 1)
    with pytest.raises(PmxFormatError):
        read_pmx(truncated)


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
