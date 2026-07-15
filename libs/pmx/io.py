"""PMX読み取り。

ボーン階層のみを保持し、それ以外のセクション(頂点・面・テクスチャ・材質・
モーフ・表示枠・剛体・ジョイント・SoftBody)は個数を読んで本体を読み飛ばす。
ボーンより前のセクションはボーンへ到達するために、後ろのセクションは構造の
妥当性検証のために走査する。
"""

import struct
from pathlib import Path

from .types import PmxBone, PmxFormatError, PmxModel, PmxWarning

_MAGIC = b"PMX "
_SUPPORTED_VERSIONS = (2.0, 2.1)
_INDEX_FMT = {1: "<b", 2: "<h", 4: "<i"}

# ボーンフラグ
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

# FK近似で解釈しない(変形特性に関わるが本実装では無視する)フラグ。
# 該当ボーンは警告に残すが読み取りは継続する。
_UNINTERPRETED_FLAGS = (
    _FLAG_IK
    | _FLAG_LOCAL_GRANT
    | _FLAG_ROT_GRANT
    | _FLAG_MOV_GRANT
    | _FLAG_PHYSICS_AFTER
    | _FLAG_EXTERNAL_PARENT
)

# SoftBodyセクションを持つPMXバージョンの下限(2.1拡張)。
_SOFTBODY_MIN_VERSION = 2.05


class _Reader:
    """前進専用のバイトカーソル。境界超過は PmxFormatError。"""

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
        """要素数(後続要素数)を読む。負値は構造異常。"""
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


def _read_textbuf(r: _Reader, encoding: str) -> tuple[bytes, str]:
    raw = r.take(r.i32())
    return raw, raw.decode(encoding, errors="replace")


def _skip_vertex(r: _Reader, add_uv: int, idx: dict) -> None:
    r.take(12 + 12 + 8)  # 位置・法線・UV
    r.take(16 * add_uv)  # 追加UV
    weight_type = r.u8()
    bone = idx["bone"]
    if weight_type == 0:  # BDEF1
        r.take(bone)
    elif weight_type == 1:  # BDEF2
        r.take(2 * bone + 4)
    elif weight_type == 2:  # BDEF4
        r.take(4 * bone + 16)
    elif weight_type == 3:  # SDEF
        r.take(2 * bone + 4 + 36)
    elif weight_type == 4:  # QDEF(2.1拡張。BDEF4と同一レイアウト)
        r.take(4 * bone + 16)
    else:
        raise PmxFormatError(f"未対応のウェイト変形方式: {weight_type}")
    r.take(4)  # エッジ倍率


def _skip_material(r: _Reader, idx: dict) -> None:
    _skip_textbuf(r)  # 材質名
    _skip_textbuf(r)  # 材質名英
    r.take(16 + 12 + 4 + 12)  # diffuse, specular, specular係数, ambient
    r.take(1)  # 描画フラグ
    r.take(16 + 4)  # エッジ色, エッジサイズ
    r.take(idx["texture"])  # 通常テクスチャ
    r.take(idx["texture"])  # スフィアテクスチャ
    r.take(1)  # スフィアモード
    if r.u8() == 0:  # 共有Toonフラグ
        r.take(idx["texture"])  # 個別Toon
    else:
        r.take(1)  # 共有Toon番号
    _skip_textbuf(r)  # メモ
    r.take(4)  # 材質に対応する面(頂点)数


def _morph_offset_size(morph_type: int, idx: dict) -> int:
    """モーフ種類1件あたりのオフセットデータのバイト数。"""
    if morph_type == 0:  # グループ
        return idx["morph"] + 4
    if morph_type == 1:  # 頂点
        return idx["vertex"] + 12
    if morph_type == 2:  # ボーン
        return idx["bone"] + 12 + 16
    if morph_type in (3, 4, 5, 6, 7):  # UV・追加UV1〜4
        return idx["vertex"] + 16
    if morph_type == 8:  # 材質(演算形式1 + float4×4 + float3×2 + float×2)
        return idx["material"] + 113
    if morph_type == 9:  # フリップ(2.1)
        return idx["morph"] + 4
    if morph_type == 10:  # インパルス(2.1)
        return idx["rigid"] + 1 + 12 + 12
    raise PmxFormatError(f"未対応のモーフ種類: {morph_type}")


def _skip_morph(r: _Reader, idx: dict) -> None:
    _skip_textbuf(r)  # モーフ名
    _skip_textbuf(r)  # モーフ名英
    r.take(1)  # 操作パネル
    morph_type = r.u8()
    n = r.count()  # オフセット数
    r.take(n * _morph_offset_size(morph_type, idx))


def _skip_display_frame(r: _Reader, idx: dict) -> None:
    _skip_textbuf(r)  # 枠名
    _skip_textbuf(r)  # 枠名英
    r.take(1)  # 特殊枠フラグ
    for _ in range(r.count()):  # 枠内要素数
        target = r.u8()
        if target == 0:  # ボーン
            r.take(idx["bone"])
        elif target == 1:  # モーフ
            r.take(idx["morph"])
        else:
            raise PmxFormatError(f"未対応の表示枠要素対象: {target}")


def _skip_rigidbody(r: _Reader, idx: dict) -> None:
    _skip_textbuf(r)  # 剛体名
    _skip_textbuf(r)  # 剛体名英
    r.take(idx["bone"])  # 関連ボーン
    r.take(1)  # グループ
    r.take(2)  # 非衝突グループフラグ
    r.take(1)  # 形状
    r.take(12 + 12 + 12)  # サイズ・位置・回転
    r.take(4 * 5)  # 質量・移動減衰・回転減衰・反発力・摩擦力
    r.take(1)  # 物理演算


def _skip_joint(r: _Reader, idx: dict) -> None:
    _skip_textbuf(r)  # Joint名
    _skip_textbuf(r)  # Joint名英
    r.take(1)  # Joint種類(種類に依らず以降は同一レイアウト)
    r.take(idx["rigid"])  # 関連剛体A
    r.take(idx["rigid"])  # 関連剛体B
    r.take(12 * 8)  # 位置・回転・移動制限下上・回転制限下上・バネ移動・バネ回転


def _skip_softbody(r: _Reader, idx: dict) -> None:
    _skip_textbuf(r)  # SoftBody名
    _skip_textbuf(r)  # SoftBody名英
    r.take(1)  # 形状
    r.take(idx["material"])  # 関連材質
    r.take(1)  # グループ
    r.take(2)  # 非衝突グループフラグ
    r.take(1)  # フラグ
    r.take(4 + 4)  # B-Link作成距離・クラスタ数
    r.take(4 + 4)  # 総質量・衝突マージン
    r.take(4)  # AeroModel
    r.take(4 * 25)  # config12 + cluster6 + iteration4 + material3
    for _ in range(r.count()):  # アンカー剛体数
        r.take(idx["rigid"])
        r.take(idx["vertex"])
        r.take(1)  # Nearモード
    r.take(r.count() * idx["vertex"])  # Pin頂点数


def _read_bone(r: _Reader, encoding: str, bone_index_size: int) -> PmxBone:
    name_raw, name = _read_textbuf(r, encoding)
    en_raw, en = _read_textbuf(r, encoding)
    position = r.f32x3()
    parent = r.index(bone_index_size)
    r.i32()  # 変形階層(FKでは未使用)
    flags = r.u16()
    # 接続先
    if flags & _FLAG_TAIL_IS_BONE:
        r.index(bone_index_size)
    else:
        r.take(12)  # 座標オフセット
    # 付与(回転付与または移動付与)
    if flags & (_FLAG_ROT_GRANT | _FLAG_MOV_GRANT):
        r.index(bone_index_size)
        r.take(4)  # 付与率
    # 軸固定
    if flags & _FLAG_FIXED_AXIS:
        r.take(12)
    # ローカル軸
    if flags & _FLAG_LOCAL_AXIS:
        r.take(24)  # X軸・Z軸の方向ベクトル
    # 外部親変形
    if flags & _FLAG_EXTERNAL_PARENT:
        r.take(4)  # Key値
    # IK
    if flags & _FLAG_IK:
        r.index(bone_index_size)  # IKターゲット
        r.take(4)  # ループ回数
        r.take(4)  # 制限角
        for _ in range(r.count()):  # IKリンク
            r.index(bone_index_size)
            if r.u8():  # 角度制限ON
                r.take(24)  # 下限・上限
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


def read_pmx(source) -> PmxModel:
    """PMXを読み、ボーン階層を共通モデルへ変換する。

    source はパス(str/Path)またはPMXバイト列。
    """
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        raise TypeError(f"read_pmx: 未対応の入力型 {type(source)!r}")

    r = _Reader(data)

    # ---- ヘッダ ----
    magic = r.take(4)
    if magic != _MAGIC:
        raise PmxFormatError(f"PMXマジック不正: {magic!r}")
    version = struct.unpack("<f", r.take(4))[0]
    if not any(abs(version - v) < 1e-4 for v in _SUPPORTED_VERSIONS):
        raise PmxFormatError(f"未対応のPMXバージョン: {version}")

    globals_count = r.u8()
    globals_bytes = r.take(globals_count)
    if globals_count < 8:
        raise PmxFormatError(f"globals数が不足: {globals_count}")
    encoding_id = globals_bytes[0]
    if encoding_id not in (0, 1):
        raise PmxFormatError(f"未対応の文字コード: {encoding_id}")
    encoding = "utf-16-le" if encoding_id == 0 else "utf-8"
    add_uv = globals_bytes[1]
    if not 0 <= add_uv <= 4:
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
        if size not in (1, 2, 4):
            raise PmxFormatError(f"{name} indexサイズ不正: {size}")

    # ---- モデル情報(名前・コメント4種) ----
    for _ in range(4):
        _skip_textbuf(r)

    # ---- 頂点 ----
    for _ in range(r.count()):
        _skip_vertex(r, add_uv, idx)

    # ---- 面 ----
    r.take(r.count() * idx["vertex"])

    # ---- テクスチャ ----
    for _ in range(r.count()):
        _skip_textbuf(r)

    # ---- 材質 ----
    for _ in range(r.count()):
        _skip_material(r, idx)

    # ---- ボーン ----
    bones: list[PmxBone] = []
    warnings: list[PmxWarning] = []
    name_to_index: dict[str, int] = {}
    for i in range(r.count()):
        bone = _read_bone(r, encoding, idx["bone"])
        bones.append(bone)
        name_to_index.setdefault(bone.name, i)
        if bone.flags & _UNINTERPRETED_FLAGS:
            warnings.append(
                PmxWarning(
                    code="uninterpreted_bone_flag",
                    message=(
                        "FK近似で解釈しないボーンフラグを含む"
                        f"(flags=0x{bone.flags:04x})"
                    ),
                    bone_index=i,
                )
            )

    # ---- ボーン以降(モーフ・表示枠・剛体・Joint・SoftBody)を構造検証のため走査 ----
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
