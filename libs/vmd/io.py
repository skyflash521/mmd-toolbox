import os
import struct
import tempfile
from dataclasses import replace
from pathlib import Path

from .types import (
    MAGIC_V1_PREFIX,
    MAGIC_V2_PREFIX,
    BoneKey,
    CameraKey,
    IkBone,
    IkPropertyKey,
    LightKey,
    MorphKey,
    SelfShadowKey,
    VmdDocument,
    VmdFormatError,
    VmdWarning,
)

_SECTIONS = ("bone", "morph", "camera", "light", "self_shadow", "ik_property")

_MAGIC_SIZE = 30
_MAGIC_PREFIX_SIZE = len(MAGIC_V2_PREFIX)
_MODEL_NAME_SIZE = 20
_BONE_NAME_SIZE = 15
_MORPH_NAME_SIZE = 15
_IK_BONE_NAME_SIZE = 20
_BONE_INTERP_SIZE = 64
_CAMERA_INTERP_SIZE = 24


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int, what: str) -> bytes:
        if self.pos + n > len(self.data):
            raise VmdFormatError(
                f"{what} の途中でデータが尽きた(offset {self.pos}, 要求 {n} bytes)"
            )
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u32(self, what: str) -> int:
        return struct.unpack("<I", self.take(4, what))[0]

    def u8(self, what: str) -> int:
        return self.take(1, what)[0]

    def f32(self, what: str, n: int = 1) -> tuple[float, ...]:
        return struct.unpack(f"<{n}f", self.take(4 * n, what))

    def at_end(self) -> bool:
        return self.pos == len(self.data)


def _check_name(
    raw: bytes,
    section: str,
    key_index: int | None,
    frame: int | None,
    warnings: list[VmdWarning],
) -> None:
    head = raw.split(b"\x00", 1)[0]
    try:
        head.decode("cp932")
    except UnicodeDecodeError:
        warnings.append(
            VmdWarning(
                code="decode-error",
                message="Shift-JISとしてデコードできない名前フィールド",
                section=section,
                key_index=key_index,
                frame=frame,
            )
        )


def read(src: str | Path | bytes) -> tuple[VmdDocument, list[VmdWarning]]:
    """戻り値のキー列はファイル内の並びのまま(ソート・重複除去をしない)。"""
    if isinstance(src, (str, Path)):
        data = Path(src).read_bytes()
    else:
        data = bytes(src)

    warnings: list[VmdWarning] = []
    r = _Reader(data)

    if len(data) < _MAGIC_SIZE:
        raise VmdFormatError("magicが不正(VMDファイルではない)")
    magic_raw = r.take(_MAGIC_SIZE, "magic")
    if magic_raw[:_MAGIC_PREFIX_SIZE] == MAGIC_V1_PREFIX:
        raise VmdFormatError(
            "v1形式(Vocaloid Motion Data file)は非対応。"
            "MMD(Multi-Model Edition以降)で保存したv2形式のみ対応する"
        )
    if magic_raw[:_MAGIC_PREFIX_SIZE] != MAGIC_V2_PREFIX:
        raise VmdFormatError("magicが不正(VMDファイルではない)")

    model_name_raw = r.take(_MODEL_NAME_SIZE, "モデル名")
    _check_name(model_name_raw, "header", None, None, warnings)

    doc = VmdDocument(magic_raw=magic_raw, model_name_raw=model_name_raw)

    count = r.u32("ボーンキー数")
    for i in range(count):
        name_raw = r.take(_BONE_NAME_SIZE, "ボーン名")
        frame = r.u32("ボーンキー frame")
        position = r.f32("ボーンキー position", 3)
        rotation = r.f32("ボーンキー rotation", 4)
        interp = r.take(_BONE_INTERP_SIZE, "ボーンキー 補間ブロック")
        _check_name(name_raw, "bone", i, frame, warnings)
        doc.bone.append(BoneKey(name_raw, frame, position, rotation, interp))

    count = r.u32("モーフキー数")
    for i in range(count):
        name_raw = r.take(_MORPH_NAME_SIZE, "モーフ名")
        frame = r.u32("モーフキー frame")
        (weight,) = r.f32("モーフキー weight")
        _check_name(name_raw, "morph", i, frame, warnings)
        doc.morph.append(MorphKey(name_raw, frame, weight))

    count = r.u32("カメラキー数")
    for _ in range(count):
        frame = r.u32("カメラキー frame")
        (distance,) = r.f32("カメラキー distance")
        position = r.f32("カメラキー position", 3)
        rotation = r.f32("カメラキー rotation", 3)
        interp = r.take(_CAMERA_INTERP_SIZE, "カメラキー 補間ブロック")
        fov = r.u32("カメラキー fov")
        perspective = r.u8("カメラキー perspective")
        doc.camera.append(
            CameraKey(frame, distance, position, rotation, interp, fov, perspective)
        )

    count = r.u32("照明キー数")
    for _ in range(count):
        frame = r.u32("照明キー frame")
        color = r.f32("照明キー color", 3)
        position = r.f32("照明キー position", 3)
        doc.light.append(LightKey(frame, color, position))

    if r.at_end():
        doc.has_self_shadow_section = False
        doc.has_ik_section = False
        warnings.append(
            VmdWarning(
                code="sections-missing",
                message="セルフ影・IKセクションが存在しない(旧版VMD)",
            )
        )
        return doc, warnings
    count = r.u32("セルフ影キー数")
    for _ in range(count):
        frame = r.u32("セルフ影キー frame")
        mode = r.u8("セルフ影キー mode")
        (distance,) = r.f32("セルフ影キー distance")
        doc.self_shadow.append(SelfShadowKey(frame, mode, distance))

    if r.at_end():
        doc.has_ik_section = False
        warnings.append(
            VmdWarning(
                code="sections-missing",
                message="IKセクションが存在しない(旧版VMD)",
            )
        )
        return doc, warnings
    count = r.u32("IKキー数")
    for i in range(count):
        frame = r.u32("IKキー frame")
        display = r.u8("IKキー display")
        ik_count = r.u32("IKボーン数")
        ik_bones = []
        for _ in range(ik_count):
            name_raw = r.take(_IK_BONE_NAME_SIZE, "IKボーン名")
            enable = r.u8("IKボーン enable")
            _check_name(name_raw, "ik_property", i, frame, warnings)
            ik_bones.append(IkBone(name_raw, enable))
        doc.ik_property.append(IkPropertyKey(frame, display, ik_bones))

    if not r.at_end():
        raise VmdFormatError(
            f"末尾に余分なデータがある(offset {r.pos}, 残り {len(data) - r.pos} bytes)"
        )
    return doc, warnings


def _fixed(raw: bytes, size: int, what: str) -> bytes:
    if len(raw) != size:
        raise VmdFormatError(f"{what} は {size} bytes 固定({len(raw)} bytes が渡された)")
    return raw


def write(doc: VmdDocument) -> bytes:
    out = bytearray()
    out += _fixed(doc.magic_raw, _MAGIC_SIZE, "magic")
    out += _fixed(doc.model_name_raw, _MODEL_NAME_SIZE, "モデル名")

    out += struct.pack("<I", len(doc.bone))
    for k in doc.bone:
        out += _fixed(k.name_raw, _BONE_NAME_SIZE, "ボーン名")
        out += struct.pack("<I3f4f", k.frame, *k.position, *k.rotation)
        out += _fixed(k.interpolation, _BONE_INTERP_SIZE, "ボーン補間ブロック")

    out += struct.pack("<I", len(doc.morph))
    for k in doc.morph:
        out += _fixed(k.name_raw, _MORPH_NAME_SIZE, "モーフ名")
        out += struct.pack("<If", k.frame, k.weight)

    out += struct.pack("<I", len(doc.camera))
    for k in doc.camera:
        out += struct.pack("<If3f3f", k.frame, k.distance, *k.position, *k.rotation)
        out += _fixed(k.interpolation, _CAMERA_INTERP_SIZE, "カメラ補間ブロック")
        out += struct.pack("<IB", k.fov, k.perspective)

    out += struct.pack("<I", len(doc.light))
    for k in doc.light:
        out += struct.pack("<I3f3f", k.frame, *k.color, *k.position)

    if doc.has_self_shadow_section:
        out += struct.pack("<I", len(doc.self_shadow))
        for k in doc.self_shadow:
            out += struct.pack("<IBf", k.frame, k.mode, k.distance)

        if doc.has_ik_section:
            out += struct.pack("<I", len(doc.ik_property))
            for k in doc.ik_property:
                out += struct.pack("<IBI", k.frame, k.display, len(k.ik_bones))
                for ik in k.ik_bones:
                    out += _fixed(ik.name_raw, _IK_BONE_NAME_SIZE, "IKボーン名")
                    out += struct.pack("<B", ik.enable)

    return bytes(out)


def write_file(doc: VmdDocument, path: str | Path) -> None:
    """書き出しが途中で失敗しても、既存の path の内容は残る。"""
    path = Path(path)
    data = write(doc)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _normalize_keys(keys, identity, sort_key, section, warnings):
    last: dict = {}
    dropped = []
    for i, k in enumerate(keys):
        ident = identity(k)
        if ident in last:
            dropped.append(last[ident])
        last[ident] = (i, k)

    for idx, k in dropped:
        warnings.append(
            VmdWarning(
                code="normalize-duplicate",
                message="同一キーの重複(後に現れたキーを採用し、このキーを破棄)",
                section=section,
                key_index=idx,
                frame=k.frame,
            )
        )

    kept_in_file_order = [k for _, k in sorted(last.values(), key=lambda t: t[0])]
    result = sorted(kept_in_file_order, key=sort_key)
    if result != kept_in_file_order:
        warnings.append(
            VmdWarning(
                code="normalize-sorted",
                message="キーを並べ替えた(ファイル内で順不同だった)",
                section=section,
            )
        )
    return result


def _by_frame(k):
    return k.frame


def _by_name_frame(k):
    return (k.name_raw, k.frame)


def normalize(
    doc: VmdDocument, sections: list[str] | None = None
) -> tuple[VmdDocument, list[VmdWarning]]:
    """sections が None なら全セクションを対象にする。対象外のセクションは渡した値のまま返す。"""
    targets = _SECTIONS if sections is None else tuple(sections)
    unknown = set(targets) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"不明なセクション指定: {sorted(unknown)}")

    keyers = {
        "bone": _by_name_frame,
        "morph": _by_name_frame,
        "camera": _by_frame,
        "light": _by_frame,
        "self_shadow": _by_frame,
        "ik_property": _by_frame,
    }
    warnings: list[VmdWarning] = []
    updates = {}
    for section in targets:
        keyer = keyers[section]
        updates[section] = _normalize_keys(
            getattr(doc, section), keyer, keyer, section, warnings
        )
    return replace(doc, **updates), warnings


def _neutral_morph(name_raw: bytes) -> MorphKey:
    return MorphKey(name_raw, 0, 0.0)


def _neutral_bone(name_raw: bytes) -> BoneKey:
    from .reduce import BONE_LINEAR_INTERP

    return BoneKey(name_raw, 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)


_NAMED_SECTION_NEUTRAL_BUILDERS = {"morph": _neutral_morph, "bone": _neutral_bone}


def ensure_frame0_neutral_keys(
    doc: VmdDocument, sections: tuple[str, ...] = ("morph",)
) -> VmdDocument:
    """補った中立キーは各セクションの末尾に並ぶ。フレーム順に揃えるには続けて normalize を呼ぶ。"""
    invalid = set(sections) - set(_NAMED_SECTION_NEUTRAL_BUILDERS)
    if invalid:
        raise ValueError(
            f"frame-0 中立キーは名前付きセクション(morph/bone)のみ対象: {sorted(invalid)} は非対応"
        )
    updates = {}
    for section in sections:
        keys = getattr(doc, section)
        have_zero = {k.name_raw for k in keys if k.frame == 0}
        missing = [nr for nr in sorted({k.name_raw for k in keys}) if nr not in have_zero]
        if missing:
            builder = _NAMED_SECTION_NEUTRAL_BUILDERS[section]
            updates[section] = list(keys) + [builder(nr) for nr in missing]
    return replace(doc, **updates) if updates else doc
