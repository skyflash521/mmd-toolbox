from dataclasses import dataclass, field

MAGIC_V2_PREFIX = b"Vocaloid Motion Data 0002"
MAGIC_V1_PREFIX = b"Vocaloid Motion Data file"


class VmdFormatError(Exception):
    pass


@dataclass
class VmdWarning:
    code: str
    message: str
    section: str | None = None
    key_index: int | None = None
    frame: int | None = None


def _decode_name(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("cp932", errors="replace")


@dataclass
class BoneKey:
    """name_raw は15バイト、interpolation は64バイト固定。rotation はクォータニオンで成分順は (x, y, z, w)。"""

    name_raw: bytes
    frame: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    interpolation: bytes

    @property
    def name(self) -> str:
        return _decode_name(self.name_raw)

    def control_points(self) -> dict[str, tuple[int, int, int, int]]:
        """チャンネル "X"・"Y"・"Z"・"R" ごとの (x1, y1, x2, y2) を返す。"""
        b = self.interpolation
        return {
            "X": (b[0], b[4], b[8], b[12]),
            "Y": (b[1], b[5], b[9], b[13]),
            "Z": (b[17], b[6], b[10], b[14]),
            "R": (b[18], b[7], b[11], b[15]),
        }


@dataclass
class MorphKey:
    """name_raw は15バイト固定。"""

    name_raw: bytes
    frame: int
    weight: float

    @property
    def name(self) -> str:
        return _decode_name(self.name_raw)


@dataclass
class CameraKey:
    """position はカメラ中心(カメラ本体の位置ではない)、rotation の単位はラジアン。

    interpolation は24バイト固定。perspective は 0 が透視投影 ON、1 が OFF。
    """

    frame: int
    distance: float
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    interpolation: bytes
    fov: int
    perspective: int


@dataclass
class LightKey:
    frame: int
    color: tuple[float, float, float]
    position: tuple[float, float, float]


@dataclass
class SelfShadowKey:
    """distance はファイル格納値のまま(距離へ換算しない)。"""

    frame: int
    mode: int
    distance: float


@dataclass
class IkBone:
    """name_raw は20バイト固定。"""

    name_raw: bytes
    enable: int

    @property
    def name(self) -> str:
        return _decode_name(self.name_raw)


@dataclass
class IkPropertyKey:
    frame: int
    display: int
    ik_bones: list[IkBone] = field(default_factory=list)


@dataclass
class VmdDocument:
    """magic_raw は30バイト、model_name_raw は20バイト固定。"""

    magic_raw: bytes = MAGIC_V2_PREFIX + b"\x00" * 5
    model_name_raw: bytes = b"\x00" * 20
    bone: list[BoneKey] = field(default_factory=list)
    morph: list[MorphKey] = field(default_factory=list)
    camera: list[CameraKey] = field(default_factory=list)
    light: list[LightKey] = field(default_factory=list)
    self_shadow: list[SelfShadowKey] = field(default_factory=list)
    ik_property: list[IkPropertyKey] = field(default_factory=list)
    has_self_shadow_section: bool = True
    has_ik_section: bool = True

    @property
    def model_name(self) -> str:
        return _decode_name(self.model_name_raw)
