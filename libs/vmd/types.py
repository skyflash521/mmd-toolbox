"""VMDデータモデル(vmd-io.md §2)。

数値・生バイトはファイル格納値をそのまま保持する。意味的解釈
(距離の負値、セルフ影distanceのエンコード等)はここでは行わない。
"""

from dataclasses import dataclass, field

# magicフィールド(30バイト)の先頭25バイト
MAGIC_V2_PREFIX = b"Vocaloid Motion Data 0002"
MAGIC_V1_PREFIX = b"Vocaloid Motion Data file"


class VmdFormatError(Exception):
    """ファイル構造の異常(vmd-io.md §7)。"""


@dataclass
class VmdWarning:
    """続行可能な事象の構造化報告(mmd_toolbox.md §3)。"""

    code: str
    message: str
    section: str | None = None
    key_index: int | None = None
    frame: int | None = None


def _decode_name(raw: bytes) -> str:
    """null終端までをcp932でデコードする(不能バイトは置換文字)。"""
    return raw.split(b"\x00", 1)[0].decode("cp932", errors="replace")


@dataclass
class BoneKey:
    name_raw: bytes  # 15バイト固定。null終端後の残バイトを含む
    frame: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # クォータニオン x,y,z,w
    interpolation: bytes  # 64バイト生

    @property
    def name(self) -> str:
        return _decode_name(self.name_raw)

    def control_points(self) -> dict[str, tuple[int, int, int, int]]:
        """チャンネル 'X'/'Y'/'Z'/'R' → (x1, y1, x2, y2)。

        Byte[2], Byte[3](Z_x1, R_x1の位置)は物理演算フラグで上書きされる
        ことがあるため、シフトコピー側(Byte[17], Byte[18])から復元する
        (vmd-io.md §2.2)。
        """
        b = self.interpolation
        return {
            "X": (b[0], b[4], b[8], b[12]),
            "Y": (b[1], b[5], b[9], b[13]),
            "Z": (b[17], b[6], b[10], b[14]),
            "R": (b[18], b[7], b[11], b[15]),
        }


@dataclass
class MorphKey:
    name_raw: bytes  # 15バイト固定
    frame: int
    weight: float

    @property
    def name(self) -> str:
        return _decode_name(self.name_raw)


@dataclass
class CameraKey:
    frame: int
    distance: float
    position: tuple[float, float, float]  # カメラ中心
    rotation: tuple[float, float, float]  # 角度(ラジアン)
    interpolation: bytes  # 24バイト生
    fov: int
    perspective: int  # 0=ON, 1=OFF


@dataclass
class LightKey:
    frame: int
    color: tuple[float, float, float]
    position: tuple[float, float, float]


@dataclass
class SelfShadowKey:
    frame: int
    mode: int
    distance: float  # 格納値そのまま


@dataclass
class IkBone:
    name_raw: bytes  # 20バイト固定
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
    magic_raw: bytes = MAGIC_V2_PREFIX + b"\x00" * 5  # 30バイト固定
    model_name_raw: bytes = b"\x00" * 20  # 20バイト固定(v2)
    bone: list[BoneKey] = field(default_factory=list)
    morph: list[MorphKey] = field(default_factory=list)
    camera: list[CameraKey] = field(default_factory=list)
    light: list[LightKey] = field(default_factory=list)
    self_shadow: list[SelfShadowKey] = field(default_factory=list)
    ik_property: list[IkPropertyKey] = field(default_factory=list)
    # 旧版VMDで省略された後方セクションの記録(vmd-io.md §2.1)
    has_self_shadow_section: bool = True
    has_ik_section: bool = True

    @property
    def model_name(self) -> str:
        return _decode_name(self.model_name_raw)
