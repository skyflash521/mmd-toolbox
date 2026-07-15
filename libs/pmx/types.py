"""PMXデータモデル。

読み取り専用。ボーン階層・基準位置・移動/回転可否のみを保持し、
FK近似に不要なデータ(頂点・材質・モーフ等)は読み飛ばす。
"""

from dataclasses import dataclass


class PmxFormatError(Exception):
    """PMXファイル構造の異常。"""


@dataclass
class PmxWarning:
    """続行可能な事象の構造化報告。"""

    code: str
    message: str
    bone_index: int | None = None


@dataclass
class PmxBone:
    name: str
    name_raw: bytes  # TextBufの生バイト(エンコードはモデルの文字コードに従う)
    english_name: str
    english_name_raw: bytes
    parent: int | None  # 親ボーンindex。非参照(-1)はNone
    position: tuple[float, float, float]  # モデル原点からの基準位置
    movable: bool
    rotatable: bool
    flags: int  # ボーンフラグ(16bit)生値


@dataclass
class PmxModel:
    bones: tuple[PmxBone, ...]
    name_to_index: dict[str, int]
    warnings: tuple[PmxWarning, ...]
