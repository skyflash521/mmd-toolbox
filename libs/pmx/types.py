from dataclasses import dataclass


class PmxFormatError(Exception):
    pass


@dataclass
class PmxWarning:
    code: str
    message: str
    bone_index: int | None = None


@dataclass
class PmxBone:
    name: str
    name_raw: bytes
    english_name: str
    english_name_raw: bytes
    parent: int | None
    position: tuple[float, float, float]
    movable: bool
    rotatable: bool
    flags: int


@dataclass
class PmxModel:
    bones: tuple[PmxBone, ...]
    name_to_index: dict[str, int]
    warnings: tuple[PmxWarning, ...]
