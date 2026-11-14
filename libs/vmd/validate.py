import math

from .types import VmdDocument, VmdValueError


def _check_finite(section: str, index: int, field: str, value) -> None:
    values = value if isinstance(value, tuple) else (value,)
    if not all(math.isfinite(v) for v in values):
        raise VmdValueError(
            f"{section} キー {index} の {field} が非有限値: {value}",
            section=section,
            index=index,
            field=field,
        )


def validate_values(doc: VmdDocument) -> None:
    """最初に見つけた不正値を VmdValueError で報告する。"""
    for i, k in enumerate(doc.bone):
        _check_finite("bone", i, "position", k.position)
        _check_finite("bone", i, "rotation", k.rotation)
        if all(c == 0.0 for c in k.rotation):
            raise VmdValueError(
                f"bone キー {i} の rotation がゼロノルム",
                section="bone",
                index=i,
                field="rotation",
            )
    for i, k in enumerate(doc.morph):
        _check_finite("morph", i, "weight", k.weight)
    for i, k in enumerate(doc.camera):
        _check_finite("camera", i, "distance", k.distance)
        _check_finite("camera", i, "position", k.position)
        _check_finite("camera", i, "rotation", k.rotation)
    for i, k in enumerate(doc.light):
        _check_finite("light", i, "color", k.color)
        _check_finite("light", i, "position", k.position)
    for i, k in enumerate(doc.self_shadow):
        _check_finite("self_shadow", i, "distance", k.distance)
