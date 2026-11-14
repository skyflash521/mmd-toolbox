from dataclasses import dataclass, field
from typing import Literal

from pmx.io import read_pmx
from pmx.types import PmxModel

from .default_profile import (
    FEATURE_BINDINGS,
    MARKER_BINDINGS,
    ROLE_TO_INDEX,
    build_default_model,
)

STANDARD_BONE_NAMES = {
    "center": "センター",
    "groove": "グルーブ",
    "lower_body": "下半身",
    "upper_body": "上半身",
    "upper_body2": "上半身2",
    "neck": "首",
    "head": "頭",
    "shoulder_l": "左肩",
    "shoulder_r": "右肩",
    "elbow_l": "左ひじ",
    "elbow_r": "右ひじ",
    "wrist_l": "左手首",
    "wrist_r": "右手首",
    "hip_l": "左足",
    "hip_r": "右足",
    "knee_l": "左ひざ",
    "knee_r": "右ひざ",
    "ankle_l": "左足首",
    "ankle_r": "右足首",
    "toe_l": "左つま先",
    "toe_r": "右つま先",
}


class MocapModelProfileError(Exception):
    pass


@dataclass
class MarkerBinding:
    """bone は MocapModelProfile.model.bones の添字。"""

    marker: str
    bone: int
    offset: tuple[float, float, float]
    category: str
    weight: float


@dataclass
class FeatureBinding:
    """特徴量は、ロール a のボーン位置からロール b のボーン位置を引いたベクトル(a − b)。"""

    feature: str
    kind: Literal["vector"]
    a: str
    b: str
    weight: float


@dataclass
class MocapModelProfile:
    """required_bones は標準ロール名から model.bones の添字への対応。"""

    model: PmxModel
    required_bones: dict[str, int]
    marker_bindings: dict[str, MarkerBinding]
    feature_bindings: dict[str, FeatureBinding]
    source: Literal["pmx", "default"]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def validate_required_roles(profile: MocapModelProfile) -> None:
    missing = set(STANDARD_BONE_NAMES) - set(profile.required_bones)
    if missing:
        raise MocapModelProfileError(
            f"必須標準ロールが不足: {sorted(missing)}"
        )


def _resolve_roles_from_model(model: PmxModel) -> dict[str, int]:
    required: dict[str, int] = {}
    for role, name in STANDARD_BONE_NAMES.items():
        idx = model.name_to_index.get(name)
        if idx is None:
            raise MocapModelProfileError(
                f"必須標準ボーンがモデルに存在しない: role={role}"
            )
        required[role] = idx
    return required


def load_mocap_profile(pmx_path: str | None) -> MocapModelProfile:
    """必須標準ボーンの欠落は MocapModelProfileError、PMX 形式の不正は PmxFormatError を送出する。"""
    if pmx_path is None:
        model = build_default_model()
        required_bones = dict(ROLE_TO_INDEX)
        source = "default"
    else:
        model = read_pmx(pmx_path)
        required_bones = _resolve_roles_from_model(model)
        source = "pmx"

    marker_bindings = {
        marker: MarkerBinding(
            marker=marker,
            bone=required_bones[spec.role],
            offset=(0.0, 0.0, 0.0),
            category=spec.category,
            weight=spec.fit_weight,
        )
        for marker, spec in MARKER_BINDINGS.items()
    }
    feature_bindings = {
        feature: FeatureBinding(
            feature=feature,
            kind="vector",
            a=spec.head_role,
            b=spec.tail_role,
            weight=spec.weight,
        )
        for feature, spec in FEATURE_BINDINGS.items()
    }

    profile = MocapModelProfile(
        model=model,
        required_bones=required_bones,
        marker_bindings=marker_bindings,
        feature_bindings=feature_bindings,
        source=source,
    )
    validate_required_roles(profile)
    return profile
