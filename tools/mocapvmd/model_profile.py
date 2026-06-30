"""mocap用モデルプロファイルの解決。

共通 PmxModel を、mocapの標準ボーンロール・マーカー/派生特徴へ対応付ける。
PMX指定時はそのモデルから標準ロールを名前で解決し、未指定時は既定モデル
プロファイルを使う。必須標準ロールが欠けると MocapModelProfileError、PMX形式
不正は共通側の PmxFormatError を透過する。
"""

from dataclasses import dataclass, field

from pmx.io import read_pmx
from pmx.types import PmxModel

from .default_profile import (
    FEATURE_BINDINGS,
    MARKER_BINDINGS,
    ROLE_TO_INDEX,
    build_default_model,
)

# 標準ロール名 -> 標準MMDボーン名。PMX指定時のロール解決に使う。
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
    """mocapモデルプロファイルの解決失敗(必須標準ロール不足など)。"""


@dataclass
class MarkerBinding:
    marker: str
    bone: int  # required_bones が指すボーンindex
    offset: tuple[float, float, float]
    category: str
    weight: float


@dataclass
class FeatureBinding:
    feature: str
    kind: str  # 現状は "vector"(2ロール間のベクトル a-b)
    a: str  # 始点ロール
    b: str  # 終点ロール
    weight: float


@dataclass
class MocapModelProfile:
    model: PmxModel
    required_bones: dict[str, int]  # 標準ロール名 -> ボーンindex
    marker_bindings: dict[str, MarkerBinding]
    feature_bindings: dict[str, FeatureBinding]
    source: str  # "pmx" | "default"
    warnings: tuple[str, ...] = field(default_factory=tuple)


def validate_required_roles(profile: MocapModelProfile) -> None:
    """必須標準ロールが揃っているか検証する。不足は MocapModelProfileError。"""
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
    """PMXパス指定時はそのモデル、未指定時は既定モデルプロファイルを解決する。"""
    if pmx_path is None:
        model = build_default_model()
        required_bones = dict(ROLE_TO_INDEX)
        source = "default"
    else:
        model = read_pmx(pmx_path)  # PmxFormatError は透過
        required_bones = _resolve_roles_from_model(model)
        source = "pmx"

    marker_bindings = {
        marker: MarkerBinding(
            marker=marker,
            bone=required_bones[role],
            offset=(0.0, 0.0, 0.0),
            category=category,
            weight=weight,
        )
        for marker, (role, category, weight) in MARKER_BINDINGS.items()
    }
    feature_bindings = {
        feature: FeatureBinding(
            feature=feature,
            kind="vector",
            a=a_role,
            b=b_role,
            weight=weight,
        )
        for feature, (a_role, b_role, weight) in FEATURE_BINDINGS.items()
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
