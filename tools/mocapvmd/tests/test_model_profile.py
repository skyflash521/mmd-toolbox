"""mocap用モデルプロファイル解決のテスト。

PMX未指定時は既定モデルプロファイル、指定時はそのPMXから標準ロールを解決し、
共通 MocapModelProfile(モデル・必須ロール索引・マーカー/派生特徴binding)を返す。
必須標準ロールが欠けると MocapModelProfileError、PMX形式不正は PmxFormatError。
"""

import struct

import pytest

from mocapvmd.default_profile import ROLE_TO_INDEX as _DEFAULT_ROLE_TO_INDEX
from mocapvmd.default_profile import build_default_model
from mocapvmd.model_profile import (
    STANDARD_BONE_NAMES,
    MocapModelProfile,
    MocapModelProfileError,
    load_mocap_profile,
    validate_required_roles,
)
from pmx.types import PmxFormatError, PmxModel

_EXPECTED_MARKERS = {
    "center", "pelvis", "chest", "head",
    "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
    "wrist_l", "wrist_r", "knee_l", "knee_r",
    "ankle_l", "ankle_r", "toe_l", "toe_r",
}
_EXPECTED_FEATURES = {
    "shoulder_line", "hip_line", "torso_axis",
    "forearm_l", "forearm_r", "shin_l", "shin_r",
}
_CATEGORIES = {"center", "torso", "head", "arms", "wrists", "legs", "feet"}


# ---------------------------------------------------------------------------
# 標準ボーン名だけを持つ最小PMXのビルダー(平坦階層)
# ---------------------------------------------------------------------------


def _textbuf(s):
    b = s.encode("utf-16-le")
    return struct.pack("<i", len(b)) + b


def _build_pmx(names):
    out = bytearray()
    out += b"PMX "
    out += struct.pack("<f", 2.0)
    out += struct.pack("<B", 8)
    out += bytes([0, 0, 1, 1, 1, 1, 1, 1])  # utf16, addUV0, 各indexサイズ1
    for _ in range(4):
        out += _textbuf("")  # モデル情報
    out += struct.pack("<i", 0)  # 頂点
    out += struct.pack("<i", 0)  # 面
    out += struct.pack("<i", 0)  # テクスチャ
    out += struct.pack("<i", 0)  # 材質
    out += struct.pack("<i", len(names))
    for i, name in enumerate(names):
        out += _textbuf(name)
        out += _textbuf("")
        out += struct.pack("<3f", 0.0, float(i), 0.0)  # 位置
        out += struct.pack("<b", -1)  # 親(なし)
        out += struct.pack("<i", 0)  # 変形階層
        out += struct.pack("<H", 0x0002 | 0x0004)  # 回転+移動可
        out += struct.pack("<3f", 0.0, 0.0, 0.0)  # 接続先オフセット
    out += struct.pack("<i", 0) * 4  # モーフ/表示枠/剛体/Joint
    return bytes(out)


def _write_pmx(tmp_path, names):
    p = tmp_path / "model.pmx"
    p.write_bytes(_build_pmx(names))
    return str(p)


# ---------------------------------------------------------------------------
# 既定モデルプロファイル
# ---------------------------------------------------------------------------


def test_default_profile_loads():
    p = load_mocap_profile(None)
    assert isinstance(p, MocapModelProfile)
    assert p.source == "default"
    assert isinstance(p.model, PmxModel)
    assert set(p.required_bones) >= set(STANDARD_BONE_NAMES)
    assert set(p.marker_bindings) == _EXPECTED_MARKERS
    assert set(p.feature_bindings) == _EXPECTED_FEATURES


def test_default_marker_bindings_valid():
    p = load_mocap_profile(None)
    n = len(p.model.bones)
    for marker, mb in p.marker_bindings.items():
        assert mb.marker == marker
        assert 0 <= mb.bone < n
        assert mb.offset == (0.0, 0.0, 0.0)
        assert mb.category in _CATEGORIES
        assert 0.0 < mb.weight <= 1.0


def test_default_feature_bindings_valid():
    p = load_mocap_profile(None)
    for feature, fb in p.feature_bindings.items():
        assert fb.feature == feature
        assert fb.kind == "vector"
        assert fb.a in p.required_bones
        assert fb.b in p.required_bones
        assert 0.0 < fb.weight <= 1.0


def test_validate_required_roles_ok_for_default():
    validate_required_roles(load_mocap_profile(None))  # 例外なし


def test_default_role_index_matches_standard_names():
    # 既定モデルでの標準名解決が default_profile の ROLE_TO_INDEX と一致する。
    model = build_default_model()
    for role, name in STANDARD_BONE_NAMES.items():
        assert _DEFAULT_ROLE_TO_INDEX[role] == model.name_to_index[name]


# ---------------------------------------------------------------------------
# PMX指定
# ---------------------------------------------------------------------------


def test_pmx_profile_loads(tmp_path):
    path = _write_pmx(tmp_path, list(STANDARD_BONE_NAMES.values()))
    p = load_mocap_profile(path)
    # PMX指定時も既定時と同じ MocapModelProfile 型・binding を返す。
    assert isinstance(p, MocapModelProfile)
    assert p.source == "pmx"
    assert isinstance(p.model, PmxModel)
    assert set(p.required_bones) >= set(STANDARD_BONE_NAMES)
    assert set(p.marker_bindings) == _EXPECTED_MARKERS
    assert set(p.feature_bindings) == _EXPECTED_FEATURES
    n = len(p.model.bones)
    for marker, mb in p.marker_bindings.items():
        assert mb.marker == marker
        assert 0 <= mb.bone < n
        assert mb.offset == (0.0, 0.0, 0.0)
        assert mb.category in _CATEGORIES
        assert 0.0 < mb.weight <= 1.0
    for feature, fb in p.feature_bindings.items():
        assert fb.feature == feature
        assert fb.kind == "vector"
        assert fb.a in p.required_bones
        assert fb.b in p.required_bones
        assert 0.0 < fb.weight <= 1.0
    # 名前解決が正しいindexを指す。
    for role, name in STANDARD_BONE_NAMES.items():
        assert p.model.bones[p.required_bones[role]].name == name


def test_missing_required_role_raises(tmp_path):
    names = [n for n in STANDARD_BONE_NAMES.values() if n != STANDARD_BONE_NAMES["wrist_r"]]
    path = _write_pmx(tmp_path, names)
    with pytest.raises(MocapModelProfileError):
        load_mocap_profile(path)


def test_pmx_format_error_propagates(tmp_path):
    bad = tmp_path / "bad.pmx"
    bad.write_bytes(b"NOTPMX")
    with pytest.raises(PmxFormatError):
        load_mocap_profile(str(bad))


def test_standard_bone_names_cover_required_roles():
    # 標準ロール名定義がマーカー/特徴の参照ロールを過不足なく含む。
    referenced = set()
    for _role in _DEFAULT_ROLE_TO_INDEX:
        referenced.add(_role)
    assert set(STANDARD_BONE_NAMES) == referenced
