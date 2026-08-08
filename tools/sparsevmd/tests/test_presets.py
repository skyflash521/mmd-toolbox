"""sparsevmd プリセット・許容誤差解決のテスト。

`resolve_tolerances(preset_name, overrides)` は、プリセット名から許容誤差の既定値を引き、
個別オプションの明示値があればそれを優先する。値の検証もここで行う:
FOV以外は0以上の有限値(0は整数フレーム完全一致を要求)、FOVは0.5以上。
NaN/Inf/負値、未知プリセット、FOV<0.5 は ValueError。
"""

import math

import pytest

from sparsevmd import presets


def test_preset_names():
    # 公開する3プリセットの名前を固定する。
    assert presets.PRESET_NAMES == ("precise", "balanced", "aggressive")


def test_balanced_values():
    # balanced プリセットの既定値を全件固定する。
    t = presets.resolve_tolerances("balanced")
    assert t.bone_pos == pytest.approx(0.01)
    assert t.bone_rot == pytest.approx(0.10)
    assert t.camera_pos == pytest.approx(0.02)
    assert t.camera_rot == pytest.approx(0.05)
    assert t.camera_distance == pytest.approx(0.02)
    assert t.camera_fov == pytest.approx(0.50)


def test_precise_values():
    # precise プリセットの既定値を全件固定する。
    t = presets.resolve_tolerances("precise")
    assert t.bone_pos == pytest.approx(0.005)
    assert t.bone_rot == pytest.approx(0.05)
    assert t.camera_pos == pytest.approx(0.01)
    assert t.camera_rot == pytest.approx(0.02)
    assert t.camera_distance == pytest.approx(0.01)
    assert t.camera_fov == pytest.approx(0.50)


def test_aggressive_values():
    # aggressive プリセットの既定値を全件固定する。
    t = presets.resolve_tolerances("aggressive")
    assert t.bone_pos == pytest.approx(0.05)
    assert t.bone_rot == pytest.approx(0.50)
    assert t.camera_pos == pytest.approx(0.10)
    assert t.camera_rot == pytest.approx(0.25)
    assert t.camera_distance == pytest.approx(0.10)
    assert t.camera_fov == pytest.approx(1.00)


def test_override_precedence():
    # 個別オプション明示はプリセットより優先。指定外はプリセット値のまま。
    t = presets.resolve_tolerances("balanced", {"camera_fov": 0.75, "bone_pos": 0.001})
    assert t.camera_fov == pytest.approx(0.75)
    assert t.bone_pos == pytest.approx(0.001)
    assert t.camera_pos == pytest.approx(0.02)  # 非指定はbalanced


def test_tolerances_fields():
    # 公開する許容誤差フィールド集合(6項目)を固定する。
    t = presets.resolve_tolerances("balanced")
    assert set(vars(t)) == {
        "bone_pos",
        "bone_rot",
        "camera_pos",
        "camera_rot",
        "camera_distance",
        "camera_fov",
    }


@pytest.mark.parametrize(
    "field", ["bone_pos", "bone_rot", "camera_pos", "camera_rot", "camera_distance"]
)
def test_zero_allowed_for_non_fov(field):
    # 視野角以外の0は許容(整数フレーム完全一致の要求)。
    t = presets.resolve_tolerances("balanced", {field: 0.0})
    assert getattr(t, field) == 0.0


# NOTE: 「単位付き文字列はエラー」は二重に守る。CLI(argparse)層が parse 時に
# 弾いて終了コード2にするのに加え、resolve_tolerances 自体も float() 不能な値を
# ValueError とする(下の test_non_numeric_override_rejected)。終了コードへの対応づけは
# CLI テストで別途検証する。


def test_override_stored_as_float():
    # override 値(int・数値文字列)は float へ変換して格納する(非floatのまま残らない)。
    t = presets.resolve_tolerances(
        "balanced", {"bone_pos": 0, "camera_fov": 1, "camera_pos": "0.03"}
    )
    assert isinstance(t.bone_pos, float) and t.bone_pos == 0.0
    assert isinstance(t.camera_fov, float) and t.camera_fov == 1.0
    assert isinstance(t.camera_pos, float) and t.camera_pos == pytest.approx(0.03)


@pytest.mark.parametrize("bad", [None, object(), "0.5deg", "abc"])
def test_non_numeric_override_rejected(bad):
    # float() できない値(None/非数値/単位付き文字列)は TypeError を漏らさず ValueError。
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"bone_pos": bad})


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        presets.resolve_tolerances("ultra")


def test_fov_floor():
    # 視野角は0.5以上必須。0.5未満は ValueError。
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"camera_fov": 0.4})
    # ちょうど0.5は許容。
    assert presets.resolve_tolerances("balanced", {"camera_fov": 0.5}).camera_fov == 0.5


def test_negative_rejected():
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"bone_pos": -0.01})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_rejected(bad):
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"camera_pos": bad})


def test_unknown_override_key_rejected():
    # タイポ等の未知キーは黙殺せずエラーにする。
    with pytest.raises(ValueError):
        presets.resolve_tolerances("balanced", {"bone_position": 0.01})
