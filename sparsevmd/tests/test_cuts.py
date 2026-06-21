"""カット閾値パースのテスト(sparsevmd.md §2.6)。

`POS,ROT,DIST`(camera)/ `POS,ROT`(bone)のCLI閾値文字列を解析する。
個数違い・非数値・負値・非有限(nan/inf)・空白混入・空要素は ValueError。
不連続検出・必須境界の本体テストは mmd_toolbox.vmd.cuts にある。
"""

import pytest

from sparsevmd.cuts import parse_cut_threshold_bone, parse_cut_threshold_camera


def test_parse_camera_threshold():
    assert parse_cut_threshold_camera("5.0,20.0,5.0") == (5.0, 20.0, 5.0)


def test_parse_bone_threshold():
    assert parse_cut_threshold_bone("1.0,30.0") == (1.0, 30.0)


@pytest.mark.parametrize("bad", ["5.0,20.0", "5.0,20.0,5.0,1.0", "abc", ""])
def test_parse_camera_threshold_wrong_arity(bad):
    with pytest.raises(ValueError):
        parse_cut_threshold_camera(bad)


@pytest.mark.parametrize(
    "bad", ["-1.0,20.0,5.0", "5.0,nan,5.0", "5.0,inf,5.0", "5.0, 20.0,5.0", "5.0,,5.0"]
)
def test_parse_camera_threshold_bad_values(bad):
    # 負値・非有限(nan/inf)・空白混入・空要素はエラー。
    with pytest.raises(ValueError):
        parse_cut_threshold_camera(bad)


@pytest.mark.parametrize("bad", ["1.0", "1.0,30.0,5.0", "abc", ""])
def test_parse_bone_threshold_wrong_arity(bad):
    with pytest.raises(ValueError):
        parse_cut_threshold_bone(bad)


@pytest.mark.parametrize("bad", ["-1.0,30.0", "1.0,nan", "1.0,inf", "1.0, 30.0", "1.0,"])
def test_parse_bone_threshold_bad_values(bad):
    with pytest.raises(ValueError):
        parse_cut_threshold_bone(bad)
