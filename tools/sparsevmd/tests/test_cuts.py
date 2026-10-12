import argparse

import pytest

from sparsevmd.cli import _CUT_THRESHOLD_BONE, _CUT_THRESHOLD_CAMERA


def test_parse_camera_threshold():
    assert _CUT_THRESHOLD_CAMERA("5.0,20.0,5.0") == (5.0, 20.0, 5.0)


def test_parse_bone_threshold():
    assert _CUT_THRESHOLD_BONE("1.0,30.0") == (1.0, 30.0)


@pytest.mark.parametrize("bad", ["5.0,20.0", "5.0,20.0,5.0,1.0", "abc", ""])
def test_parse_camera_threshold_wrong_arity(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        _CUT_THRESHOLD_CAMERA(bad)


@pytest.mark.parametrize("bad", ["-1.0,20.0,5.0", "5.0,nan,5.0", "5.0,inf,5.0", "5.0,,5.0"])
def test_parse_camera_threshold_bad_values(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        _CUT_THRESHOLD_CAMERA(bad)


@pytest.mark.parametrize("bad", ["1.0", "1.0,30.0,5.0", "abc", ""])
def test_parse_bone_threshold_wrong_arity(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        _CUT_THRESHOLD_BONE(bad)


@pytest.mark.parametrize("bad", ["-1.0,30.0", "1.0,nan", "1.0,inf", "1.0,"])
def test_parse_bone_threshold_bad_values(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        _CUT_THRESHOLD_BONE(bad)
