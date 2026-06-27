"""表現空間ノイズ除去オーケストレーションのテスト。

入力ボーンキーを、既定モデルプロファイル(または指定PMX)のFK・マーカー平滑化・
姿勢フィットを通して密キー列へ変換する。モデルが扱うボーンは密キー化し、扱わない
ボーン(指など)は原キーのまま通す。
"""

import pytest

from mmd_toolbox.vmd.reduce import BONE_LINEAR_INTERP
from mmd_toolbox.vmd.types import BoneKey

from .helpers import BONE_NONLINEAR, bone

try:
    from mocapvmd.pose_denoise import apply_pose_denoise

    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False

# impl pending: pose-denoise Step7 pose_denoise
pytestmark = (
    []
    if _IMPORT_OK
    else pytest.mark.skip(reason="impl pending: pose-denoise Step7 pose_denoise")
)


def test_empty_input_returns_empty():
    assert apply_pose_denoise([], pmx_path=None) == []


def test_dense_keys_for_model_bones():
    # センターは 2..5、頭は 0..10 に入力。全体フレーム範囲(0..10)で密化される。
    keys = [
        bone("センター", 2, pos=(0.2, 0.0, 0.0)),
        bone("センター", 5, pos=(0.5, 0.0, 0.0)),
        bone("頭", 0),
        bone("頭", 10, rot=(0.0, 0.0, 0.05, 0.99875)),
    ]
    out = apply_pose_denoise(keys, pmx_path=None)
    # 入力tracksにあるモデルボーンだけを出す(余計なモデルボーンを密キー化しない)。
    assert {k.name for k in out} == {"センター", "頭"}
    # 全体フレーム範囲で密化(ボーンごとの min..max ではない)。
    center_frames = sorted(k.frame for k in out if k.name == "センター")
    assert center_frames == list(range(0, 11))
    head_frames = sorted(k.frame for k in out if k.name == "頭")
    assert head_frames == list(range(0, 11))


def test_dense_keys_use_linear_interpolation():
    keys = [bone("センター", 0), bone("センター", 6, pos=(0.5, 0.0, 0.0))]
    out = apply_pose_denoise(keys, pmx_path=None)
    center = [k for k in out if k.name == "センター"]
    assert center
    for k in center:
        assert k.interpolation == BONE_LINEAR_INTERP


def test_non_model_bone_passes_through():
    # モデル外ボーンは name_raw・position・rotation・補間まで原キーのまま通す。
    thumb_key = bone(
        "左親指１", 0, pos=(0.2, 0.1, 0.0), rot=(0.0, 0.0, 0.1, 0.995), interp=BONE_NONLINEAR
    )
    keys = [bone("センター", 0), bone("センター", 5, pos=(0.3, 0.0, 0.0)), thumb_key]
    out = apply_pose_denoise(keys, pmx_path=None)
    thumb = [k for k in out if k.name == "左親指１"]
    assert thumb == [thumb_key]  # 完全一致(再構築されていない)


def test_static_motion_is_preserved():
    # 動きの無い静止モーションは平滑化・フィットでほぼ変わらない。
    keys = [bone("センター", 0), bone("センター", 8)]
    out = apply_pose_denoise(keys, pmx_path=None)
    for k in out:
        if k.name == "センター":
            assert k.position == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
            assert k.rotation == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-6)


def test_output_keys_are_valid_bonekeys():
    keys = [bone("頭", 0), bone("頭", 4, rot=(0.0, 0.0, 0.05, 0.9987))]
    out = apply_pose_denoise(keys, pmx_path=None)
    assert out
    for k in out:
        assert isinstance(k, BoneKey)
        assert isinstance(k.frame, int)
        assert len(k.position) == 3
        assert len(k.rotation) == 4
        assert isinstance(k.interpolation, bytes)
        assert len(k.interpolation) == 64
