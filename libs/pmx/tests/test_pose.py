"""FK評価のテスト(pmx.md §7)。

PmxModel/PmxBone はデータモデルを直接組み立て、BoneKey のトラックを与えて
ローカル姿勢サンプリングと前方運動学を検証する。
"""

import math

import pytest

from vmd.types import BoneKey

from pmx.pose import (
    evaluate_fk,
    evaluate_fk_range,
    sample_local_poses,
)
from pmx.types import PmxBone, PmxModel

# Z軸まわり90度のクォータニオン (x,y,z,w)
_SIN45 = math.sin(math.pi / 4)
_RZ90 = (0.0, 0.0, _SIN45, _SIN45)


# ---------------------------------------------------------------------------
# データ構築ヘルパ
# ---------------------------------------------------------------------------


def _bone(
    name, *, parent=None, position=(0.0, 0.0, 0.0), movable=True, rotatable=True
):
    return PmxBone(
        name=name,
        name_raw=name.encode("utf-16-le"),
        english_name="",
        english_name_raw=b"",
        parent=parent,
        position=position,
        movable=movable,
        rotatable=rotatable,
        flags=0,
    )


def _model(bones):
    name_to_index = {}
    for i, b in enumerate(bones):
        name_to_index.setdefault(b.name, i)
    return PmxModel(
        bones=tuple(bones), name_to_index=name_to_index, warnings=()
    )


def _key(name, *, frame=0, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(
        name_raw=name.encode("cp932").ljust(15, b"\x00")[:15],
        frame=frame,
        position=position,
        rotation=rotation,
        interpolation=bytes(64),
    )


def _approx_vec(v):
    return pytest.approx(v, abs=1e-6)


# ---------------------------------------------------------------------------
# sample_local_poses
# ---------------------------------------------------------------------------


def test_sample_uses_vmd_values():
    model = _model([_bone("P")])
    tracks = {"P": [_key("P", position=(1.0, 2.0, 3.0), rotation=_RZ90)]}
    poses = sample_local_poses(model, tracks, 0)
    assert len(poses) == 1
    assert poses[0].position == _approx_vec((1.0, 2.0, 3.0))
    assert poses[0].rotation == _approx_vec(_RZ90)


def test_sample_missing_bone_is_base_pose():
    model = _model([_bone("P"), _bone("C", parent=0)])
    poses = sample_local_poses(model, {}, 0)
    assert poses[0].position == _approx_vec((0.0, 0.0, 0.0))
    assert poses[0].rotation == _approx_vec((0.0, 0.0, 0.0, 1.0))
    assert poses[1].position == _approx_vec((0.0, 0.0, 0.0))
    assert poses[1].rotation == _approx_vec((0.0, 0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# evaluate_fk
# ---------------------------------------------------------------------------


def test_parent_rotation_moves_child_world_position():
    model = _model([_bone("P"), _bone("C", parent=0, position=(1.0, 0.0, 0.0))])
    tracks = {"P": [_key("P", rotation=_RZ90)]}
    world = evaluate_fk(model, sample_local_poses(model, tracks, 0))
    # 親をZ90回転 → 子(基準オフセット (1,0,0))はワールドで (0,1,0)
    assert world[0].position == _approx_vec((0.0, 0.0, 0.0))
    assert world[1].position == _approx_vec((0.0, 1.0, 0.0))


def test_ancestor_rotation_propagates_to_descendant():
    model = _model(
        [
            _bone("A"),
            _bone("B", parent=0, position=(1.0, 0.0, 0.0)),
            _bone("C", parent=1, position=(2.0, 0.0, 0.0)),
        ]
    )
    tracks = {"A": [_key("A", rotation=_RZ90)]}
    world = evaluate_fk(model, sample_local_poses(model, tracks, 0))
    assert world[1].position == _approx_vec((0.0, 1.0, 0.0))
    assert world[2].position == _approx_vec((0.0, 2.0, 0.0))


def test_immovable_bone_ignores_vmd_position():
    model = _model([_bone("M", movable=False)])
    tracks = {"M": [_key("M", position=(5.0, 0.0, 0.0))]}
    world = evaluate_fk(model, sample_local_poses(model, tracks, 0))
    assert world[0].position == _approx_vec((0.0, 0.0, 0.0))


def test_unrotatable_bone_ignores_vmd_rotation():
    model = _model(
        [
            _bone("P", rotatable=False),
            _bone("C", parent=0, position=(1.0, 0.0, 0.0)),
        ]
    )
    tracks = {"P": [_key("P", rotation=_RZ90)]}
    world = evaluate_fk(model, sample_local_poses(model, tracks, 0))
    # 親の回転が無視されるので子は回らない
    assert world[1].position == _approx_vec((1.0, 0.0, 0.0))


def test_missing_keys_give_base_pose():
    model = _model([_bone("A"), _bone("B", parent=0, position=(3.0, 0.0, 0.0))])
    world = evaluate_fk(model, sample_local_poses(model, {}, 0))
    assert world[1].position == _approx_vec((3.0, 0.0, 0.0))
    assert world[1].rotation == _approx_vec((0.0, 0.0, 0.0, 1.0))


def test_root_has_no_parent_uses_origin():
    model = _model([_bone("R", position=(2.0, 1.0, 0.0))])
    world = evaluate_fk(model, sample_local_poses(model, {}, 0))
    # 親なしボーンはモデル原点を親とする → 基準位置がそのままワールド
    assert world[0].position == _approx_vec((2.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# evaluate_fk_range
# ---------------------------------------------------------------------------


def test_fk_range_is_per_frame_and_deterministic():
    model = _model([_bone("P")])
    tracks = {
        "P": [
            _key("P", frame=0, position=(0.0, 0.0, 0.0)),
            _key("P", frame=1, position=(1.0, 0.0, 0.0)),
            _key("P", frame=2, position=(2.0, 0.0, 0.0)),
        ]
    }
    frames = range(0, 3)
    result = evaluate_fk_range(model, tracks, frames)
    assert len(result) == 3
    for f in (0, 1, 2):
        assert result[f][0].position == _approx_vec((float(f), 0.0, 0.0))
    # 決定論性
    assert evaluate_fk_range(model, tracks, frames)[1][0].position == _approx_vec(
        (1.0, 0.0, 0.0)
    )
