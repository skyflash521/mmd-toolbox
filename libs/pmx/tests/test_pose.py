import math

import pytest

from pmx.pose import LocalBonePose, evaluate_fk
from pmx.types import PmxBone, PmxModel

_SIN45 = math.sin(math.pi / 4)
_ROT_Z_90 = (0.0, 0.0, _SIN45, _SIN45)


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


def _local(model, poses):
    return tuple(
        LocalBonePose(
            position=poses.get(b.name, {}).get("position", (0.0, 0.0, 0.0)),
            rotation=poses.get(b.name, {}).get("rotation", (0.0, 0.0, 0.0, 1.0)),
        )
        for b in model.bones
    )


def _approx_vec(v):
    return pytest.approx(v, abs=1e-6)


def test_parent_rotation_moves_child_world_position():
    model = _model([_bone("P"), _bone("C", parent=0, position=(1.0, 0.0, 0.0))])
    poses = {"P": dict(rotation=_ROT_Z_90)}
    world = evaluate_fk(model, _local(model, poses))
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
    poses = {"A": dict(rotation=_ROT_Z_90)}
    world = evaluate_fk(model, _local(model, poses))
    assert world[1].position == _approx_vec((0.0, 1.0, 0.0))
    assert world[2].position == _approx_vec((0.0, 2.0, 0.0))


def test_parent_placed_after_child_is_resolved():
    model = _model([_bone("C", parent=1, position=(1.0, 0.0, 0.0)), _bone("P")])
    poses = {"P": dict(rotation=_ROT_Z_90)}
    world = evaluate_fk(model, _local(model, poses))
    assert world[0].position == _approx_vec((0.0, 1.0, 0.0))


def test_non_unit_quaternion_is_normalized_before_applying():
    model = _model([_bone("P"), _bone("C", parent=0, position=(1.0, 0.0, 0.0))])
    scaled = tuple(3.0 * c for c in _ROT_Z_90)
    poses = {"P": dict(rotation=scaled)}
    world = evaluate_fk(model, _local(model, poses))
    assert world[1].position == _approx_vec((0.0, 1.0, 0.0))


def test_immovable_bone_ignores_local_position():
    model = _model([_bone("M", movable=False)])
    poses = {"M": dict(position=(5.0, 0.0, 0.0))}
    world = evaluate_fk(model, _local(model, poses))
    assert world[0].position == _approx_vec((0.0, 0.0, 0.0))


def test_unrotatable_bone_ignores_local_rotation():
    model = _model(
        [
            _bone("P", rotatable=False),
            _bone("C", parent=0, position=(1.0, 0.0, 0.0)),
        ]
    )
    poses = {"P": dict(rotation=_ROT_Z_90)}
    world = evaluate_fk(model, _local(model, poses))
    assert world[1].position == _approx_vec((1.0, 0.0, 0.0))


def test_missing_keys_give_base_pose():
    model = _model([_bone("A"), _bone("B", parent=0, position=(3.0, 0.0, 0.0))])
    world = evaluate_fk(model, _local(model, {}))
    assert world[1].position == _approx_vec((3.0, 0.0, 0.0))
    assert world[1].rotation == _approx_vec((0.0, 0.0, 0.0, 1.0))


def test_root_without_parent_is_placed_at_its_base_position():
    model = _model([_bone("R", position=(2.0, 1.0, 0.0))])
    world = evaluate_fk(model, _local(model, {}))
    assert world[0].position == _approx_vec((2.0, 1.0, 0.0))
