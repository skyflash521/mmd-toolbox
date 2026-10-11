import struct

import pytest

from pmx.io import read_pmx
from pmx.pose import LocalBonePose, evaluate_fk
from pmx.types import PmxBone, PmxFormatError, PmxModel

_FLAGS = 0x0002 | 0x0004
_IDX_FMT = {1: "<b", 2: "<h", 4: "<i"}


def _i32(v: int) -> bytes:
    return struct.pack("<i", v)


def _textbuf(s: str) -> bytes:
    raw = s.encode("utf-16-le")
    return _i32(len(raw)) + raw


def _bone_bytes(name: str, parent: int, bone_index_size: int) -> bytes:
    return (
        _textbuf(name)
        + _textbuf("")
        + struct.pack("<3f", 0.0, 0.0, 0.0)
        + struct.pack(_IDX_FMT[bone_index_size], parent)
        + _i32(0)
        + struct.pack("<H", _FLAGS)
        + struct.pack("<3f", 0.0, 0.0, 0.0)
    )


def _build_pmx(parents: list[int], *, bone_index_size: int = 1) -> bytes:
    globals_bytes = bytes([0, 0, 1, 1, 1, bone_index_size, 1, 1])
    out = bytearray(b"PMX " + struct.pack("<f", 2.0))
    out += bytes([len(globals_bytes)]) + globals_bytes
    for s in ("model", "model_en", "comment", "comment_en"):
        out += _textbuf(s)
    out += _i32(0)
    out += _i32(0)
    out += _i32(0)
    out += _i32(0)
    out += _i32(len(parents))
    for i, p in enumerate(parents):
        out += _bone_bytes(f"b{i}", p, bone_index_size)
    for _ in range(4):
        out += _i32(0)
    return bytes(out)


@pytest.mark.parametrize("bone_index_size", [1, 2, 4])
def test_no_parent_and_valid_hierarchy_are_accepted(bone_index_size):
    model = read_pmx(_build_pmx([-1, 0, 1, 0, -1], bone_index_size=bone_index_size))
    assert [b.parent for b in model.bones] == [None, 0, 1, 0, None]


def test_parent_declared_after_child_is_accepted():
    model = read_pmx(_build_pmx([1, -1]))
    assert [b.parent for b in model.bones] == [1, None]


@pytest.mark.parametrize("bone_index_size", [1, 2, 4])
def test_parent_index_equal_to_bone_count_is_rejected(bone_index_size):
    with pytest.raises(PmxFormatError):
        read_pmx(_build_pmx([-1, 0, 3], bone_index_size=bone_index_size))


def test_parent_index_far_beyond_bone_count_is_rejected():
    with pytest.raises(PmxFormatError):
        read_pmx(_build_pmx([-1, 100]))


def test_parent_index_below_minus_one_is_rejected():
    with pytest.raises(PmxFormatError):
        read_pmx(_build_pmx([-1, -2]))


def test_single_bone_with_out_of_range_parent_is_rejected():
    with pytest.raises(PmxFormatError):
        read_pmx(_build_pmx([1]))


def test_bone_that_is_its_own_parent_is_rejected():
    with pytest.raises(PmxFormatError):
        read_pmx(_build_pmx([-1, 1]))


def test_two_bone_cycle_is_rejected():
    with pytest.raises(PmxFormatError):
        read_pmx(_build_pmx([1, 0]))


def test_longer_cycle_away_from_root_is_rejected():
    with pytest.raises(PmxFormatError):
        read_pmx(_build_pmx([-1, 3, 1, 2]))


def test_cycle_in_one_branch_is_rejected_even_if_other_branch_is_valid():
    with pytest.raises(PmxFormatError):
        read_pmx(_build_pmx([-1, 0, 3, 2]))


def _chain_model(depth: int) -> PmxModel:
    bones = tuple(
        PmxBone(
            name=f"b{i}",
            name_raw=b"",
            english_name="",
            english_name_raw=b"",
            parent=None if i == 0 else i - 1,
            position=(float(i), 0.0, 0.0),
            movable=True,
            rotatable=True,
            flags=0,
        )
        for i in range(depth)
    )
    return PmxModel(bones=bones, name_to_index={}, warnings=())


def test_fk_on_deep_chain_does_not_hit_recursion_limit():
    depth = 5000
    model = _chain_model(depth)
    local = [
        LocalBonePose(position=(0.0, 1.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0))
        for _ in range(depth)
    ]
    world = evaluate_fk(model, local)
    assert len(world) == depth
    for i in (0, 1, 100, 2500, depth - 1):
        assert world[i].position == pytest.approx((float(i), float(i + 1), 0.0), abs=1e-6)
        assert world[i].rotation == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-9)


def test_fk_on_deep_chain_with_children_listed_before_parents():
    depth = 5000
    chain = _chain_model(depth)
    reversed_bones = tuple(
        PmxBone(
            name=b.name,
            name_raw=b.name_raw,
            english_name="",
            english_name_raw=b"",
            parent=None if b.parent is None else depth - 1 - b.parent,
            position=b.position,
            movable=True,
            rotatable=True,
            flags=0,
        )
        for b in reversed(chain.bones)
    )
    model = PmxModel(bones=reversed_bones, name_to_index={}, warnings=())
    local = [
        LocalBonePose(position=(0.0, 0.0, 1.0), rotation=(0.0, 0.0, 0.0, 1.0))
        for _ in range(depth)
    ]
    world = evaluate_fk(model, local)
    for i in (0, 1, 2500, depth - 1):
        assert world[depth - 1 - i].position == pytest.approx(
            (float(i), 0.0, float(i + 1)), abs=1e-6
        )
