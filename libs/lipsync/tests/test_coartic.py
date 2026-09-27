import math

import pytest

import lipsync
from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape, generate

_WIDE = GenerationParams(coartic_overlap_max=4)
_NONE = ConsonantClass.NONE


def _envelope(events, params):
    by_morph: dict[str, list[tuple[int, float]]] = {}
    for k in lipsync.generate_morph_keys(events, params):
        by_morph.setdefault(k.name, []).append((k.frame, k.weight))
    for name in by_morph:
        by_morph[name].sort()
    return by_morph


def _approx_envelope(actual, expected):
    assert [f for f, _ in actual] == [f for f, _ in expected]
    for (_, aw), (_, ew) in zip(actual, expected, strict=True):
        assert aw == pytest.approx(ew)


def test_shape_diff_identical_is_zero():
    assert generate._shape_diff(
        MouthShape.A, _NONE, MouthShape.A, _NONE, GenerationParams()
    ) == pytest.approx(0.0)


def test_shape_diff_disjoint_is_one():
    assert generate._shape_diff(
        MouthShape.A, _NONE, MouthShape.U, _NONE, GenerationParams()
    ) == pytest.approx(1.0)


def test_shape_diff_includes_consonant_modulation():
    cosine = 1.0 / math.sqrt(1.0 + 0.3**2)
    expected = math.sqrt(2.0 - 2.0 * cosine) / math.sqrt(2.0)
    assert generate._shape_diff(
        MouthShape.A, _NONE, MouthShape.A, ConsonantClass.ROUNDED, GenerationParams()
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "shorter_len,overlap_max,expected",
    [
        pytest.param(10.0, 4, 4, id="base_length_when_room"),
        pytest.param(4.0, 4, 2, id="capped_at_half_shorter"),
        pytest.param(10.0, 2, 2, id="default_base_length"),
        pytest.param(10.0, 1, 1, id="base_length_one"),
        pytest.param(1.0, 4, 1, id="raised_to_minimum_one"),
    ],
)
def test_transition_frames_independent_of_shape_diff(shorter_len, overlap_max, expected):
    p = GenerationParams(coartic_overlap_max=overlap_max)
    for diff in (0.0, 0.5, 1.0):
        assert generate._transition_frames(diff, shorter_len, p) == expected


def test_coarticulation_boundary_holds_midpoint_of_both_shapes():
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)],
        _WIDE,
    )
    at10 = {name: w for name, keys in env.items() for f, w in keys if f == 10}
    assert at10 == pytest.approx({"あ": 0.25, "う": 0.25})


def test_coarticulation_full_envelopes():
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)],
        _WIDE,
    )
    assert set(env) == {"あ", "う"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.25), (12, 0.0)])
    _approx_envelope(env["う"], [(8, 0.0), (10, 0.25), (12, 0.5), (18, 0.5), (20, 0.0)])


def test_bilabial_between_vowels_blocks_coarticulation_and_emits_no_keys():
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.BILABIAL, 10.0, 14.0),
            MouthEvent(MouthShape.U, 14.0, 24.0, 0.5),
        ],
        GenerationParams(),
    )
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (10, 0.5), (11, 0.0)])
    _approx_envelope(env["う"], [(13, 0.0), (14, 0.5), (22, 0.5), (24, 0.0)])
