import pytest

import lipsync
from lipsync import ApertureClass, ConsonantClass, GenerationParams, MouthEvent, MouthShape


def _envelope(events, params=None):
    params = params or GenerationParams()
    keys = lipsync.generate_morph_keys(events, params)
    by_morph: dict[str, list[tuple[int, float]]] = {}
    for k in keys:
        by_morph.setdefault(k.name, []).append((k.frame, k.weight))
    for name in by_morph:
        by_morph[name].sort()
    return by_morph


def _approx_envelope(actual, expected):
    assert [f for f, _ in actual] == [f for f, _ in expected]
    for (_, aw), (_, ew) in zip(actual, expected, strict=True):
        assert aw == pytest.approx(ew)


def test_two_same_vowels_merge_into_one_envelope_with_midpoint_nodes():
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.4), MouthEvent(MouthShape.A, 10.0, 20.0, 0.7)]
    )
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"], [(0, 0.0), (2, 0.4), (5, 0.4), (15, 0.7), (18, 0.7), (20, 0.0)]
    )


def test_no_key_at_internal_boundary_of_merged_vowels():
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.4), MouthEvent(MouthShape.A, 10.0, 20.0, 0.7)]
    )
    keys = env["あ"]
    assert 10 not in [f for f, _ in keys]
    assert all(weight > 0.0 for frame, weight in keys if 0 < frame < 20)


def test_three_same_vowels_single_attack_release():
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 12.0, 0.3),
            MouthEvent(MouthShape.A, 12.0, 24.0, 0.5),
            MouthEvent(MouthShape.A, 24.0, 36.0, 0.7),
        ],
        GenerationParams(vibrato_amp=0.0),
    )
    _approx_envelope(
        env["あ"],
        [(0, 0.0), (2, 0.3), (6, 0.3), (18, 0.5), (30, 0.7), (34, 0.7), (36, 0.0)],
    )


def test_same_vowel_different_consonant_merges_and_aux_fades_to_zero():
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5, ConsonantClass.ROUNDED),
            MouthEvent(MouthShape.A, 10.0, 20.0, 0.5, ConsonantClass.NONE),
        ]
    )
    assert set(env) == {"あ", "う"}
    assert all(w > 0.0 for f, w in env["あ"] if 0 < f < 20)
    assert env["う"][-1][1] == pytest.approx(0.0)
    assert max(w for _, w in env["う"]) == pytest.approx(0.15)


def test_same_vowel_same_profile_different_consonant_merges():
    env = _envelope(
        [
            MouthEvent(MouthShape.I, 0.0, 4.0, 0.5, ConsonantClass.NONE),
            MouthEvent(MouthShape.I, 4.0, 8.0, 0.5, ConsonantClass.SPREAD),
        ]
    )
    assert set(env) == {"い"}
    assert all(w > 0.0 for f, w in env["い"] if 0 < f < 8)


def test_short_same_vowel_different_consonant_stays_open_across_boundary():
    env = _envelope(
        [
            MouthEvent(MouthShape.U, 0.0, 4.0, 0.5, ConsonantClass.SPREAD),
            MouthEvent(MouthShape.U, 4.0, 8.0, 0.5, ConsonantClass.NONE),
        ]
    )
    assert "う" in env and "い" in env
    assert all(w > 0.0 for f, w in env["う"] if 0 < f < 8)
    assert env["い"][-1][1] == pytest.approx(0.0)


def test_merged_nodes_keep_per_segment_aperture_decay():
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.4, ConsonantClass.NONE, ApertureClass.FIRM_CLOSURE),
            MouthEvent(MouthShape.A, 10.0, 20.0, 0.4, ConsonantClass.NONE, ApertureClass.NONE),
        ]
    )
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"], [(0, 0.0), (2, 0.4 * 0.75), (5, 0.4 * 0.75), (15, 0.4), (18, 0.4), (20, 0.0)]
    )


def test_different_adjacent_vowels_not_merged():
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)]
    )
    assert set(env) == {"あ", "う"}
    assert any(f < 10 and w == pytest.approx(0.5) for f, w in env["あ"])
    assert any(f > 10 and w == pytest.approx(0.5) for f, w in env["う"])
