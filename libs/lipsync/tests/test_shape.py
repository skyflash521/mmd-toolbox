import pytest

import lipsync
from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape


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


def test_isolated_vowel_opens_from_zero_holds_flat_and_closes_to_zero():
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)])
    assert set(env) == {"あ"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])


def test_attack_release_frames_honored():
    p = GenerationParams(attack_frames=3, release_frames=1)
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 12.0, 0.5)], p)
    _approx_envelope(env["あ"], [(0, 0.0), (3, 0.5), (11, 0.5), (12, 0.0)])


def test_each_composed_morph_gets_own_envelope():
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5, ConsonantClass.ROUNDED)])
    assert set(env) == {"あ", "う"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])
    _approx_envelope(env["う"], [(0, 0.0), (2, 0.15), (8, 0.15), (10, 0.0)])


def test_envelope_respects_interval_start():
    env = _envelope([MouthEvent(MouthShape.A, 10.0, 20.0, 0.5)])
    _approx_envelope(env["あ"], [(10, 0.0), (12, 0.5), (18, 0.5), (20, 0.0)])


def test_plateau_holds_total_clamped_weight():
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 1.0, ConsonantClass.ROUNDED)])
    held = 0.8 * 0.8 / 1.04
    assert set(env) == {"あ", "う"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, held), (8, held), (10, 0.0)])
