import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape, generate


def _envelope(events, params=None):
    params = params or GenerationParams()
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


def test_half_frame_targets_round_up_and_later_target_wins_collision():
    events = [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)]
    p = GenerationParams(coartic_overlap_max=1)
    keys = lipsync.generate_morph_keys(events, p)
    assert len({(k.name_raw, k.frame) for k in keys}) == len(keys)
    env = _envelope(events, p)
    assert set(env) == {"あ", "う"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (10, 0.25), (11, 0.0)])
    _approx_envelope(env["う"], [(10, 0.25), (11, 0.5), (18, 0.5), (20, 0.0)])


def test_integer_coarticulation_window_has_no_collision():
    p = GenerationParams(coartic_overlap_max=4)
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)], p
    )
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.25), (12, 0.0)])
    _approx_envelope(env["う"], [(8, 0.0), (10, 0.25), (12, 0.5), (18, 0.5), (20, 0.0)])


def test_quantize_rounds_half_up_and_keeps_latest_target_per_frame():
    targets = [
        generate._Target("あ", 0.5, 0.1),
        generate._Target("あ", 2.5, 0.2),
        generate._Target("あ", 4.4, 0.3),
        generate._Target("あ", 3.6, 0.4),
        generate._Target("あ", 6.0, 0.5),
        generate._Target("あ", 6.0, 0.6),
        generate._Target("い", 1.5, 0.7),
    ]
    keys = generate._quantize_targets(targets)
    assert [(k.name, k.frame, k.weight) for k in keys] == [
        ("あ", 1, 0.1),
        ("い", 2, 0.7),
        ("あ", 3, 0.2),
        ("あ", 4, 0.3),
        ("あ", 6, 0.6),
    ]
