import pytest

import lipsync
from lipsync import ApertureClass, ConsonantClass, GenerationParams, MouthEvent, MouthShape, generate


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


def _total_open(env, frame):
    def interp(keys, f):
        if not keys:
            return 0.0
        if f <= keys[0][0]:
            return keys[0][1]
        if f >= keys[-1][0]:
            return keys[-1][1]
        for (f0, v0), (f1, v1) in zip(keys, keys[1:], strict=False):
            if f0 <= f <= f1:
                return v0 if f1 == f0 else v0 + (v1 - v0) * (f - f0) / (f1 - f0)
        return keys[-1][1]

    return sum(interp(keys, frame) for keys in env.values())


def _span(group):
    return (group.shape, group.start, group.end)


@pytest.mark.parametrize(
    "end,attack,release,expected",
    [
        pytest.param(
            7.0, 2, 6, [(0, 0.0), (1, 0.5), (4, 0.5), (7, 0.0)], id="proportional_1_to_3"
        ),
        pytest.param(
            5.0, 3, 9, [(0, 0.0), (1, 0.5), (4, 0.5), (5, 0.0)], id="attack_raised_to_one"
        ),
        pytest.param(
            5.0, 9, 3, [(0, 0.0), (1, 0.5), (4, 0.5), (5, 0.0)], id="release_raised_to_one"
        ),
    ],
)
def test_competition_shortening_keeps_min_hold(end, attack, release, expected):
    p = GenerationParams(attack_frames=attack, release_frames=release)
    env = _envelope([MouthEvent(MouthShape.A, 0.0, end, 0.5)], p)
    _approx_envelope(env["あ"], expected)


def test_normal_length_group_not_shortened():
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)])
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])


def test_too_short_group_without_neighbor_is_dropped():
    assert _envelope([MouthEvent(MouthShape.A, 0.0, 1.5, 0.5)]) == {}


def test_short_vowel_kept_as_triangle_peak():
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 4.0, 0.5)])
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (4, 0.0)])


def test_triangle_has_no_anticipation_or_release_lag():
    env = _envelope(
        [
            MouthEvent(MouthShape.SILENCE, 0.0, 10.0),
            MouthEvent(MouthShape.A, 10.0, 14.0, 0.5),
            MouthEvent(MouthShape.SILENCE, 14.0, 24.0),
        ],
        GenerationParams(anticipation_frames=3),
    )
    _approx_envelope(env["あ"], [(10, 0.0), (12, 0.5), (14, 0.0)])


def test_short_vowel_between_same_vowels_absorbed_and_merged():
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.8),
            MouthEvent(MouthShape.I, 10.0, 11.0, 0.5),
            MouthEvent(MouthShape.A, 11.0, 21.0, 0.3),
        ]
    )
    assert set(env) == {"あ"}
    keys = env["あ"]
    assert keys[0] == (0, pytest.approx(0.0))
    assert keys[-1] == (21, pytest.approx(0.0))
    assert all(weight > 0.0 for frame, weight in keys if 0 < frame < 21)


def test_absorbed_by_neighbor_with_larger_final_weight_after_aperture_decay():
    groups = generate._normalize_groups(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.6),
            MouthEvent(MouthShape.I, 10.0, 11.0, 0.5),
            MouthEvent(MouthShape.U, 11.0, 21.0, 0.65, ConsonantClass.NONE, ApertureClass.FIRM_CLOSURE),
        ],
        GenerationParams(),
    )
    assert [_span(g) for g in groups] == [(MouthShape.A, 0.0, 11.0), (MouthShape.U, 11.0, 21.0)]


def test_absorbed_by_neighbor_with_larger_final_weight_after_total_clamp():
    groups = generate._normalize_groups(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.7),
            MouthEvent(MouthShape.I, 10.0, 11.0, 0.5),
            MouthEvent(MouthShape.E, 11.0, 21.0, 0.75, ConsonantClass.ROUNDED),
        ],
        GenerationParams(),
    )
    assert [_span(g) for g in groups] == [(MouthShape.A, 0.0, 11.0), (MouthShape.E, 11.0, 21.0)]


def test_absorbed_by_longer_neighbor_when_final_weights_tie():
    groups = generate._normalize_groups(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.I, 10.0, 11.0, 0.5),
            MouthEvent(MouthShape.U, 11.0, 31.0, 0.5),
        ],
        GenerationParams(),
    )
    assert [_span(g) for g in groups] == [(MouthShape.A, 0.0, 10.0), (MouthShape.U, 10.0, 31.0)]


def test_consecutive_short_run_absorbed_as_one_span():
    groups = generate._normalize_groups(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.8),
            MouthEvent(MouthShape.I, 10.0, 11.0, 0.5),
            MouthEvent(MouthShape.U, 11.0, 12.0, 0.5),
            MouthEvent(MouthShape.E, 12.0, 22.0, 0.3),
        ],
        GenerationParams(),
    )
    assert [_span(g) for g in groups] == [(MouthShape.A, 0.0, 12.0), (MouthShape.E, 12.0, 22.0)]


def test_one_sided_absorption_reclassifies_triangle_to_normal():
    groups = generate._normalize_groups(
        [
            MouthEvent(MouthShape.A, 0.0, 4.0, 0.5),
            MouthEvent(MouthShape.I, 4.0, 5.5, 0.5),
        ],
        GenerationParams(),
    )
    assert len(groups) == 1
    g = groups[0]
    assert (g.start, g.end) == (0.0, 5.5)
    assert g.triangle is False
    assert g.attack > 0.0 and g.release > 0.0


def test_merge_after_absorption_reclassifies_triangles_to_normal():
    groups = generate._normalize_groups(
        [
            MouthEvent(MouthShape.A, 0.0, 3.0, 0.5, ConsonantClass.ROUNDED),
            MouthEvent(MouthShape.I, 3.0, 4.0, 0.5),
            MouthEvent(MouthShape.A, 4.0, 7.0, 0.5, ConsonantClass.NONE),
        ],
        GenerationParams(),
    )
    assert len(groups) == 1
    g = groups[0]
    assert (g.start, g.end) == (0.0, 7.0)
    assert g.triangle is False
    assert g.attack > 0.0 and g.release > 0.0


def test_absorbed_span_extends_effective_boundary_for_valley_and_midpoints():
    events = [
        MouthEvent(MouthShape.A, 0.0, 4.5, 0.8, ConsonantClass.NONE, ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.I, 4.5, 5.0, 0.5),
        MouthEvent(MouthShape.A, 5.0, 9.0, 0.8, ConsonantClass.NONE, ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.6), (3, 0.6), (5, 0.45), (7, 0.6), (9, 0.0)])


def test_triangle_multi_segment_uses_length_weighted_average_no_valley():
    events = [
        MouthEvent(MouthShape.A, 0.0, 1.0, 0.4),
        MouthEvent(MouthShape.A, 1.0, 3.0, 0.8, ConsonantClass.NONE, ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, (1 * 0.4 + 2 * 0.8 * 0.75) / 3), (3, 0.0)])


def test_triangle_coarticulation_clamps_with_whole_group_length():
    events = [
        MouthEvent(MouthShape.A, 0.0, 2.0, 0.4),
        MouthEvent(MouthShape.A, 2.0, 3.0, 0.6),
        MouthEvent(MouthShape.I, 3.0, 13.0, 0.5),
    ]
    env = _envelope(events)
    assert 2 in dict(env["い"])


def test_consonant_at_uses_nearest_event_outside_event_spans():
    events = [
        MouthEvent(MouthShape.A, 0.0, 30.0, 0.5, ConsonantClass.ROUNDED),
        MouthEvent(MouthShape.A, 31.0, 60.0, 0.5, ConsonantClass.NONE),
    ]
    assert generate._consonant_at(events, 15.0) is ConsonantClass.ROUNDED
    assert generate._consonant_at(events, 45.0) is ConsonantClass.NONE
    assert generate._consonant_at(events, 30.1) is ConsonantClass.ROUNDED
    assert generate._consonant_at(events, 30.9) is ConsonantClass.NONE
    assert generate._consonant_at(events, -5.0) is ConsonantClass.ROUNDED
    assert generate._consonant_at(events, 70.0) is ConsonantClass.NONE


def test_adjacent_triangles_stay_open_between_and_close_at_timeline_ends():
    env = _envelope(
        [MouthEvent(MouthShape.U, 0.0, 4.0, 0.5), MouthEvent(MouthShape.E, 4.0, 8.0, 0.5)]
    )
    for f in range(1, 8):
        assert _total_open(env, f) > 0.05
    assert _total_open(env, 0) == pytest.approx(0.0)
    assert _total_open(env, 8) == pytest.approx(0.0)


def test_triangle_closes_on_silence_side_and_connects_on_vowel_side():
    env = _envelope(
        [
            MouthEvent(MouthShape.SILENCE, 0.0, 4.0),
            MouthEvent(MouthShape.U, 4.0, 8.0, 0.5),
            MouthEvent(MouthShape.E, 8.0, 16.0, 0.5),
        ]
    )
    assert _total_open(env, 4) == pytest.approx(0.0)
    for f in range(6, 15):
        assert _total_open(env, f) > 0.05


def test_triangles_across_legato_gap_stay_open():
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 4.0, 0.5),
            MouthEvent(MouthShape.LEGATO_GAP, 4.0, 6.0),
            MouthEvent(MouthShape.A, 6.0, 10.0, 0.5),
        ]
    )
    for f in range(1, 10):
        assert _total_open(env, f) > 0.05
    assert _total_open(env, 0) == pytest.approx(0.0)
    assert _total_open(env, 10) == pytest.approx(0.0)
