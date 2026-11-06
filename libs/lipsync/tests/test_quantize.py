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


@pytest.mark.parametrize("boundary", [10.0, 10.25], ids=["integer", "fraction_below_half"])
def test_one_frame_transition_keeps_both_end_values_and_omits_midpoint(boundary):
    events = [
        MouthEvent(MouthShape.A, 0.0, boundary, 0.5),
        MouthEvent(MouthShape.U, boundary, 20.0, 0.5),
    ]
    p = GenerationParams(coartic_overlap_max=1)
    keys = lipsync.generate_morph_keys(events, p)
    assert len({(k.name_raw, k.frame) for k in keys}) == len(keys)
    env = _envelope(events, p)
    assert set(env) == {"あ", "う"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (10, 0.5), (11, 0.0)])
    _approx_envelope(env["う"], [(10, 0.0), (11, 0.5), (18, 0.5), (20, 0.0)])


def test_morph_stays_closed_through_silence_before_transition_out_of_two_frame_shape():
    events = [
        MouthEvent(MouthShape.I, 0.0, 30.0, 0.5),
        MouthEvent(MouthShape.SILENCE, 30.0, 300.0),
        MouthEvent(MouthShape.U, 300.0, 317.0, 0.5),
        MouthEvent(MouthShape.A, 317.0, 319.0, 0.5),
        MouthEvent(MouthShape.I, 319.0, 345.0, 0.5),
        MouthEvent(MouthShape.SILENCE, 345.0, 400.0),
    ]
    reopened = [key for key in _envelope(events)["い"] if key[0] > 100]
    _approx_envelope(reopened[:2], [(319, 0.0), (320, 0.5)])


@pytest.mark.parametrize(
    "opening,params,expected",
    [
        pytest.param(
            [
                MouthEvent(MouthShape.A, 300.0, 310.0, 0.5),
                MouthEvent(MouthShape.U, 310.0, 310.25, 0.5),
                MouthEvent(MouthShape.U, 310.25, 340.0, 0.5),
            ],
            GenerationParams(coartic_overlap_max=1, vibrato_amp=0.0),
            [(310, 0.0), (311, 0.5)],
            id="coarticulation",
        ),
        pytest.param(
            [
                MouthEvent(MouthShape.U, 300.0, 300.25, 0.5),
                MouthEvent(MouthShape.U, 300.25, 340.0, 0.5),
            ],
            GenerationParams(anticipation_frames=0, vibrato_amp=0.0),
            [(300, 0.0), (302, 0.5)],
            id="attack",
        ),
        pytest.param(
            [
                MouthEvent(MouthShape.A, 300.0, 310.6, 0.5),
                MouthEvent(MouthShape.LEGATO_GAP, 310.6, 310.8),
                MouthEvent(MouthShape.U, 310.8, 311.0, 0.5),
                MouthEvent(MouthShape.U, 311.0, 340.0, 0.5),
            ],
            GenerationParams(vibrato_amp=0.0),
            [(311, 0.0), (312, 0.5)],
            id="legato",
        ),
    ],
)
def test_midpoint_of_short_first_event_does_not_replace_closed_key(opening, params, expected):
    events = [
        MouthEvent(MouthShape.U, 0.0, 30.0, 0.5),
        MouthEvent(MouthShape.SILENCE, 30.0, 300.0),
        *opening,
    ]
    reopened = [key for key in _envelope(events, params)["う"] if key[0] > 100]
    _approx_envelope(reopened[:2], expected)


def test_long_shape_holds_its_value_up_to_transition_into_two_frame_shape():
    events = [
        MouthEvent(MouthShape.A, 0.0, 100.0, 0.5),
        MouthEvent(MouthShape.I, 100.0, 102.0, 0.5),
        MouthEvent(MouthShape.U, 102.0, 130.0, 0.5),
    ]
    assert dict(_envelope(events)["あ"])[100] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "gap_end",
    [
        pytest.param(11.6, id="gap_start_and_valley_share_frame"),
        pytest.param(11.2, id="whole_gap_within_one_frame"),
    ],
)
def test_short_legato_gap_keeps_both_end_values(gap_end):
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.6, 0.5),
        MouthEvent(MouthShape.LEGATO_GAP, 10.6, gap_end),
        MouthEvent(MouthShape.U, gap_end, 30.0, 0.5),
    ]
    env = _envelope(events)
    _approx_envelope(env["あ"][-2:], [(11, 0.5), (12, 0.0)])
    _approx_envelope(env["う"][:2], [(11, 0.0), (12, 0.5)])


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
