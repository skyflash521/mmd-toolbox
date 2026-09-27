import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape


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


def _a_gap_i(gap_shape, gap_len):
    return [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
        MouthEvent(gap_shape, 10.0, 10.0 + gap_len),
        MouthEvent(MouthShape.I, 10.0 + gap_len, 20.0 + gap_len, 0.5),
    ]


def test_legato_gap_bridges_with_overlapping_valley():
    env = _envelope(_a_gap_i(MouthShape.LEGATO_GAP, 4.0))
    assert set(env) == {"あ", "い"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (10, 0.5), (12, 0.15), (14, 0.0)])
    _approx_envelope(env["い"], [(10, 0.0), (12, 0.15), (14, 0.5), (22, 0.5), (24, 0.0)])


@pytest.mark.parametrize(
    "gap_len,mid_frame,depth",
    [
        pytest.param(2.0, 11, 0.35, id="short_gap_shallow"),
        pytest.param(8.0, 14, 0.2, id="long_gap_deep"),
        pytest.param(12.0, 16, 0.2, id="depth_clamped_at_deep"),
    ],
)
def test_legato_valley_depth_decreases_linearly_with_gap_length(gap_len, mid_frame, depth):
    env = _envelope(_a_gap_i(MouthShape.LEGATO_GAP, gap_len))
    assert dict(env["あ"])[mid_frame] == pytest.approx(depth * 0.5)


def test_silence_gap_closes_instead_of_valley():
    s_a = dict(_envelope(_a_gap_i(MouthShape.SILENCE, 4.0))["あ"])
    l_a = dict(_envelope(_a_gap_i(MouthShape.LEGATO_GAP, 4.0))["あ"])
    assert s_a.get(11) == pytest.approx(0.0)
    assert 12 not in s_a
    assert l_a.get(12) == pytest.approx(0.15)


def test_gap_mixing_legato_and_silence_closes_without_valley():
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.LEGATO_GAP, 10.0, 12.0),
            MouthEvent(MouthShape.SILENCE, 12.0, 14.0),
            MouthEvent(MouthShape.I, 14.0, 24.0, 0.5),
        ]
    )
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])
    _approx_envelope(env["い"], [(13, 0.0), (14, 0.5), (22, 0.5), (24, 0.0)])
