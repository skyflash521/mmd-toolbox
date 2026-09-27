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


def test_anticipation_after_silence_reaches_hold_at_note_start():
    env = _envelope(
        [MouthEvent(MouthShape.SILENCE, 0.0, 10.0), MouthEvent(MouthShape.A, 10.0, 20.0, 0.5)]
    )
    _approx_envelope(env["あ"], [(9, 0.0), (10, 0.5), (18, 0.5), (20, 0.0)])


def test_anticipation_capped_at_half_of_short_silence():
    p = GenerationParams(anticipation_frames=2)
    env = _envelope(
        [MouthEvent(MouthShape.SILENCE, 0.0, 2.0), MouthEvent(MouthShape.A, 2.0, 12.0, 0.5)], p
    )
    _approx_envelope(env["あ"], [(1, 0.0), (2, 0.5), (10, 0.5), (12, 0.0)])


def test_release_lag_before_silence_holds_to_note_end():
    p = GenerationParams(anticipation_frames=3)
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.SILENCE, 10.0, 20.0)], p
    )
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (10, 0.5), (12, 0.0)])


@pytest.mark.parametrize(
    "open_amount,ramp_start_frame",
    [
        pytest.param(0.8, 16, id="full_open_ramp_4_frames"),
        pytest.param(0.2, 19, id="quarter_open_ramp_1_frame"),
    ],
)
def test_anticipation_scales_with_opening(open_amount, ramp_start_frame):
    env = _envelope(
        [
            MouthEvent(MouthShape.SILENCE, 0.0, 20.0),
            MouthEvent(MouthShape.A, 20.0, 40.0, open_amount),
        ],
        GenerationParams(anticipation_frames=4),
    )
    assert env["あ"][0][0] == ramp_start_frame


def test_no_anticipation_at_timeline_start():
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)])
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])


def test_anticipation_after_bilabial():
    env = _envelope(
        [
            MouthEvent(MouthShape.SILENCE, 0.0, 8.0),
            MouthEvent(MouthShape.BILABIAL, 8.0, 10.0),
            MouthEvent(MouthShape.A, 10.0, 20.0, 0.5),
        ]
    )
    _approx_envelope(env["あ"], [(9, 0.0), (10, 0.5), (18, 0.5), (20, 0.0)])


def test_release_lag_before_bilabial():
    p = GenerationParams(anticipation_frames=3)
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.BILABIAL, 10.0, 20.0)], p
    )
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (10, 0.5), (12, 0.0)])


def test_no_anticipation_when_silence_shorter_than_two_frames():
    p = GenerationParams(anticipation_frames=2)
    env = _envelope(
        [MouthEvent(MouthShape.SILENCE, 0.0, 1.0), MouthEvent(MouthShape.A, 1.0, 11.0, 0.5)], p
    )
    _approx_envelope(env["あ"], [(1, 0.0), (3, 0.5), (9, 0.5), (11, 0.0)])
