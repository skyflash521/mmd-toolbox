import pytest

import lipsync
from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape


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


def test_long_vowel_vibrato_known_values():
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 40.0, 0.5)], GenerationParams())
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),
            (6, 0.55),
            (13, 0.45),
            (21, 0.55),
            (28, 0.45),
            (36, 0.55),
            (38, 0.5),
            (40, 0.0),
        ],
    )


def test_short_plateau_no_vibrato():
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 20.0, 0.5)], GenerationParams())
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (18, 0.5), (20, 0.0)])


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(GenerationParams(vibrato_amp=0.0), id="amp_zero"),
        pytest.param(GenerationParams(vibrato_period=0), id="period_zero"),
    ],
)
def test_vibrato_disabled_when_amp_or_period_not_positive(params):
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 40.0, 0.5)], params)
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (38, 0.5), (40, 0.0)])


def test_vibrato_follows_interpolated_aperture_scale():
    events = [
        MouthEvent(
            MouthShape.A, 0.0, 20.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE
        ),
        MouthEvent(
            MouthShape.A, 20.0, 40.0, 0.5, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
    ]
    env = _envelope(events, GenerationParams(mora_valley_frames=2.0))
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),
            (6, 0.55),
            (10, 0.5),
            (18, 0.45),
            (20, 0.328125),
            (22, 0.425),
            (30, 0.375),
            (36, 0.4125),
            (38, 0.375),
            (40, 0.0),
        ],
    )


def test_vibrato_aperture_decay_applied_after_shrink_not_before():
    events = [
        MouthEvent(MouthShape.A, 0.0, 40.0, 0.6, ConsonantClass.SPREAD, lipsync.ApertureClass.NARROW_CHANNEL)
    ]
    env = _envelope(events)
    assert set(env) == {"あ", "い"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.51),
            (6, 0.5230769230769231),
            (13, 0.4675),
            (21, 0.5230769230769231),
            (28, 0.4675),
            (36, 0.5230769230769231),
            (38, 0.51),
            (40, 0.0),
        ],
    )
    _approx_envelope(
        env["い"],
        [
            (0, 0.0),
            (2, 0.153),
            (6, 0.15692307692307692),
            (13, 0.14025),
            (21, 0.15692307692307692),
            (28, 0.14025),
            (36, 0.15692307692307692),
            (38, 0.153),
            (40, 0.0),
        ],
    )


def test_vibrato_extremum_exactly_at_half_period_margin_is_suppressed():
    events = [
        MouthEvent(
            MouthShape.A, 0.0, 20.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE
        ),
        MouthEvent(
            MouthShape.A, 20.0, 40.0, 0.5, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
    ]
    env = _envelope(events, GenerationParams(mora_valley_frames=2.0, vibrato_period=24))
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),
            (10, 0.5),
            (18, 0.45),
            (20, 0.328125),
            (22, 0.425),
            (30, 0.375),
            (38, 0.375),
            (40, 0.0),
        ],
    )


def test_mora_valley_none_boundary_does_not_disturb_vibrato():
    events = [
        MouthEvent(
            MouthShape.A, 0.0, 20.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE
        ),
        MouthEvent(
            MouthShape.A, 20.0, 40.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE
        ),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),
            (6, 0.55),
            (10, 0.5),
            (13, 0.45),
            (21, 0.55),
            (28, 0.45),
            (30, 0.5),
            (36, 0.55),
            (38, 0.5),
            (40, 0.0),
        ],
    )


def test_vibrato_amplitude_limited_to_hold_value():
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 40.0, 0.03)])
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.03),
            (6, 0.06),
            (13, 0.0),
            (21, 0.06),
            (28, 0.0),
            (36, 0.06),
            (38, 0.03),
            (40, 0.0),
        ],
    )


def test_vibrato_follows_interpolated_hold_between_midpoints():
    events = [MouthEvent(MouthShape.A, 0.0, 20.0, 0.3), MouthEvent(MouthShape.A, 20.0, 40.0, 0.6)]
    env = _envelope(events)
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.3),
            (6, 0.35),
            (10, 0.3),
            (13, 0.29875),
            (21, 0.51125),
            (28, 0.52375),
            (30, 0.6),
            (36, 0.65),
            (38, 0.6),
            (40, 0.0),
        ],
    )


def test_vibrato_phase_starts_at_coarticulation_window_end():
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 50.0, 0.5)]
    )
    _approx_envelope(
        env["う"],
        [
            (9, 0.0),
            (10, 0.25),
            (11, 0.5),
            (15, 0.55),
            (22, 0.45),
            (30, 0.55),
            (37, 0.45),
            (45, 0.55),
            (48, 0.5),
            (50, 0.0),
        ],
    )


def test_vibrato_suppression_margin_extends_two_frames_beyond_valley_half_width():
    events = [
        MouthEvent(MouthShape.A, 0.0, 30.0, 0.5),
        MouthEvent(MouthShape.A, 30.0, 60.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, GenerationParams(vibrato_period=6))
    frames_near_valley = [f for f, _ in env["あ"] if 17 <= f <= 44]
    assert frames_near_valley == [18, 30, 42]
