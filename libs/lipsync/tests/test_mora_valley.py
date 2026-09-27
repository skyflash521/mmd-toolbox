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


def test_basic_valley_between_equal_weight_moras():
    events = [
        MouthEvent(
            MouthShape.A, 0.0, 10.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE
        ),
        MouthEvent(
            MouthShape.A, 10.0, 20.0, 0.8, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.6),
            (5, 0.6),
            (10, 0.45),
            (15, 0.6),
            (18, 0.6),
            (20, 0.0),
        ],
    )


def test_three_moras_all_boundaries_get_valleys_when_well_spaced():
    events = [
        MouthEvent(MouthShape.A, 0.0, 30.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 30.0, 60.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 60.0, 90.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, GenerationParams(vibrato_amp=0.0))
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.6),
            (15, 0.6),
            (18, 0.6),
            (30, 0.45),
            (42, 0.6),
            (45, 0.6),
            (48, 0.6),
            (60, 0.45),
            (72, 0.6),
            (75, 0.6),
            (88, 0.6),
            (90, 0.0),
        ],
    )


def test_valley_depth_scales_with_aperture_class():
    def _valley_center(aperture_class, aperture_scale):
        open_amount = 0.6 / aperture_scale
        events = [
            MouthEvent(MouthShape.A, 0.0, 10.0, open_amount, ConsonantClass.NONE, aperture_class),
            MouthEvent(MouthShape.A, 10.0, 20.0, open_amount, ConsonantClass.NONE, aperture_class),
        ]
        env = _envelope(events)
        by_frame = dict(env["あ"])
        return by_frame[10]

    firm = _valley_center(lipsync.ApertureClass.FIRM_CLOSURE, 0.75)
    slight = _valley_center(lipsync.ApertureClass.SLIGHT_CLOSURE, 0.92)
    assert firm == pytest.approx(0.45)
    assert slight == pytest.approx(0.552)
    assert firm < slight


def test_none_aperture_produces_no_valley_points():
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(MouthShape.A, 10.0, 20.0, 0.3, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [(0, 0.0), (2, 0.8), (5, 0.8), (15, 0.3), (18, 0.3), (20, 0.0)],
    )


def test_valley_omitted_when_half_width_floors_to_zero():
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.A, 10.0, 11.0, 0.6, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
        MouthEvent(MouthShape.A, 11.0, 20.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),
            (5, 0.5),
            (11, 0.45),
            (16, 0.5),
            (18, 0.5),
            (20, 0.0),
        ],
    )


def test_valley_omitted_when_half_width_floors_to_one():
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.A, 10.0, 13.0, 0.6, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
        MouthEvent(MouthShape.A, 13.0, 23.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
    ]
    env = _envelope(events, GenerationParams(vibrato_amp=0.0))
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.8),
            (5, 0.8),
            (12, 0.45),
            (18, 0.5),
            (21, 0.5),
            (23, 0.0),
        ],
    )


def test_valley_uses_lerp_not_raw_endpoints_for_unequal_weights():
    events = [
        MouthEvent(MouthShape.A, 0.0, 6.0, 0.15, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.A, 6.0, 20.0, 0.8, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.15),
            (3, 0.15),
            (6, 0.21375),
            (9, 0.42),
            (13, 0.6),
            (18, 0.6),
            (20, 0.0),
        ],
    )


def test_density_avoidance_prioritizes_total_displacement_over_greedy():
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.6, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.A, 10.0, 18.0, 0.6 / 0.85, ConsonantClass.NONE,
            lipsync.ApertureClass.NARROW_CHANNEL,
        ),
        MouthEvent(
            MouthShape.A, 18.0, 26.0, 0.6 / 0.75, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
        MouthEvent(
            MouthShape.A, 26.0, 36.0, 0.6 / 0.85, ConsonantClass.NONE,
            lipsync.ApertureClass.NARROW_CHANNEL,
        ),
    ]
    env = _envelope(events, params)
    by_frame = dict(env["あ"])
    assert by_frame[10] == pytest.approx(0.51)
    assert by_frame[26] == pytest.approx(0.51)
    assert 18 not in by_frame
    assert by_frame[14] == pytest.approx(0.6) and by_frame[22] == pytest.approx(0.6)


def test_density_avoidance_uses_multi_morph_sum_not_single_morph_max():
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.E, 0.0, 10.0, 0.2, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.E, 10.0, 18.0, 0.2, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
        MouthEvent(
            MouthShape.E, 18.0, 22.0, 0.7, ConsonantClass.ROUNDED,
            lipsync.ApertureClass.SLIGHT_CLOSURE,
        ),
        MouthEvent(MouthShape.E, 22.0, 30.0, 0.3, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
    ]
    env = _envelope(events, params)
    by_frame_e = dict(env["え"])
    by_frame_u = dict(env["う"])
    assert 10 not in by_frame_e
    assert by_frame_e[18] == pytest.approx(0.3932410256410257)
    assert by_frame_u[18] == pytest.approx(0.10417230769230772)


def test_density_avoidance_uses_quantized_frames_for_gap_check():
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.A, 0.0, 8.0, 0.6, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(MouthShape.A, 8.0, 20.02, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 20.02, 28.02, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, params)
    by_frame = dict(env["あ"])
    assert by_frame[20] == pytest.approx(0.45)
    assert 8 not in by_frame


def test_density_avoidance_prioritizes_actual_displacement_over_aperture_class_alone():
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.A, 0.0, 8.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(MouthShape.A, 8.0, 12.0, 0.02, ConsonantClass.NONE, lipsync.ApertureClass.SLIGHT_CLOSURE),
        MouthEvent(MouthShape.A, 12.0, 20.0, 0.02, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, params)
    by_frame = dict(env["あ"])
    assert by_frame[8] == pytest.approx(0.2566186666666667)
    assert 12 not in by_frame


def test_valley_not_generated_when_mora_valley_frames_below_two():
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 10.0, 20.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, GenerationParams(mora_valley_frames=1.9))
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.6), (5, 0.6), (15, 0.6), (18, 0.6), (20, 0.0)])


def test_touching_valleys_rejected_with_zero_min_gap_and_tie_adopts_later_boundary():
    events = [
        MouthEvent(MouthShape.A, 0.0, 8.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 8.0, 16.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 16.0, 24.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=0.0))
    _approx_envelope(
        env["あ"],
        [(0, 0.0), (2, 0.6), (4, 0.6), (12, 0.6), (16, 0.45), (20, 0.6), (22, 0.6), (24, 0.0)],
    )
