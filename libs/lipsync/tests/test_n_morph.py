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


def _assert_n_mirrors_a(n_events, a_events, params=None):
    params = params or GenerationParams()
    nk = lipsync.generate_morph_keys(n_events, params)
    ak = lipsync.generate_morph_keys(a_events, params)
    n_mapped = sorted((("あ" if k.name == "ん" else k.name), k.frame, k.weight) for k in nk)
    a_sorted = sorted((k.name, k.frame, k.weight) for k in ak)
    assert [(n, f) for n, f, _ in n_mapped] == [(n, f) for n, f, _ in a_sorted]
    for (_, _, nw), (_, _, aw) in zip(n_mapped, a_sorted, strict=True):
        assert nw == pytest.approx(aw)


def test_single_n_envelope_on_n_morph_only():
    env = _envelope([MouthEvent(MouthShape.N, 0.0, 10.0, 0.5)])
    assert set(env) == {"ん"}
    _approx_envelope(env["ん"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])


@pytest.mark.parametrize(
    "events",
    [
        pytest.param([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)], id="a"),
        pytest.param([MouthEvent(MouthShape.I, 0.0, 10.0, 0.5)], id="i"),
        pytest.param([MouthEvent(MouthShape.E, 0.0, 10.0, 1.0)], id="e_clamped"),
        pytest.param(
            [
                MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
                MouthEvent(MouthShape.I, 10.0, 20.0, 0.5),
            ],
            id="coarticulation",
        ),
        pytest.param(
            [
                MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
                MouthEvent(MouthShape.SILENCE, 10.0, 14.0, 0.0),
                MouthEvent(MouthShape.O, 14.0, 24.0, 0.6),
            ],
            id="silence_between",
        ),
        pytest.param(
            [
                MouthEvent(MouthShape.U, 0.0, 8.0, 0.4),
                MouthEvent(MouthShape.BILABIAL, 8.0, 11.0, 0.0),
                MouthEvent(MouthShape.A, 11.0, 21.0, 0.7),
            ],
            id="bilabial_between",
        ),
    ],
)
def test_vowel_inputs_emit_no_n_morph(events):
    keys = lipsync.generate_morph_keys(events, GenerationParams())
    assert "ん" not in {k.name for k in keys}


def test_consecutive_n_merged_like_vowel():
    n_events = [MouthEvent(MouthShape.N, 0.0, 10.0, 0.5), MouthEvent(MouthShape.N, 10.0, 20.0, 0.5)]
    a_events = [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.A, 10.0, 20.0, 0.5)]
    _assert_n_mirrors_a(n_events, a_events)
    keys = lipsync.generate_morph_keys(n_events, GenerationParams())
    assert {k.name for k in keys} == {"ん"}
    assert {k.frame for k in keys if k.weight == pytest.approx(0.0)} == {0, 20}


def test_long_n_gets_vibrato_like_vowel():
    n_events = [MouthEvent(MouthShape.N, 0.0, 40.0, 0.5)]
    a_events = [MouthEvent(MouthShape.A, 0.0, 40.0, 0.5)]
    _assert_n_mirrors_a(n_events, a_events)
    keys = lipsync.generate_morph_keys(n_events, GenerationParams())
    assert len([k for k in keys if k.name == "ん"]) > 4


def test_n_anticipation_after_silence_like_vowel():
    n_events = [MouthEvent(MouthShape.SILENCE, 0.0, 6.0), MouthEvent(MouthShape.N, 6.0, 16.0, 0.5)]
    a_events = [MouthEvent(MouthShape.SILENCE, 0.0, 6.0), MouthEvent(MouthShape.A, 6.0, 16.0, 0.5)]
    _assert_n_mirrors_a(n_events, a_events)


def test_short_n_absorbed_into_neighbor():
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
        MouthEvent(MouthShape.N, 10.0, 11.0, 0.5),
        MouthEvent(MouthShape.A, 11.0, 21.0, 0.5),
    ]
    keys = lipsync.generate_morph_keys(events, GenerationParams())
    assert {k.name for k in keys} == {"あ"}
    assert {k.frame for k in keys if k.weight == pytest.approx(0.0)} == {0, 21}


def test_n_to_vowel_boundary_coarticulates():
    env = _envelope(
        [MouthEvent(MouthShape.N, 0.0, 10.0, 0.5), MouthEvent(MouthShape.A, 10.0, 20.0, 0.5)]
    )
    assert set(env) == {"ん", "あ"}
    _approx_envelope(env["ん"], [(0, 0.0), (2, 0.5), (9, 0.5), (10, 0.25), (11, 0.0)])
    _approx_envelope(env["あ"], [(9, 0.0), (10, 0.25), (11, 0.5), (18, 0.5), (20, 0.0)])
