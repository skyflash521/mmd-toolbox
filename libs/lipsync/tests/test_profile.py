import pytest

import lipsync
from lipsync import ApertureClass, ConsonantClass, GenerationParams, MouthEvent, MouthShape

_DEFAULT_CAP = GenerationParams().open_cap
_CONSONANT_AUX_GAIN = 0.3


def _peak_weights(events, params=None):
    params = params or GenerationParams()
    keys = lipsync.generate_morph_keys(events, params)
    weights: dict[str, float] = {}
    for k in keys:
        weights[k.name] = max(weights.get(k.name, 0.0), k.weight)
    return weights


def _vowel(shape, open_amount, consonant=ConsonantClass.NONE, aperture=ApertureClass.NONE):
    return [MouthEvent(shape, 0.0, 10.0, open_amount, consonant, aperture)]


@pytest.mark.parametrize(
    "shape,main",
    [
        (MouthShape.A, "あ"),
        (MouthShape.I, "い"),
        (MouthShape.U, "う"),
        (MouthShape.E, "え"),
        (MouthShape.O, "お"),
        (MouthShape.N, "ん"),
    ],
)
def test_pure_vowel_is_main_morph_only(shape, main):
    assert _peak_weights(_vowel(shape, 0.4)) == pytest.approx({main: 0.4})


def test_neutral_consonant_is_pure_vowel():
    assert _peak_weights(_vowel(MouthShape.A, 0.5, ConsonantClass.NEUTRAL)) == pytest.approx(
        {"あ": 0.5}
    )


def test_rounded_consonant_adds_u_morph():
    w = _peak_weights(_vowel(MouthShape.A, 0.5, ConsonantClass.ROUNDED))
    assert w == pytest.approx({"あ": 0.5, "う": 0.5 * _CONSONANT_AUX_GAIN})


def test_spread_consonant_adds_i_morph():
    w = _peak_weights(_vowel(MouthShape.A, 0.5, ConsonantClass.SPREAD))
    assert w == pytest.approx({"あ": 0.5, "い": 0.5 * _CONSONANT_AUX_GAIN})


@pytest.mark.parametrize(
    "shape,consonant,main",
    [
        pytest.param(MouthShape.U, ConsonantClass.ROUNDED, "う", id="rounded_u"),
        pytest.param(MouthShape.I, ConsonantClass.SPREAD, "い", id="spread_i"),
    ],
)
def test_consonant_aux_not_added_to_main_morph(shape, consonant, main):
    assert _peak_weights(_vowel(shape, 0.5, consonant)) == pytest.approx({main: 0.5})


def test_hold_clamped_to_open_cap():
    assert _peak_weights(_vowel(MouthShape.A, 1.0)) == pytest.approx({"あ": _DEFAULT_CAP})


def test_vowel_scale_applied_then_clamped():
    p = GenerationParams(vowel_scale=(2.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    assert _peak_weights(_vowel(MouthShape.A, 0.5), p) == pytest.approx({"あ": _DEFAULT_CAP})


def test_total_over_open_cap_is_scaled_proportionally():
    w = _peak_weights(_vowel(MouthShape.A, 1.0, ConsonantClass.ROUNDED))
    raw_a, raw_u = _DEFAULT_CAP, _DEFAULT_CAP * _CONSONANT_AUX_GAIN
    factor = _DEFAULT_CAP / (raw_a + raw_u)
    assert w == pytest.approx({"あ": raw_a * factor, "う": raw_u * factor})
    assert sum(w.values()) == pytest.approx(_DEFAULT_CAP)


def test_open_cap_param_used_for_hold_clamp():
    p = GenerationParams(open_cap=0.6)
    assert _peak_weights(_vowel(MouthShape.A, 1.0), p) == pytest.approx({"あ": 0.6})


def test_open_cap_param_used_for_total_clamp():
    p = GenerationParams(open_cap=0.6)
    w = _peak_weights(_vowel(MouthShape.A, 1.0, ConsonantClass.ROUNDED), p)
    factor = 0.6 / (0.6 + 0.6 * _CONSONANT_AUX_GAIN)
    assert w == pytest.approx({"あ": 0.6 * factor, "う": 0.6 * _CONSONANT_AUX_GAIN * factor})
    assert sum(w.values()) == pytest.approx(0.6)


def test_exaggeration_scales_consonant_aux_only():
    p = GenerationParams(exaggeration=2.0)
    w = _peak_weights(_vowel(MouthShape.A, 0.5, ConsonantClass.ROUNDED), p)
    assert w == pytest.approx({"あ": 0.5, "う": 0.5 * _CONSONANT_AUX_GAIN * 2.0})


def test_exaggerated_aux_included_in_total_clamp():
    p = GenerationParams(exaggeration=2.0)
    w = _peak_weights(_vowel(MouthShape.A, 1.0, ConsonantClass.ROUNDED), p)
    raw_a, raw_u = _DEFAULT_CAP, _DEFAULT_CAP * _CONSONANT_AUX_GAIN * 2.0
    factor = _DEFAULT_CAP / (raw_a + raw_u)
    assert w == pytest.approx({"あ": raw_a * factor, "う": raw_u * factor})
    assert sum(w.values()) == pytest.approx(_DEFAULT_CAP)


@pytest.mark.parametrize(
    "shape,index,main",
    [
        (MouthShape.A, 0, "あ"),
        (MouthShape.I, 1, "い"),
        (MouthShape.U, 2, "う"),
        (MouthShape.E, 3, "え"),
        (MouthShape.O, 4, "お"),
        (MouthShape.N, 5, "ん"),
    ],
)
def test_vowel_scale_indexed_per_vowel(shape, index, main):
    scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    scale[index] = 1.5
    p = GenerationParams(vowel_scale=tuple(scale))
    assert _peak_weights(_vowel(shape, 0.3), p) == pytest.approx({main: 0.3 * 1.5})


@pytest.mark.parametrize(
    "aperture,scale",
    [
        (ApertureClass.NONE, 1.0),
        (ApertureClass.FIRM_CLOSURE, 0.75),
        (ApertureClass.NARROW_CHANNEL, 0.85),
        (ApertureClass.SLIGHT_CLOSURE, 0.92),
    ],
)
def test_aperture_scale_multiplies_final_weight(aperture, scale):
    assert _peak_weights(_vowel(MouthShape.A, 0.5, aperture=aperture)) == pytest.approx(
        {"あ": 0.5 * scale}
    )


def test_aperture_decay_applied_after_hold_clamp():
    p = GenerationParams(vowel_scale=(1.0, 1.0, 1.0, 1.0, 1.70, 1.0), open_cap=0.90)
    events = _vowel(MouthShape.O, 0.75, aperture=ApertureClass.FIRM_CLOSURE)
    assert _peak_weights(events, p) == pytest.approx({"お": 0.90 * 0.75})


def test_aperture_decay_applied_after_total_clamp_shrink():
    raw_a, raw_i = _DEFAULT_CAP, _DEFAULT_CAP * _CONSONANT_AUX_GAIN
    factor = _DEFAULT_CAP / (raw_a + raw_i)
    w_none = _peak_weights(_vowel(MouthShape.A, 1.0, ConsonantClass.SPREAD, ApertureClass.NONE))
    w_narrow = _peak_weights(
        _vowel(MouthShape.A, 1.0, ConsonantClass.SPREAD, ApertureClass.NARROW_CHANNEL)
    )
    assert w_none == pytest.approx({"あ": raw_a * factor, "い": raw_i * factor})
    assert w_narrow == pytest.approx({"あ": raw_a * factor * 0.85, "い": raw_i * factor * 0.85})


def test_consonant_and_aperture_apply_independently():
    w = _peak_weights(_vowel(MouthShape.A, 0.5, ConsonantClass.SPREAD, ApertureClass.NARROW_CHANNEL))
    assert w == pytest.approx({"あ": 0.5 * 0.85, "い": 0.5 * _CONSONANT_AUX_GAIN * 0.85})
