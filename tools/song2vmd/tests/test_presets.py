from typing import NamedTuple

import pytest

from song2vmd import presets


class _PresetRow(NamedTuple):
    open_lo: float
    open_hi: float
    attack: int
    release: int
    coarticulation: int
    anticipation: int
    min_hold: int
    exaggeration: float
    open_max: float


_EXPECTED = {
    "pop": _PresetRow(0.30, 0.75, 2, 2, 6, 11, 1, 1.0, 0.90),
    "ballad": _PresetRow(0.20, 0.55, 3, 3, 3, 1, 4, 0.8, 0.70),
    "powerful": _PresetRow(0.40, 0.95, 1, 1, 2, 2, 3, 1.3, 0.97),
    "whisper": _PresetRow(0.10, 0.35, 2, 2, 2, 1, 3, 0.7, 0.50),
    "rap": _PresetRow(0.30, 0.70, 1, 1, 1, 1, 2, 1.0, 0.85),
}


def test_style_names_are_the_five_presets():
    assert set(presets.STYLE_NAMES) == set(_EXPECTED)


@pytest.mark.parametrize("style", list(_EXPECTED))
def test_resolve_without_overrides_matches_preset_table(style):
    row = _EXPECTED[style]
    openness, gen = presets.resolve(style)
    assert openness.open_lo == pytest.approx(row.open_lo)
    assert openness.open_hi == pytest.approx(row.open_hi)
    assert openness.open_max == pytest.approx(row.open_max)
    assert gen.attack_frames == row.attack
    assert gen.release_frames == row.release
    assert gen.coartic_overlap_max == row.coarticulation
    assert gen.anticipation_frames == row.anticipation
    assert gen.min_hold_frames == row.min_hold
    assert gen.exaggeration == pytest.approx(row.exaggeration)


def test_different_styles_give_different_open_hi_and_exaggeration():
    pop_openness, pop_gen = presets.resolve("pop")
    whisper_openness, whisper_gen = presets.resolve("whisper")
    assert pop_openness.open_hi != whisper_openness.open_hi
    assert pop_gen.exaggeration != whisper_gen.exaggeration


def test_cli_override_open_max_replaces_only_the_upper_bound():
    openness, _gen = presets.resolve("pop", open_max=0.5)
    assert openness.open_max == pytest.approx(0.5)
    assert openness.open_lo == pytest.approx(0.30)
    assert openness.open_hi == pytest.approx(0.75)


def test_cli_override_coarticulation_takes_precedence_over_preset():
    _openness, gen = presets.resolve("pop", coarticulation=9)
    assert gen.coartic_overlap_max == 9


def test_cli_override_anticipation_takes_precedence_over_preset():
    _openness, gen = presets.resolve("pop", anticipation=9)
    assert gen.anticipation_frames == 9


def test_cli_override_min_hold_takes_precedence_over_preset():
    _openness, gen = presets.resolve("pop", min_hold=9)
    assert gen.min_hold_frames == 9


def test_no_overrides_use_preset_values():
    openness, gen = presets.resolve("ballad")
    assert openness.open_max == pytest.approx(0.70)
    assert gen.coartic_overlap_max == 3
    assert gen.anticipation_frames == 1
    assert gen.min_hold_frames == 4


@pytest.mark.parametrize("override", [
    pytest.param({"attack": 5}, id="attack"),
    pytest.param({"release": 5}, id="release"),
    pytest.param({"exaggeration": 2.0}, id="exaggeration"),
])
def test_resolve_rejects_override_of_attack_release_and_exaggeration(override):
    with pytest.raises(TypeError):
        presets.resolve("powerful", **override)


def test_powerful_preset_fixes_attack_release_and_exaggeration():
    _openness, gen = presets.resolve("powerful")
    assert gen.attack_frames == 1
    assert gen.release_frames == 1
    assert gen.exaggeration == pytest.approx(1.3)


def test_pop_vowel_scale_matches_tuned_values():
    _openness, gen = presets.resolve("pop")
    assert gen.vowel_scale == (1.30, 1.20, 1.70, 0.80, 1.70, 1.00)


def test_vowel_gain_multiplies_preset_vowel_scale_elementwise_leaving_n_unchanged():
    _openness, gen = presets.resolve("pop", vowel_gain=(2.0, 1.0, 0.5, 1.0, 1.0))
    assert gen.vowel_scale[0] == pytest.approx(1.30 * 2.0)
    assert gen.vowel_scale[1] == pytest.approx(1.20)
    assert gen.vowel_scale[2] == pytest.approx(1.70 * 0.5)
    assert gen.vowel_scale[3] == pytest.approx(0.80)
    assert gen.vowel_scale[4] == pytest.approx(1.70)
    assert gen.vowel_scale[5] == pytest.approx(1.00)


def test_pop_carries_tuned_generation_parameters_explicitly():
    _openness, gen = presets.resolve("pop")
    assert gen.triangle_min_frames == pytest.approx(2.0)
    assert (gen.vibrato_threshold, gen.vibrato_amp, gen.vibrato_period) == (10, 0.05, 22)
    assert gen.legato_valley_shallow == pytest.approx(0.45)
    assert gen.legato_valley_deep == pytest.approx(0.30)
    assert gen.legato_valley_slope == pytest.approx(0.02)


@pytest.mark.parametrize("style", ["ballad", "powerful", "whisper", "rap"])
def test_non_pop_styles_share_the_untuned_generation_parameters(style):
    _openness, gen = presets.resolve(style)
    assert gen.vowel_scale == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert gen.triangle_min_frames == pytest.approx(2.0)
    assert (gen.vibrato_threshold, gen.vibrato_amp, gen.vibrato_period) == (18, 0.05, 15)
    assert gen.legato_valley_shallow == pytest.approx(0.4)
    assert gen.legato_valley_deep == pytest.approx(0.2)
    assert gen.legato_valley_slope == pytest.approx(0.025)
