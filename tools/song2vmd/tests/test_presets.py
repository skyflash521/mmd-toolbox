import pytest

from song2vmd import presets

_STYLE_NAMES = ("pop", "ballad", "powerful", "whisper", "rap")


def test_style_names_are_the_five_presets():
    assert set(presets.STYLE_NAMES) == set(_STYLE_NAMES)


@pytest.mark.parametrize("style", _STYLE_NAMES)
def test_resolve_without_overrides_applies_the_described_preset_values(style):
    described = presets.describe_values()[style]
    openness, gen = presets.resolve(style)
    assert openness.open_max == pytest.approx(described["open_max"])
    assert gen.coartic_overlap_max == described["coarticulation"]
    assert gen.anticipation_frames == described["anticipation"]
    assert gen.min_hold_frames == described["min_hold"]


def test_cli_override_open_max_replaces_only_the_upper_bound():
    preset_openness, _gen = presets.resolve("pop")
    openness, _gen = presets.resolve("pop", open_max=0.5)
    assert openness.open_max == pytest.approx(0.5)
    assert openness.open_lo == preset_openness.open_lo
    assert openness.open_hi == preset_openness.open_hi


def test_cli_override_coarticulation_takes_precedence_over_preset():
    _openness, gen = presets.resolve("pop", coarticulation=9)
    assert gen.coartic_overlap_max == 9


def test_cli_override_anticipation_takes_precedence_over_preset():
    _openness, gen = presets.resolve("pop", anticipation=9)
    assert gen.anticipation_frames == 9


def test_cli_override_min_hold_takes_precedence_over_preset():
    _openness, gen = presets.resolve("pop", min_hold=9)
    assert gen.min_hold_frames == 9


def test_vowel_gain_multiplies_preset_vowel_scale_elementwise_leaving_n_unchanged():
    _openness, base = presets.resolve("pop")
    _openness, gen = presets.resolve("pop", vowel_gain=(2.0, 1.0, 0.5, 1.0, 1.0))
    gains = (2.0, 1.0, 0.5, 1.0, 1.0, 1.0)
    assert gen.vowel_scale == pytest.approx(tuple(s * g for s, g in zip(base.vowel_scale, gains, strict=True)))
