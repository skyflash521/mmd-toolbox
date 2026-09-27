import pytest

from vpr2vmd import openness

_LO, _HI = 0.30, 0.75
_OPEN_MAX = 0.90
_DEFAULT = (_LO + _HI) / 2


def test_velocity_max_maps_to_high():
    assert openness.velocity_to_open(127, lo=_LO, hi=_HI, open_max=_OPEN_MAX) == pytest.approx(_HI)


def test_velocity_zero_maps_to_low():
    assert openness.velocity_to_open(0, lo=_LO, hi=_HI, open_max=_OPEN_MAX) == pytest.approx(_LO)


def test_velocity_mid_follows_power_curve():
    n = 64 / 127
    expected = _LO + (_HI - _LO) * (n ** openness.GAMMA_DEFAULT)
    assert openness.velocity_to_open(64, lo=_LO, hi=_HI, open_max=_OPEN_MAX) == pytest.approx(expected)


def test_velocity_gamma_parameter_changes_curve():
    n = 64 / 127
    linear = _LO + (_HI - _LO) * n
    assert openness.velocity_to_open(
        64, lo=_LO, hi=_HI, open_max=_OPEN_MAX, gamma=1.0
    ) == pytest.approx(linear)
    assert openness.velocity_to_open(
        64, lo=_LO, hi=_HI, open_max=_OPEN_MAX
    ) != pytest.approx(linear)


def test_velocity_monotonic_increasing():
    vals = [openness.velocity_to_open(v, lo=_LO, hi=_HI, open_max=_OPEN_MAX) for v in (0, 32, 64, 96, 127)]
    assert vals == sorted(vals)
    assert vals[0] < vals[-1]


def test_open_max_caps_below_high():
    assert openness.velocity_to_open(127, lo=_LO, hi=_HI, open_max=0.50) == pytest.approx(0.50)


def test_open_amounts_empty():
    assert openness.open_amounts([], lo=_LO, hi=_HI, open_max=_OPEN_MAX, default_open=_DEFAULT) == []


def test_open_amounts_uniform_uses_default():
    result = openness.open_amounts(
        [64, 64, 64], lo=_LO, hi=_HI, open_max=_OPEN_MAX, default_open=_DEFAULT
    )
    assert result == [pytest.approx(_DEFAULT)] * 3


def test_open_amounts_single_note_is_uniform_default():
    assert openness.open_amounts(
        [80], lo=_LO, hi=_HI, open_max=_OPEN_MAX, default_open=_DEFAULT
    ) == [pytest.approx(_DEFAULT)]


def test_open_amounts_default_is_capped_by_open_max():
    result = openness.open_amounts(
        [70, 70], lo=_LO, hi=_HI, open_max=0.50, default_open=0.80
    )
    assert result == [pytest.approx(0.50), pytest.approx(0.50)]


def test_open_amounts_default_above_hi_is_kept_below_open_max():
    result = openness.open_amounts(
        [70, 70], lo=_LO, hi=0.75, open_max=0.90, default_open=0.85
    )
    assert result == [pytest.approx(0.85), pytest.approx(0.85)]


def test_open_amounts_default_below_open_max_is_unchanged():
    result = openness.open_amounts(
        [70, 70], lo=_LO, hi=_HI, open_max=0.90, default_open=0.40
    )
    assert result == [pytest.approx(0.40), pytest.approx(0.40)]


def test_open_amounts_varying_maps_each_velocity():
    result = openness.open_amounts(
        [0, 127], lo=_LO, hi=_HI, open_max=_OPEN_MAX, default_open=_DEFAULT
    )
    assert result == [pytest.approx(_LO), pytest.approx(_HI)]
