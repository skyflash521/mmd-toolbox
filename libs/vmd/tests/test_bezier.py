import math

import pytest

from vmd import interp
from vmd.fit import fit_bezier_curve


def _internal_xs(n=9):
    return [(i + 1) / (n + 1) for i in range(n)]


def test_linear_data_fits_with_zero_error():
    xs = _internal_xs()
    ys = list(xs)
    cp, err = fit_bezier_curve(xs, ys)
    assert err < 1e-6
    assert len(cp) == 4
    assert all(0 <= c <= 127 for c in cp)
    assert cp[0] <= cp[2]


def test_known_quantized_curve_reproduced_within_quantization_error():
    xs = _internal_xs()
    ys = [interp._solve_factor(40, 10, 90, 118, x) for x in xs]
    cp, err = fit_bezier_curve(xs, ys)
    assert err < 0.01
    assert cp[0] <= cp[2]


def test_asymmetric_ease_curve_low_error():
    xs = _internal_xs()
    ys = [interp._solve_factor(0, 0, 127, 64, x) for x in xs]
    cp, err = fit_bezier_curve(xs, ys)
    assert err < 0.03


def test_nonmonotonic_values_have_large_error():
    xs = _internal_xs()
    ys = [math.sin(math.pi * x) for x in xs]
    cp, err = fit_bezier_curve(xs, ys)
    assert err > 0.1


def test_quantized_cp_in_range_and_monotonic():
    xs = _internal_xs()
    ys = [interp._solve_factor(20, 40, 107, 100, x) for x in xs]
    cp, err = fit_bezier_curve(xs, ys)
    assert all(isinstance(c, int) and 0 <= c <= 127 for c in cp)
    assert cp[0] <= cp[2]


def test_empty_internal_returns_valid_zero_error():
    cp, err = fit_bezier_curve([], [])
    assert err == 0.0
    assert len(cp) == 4
    assert all(isinstance(c, int) and 0 <= c <= 127 for c in cp)
    assert cp[0] <= cp[2]


def test_error_is_measured_on_quantized_curve():
    xs = _internal_xs()
    ys = [interp._solve_factor(30, 5, 100, 120, x) for x in xs]
    cp, err = fit_bezier_curve(xs, ys)
    recomputed = max(abs(interp._solve_factor(*cp, x) - y) for x, y in zip(xs, ys, strict=True))
    assert err == pytest.approx(recomputed, abs=1e-9)
