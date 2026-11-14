from vmd import fit, interp

EASE = (96, 0, 96, 30)


def _eased(v0, v1, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*EASE, f / span) for f in range(n)]


def test_memoized_curve_matches_first_fit_scalar():
    ch = fit.LinearScalarChannel(0, _eased(0.0, 100.0), tol=1.0, mode="bezier")
    first = ch.curve(0, 10)
    second = ch.curve(0, 10)
    assert first == second == EASE
