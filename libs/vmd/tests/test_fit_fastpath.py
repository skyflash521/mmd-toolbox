from vmd import fit, interp

LIN = fit._BEZIER_LINEAR_CP
STRONG_EASE = (96, 0, 96, 30)


def _count_least_squares(monkeypatch):
    counter = {"n": 0}
    orig = fit.least_squares

    def wrapper(*args, **kwargs):
        counter["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(fit, "least_squares", wrapper)
    return counter


def test_linear_scalar_skips_least_squares(monkeypatch):
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 12 for i in range(11)]
    ys = list(xs)
    cp, err = fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    assert calls["n"] == 0
    assert cp == LIN
    assert err <= 0.01


def test_coeff_linear_skips_least_squares(monkeypatch):
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 6 for i in range(5)]

    def resid_at(coeff):
        return [coeff(x) - x for x in xs]

    cp = fit._fit_coeff_curve(xs, resid_at, early_exit_err=0.01)
    assert calls["n"] == 0
    assert cp == LIN


def test_nonlinear_still_calls_least_squares(monkeypatch):
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 12 for i in range(11)]
    ys = [interp._solve_factor(*STRONG_EASE, x) for x in xs]
    fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    assert calls["n"] >= 1


def test_no_early_exit_err_takes_no_fastpath(monkeypatch):
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 12 for i in range(11)]
    ys = list(xs)
    fit.fit_bezier_curve(xs, ys)
    assert calls["n"] >= 1


_CURVED_XS = [(i + 1) / 12 for i in range(11)]
_CURVED_YS = [interp._solve_factor(*STRONG_EASE, x) for x in _CURVED_XS]


def _linear_err(xs, ys):
    return max(abs(interp._solve_factor(*LIN, x) - y) for x, y in zip(xs, ys, strict=True))


def _is_identity_curve(cp, xs):
    return max(abs(interp._solve_factor(*cp, x) - x) for x in xs) < 1e-3


def test_skip_fastpath_forces_real_bezier(monkeypatch):
    thr = _linear_err(_CURVED_XS, _CURVED_YS) + 1e-6
    cp_default, _ = fit.fit_bezier_curve(_CURVED_XS, _CURVED_YS, early_exit_err=thr)
    assert cp_default == LIN
    calls = _count_least_squares(monkeypatch)
    cp_skip, err_skip = fit.fit_bezier_curve(
        _CURVED_XS, _CURVED_YS, early_exit_err=thr, skip_fastpath=True
    )
    assert calls["n"] >= 1
    assert cp_skip != LIN


def test_skip_fastpath_coeff_forces_real_bezier(monkeypatch):
    def resid_at(coeff):
        return [coeff(x) - y for x, y in zip(_CURVED_XS, _CURVED_YS, strict=True)]

    lin_res = max(abs(r) for r in resid_at(lambda x: interp._solve_factor(*LIN, x)))
    thr = lin_res + 1e-6
    cp_default = fit._fit_coeff_curve(_CURVED_XS, resid_at, early_exit_err=thr)
    assert cp_default == LIN
    calls = _count_least_squares(monkeypatch)
    cp_skip = fit._fit_coeff_curve(_CURVED_XS, resid_at, early_exit_err=thr, skip_fastpath=True)
    assert calls["n"] >= 1
    assert cp_skip != LIN


def test_skip_fastpath_cheap_accept_also_skipped(monkeypatch):
    cand = fit._CHEAP_EASE_CPS[0]
    ys = [interp._solve_factor(*cand, x) for x in _CURVED_XS]
    cand_err = max(abs(interp._solve_factor(*cand, x) - y) for x, y in zip(_CURVED_XS, ys, strict=True))
    thr = cand_err + 1e-6
    calls_default = _count_least_squares(monkeypatch)
    cp_default, _ = fit.fit_bezier_curve(_CURVED_XS, ys, early_exit_err=thr)
    assert calls_default["n"] == 0
    assert cp_default == cand
    calls_skip = _count_least_squares(monkeypatch)
    fit.fit_bezier_curve(_CURVED_XS, ys, early_exit_err=thr, skip_fastpath=True)
    assert calls_skip["n"] >= 1


def test_skip_fastpath_linear_sample_still_identity_curve():
    xs = [(i + 1) / 12 for i in range(11)]
    ys = list(xs)
    cp, _ = fit.fit_bezier_curve(xs, ys, early_exit_err=0.01, skip_fastpath=True)
    assert _is_identity_curve(cp, xs)


def test_skip_fastpath_empty_returns_linear():
    assert fit.fit_bezier_curve([], [], skip_fastpath=True)[0] == LIN
    assert fit._fit_coeff_curve([], lambda c: [], skip_fastpath=True) == LIN


def test_axis_curve_passes_skip_fastpath_to_fit():
    a, b = 0, 11
    a0, a1 = 0.0, 1.0

    def sample_fn(f):
        return a0 + (a1 - a0) * interp._solve_factor(*STRONG_EASE, (f - a) / (b - a))

    internal = range(a + 1, b)
    xs = [(f - a) / (b - a) for f in internal]
    ys = [(sample_fn(f) - a0) / (a1 - a0) for f in internal]
    thr = _linear_err(xs, ys) + 1e-6
    cp_default = fit._axis_curve(a0, a1, a, b, sample_fn, early_exit_err=thr)
    assert cp_default == LIN
    cp_skip = fit._axis_curve(a0, a1, a, b, sample_fn, early_exit_err=thr, skip_fastpath=True)
    assert cp_skip != LIN
