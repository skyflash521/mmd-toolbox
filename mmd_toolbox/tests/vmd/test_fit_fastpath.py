"""線形ファストパスのテスト(reduce-fit-perf-plan.md レバー1, mmd_toolbox.md §6.3)。

線形制御点で許容内(`early_exit_err` 以内)に収まる区間は `least_squares` を呼ばず即採用し、
最適化の呼び出し回数そのものを減らす。許容(`early_exit_err`)が無い全探索ではファストパスを
取らない(閾値が無い)。線形で収まらない区間は従来どおり `least_squares` を回す(過剰発火しない)。
"""

import pytest

from mmd_toolbox.vmd import fit, interp

LIN = fit._BEZIER_LINEAR_CP  # 対角制御点(20,20,107,107)=正規化空間で y=x の直線


def _count_least_squares(monkeypatch):
    counter = {"n": 0}
    orig = fit.least_squares

    def wrapper(*args, **kwargs):
        counter["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(fit, "least_squares", wrapper)
    return counter


@pytest.mark.xfail(reason="impl pending: レバー1-線形ファストパス", strict=True)
def test_linear_scalar_skips_least_squares(monkeypatch):
    # 完全に線形なサンプル(線形制御点で誤差ほぼ0)は least_squares を呼ばずに即採用する。
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 12 for i in range(11)]
    ys = list(xs)
    cp, err = fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    assert calls["n"] == 0
    assert cp == LIN
    assert err <= 0.01


@pytest.mark.xfail(reason="impl pending: レバー1-線形ファストパス", strict=True)
def test_coeff_linear_skips_least_squares(monkeypatch):
    # 回転の係数曲線も、係数が線形なら least_squares を呼ばず線形制御点で即採用する。
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 6 for i in range(5)]

    def resid_at(coeff):
        return [coeff(x) - x for x in xs]

    cp = fit._fit_coeff_curve(xs, resid_at, early_exit_err=0.01)
    assert calls["n"] == 0
    assert cp == LIN


def test_nonlinear_still_calls_least_squares(monkeypatch):
    # 線形で許容内に収まらない曲線データはファストパスを取らず least_squares を回す(過剰発火しない)。
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 12 for i in range(11)]
    ys = [interp._solve_factor(96, 0, 96, 30, x) for x in xs]  # 強いイージング
    fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    assert calls["n"] >= 1


def test_no_early_exit_err_takes_no_fastpath(monkeypatch):
    # early_exit_err が無い(全探索)ときは閾値が無いのでファストパスを取らず least_squares を回す。
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 12 for i in range(11)]
    ys = list(xs)
    fit.fit_bezier_curve(xs, ys)
    assert calls["n"] >= 1
