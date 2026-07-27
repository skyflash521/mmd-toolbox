"""線形ファストパスのテスト。

線形制御点で許容内(`early_exit_err` 以内)に収まる区間は `least_squares` を呼ばず即採用し、
最適化の呼び出し回数そのものを減らす。許容(`early_exit_err`)が無い全探索ではファストパスを
取らない(閾値が無い)。線形で収まらない区間は従来どおり `least_squares` を回す(過剰発火しない)。
"""

from vmd import fit, interp

LIN = fit._BEZIER_LINEAR_CP  # 対角制御点(20,20,107,107)=正規化空間で y=x の直線


def _count_least_squares(monkeypatch):
    counter = {"n": 0}
    orig = fit.least_squares

    def wrapper(*args, **kwargs):
        counter["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(fit, "least_squares", wrapper)
    return counter


def test_linear_scalar_skips_least_squares(monkeypatch):
    # 完全に線形なサンプル(線形制御点で誤差ほぼ0)は least_squares を呼ばずに即採用する。
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 12 for i in range(11)]
    ys = list(xs)
    cp, err = fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    assert calls["n"] == 0
    assert cp == LIN
    assert err <= 0.01


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


# --- skip_fastpath: ファストパスのオプトアウト -----------------

# 「線形でも許容内に収まるが実際は曲がっている」サンプル(強いイージング)。
_CURVED_XS = [(i + 1) / 12 for i in range(11)]
_CURVED_YS = [interp._solve_factor(96, 0, 96, 30, x) for x in _CURVED_XS]


def _linear_err(xs, ys):
    return max(abs(interp._solve_factor(*LIN, x) - y) for x, y in zip(xs, ys, strict=True))


def _is_linear_curve(cp, xs):
    # 制御点 cp の曲線が数学的に恒等(y=x)か。skip_fastpath の実フィットは線形サンプルでも
    # LIN(20,20,107,107)と別の対角制御点(例 39,39,124,124)を返しうるので、等値でなく曲線で見る。
    return max(abs(interp._solve_factor(*cp, x) - x) for x in xs) < 1e-3


def test_skip_fastpath_forces_real_bezier(monkeypatch):
    # 線形が許容内に収まる閾値でも、skip_fastpath=True なら線形ファストパスを切り
    # least_squares 由来の非線形制御点を返す。既定は従来どおり線形を即採用。
    thr = _linear_err(_CURVED_XS, _CURVED_YS) + 1e-6
    cp_default, _ = fit.fit_bezier_curve(_CURVED_XS, _CURVED_YS, early_exit_err=thr)
    assert cp_default == LIN  # 既定はファストパスで線形を即採用(前提=線形が許容内)
    calls = _count_least_squares(monkeypatch)
    cp_skip, err_skip = fit.fit_bezier_curve(
        _CURVED_XS, _CURVED_YS, early_exit_err=thr, skip_fastpath=True
    )
    assert calls["n"] >= 1  # ファストパスを切るので least_squares を回す
    assert cp_skip != LIN   # 実フィット由来の非線形制御点


def test_skip_fastpath_coeff_forces_real_bezier(monkeypatch):
    # 回転の係数曲線でも、skip_fastpath=True で線形ファストパスを切り非線形制御点を返す。
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
    # cheap accept だけ残らないこと: 線形は外すが固定 ease が許容内に収まる閾値でも、
    # skip_fastpath=True なら cheap accept も切って least_squares を回す。
    # 固定 ease 候補 (53,0,127,127) で生成した曲線は、線形では大きく外れるが当該 ease では誤差0。
    cand = fit._CHEAP_EASE_CPS[0]
    ys = [interp._solve_factor(*cand, x) for x in _CURVED_XS]
    cand_err = max(abs(interp._solve_factor(*cand, x) - y) for x, y in zip(_CURVED_XS, ys, strict=True))
    thr = cand_err + 1e-6
    # 既定: 線形は外れるが cheap accept が発火して固定 ease を即採用(least_squares なし)
    calls_default = _count_least_squares(monkeypatch)
    cp_default, _ = fit.fit_bezier_curve(_CURVED_XS, ys, early_exit_err=thr)
    assert calls_default["n"] == 0
    assert cp_default == cand
    # skip_fastpath: cheap accept も切るので least_squares を回す
    calls_skip = _count_least_squares(monkeypatch)
    fit.fit_bezier_curve(_CURVED_XS, ys, early_exit_err=thr, skip_fastpath=True)
    assert calls_skip["n"] >= 1


def test_skip_fastpath_linear_sample_still_linear():
    # 数学的に線形なサンプルは skip_fastpath=True でも線形曲線を返す(退化ではなく正解)。
    # 実フィットは LIN と別の対角制御点を返しうるので、等値でなく曲線が恒等かで見る。
    xs = [(i + 1) / 12 for i in range(11)]
    ys = list(xs)
    cp, _ = fit.fit_bezier_curve(xs, ys, early_exit_err=0.01, skip_fastpath=True)
    assert _is_linear_curve(cp, xs)


def test_skip_fastpath_empty_returns_linear():
    # 内部点なしは両モードで線形制御点。
    assert fit.fit_bezier_curve([], [], skip_fastpath=True)[0] == LIN
    assert fit._fit_coeff_curve([], lambda c: [], skip_fastpath=True) == LIN


def test_axis_curve_passthrough_skip_fastpath():
    # _axis_curve が skip_fastpath を fit_bezier_curve へ素通しすること(曲がった軸で非線形化)。
    a, b = 0, 11
    a0, a1 = 0.0, 1.0
    # 強いイージングの軸サンプル(端点 a0,a1、内部は曲線)。
    def sample_fn(f):
        return a0 + (a1 - a0) * interp._solve_factor(96, 0, 96, 30, (f - a) / (b - a))

    # 線形がぎりぎり許容内に収まる閾値。
    internal = range(a + 1, b)
    xs = [(f - a) / (b - a) for f in internal]
    ys = [(sample_fn(f) - a0) / (a1 - a0) for f in internal]
    thr = _linear_err(xs, ys) + 1e-6
    cp_default = fit._axis_curve(a0, a1, a, b, sample_fn, early_exit_err=thr)
    assert cp_default == LIN
    cp_skip = fit._axis_curve(a0, a1, a, b, sample_fn, early_exit_err=thr, skip_fastpath=True)
    assert cp_skip != LIN
