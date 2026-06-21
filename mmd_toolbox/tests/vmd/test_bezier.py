"""スカラー ベジェ曲線フィットのテスト(sparsevmd.md §5.2, §5.4)。

fit_bezier_curve(xs, ys) は、正規化時間 xs(0<x<1)と正規化値 ys に対し、VMD補間曲線
(制御点 (x1,y1,x2,y2) を 0..127 整数で量子化)が x->y を近似するよう制御点を最小二乗で
探索し、量子化後の (制御点, 最大絶対誤差) を返す。最終誤差は量子化後の曲線で再評価する。
"""

import math

import numpy as np
import pytest

from mmd_toolbox.vmd import interp
from mmd_toolbox.vmd import fit
from mmd_toolbox.vmd.fit import fit_bezier_curve


def _internal_xs(n=9):
    # 内部正規化時間 1/10..9/10。
    return [(i + 1) / (n + 1) for i in range(n)]


def test_linear_data_fits_with_zero_error():
    xs = _internal_xs()
    ys = list(xs)  # y = x(線形)
    cp, err = fit_bezier_curve(xs, ys)
    # 線形は線形制御点(x1==y1, x2==y2)で y=x を厳密表現でき、誤差は実質0(§10.2)。
    assert err < 1e-6
    assert len(cp) == 4
    assert all(0 <= c <= 127 for c in cp)
    assert cp[0] <= cp[2]  # x1 <= x2(X単調)


def test_known_curve_reproduced():
    # 既知の量子化曲線 (40,10,90,118) からサンプルを生成すると、フィットは量子化誤差内で再現する。
    xs = _internal_xs()
    ys = [interp._solve_factor(40, 10, 90, 118, x) for x in xs]
    cp, err = fit_bezier_curve(xs, ys)
    assert err < 0.01  # 量子化誤差内(同一の量子化曲線を復元できる)
    assert cp[0] <= cp[2]


def test_ease_curve_low_error():
    # 強い ease(0,0,1,1 風でない非対称)でも低誤差で表現できる。
    xs = _internal_xs()
    ys = [interp._solve_factor(0, 0, 127, 64, x) for x in xs]
    cp, err = fit_bezier_curve(xs, ys)
    assert err < 0.03


def test_nonmonotonic_values_have_large_error():
    # 非単調な y(中央で山→戻る)は、x:0→1 に対し y を 0→1 へ写す1本のVMD曲線では
    # 表現できず誤差が大きい。(端点同値 v0==v1 の正規化不能ケースは fit_bezier_curve では
    # なくチャンネル側 §5.2 の責務。ここは正規化済みの非単調 ys を渡す。)
    xs = _internal_xs()
    ys = [math.sin(math.pi * x) for x in xs]  # 内部点で 0.31→1→0.31 の山
    cp, err = fit_bezier_curve(xs, ys)
    assert err > 0.1


def test_quantized_cp_in_range_and_monotonic():
    xs = _internal_xs()
    ys = [interp._solve_factor(20, 40, 107, 100, x) for x in xs]
    cp, err = fit_bezier_curve(xs, ys)
    assert all(isinstance(c, int) and 0 <= c <= 127 for c in cp)
    assert cp[0] <= cp[2]


def test_empty_internal_returns_valid_zero_error():
    # 内部点が無い(隣接区間)場合は誤差0で、量子化済みの有効な制御点を返す。
    cp, err = fit_bezier_curve([], [])
    assert err == 0.0
    assert len(cp) == 4
    assert all(isinstance(c, int) and 0 <= c <= 127 for c in cp)
    assert cp[0] <= cp[2]


def test_error_is_measured_on_quantized_curve():
    # 返す誤差は量子化後の曲線を interp._solve_factor で再評価した値と整合する。
    xs = _internal_xs()
    ys = [interp._solve_factor(30, 5, 100, 120, x) for x in xs]
    cp, err = fit_bezier_curve(xs, ys)
    recomputed = max(abs(interp._solve_factor(*cp, x) - y) for x, y in zip(xs, ys))
    assert err == pytest.approx(recomputed, abs=1e-9)


# --- Step B: _bezier_y_at のベクトル化 -------------------------------------
#
# _bezier_y_at_many(px1, py1, px2, py2, xs) は単点版 _bezier_y_at を numpy 配列 xs で
# 一括評価したもの。両者は同一の Newton 法+二分法フォールバックなので結果が一致する。

# 正規化制御点 (px1,py1,px2,py2)(各 [0,1])。§B 注記の端点・線形・ease・端点近傍を網羅。
_BEZIER_MANY_CASES = [
    (20.0 / 127, 20.0 / 127, 107.0 / 127, 107.0 / 127),  # 線形(既定)
    (0.42, 0.0, 1.0, 1.0),   # ease-in(x2=1)
    (0.0, 0.0, 0.58, 1.0),   # ease-out(x1=0)
    (0.42, 0.0, 0.58, 1.0),  # ease-in-out
    (0.0, 0.5, 1.0, 0.5),    # x1=0, x2=1 端点
    (0.1, 0.9, 0.2, 0.95),   # 急峻
]


@pytest.mark.xfail(reason="impl pending: Step B vectorize", strict=False)
@pytest.mark.parametrize("cp", _BEZIER_MANY_CASES)
def test_bezier_y_at_many_matches_scalar(cp):
    # 端点 0,1 と内部点・端点近傍を含む xs で単点版と一致する。
    xs = np.array([0.0, 1e-6, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0 - 1e-6, 1.0])
    many = fit._bezier_y_at_many(*cp, xs)
    scalar = np.array([fit._bezier_y_at(*cp, float(x)) for x in xs])
    assert many.shape == xs.shape
    np.testing.assert_allclose(many, scalar, atol=1e-9, rtol=0.0)


@pytest.mark.xfail(reason="impl pending: Step B vectorize", strict=False)
def test_bezier_y_at_many_endpoints_exact():
    # x<=0 は 0.0、x>=1 は 1.0(クランプ)。
    cp = (0.42, 0.0, 0.58, 1.0)
    out = fit._bezier_y_at_many(*cp, np.array([-0.5, 0.0, 1.0, 1.5]))
    np.testing.assert_array_equal(out, np.array([0.0, 0.0, 1.0, 1.0]))


@pytest.mark.xfail(reason="impl pending: Step B vectorize", strict=False)
def test_bezier_y_at_many_random_agrees():
    # ランダム xs・複数制御点で広く一致を確認する(決定論シード)。
    rng = np.random.default_rng(1234567)
    xs = rng.random(200)
    for cp in _BEZIER_MANY_CASES:
        many = fit._bezier_y_at_many(*cp, xs)
        scalar = np.array([fit._bezier_y_at(*cp, float(x)) for x in xs])
        np.testing.assert_allclose(many, scalar, atol=1e-9, rtol=0.0)


@pytest.mark.xfail(reason="impl pending: Step B vectorize", strict=False)
def test_bezier_y_at_many_returns_ndarray_preserving_shape():
    # 戻り値は xs と同形の numpy.ndarray(flatten しない)。
    cp = (0.42, 0.0, 0.58, 1.0)
    xs = np.array([[0.1, 0.5], [0.7, 0.9]])
    out = fit._bezier_y_at_many(*cp, xs)
    assert isinstance(out, np.ndarray)
    assert out.shape == xs.shape


@pytest.mark.xfail(reason="impl pending: Step B vectorize", strict=False)
def test_fit_bezier_curve_residual_uses_vectorized(monkeypatch):
    # フィットの残差評価は配列版を使い、単点版 _bezier_y_at を呼ばない(性能目的の契約)。
    # 単点版を素通しのスパイに差し替え、フィット完走後に呼び出しが無いことを固定する
    # (fit_bezier_curve は least_squares を except でくるむため raise では検出できない)。
    real = fit._bezier_y_at
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(fit, "_bezier_y_at", spy)
    xs = _internal_xs()
    ys = [interp._solve_factor(30, 5, 100, 120, x) for x in xs]
    cp, err = fit_bezier_curve(xs, ys)
    assert len(cp) == 4
    assert calls == []  # 残差評価は配列版のみを使う
