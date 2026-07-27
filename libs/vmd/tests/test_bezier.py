"""スカラー ベジェ曲線フィットのテスト。

fit_bezier_curve(xs, ys) は、正規化時間 xs(0<x<1)と正規化値 ys に対し、VMD補間曲線
(制御点 (x1,y1,x2,y2) を 0..127 整数で量子化)が x->y を近似するよう制御点を最小二乗で
探索し、量子化後の (制御点, 最大絶対誤差) を返す。最終誤差は量子化後の曲線で再評価する。
"""

import math

import pytest

from vmd import interp
from vmd.fit import fit_bezier_curve


def _internal_xs(n=9):
    # 内部正規化時間 1/10..9/10。
    return [(i + 1) / (n + 1) for i in range(n)]


def test_linear_data_fits_with_zero_error():
    xs = _internal_xs()
    ys = list(xs)  # y = x(線形)
    cp, err = fit_bezier_curve(xs, ys)
    # 線形は線形制御点(x1==y1, x2==y2)で y=x を厳密表現でき、誤差は実質0。
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
    # なくチャンネル側の責務。ここは正規化済みの非単調 ys を渡す。)
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
