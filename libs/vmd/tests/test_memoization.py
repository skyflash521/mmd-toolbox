"""区間フィット結果のメモ化テスト。

curve(a,b) と residual(a,b) は同一区間に対して同じベジェ曲線フィット
(fit_bezier_curve / _fit_coeff_curve)を計算する。チャンネルはインスタンス単位で
区間フィット結果をキャッシュし、同一区間の再フィットを避ける(挙動は不変=制御点は同一、
フィット呼び出し回数だけが減る)。

ここではモジュール関数の呼び出し回数を数え、同一区間 (a,b) に対する residual→curve→curve の
連続呼び出しで実フィットがちょうど1回になることを確認する。
"""

import math

import pytest

from vmd import fit, interp

EASE = (96, 0, 96, 30)


def _eased(v0, v1, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*EASE, f / span) for f in range(n)]


def _count_calls(monkeypatch, name):
    """fit モジュールの関数 name をラップして呼び出し回数を数える。"""
    counter = {"n": 0}
    orig = getattr(fit, name)

    def wrapper(*args, **kwargs):
        counter["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(fit, name, wrapper)
    return counter


def test_scalar_curve_fit_memoized(monkeypatch):
    calls = _count_calls(monkeypatch, "fit_bezier_curve")
    ch = fit.LinearScalarChannel(0, _eased(0.0, 100.0), tol=1.0, mode="bezier")
    ch.residual(0, 10)
    ch.curve(0, 10)
    ch.curve(0, 10)
    assert calls["n"] == 1


def test_fov_curve_fit_memoized(monkeypatch):
    calls = _count_calls(monkeypatch, "fit_bezier_curve")
    ch = fit.FovChannel(0, _eased(30.0, 80.0), tol=1.0, mode="bezier")
    ch.residual(0, 10)
    ch.curve(0, 10)
    ch.curve(0, 10)
    assert calls["n"] == 1


def test_euclidean_axis_fit_memoized(monkeypatch):
    # Y 軸だけが動く(X/Z は端点同値=正規化不能でフィットを呼ばない)。
    # よって実フィットは Y 軸の1区間ぶんで、residual→curve→curve で1回に収束する。
    calls = _count_calls(monkeypatch, "fit_bezier_curve")
    ys = _eased(0.0, 100.0)
    vecs = [(0.0, y, 0.0) for y in ys]
    ch = fit.EuclideanVectorChannel(0, vecs, tol=1.0, mode="bezier")
    ch.residual(0, 10)
    ch.curve(0, 10)
    ch.curve(0, 10)
    assert calls["n"] == 1


def test_camera_rotation_coeff_fit_memoized(monkeypatch):
    calls = _count_calls(monkeypatch, "_fit_coeff_curve")
    c = [interp._solve_factor(*EASE, f / 10) for f in range(11)]
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    eulers = [(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f]) for f in range(11)]
    ch = fit.CameraRotationChannel(0, eulers, tol=0.1, mode="bezier")
    ch.residual(0, 10)
    ch.curve(0, 10)
    ch.curve(0, 10)
    assert calls["n"] == 1


def test_bone_rotation_coeff_fit_memoized(monkeypatch):
    calls = _count_calls(monkeypatch, "_fit_coeff_curve")

    def _z(deg):
        h = math.radians(deg) / 2.0
        return (0.0, 0.0, math.sin(h), math.cos(h))

    c = [interp._solve_factor(*EASE, f / 10) for f in range(11)]
    quats = [_z(90.0 * c[f]) for f in range(11)]
    ch = fit.BoneRotationChannel(0, quats, tol=0.1, mode="bezier")
    ch.residual(0, 10)
    ch.curve(0, 10)
    ch.curve(0, 10)
    assert calls["n"] == 1


def test_memoized_curve_matches_unmemoized_scalar():
    # メモ化しても制御点は不変(キャッシュ有無で同じ結果)。
    ch = fit.LinearScalarChannel(0, _eased(0.0, 100.0), tol=1.0, mode="bezier")
    first = ch.curve(0, 10)
    second = ch.curve(0, 10)
    assert first == second == EASE
