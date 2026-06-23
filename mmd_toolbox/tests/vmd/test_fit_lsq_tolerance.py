"""least_squares 収束許容緩和のテスト(reduce-fit-perf-plan.md レバー2, mmd_toolbox.md §6.3)。

采否は量子化後誤差で判定するので scipy 既定精度(~1e-8)まで詰める必要はない。フィットの
least_squares 呼び出しは ftol/xtol/gtol を 1e-3 へ緩めて反復を減らす。出力(制御点)は変わりうるが、
采否の不変条件(出力後の最大正規化誤差 ≤ 1.0)は保つ。
"""

import math

import pytest

from mmd_toolbox.vmd import fit, interp
from mmd_toolbox.vmd import reduce as vreduce
from mmd_toolbox.vmd.reduce import build_bone_tolerances, reduce_bone_track, verify_bone_track
from mmd_toolbox.vmd.types import BoneKey

_LOOSE = 1e-3


def _capture_least_squares(monkeypatch):
    captured = []
    orig = fit.least_squares

    def cap(fun, x0, **kwargs):
        captured.append(kwargs)
        return orig(fun, x0, **kwargs)

    monkeypatch.setattr(fit, "least_squares", cap)
    return captured


@pytest.mark.xfail(reason="impl pending: レバー2-収束許容緩和", strict=True)
def test_fit_bezier_loosens_least_squares_tolerance(monkeypatch):
    cap = _capture_least_squares(monkeypatch)
    xs = [(i + 1) / 12 for i in range(11)]
    ys = [interp._solve_factor(96, 0, 96, 30, x) for x in xs]  # 線形で収まらない=least_squares 経由
    fit.fit_bezier_curve(xs, ys)  # early_exit_err=None: ファストパスを取らず least_squares を回す
    assert cap, "least_squares が呼ばれていない"
    assert all(
        kw.get("ftol") == _LOOSE and kw.get("xtol") == _LOOSE and kw.get("gtol") == _LOOSE
        for kw in cap
    )


@pytest.mark.xfail(reason="impl pending: レバー2-収束許容緩和", strict=True)
def test_fit_coeff_loosens_least_squares_tolerance(monkeypatch):
    cap = _capture_least_squares(monkeypatch)
    xs = [(i + 1) / 10 for i in range(9)]
    targets = [math.sin(math.pi * x) for x in xs]  # 山型=1本のベジェで表現不可

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets)]

    fit._fit_coeff_curve(xs, resid_at)
    assert cap, "least_squares が呼ばれていない"
    assert all(
        kw.get("ftol") == _LOOSE and kw.get("xtol") == _LOOSE and kw.get("gtol") == _LOOSE
        for kw in cap
    )


def test_loosened_tolerance_keeps_acceptance_invariant():
    # 許容緩和後も采否の不変条件(出力後検証が空=全区間が実許容内)を保つ。曲線的なボーンを bezier 疎化し
    # verify_bone_track が空であることを確認する(実装前後どちらでも成立すべきガード)。
    keys = []
    for f in range(40):
        t = f / 10.0
        pos = (math.sin(t) * 3.0, math.cos(t * 0.7) * 2.0, math.sin(t * 1.3) * 1.5)
        ang = math.radians(30.0 * math.sin(t * 0.5))
        quat = (0.0, 0.0, math.sin(ang / 2), math.cos(ang / 2))
        keys.append(BoneKey(b"\x00" * 15, f, pos, quat, vreduce.BONE_LINEAR_INTERP))
    tols = build_bone_tolerances(0.02, 0.20)
    out = reduce_bone_track(
        keys, [(0, 39)], tols,
        cut_thresholds=(1.0, 30.0), keep_frames=[], no_cut_detect=True,
        min_seg=1, max_seg=180, strict=False, curve_mode="bezier",
    )
    assert verify_bone_track(keys, out, [(0, 39)], tols) == []
