import math

from vmd import fit, interp
from vmd import reduce as vreduce
from vmd.reduce import build_bone_tolerances, reduce_bone_track, verify_bone_track
from vmd.types import BoneKey

_LOOSE = fit._LSQ_LOOSE_TOL
_N_AT_THRESHOLD = fit._LSQ_LOOSEN_MIN_SAMPLES
_N_BELOW_THRESHOLD = fit._LSQ_LOOSEN_MIN_SAMPLES - 1


def _capture_least_squares(monkeypatch):
    captured = []
    orig = fit.least_squares

    def cap(fun, x0, **kwargs):
        captured.append(kwargs)
        return orig(fun, x0, **kwargs)

    monkeypatch.setattr(fit, "least_squares", cap)
    return captured


def _eased_xy(n):
    xs = [(i + 1) / (n + 2) for i in range(n)]
    return xs, [interp._solve_factor(96, 0, 96, 30, x) for x in xs]


def test_fit_bezier_loosens_tolerance_at_sample_threshold(monkeypatch):
    cap = _capture_least_squares(monkeypatch)
    xs, ys = _eased_xy(_N_AT_THRESHOLD)
    fit.fit_bezier_curve(xs, ys)
    assert cap, "least_squares が呼ばれていない"
    assert all(
        kw.get("ftol") == _LOOSE and kw.get("xtol") == _LOOSE and kw.get("gtol") == _LOOSE
        for kw in cap
    )


def test_fit_bezier_keeps_default_tolerance_below_sample_threshold(monkeypatch):
    cap = _capture_least_squares(monkeypatch)
    xs, ys = _eased_xy(_N_BELOW_THRESHOLD)
    fit.fit_bezier_curve(xs, ys)
    assert cap, "least_squares が呼ばれていない"
    assert all("ftol" not in kw and "xtol" not in kw and "gtol" not in kw for kw in cap)


def test_fit_coeff_loosens_tolerance_at_sample_threshold(monkeypatch):
    cap = _capture_least_squares(monkeypatch)
    n = _N_AT_THRESHOLD
    xs = [(i + 1) / (n + 2) for i in range(n)]
    targets = [math.sin(math.pi * x) for x in xs]

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets, strict=True)]

    fit._fit_coeff_curve(xs, resid_at)
    assert cap, "least_squares が呼ばれていない"
    assert all(
        kw.get("ftol") == _LOOSE and kw.get("xtol") == _LOOSE and kw.get("gtol") == _LOOSE
        for kw in cap
    )


def _curvy_bone_keys_with_segment_above_threshold():
    keys = []
    for f in range(40):
        t = f / 10.0
        pos = (math.sin(t) * 3.0, math.cos(t * 0.7) * 2.0, math.sin(t * 1.3) * 1.5)
        ang = math.radians(30.0 * math.sin(t * 0.5))
        quat = (0.0, 0.0, math.sin(ang / 2), math.cos(ang / 2))
        keys.append(BoneKey(b"\x00" * 15, f, pos, quat, vreduce.BONE_LINEAR_INTERP))
    return keys


_REDUCE_ARGS = dict(
    cut_thresholds=(1.0, 30.0), keep_frames=[], no_cut_detect=True,
    min_seg=1, max_seg=180, strict=False, curve_mode="bezier",
)


def test_loosening_fires_in_reduce_without_compression_regression_vs_default(monkeypatch):
    keys = _curvy_bone_keys_with_segment_above_threshold()
    tols = build_bone_tolerances(0.02, 0.20)

    cap = _capture_least_squares(monkeypatch)
    loosened = reduce_bone_track(keys, [(0, 39)], tols, **_REDUCE_ARGS)
    assert any(kw.get("ftol") == _LOOSE for kw in cap)
    assert verify_bone_track(keys, loosened, [(0, 39)], tols) == []

    monkeypatch.setattr(fit, "_lsq_kwargs", lambda n_samples: {})
    tight = reduce_bone_track(keys, [(0, 39)], tols, **_REDUCE_ARGS)
    assert verify_bone_track(keys, tight, [(0, 39)], tols) == []
    assert len(loosened) <= len(tight)
