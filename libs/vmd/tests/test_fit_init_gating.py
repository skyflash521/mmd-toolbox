import math

from vmd import fit
from vmd import reduce as vreduce
from vmd.reduce import build_bone_tolerances, reduce_bone_track, verify_bone_track
from vmd.types import BoneKey

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


def _hill_xy(n):
    xs = [(i + 1) / (n + 2) for i in range(n)]
    return xs, [math.sin(math.pi * x) for x in xs]


def test_fit_bezier_uses_reduced_inits_at_sample_threshold(monkeypatch):
    cap = _capture_least_squares(monkeypatch)
    xs, ys = _hill_xy(_N_AT_THRESHOLD)
    fit.fit_bezier_curve(xs, ys)
    assert len(cap) == len(fit._BEZIER_INITS_LARGE)


def test_fit_bezier_uses_all_inits_below_sample_threshold(monkeypatch):
    cap = _capture_least_squares(monkeypatch)
    xs, ys = _hill_xy(_N_BELOW_THRESHOLD)
    fit.fit_bezier_curve(xs, ys)
    assert len(cap) == len(fit._BEZIER_INITS)


def test_fit_coeff_uses_reduced_inits_at_sample_threshold(monkeypatch):
    cap = _capture_least_squares(monkeypatch)
    xs, targets = _hill_xy(_N_AT_THRESHOLD)

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets, strict=True)]

    fit._fit_coeff_curve(xs, resid_at)
    assert len(cap) == len(fit._BEZIER_INITS_LARGE)


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


def test_init_reduction_no_compression_regression_vs_full(monkeypatch):
    keys = _curvy_bone_keys_with_segment_above_threshold()
    tols = build_bone_tolerances(0.02, 0.20)

    reduced = reduce_bone_track(keys, [(0, 39)], tols, **_REDUCE_ARGS)
    assert verify_bone_track(keys, reduced, [(0, 39)], tols) == []

    monkeypatch.setattr(fit, "_bezier_inits", lambda n_samples: fit._BEZIER_INITS)
    full = reduce_bone_track(keys, [(0, 39)], tols, **_REDUCE_ARGS)
    assert verify_bone_track(keys, full, [(0, 39)], tols) == []
    assert len(reduced) <= len(full)
