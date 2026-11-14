import math

from vmd import fit
from vmd import reduce as vreduce
from vmd.reduce import build_bone_tolerances, reduce_bone_track, verify_bone_track
from vmd.types import BoneKey

_LOOSE = fit._LSQ_LOOSE_TOL


def _capture_least_squares(monkeypatch):
    captured = []
    orig = fit.least_squares

    def cap(fun, x0, **kwargs):
        captured.append(kwargs)
        return orig(fun, x0, **kwargs)

    monkeypatch.setattr(fit, "least_squares", cap)
    return captured


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
