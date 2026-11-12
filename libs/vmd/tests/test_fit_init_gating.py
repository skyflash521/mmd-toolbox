import math

from vmd import fit
from vmd import reduce as vreduce
from vmd.reduce import build_bone_tolerances, reduce_bone_track, verify_bone_track
from vmd.types import BoneKey


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
