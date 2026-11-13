import math
import signal
import sys

import pytest

from mocapvmd import presets
from mocapvmd import reduce as mreduce
from vmd.reduce import build_bone_tolerances, measure_bone_errors, reduce_bone_track

from .helpers import BONE_NONLINEAR, bone

_FIXED_REDUCE_ARGS = dict(
    cut_thresholds=(1.0, 30.0), keep_frames=[], no_cut_detect=False,
    min_seg=1, strict=False,
)

_MIXED_CATEGORY_PREFIXES = ("右人指", "右腕", "右足", "上半身")


def _dense_parabolic_track(name, n=11, k=0.01):
    return [bone(name, f, pos=(round(k * f * f, 6), 0.0, 0.0)) for f in range(n)]


def _single_key_tracks(count):
    return [bone(f"単{i:02d}", 0, pos=(float(i), 0.0, 0.0)) for i in range(count)]


def _reduce_with_building_blocks(
    track_keys, preset, category, curve_mode="bezier", override_pos=None, override_rot=None
):
    t = presets.resolve_reduction_tolerances(
        preset, category, override_pos=override_pos, override_rot=override_rot
    )
    tols = build_bone_tolerances(t["bone_pos"], t["bone_rot"])
    f0, f1 = track_keys[0].frame, track_keys[-1].frame
    return reduce_bone_track(track_keys, [(f0, f1)], tols, curve_mode=curve_mode, max_seg=None, **_FIXED_REDUCE_ARGS)


def _sorted_track(out, name):
    return sorted((k for k in out if k.name == name), key=lambda k: k.frame)


def test_each_track_reduced_over_its_own_frame_range_with_its_category_tolerance_and_no_extra_keys():
    center = _dense_parabolic_track("センター")
    fingers = [bone("右人指1", f, pos=(round(0.01 * f * f, 6), 0.0, 0.0)) for f in range(5, 16)]
    out = mreduce.reduce_bones(center + fingers, "medium")
    assert _sorted_track(out, "センター") == list(_reduce_with_building_blocks(center, "medium", "center"))
    assert _sorted_track(out, "右人指1") == list(_reduce_with_building_blocks(fingers, "medium", "fingers"))
    expected_all = (
        list(_reduce_with_building_blocks(center, "medium", "center"))
        + list(_reduce_with_building_blocks(fingers, "medium", "fingers"))
    )
    assert sorted(out, key=lambda k: (k.name, k.frame)) == sorted(expected_all, key=lambda k: (k.name, k.frame))


def test_single_key_track_kept_verbatim_alongside_multikey():
    single = [bone("右足ＩＫ", 0, pos=(1.0, 2.0, 3.0), interp=BONE_NONLINEAR)]
    multi = _dense_parabolic_track("センター")
    out = mreduce.reduce_bones(single + multi, "medium")
    assert _sorted_track(out, "右足ＩＫ") == single
    assert _sorted_track(out, "センター") == list(_reduce_with_building_blocks(multi, "medium", "center"))


@pytest.mark.parametrize("preset", ["slower"])
def test_reconstruction_error_stays_within_resolved_tolerance(preset):
    keys = _dense_parabolic_track("センター", n=31, k=0.05)
    out = mreduce.reduce_bones(keys, preset)
    tol = presets.resolve_reduction_tolerances(preset, "center")
    errors = measure_bone_errors(keys, _sorted_track(out, "センター"), [(0, 30)])
    assert max(errors["pos_x"], errors["pos_y"], errors["pos_z"]) <= tol["bone_pos"]
    assert errors["rot_deg"] <= tol["bone_rot"]


def _expected_cut_count(track_keys, tol):
    diag = {}
    f0, f1 = track_keys[0].frame, track_keys[-1].frame
    reduce_bone_track(
        track_keys, [(f0, f1)],
        build_bone_tolerances(tol["bone_pos"], tol["bone_rot"]),
        diagnostics=diag, max_seg=None, **_FIXED_REDUCE_ARGS,
    )
    return len(diag["cuts"])


def test_diagnostics_out_records_keys_category_tolerance_cuts_errors_per_track():
    center = _dense_parabolic_track("センター")
    fingers = _dense_parabolic_track("右人指1")
    diag = {}
    out = mreduce.reduce_bones(center + fingers, "medium", diagnostics_out=diag)
    assert set(diag) == {"センター", "右人指1"}
    for name, cat in (("センター", "center"), ("右人指1", "fingers")):
        src = _sorted_track(center + fingers, name)
        reduced = _sorted_track(out, name)
        tol = presets.resolve_reduction_tolerances("medium", cat)
        d = diag[name]
        assert d["input_keys"] == len(src)
        assert d["output_keys"] == len(reduced)
        assert d["tol_pos"] == tol["bone_pos"]
        assert d["tol_rot"] == tol["bone_rot"]
        assert d["cuts"] == _expected_cut_count(src, tol)
        assert d["errors"] == measure_bone_errors(src, reduced, [(src[0].frame, src[-1].frame)])


def _quat_y(deg):
    half = math.radians(deg) / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


@pytest.mark.parametrize("channel", ["override_pos", "override_rot"])
def test_override_tolerance_is_used_for_reduction(channel):
    keys = [
        bone("センター", f, pos=(round(0.05 * f * f, 6), 0.0, 0.0), rot=_quat_y(0.05 * f * f))
        for f in range(31)
    ]
    override = {"override_pos": 0.0001, "override_rot": 0.01}[channel]
    out = mreduce.reduce_bones(keys, "medium", **{channel: override})
    expected = list(_reduce_with_building_blocks(keys, "medium", "center", **{channel: override}))
    assert expected != list(_reduce_with_building_blocks(keys, "medium", "center"))
    assert _sorted_track(out, "センター") == expected


def test_diagnostics_out_tolerance_reflects_override():
    keys = _dense_parabolic_track("センター")
    diag = {}
    mreduce.reduce_bones(keys, "medium", override_pos=0.005, override_rot=0.05, diagnostics_out=diag)
    tol = presets.resolve_reduction_tolerances("medium", "center", override_pos=0.005, override_rot=0.05)
    assert diag["センター"]["tol_pos"] == tol["bone_pos"]
    assert diag["センター"]["tol_rot"] == tol["bone_rot"]


def test_diagnostics_out_records_single_key_track_as_unreduced_alongside_multikey():
    single = [bone("右足ＩＫ", 0, pos=(1.0, 2.0, 3.0), interp=BONE_NONLINEAR)]
    multi = _dense_parabolic_track("センター")
    diag = {}
    out = mreduce.reduce_bones(single + multi, "medium", diagnostics_out=diag)
    assert set(diag) == {"右足ＩＫ", "センター"}
    assert _sorted_track(out, "右足ＩＫ") == single
    d = diag["右足ＩＫ"]
    tol = presets.resolve_reduction_tolerances("medium", "foot_ik")
    assert d["input_keys"] == 1
    assert d["output_keys"] == 1
    assert d["tol_pos"] == tol["bone_pos"]
    assert d["tol_rot"] == tol["bone_rot"]
    assert d["cuts"] == 0
    assert d["errors"] == {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0, "rot_deg": 0.0}


def test_diagnostics_out_does_not_change_output():
    keys = _dense_parabolic_track("右足ＩＫ")
    diag = {}
    assert mreduce.reduce_bones(keys, "medium", diagnostics_out=diag) == mreduce.reduce_bones(keys, "medium")


def _mixed_category_tracks(n_tracks, n=11):
    keys = []
    for i in range(n_tracks):
        name = f"{_MIXED_CATEGORY_PREFIXES[i % len(_MIXED_CATEGORY_PREFIXES)]}{i:02d}"
        keys.extend(_dense_parabolic_track(name, n=n, k=0.01 + 0.001 * i))
    return keys


class _ReversedCompletionCountingPool:
    def __init__(self, recorder):
        self._rec = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, func, items, chunksize=1):
        items = list(items)
        self._rec["dispatched"] = len(items)
        return [func(it) for it in reversed(items)]


def test_parallel_output_matches_serial_exact():
    keys = _mixed_category_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)
    parallel = mreduce.reduce_bones(keys, "medium", workers=2)
    serial = mreduce.reduce_bones(keys, "medium", workers=1)
    assert parallel == serial


def test_parallel_diagnostics_match_serial_in_values_and_order():
    keys = _mixed_category_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)
    d_par, d_ser = {}, {}
    mreduce.reduce_bones(keys, "medium", workers=2, diagnostics_out=d_par)
    mreduce.reduce_bones(keys, "medium", workers=1, diagnostics_out=d_ser)
    assert list(d_par.items()) == list(d_ser.items())


def test_parallel_fires_at_threshold_and_restores_first_seen_order_from_reversed_completion(monkeypatch):
    rec = {}

    def fake_make_pool(workers):
        rec["workers"] = workers
        return _ReversedCompletionCountingPool(rec)

    monkeypatch.setattr(mreduce, "_make_pool", fake_make_pool)
    n_at_threshold = mreduce._MIN_PARALLEL_TRACKS
    keys = _mixed_category_tracks(n_at_threshold)
    d_par, d_ser = {}, {}
    out = mreduce.reduce_bones(keys, "medium", workers=3, diagnostics_out=d_par)
    assert rec["workers"] == 3
    assert rec["dispatched"] == n_at_threshold
    serial = mreduce.reduce_bones(keys, "medium", workers=1, diagnostics_out=d_ser)
    assert out == serial
    assert list(d_par.items()) == list(d_ser.items())


def test_below_threshold_creates_no_pool_and_matches_serial(monkeypatch):
    made = {"pool": False}

    def fake_make_pool(workers):
        made["pool"] = True
        return _ReversedCompletionCountingPool({})

    monkeypatch.setattr(mreduce, "_make_pool", fake_make_pool)
    keys = _mixed_category_tracks(mreduce._MIN_PARALLEL_TRACKS - 1)
    out = mreduce.reduce_bones(keys, "medium", workers=4)
    assert made["pool"] is False
    assert out == mreduce.reduce_bones(keys, "medium", workers=1)


def test_threshold_counts_multikey_tracks_only_not_single_key_tracks(monkeypatch):
    made = {"pool": False}

    def fake_make_pool(workers):
        made["pool"] = True
        return _ReversedCompletionCountingPool({})

    monkeypatch.setattr(mreduce, "_make_pool", fake_make_pool)
    multikey_below_threshold = _mixed_category_tracks(mreduce._MIN_PARALLEL_TRACKS - 1)
    keys = multikey_below_threshold + _single_key_tracks(mreduce._MIN_PARALLEL_TRACKS)
    out = mreduce.reduce_bones(keys, "medium", workers=4)
    assert made["pool"] is False
    assert out == mreduce.reduce_bones(keys, "medium", workers=1)


def test_parallel_interleaves_single_key_tracks_in_first_seen_order(monkeypatch):
    rec = {}
    monkeypatch.setattr(mreduce, "_make_pool", lambda workers: _ReversedCompletionCountingPool(rec))
    keys = []
    for i in range(mreduce._MIN_PARALLEL_TRACKS):
        name = f"{_MIXED_CATEGORY_PREFIXES[i % len(_MIXED_CATEGORY_PREFIXES)]}{i:02d}"
        keys.extend(_dense_parabolic_track(name, k=0.01 + 0.001 * i))
        keys.append(bone(f"単{i:02d}", 0, pos=(float(i), 0.0, 0.0)))
    d_par, d_ser = {}, {}
    out = mreduce.reduce_bones(keys, "medium", workers=3, diagnostics_out=d_par)
    serial = mreduce.reduce_bones(keys, "medium", workers=1, diagnostics_out=d_ser)
    assert out == serial
    assert list(d_par.items()) == list(d_ser.items())


def test_progress_does_not_change_output_or_diagnostics_serial_or_parallel():
    keys = _mixed_category_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)
    for workers in (1, 2):
        d_cb, d_no = {}, {}
        with_cb = mreduce.reduce_bones(
            keys, "medium", workers=workers, progress=lambda d, t: None, diagnostics_out=d_cb
        )
        without = mreduce.reduce_bones(keys, "medium", workers=workers, diagnostics_out=d_no)
        assert with_cb == without
        assert list(d_cb.items()) == list(d_no.items())


def test_serial_and_fallback_progress_counts_multikey_tracks_from_zero_to_total():
    multi_below_threshold = mreduce._MIN_PARALLEL_TRACKS - 1
    keys = _mixed_category_tracks(multi_below_threshold) + _single_key_tracks(3)
    for workers in (1, 4):
        calls = []
        mreduce.reduce_bones(keys, "medium", workers=workers, progress=lambda d, t: calls.append((d, t)))
        assert calls == [(d, multi_below_threshold) for d in range(multi_below_threshold + 1)]


def test_progress_reports_zero_of_zero_once_when_no_multikey():
    keys = _single_key_tracks(5)
    calls = []
    mreduce.reduce_bones(keys, "medium", workers=4, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(0, 0)]


def test_parallel_progress_counts_multikey_tracks_from_zero_to_total(monkeypatch):
    monkeypatch.setattr(mreduce, "_make_pool", lambda workers: _ReversedCompletionCountingPool({}))
    multi_at_threshold = mreduce._MIN_PARALLEL_TRACKS
    keys = _mixed_category_tracks(multi_at_threshold) + _single_key_tracks(3)
    calls = []
    mreduce.reduce_bones(keys, "medium", workers=3, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(d, multi_at_threshold) for d in range(multi_at_threshold + 1)]


def test_pool_workers_ignore_sigint_and_sigbreak():
    with mreduce._make_pool(1) as pool:
        assert pool.apply(signal.getsignal, (signal.SIGINT,)) == signal.SIG_IGN
        if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
            assert pool.apply(signal.getsignal, (signal.SIGBREAK,)) == signal.SIG_IGN


def test_repeated_calls_give_identical_output():
    keys = _dense_parabolic_track("右足ＩＫ")
    assert mreduce.reduce_bones(keys, "medium") == mreduce.reduce_bones(keys, "medium")
