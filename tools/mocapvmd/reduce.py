import os
import signal
import sys
from collections.abc import Callable
from multiprocessing import get_context

from vmd.reduce import build_bone_tolerances, measure_bone_errors, reduce_bone_track
from vmd.types import BoneKey

from . import classify, presets

_CUT_POS_THRESHOLD = 1.0
_CUT_ROT_THRESHOLD_DEG = 30.0
_MIN_SEG = 1
_UNLIMITED_MAX_SEG = None
_MIN_PARALLEL_TRACKS = 8

_ZERO_ERRORS = {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0, "rot_deg": 0.0}


def _reduce_track(ks, tol_pos, tol_rot, curve_mode, want_diag) -> tuple[list[BoneKey], dict | None]:
    tols = build_bone_tolerances(tol_pos, tol_rot)
    diag = {} if want_diag else None
    reduced = list(
        reduce_bone_track(
            ks,
            [(ks[0].frame, ks[-1].frame)],
            tols,
            cut_thresholds=(_CUT_POS_THRESHOLD, _CUT_ROT_THRESHOLD_DEG),
            keep_frames=[],
            no_cut_detect=False,
            min_seg=_MIN_SEG,
            max_seg=_UNLIMITED_MAX_SEG,
            strict=False,
            curve_mode=curve_mode,
            diagnostics=diag,
        )
    )
    if not want_diag:
        return reduced, None
    return reduced, {
        "cuts": len(diag["cuts"]),
        "errors": measure_bone_errors(ks, reduced, [(ks[0].frame, ks[-1].frame)]),
    }


# spawn の子プロセスへ pickle で渡すため、モジュール直下の関数でなければならない。
def _reduce_one(item):
    name, ks, tol_pos, tol_rot, curve_mode, want_diag = item
    reduced, payload = _reduce_track(ks, tol_pos, tol_rot, curve_mode, want_diag)
    return name, reduced, payload


def _ignore_interrupts_in_worker():
    # Windows の CTRL_C_EVENT は同じコンソールの全プロセスへ配送される。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
        # CTRL_BREAK_EVENT はプロセスグループの子孫へも配送され、ハンドラが無いと OS 既定で即終了する。
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)


def _make_pool(workers):
    return get_context("spawn").Pool(processes=workers, initializer=_ignore_interrupts_in_worker)


def _resolve_workers(workers):
    if workers is None:
        return os.cpu_count() or 1
    return max(1, int(workers))


def reduce_bones(
    cleaned_keys, preset, *, override_pos=None, override_rot=None, curve_mode="bezier",
    diagnostics_out: dict[str, dict] | None = None, workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
):
    order = []
    groups = {}
    for k in cleaned_keys:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)

    want_diag = diagnostics_out is not None
    sorted_tracks = {name: sorted(groups[name], key=lambda k: k.frame) for name in order}
    multikey = [name for name in order if len(sorted_tracks[name]) > 1]

    total = len(multikey)
    done = 0
    if progress is not None:
        progress(done, total)

    def _tol(name):
        return presets.resolve_reduction_tolerances(
            preset, classify.classify(name), override_pos=override_pos, override_rot=override_rot
        )

    precomputed = {}
    n_workers = _resolve_workers(workers)
    if n_workers > 1 and len(multikey) >= _MIN_PARALLEL_TRACKS:
        work = []
        for name in multikey:
            tol = _tol(name)
            work.append((name, sorted_tracks[name], tol["bone_pos"], tol["bone_rot"], curve_mode, want_diag))
        with _make_pool(n_workers) as pool:
            for name, reduced, payload in pool.imap_unordered(_reduce_one, work, chunksize=1):
                precomputed[name] = (reduced, payload)
                done += 1
                if progress is not None:
                    progress(done, total)

    out = []
    for name in order:
        ks = sorted_tracks[name]
        if len(ks) <= 1:
            out.extend(ks)
            if want_diag:
                tol = _tol(name)
                diagnostics_out[name] = {
                    "input_keys": len(ks),
                    "output_keys": len(ks),
                    "tol_pos": tol["bone_pos"],
                    "tol_rot": tol["bone_rot"],
                    "cuts": 0,
                    "errors": dict(_ZERO_ERRORS),
                }
            continue
        tol = _tol(name)
        if name in precomputed:
            reduced, payload = precomputed[name]
        else:
            reduced, payload = _reduce_track(ks, tol["bone_pos"], tol["bone_rot"], curve_mode, want_diag)
            done += 1
            if progress is not None:
                progress(done, total)
        out.extend(reduced)
        if want_diag:
            diagnostics_out[name] = {
                "input_keys": len(ks),
                "output_keys": len(reduced),
                "tol_pos": tol["bone_pos"],
                "tol_rot": tol["bone_rot"],
                "cuts": payload["cuts"],
                "errors": payload["errors"],
            }
    return out
