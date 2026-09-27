import dataclasses
import math
import os

from vmd import interp
from vmd.cuts import (
    assemble_boundaries,
    detect_cuts_bone,
    detect_cuts_camera,
    perspective_cut_frames,
)
from vmd.fit import (
    BoneRotationChannel,
    CameraRotationChannel,
    EuclideanVectorChannel,
    FovChannel,
    LinearScalarChannel,
    _quat_angle_deg,
    _round_half_up,
    read_fit_counters,
    read_fit_counters_by_category,
    reset_fit_counters,
)
from vmd.sample import perspective_series
from vmd.types import BoneKey, CameraKey


@dataclasses.dataclass
class Tolerances:
    """bone_pos・camera_pos・camera_distance の単位は MMD の距離単位、bone_rot・camera_rot・camera_fov の単位は度。"""

    bone_pos: float
    bone_rot: float
    camera_pos: float
    camera_rot: float
    camera_distance: float
    camera_fov: float


def build_bone_tolerances(bone_pos, bone_rot):
    """値の検証は行わない。"""
    return Tolerances(
        bone_pos=bone_pos,
        bone_rot=bone_rot,
        camera_pos=0.0,
        camera_rot=0.0,
        camera_distance=0.0,
        camera_fov=0.0,
    )


def camera_interp_bytes(cp_pos_x, cp_pos_y, cp_pos_z, cp_rot, cp_distance, cp_fov):
    out = bytearray()
    for x1, y1, x2, y2 in (cp_pos_x, cp_pos_y, cp_pos_z, cp_rot, cp_distance, cp_fov):
        out += bytes([x1, x2, y1, y2])
    return bytes(out)


def bone_interp_bytes(pos_x_cp, pos_y_cp, pos_z_cp, rot_cp):
    first = [cp[i] for i in range(4) for cp in (pos_x_cp, pos_y_cp, pos_z_cp, rot_cp)]
    b = bytearray(64)
    b[0:16] = bytes(first)
    b[16:31] = bytes(first[1:16])
    b[32:46] = bytes(first[2:16])
    b[48:61] = bytes(first[3:16])
    # 詰めパッドの値は MMD の版によって異なる。
    return bytes(b)


_LINEAR_CP = (20, 20, 107, 107)
CAMERA_LINEAR_INTERP = camera_interp_bytes(*([_LINEAR_CP] * 6))
BONE_LINEAR_INTERP = bone_interp_bytes(_LINEAR_CP, _LINEAR_CP, _LINEAR_CP, _LINEAR_CP)


def build_camera_keys(source_keys, frames, segment_interp=None):
    """segment_interp(前キーのフレーム, 当キーのフレーム) が返す補間ブロックを2番目以降のキーに格納する。

    segment_interp が無いか None を返したキーは線形の補間ブロックになる。
    """
    ordered = sorted(frames)
    keys = []
    for i, f in enumerate(ordered):
        interp_block = CAMERA_LINEAR_INTERP
        if segment_interp is not None and i > 0:
            block = segment_interp(ordered[i - 1], f)
            if block is not None:
                interp_block = block
        keys.append(
            CameraKey(
                frame=f,
                distance=interp.sample(source_keys, "distance", f),
                position=(
                    interp.sample(source_keys, "pos_x", f),
                    interp.sample(source_keys, "pos_y", f),
                    interp.sample(source_keys, "pos_z", f),
                ),
                rotation=interp.sample(source_keys, "rot", f),
                interpolation=interp_block,
                fov=_round_half_up(interp.sample(source_keys, "fov", f)),
                perspective=perspective_series(source_keys, f, f)[0],
            )
        )
    return keys


def build_bone_keys(source_keys, frames, segment_interp=None):
    """segment_interp の扱いは build_camera_keys と同じ。"""
    name_raw = source_keys[0].name_raw
    ordered = sorted(frames)
    keys = []
    for i, f in enumerate(ordered):
        interp_block = BONE_LINEAR_INTERP
        if segment_interp is not None and i > 0:
            block = segment_interp(ordered[i - 1], f)
            if block is not None:
                interp_block = block
        keys.append(
            BoneKey(
                name_raw=name_raw,
                frame=f,
                position=(
                    interp.sample(source_keys, "pos_x", f),
                    interp.sample(source_keys, "pos_y", f),
                    interp.sample(source_keys, "pos_z", f),
                ),
                rotation=interp.sample(source_keys, "rot", f),
                interpolation=interp_block,
            )
        )
    return keys


class StrictError(Exception):
    pass


def _angle_diff_deg(a, b):
    d = a - b
    return abs(math.degrees(math.atan2(math.sin(d), math.cos(d))))


def verify_camera_track(source_keys, output_keys, ranges, tols):
    """許容を超えたフレームを昇順のリストで返す。"""
    bad = set()
    for f0, f1 in ranges:
        for f in range(f0, f1 + 1):
            s = interp.sample_camera(source_keys, f)
            o = interp.sample_camera(output_keys, f)
            if math.dist(s["position"], o["position"]) > tols.camera_pos:
                bad.add(f)
            elif abs(s["distance"] - o["distance"]) > tols.camera_distance:
                bad.add(f)
            elif abs(o["fov"] - s["fov"]) > tols.camera_fov:
                bad.add(f)
            elif (
                max(_angle_diff_deg(s["rotation"][i], o["rotation"][i]) for i in range(3))
                > tols.camera_rot
            ):
                bad.add(f)
    return sorted(bad)


def verify_bone_track(source_keys, output_keys, ranges, tols):
    """許容を超えたフレームを昇順のリストで返す。"""
    bad = set()
    for f0, f1 in ranges:
        for f in range(f0, f1 + 1):
            sp = (
                interp.sample(source_keys, "pos_x", f),
                interp.sample(source_keys, "pos_y", f),
                interp.sample(source_keys, "pos_z", f),
            )
            op = (
                interp.sample(output_keys, "pos_x", f),
                interp.sample(output_keys, "pos_y", f),
                interp.sample(output_keys, "pos_z", f),
            )
            if math.dist(sp, op) > tols.bone_pos:
                bad.add(f)
            elif (
                _quat_angle_deg(
                    interp.sample(source_keys, "rot", f), interp.sample(output_keys, "rot", f)
                )
                > tols.bone_rot
            ):
                bad.add(f)
    return sorted(bad)


def _verify_record(f0, f1, bad_counts, added_counts):
    return {
        "range": [f0, f1],
        "iterations": len(bad_counts),
        "bad_counts": bad_counts,
        "added_counts": added_counts,
        "added_total": sum(added_counts),
    }


def _interval_clear_of_ranges(ranges, lo, hi):
    return not any(g0 < hi and g1 > lo for g0, g1 in ranges)


def measure_camera_errors(source_keys, output_keys, ranges):
    m = {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0, "distance": 0.0, "fov": 0.0, "rot_deg": 0.0}
    axes = ("pos_x", "pos_y", "pos_z")
    for f0, f1 in ranges:
        for f in range(f0, f1 + 1):
            s = interp.sample_camera(source_keys, f)
            o = interp.sample_camera(output_keys, f)
            for i, ax in enumerate(axes):
                m[ax] = max(m[ax], abs(s["position"][i] - o["position"][i]))
            m["distance"] = max(m["distance"], abs(s["distance"] - o["distance"]))
            m["fov"] = max(m["fov"], abs(o["fov"] - s["fov"]))
            m["rot_deg"] = max(
                m["rot_deg"],
                max(_angle_diff_deg(s["rotation"][i], o["rotation"][i]) for i in range(3)),
            )
    return m


def measure_bone_errors(source_keys, output_keys, ranges):
    m = {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0, "rot_deg": 0.0}
    axes = ("pos_x", "pos_y", "pos_z")
    for f0, f1 in ranges:
        for f in range(f0, f1 + 1):
            for ax in axes:
                m[ax] = max(
                    m[ax], abs(interp.sample(source_keys, ax, f) - interp.sample(output_keys, ax, f))
                )
            m["rot_deg"] = max(
                m["rot_deg"],
                _quat_angle_deg(
                    interp.sample(source_keys, "rot", f), interp.sample(output_keys, "rot", f)
                ),
            )
    return m


def _nearest_source_before(source_keys, frame):
    cands = [k.frame for k in source_keys if k.frame < frame]
    return max(cands) if cands else None


def _camera_seam_interp(source_keys, a, b, tols, force_bezier=False):
    rng = range(a, b + 1)
    positions = [
        (
            interp.sample(source_keys, "pos_x", f),
            interp.sample(source_keys, "pos_y", f),
            interp.sample(source_keys, "pos_z", f),
        )
        for f in rng
    ]
    distances = [interp.sample(source_keys, "distance", f) for f in rng]
    fovs = [interp.sample(source_keys, "fov", f) for f in rng]
    eulers = [interp.sample(source_keys, "rot", f) for f in rng]
    pos_ch = EuclideanVectorChannel(a, positions, tols.camera_pos, mode="bezier", force_bezier=force_bezier)
    dist_ch = LinearScalarChannel(a, distances, tols.camera_distance, mode="bezier", force_bezier=force_bezier)
    fov_ch = FovChannel(a, fovs, tols.camera_fov, mode="bezier", force_bezier=force_bezier)
    rot_ch = CameraRotationChannel(a, eulers, tols.camera_rot, mode="bezier", force_bezier=force_bezier)
    pos_ch.label, dist_ch.label, fov_ch.label, rot_ch.label = (
        "position", "distance", "fov", "rotation",
    )
    cp_x, cp_y, cp_z = pos_ch.curve(a, b)
    return camera_interp_bytes(
        cp_x, cp_y, cp_z, rot_ch.curve(a, b), dist_ch.curve(a, b), fov_ch.curve(a, b)
    )


_C1_SMOOTHING = os.environ.get("MOCAP_C1_SMOOTHING", "1") != "0"


def _bone_channel_cp(interp_bytes, c):
    b = interp_bytes
    return [b[c], b[4 + c], b[8 + c], b[12 + c]]


def _arrival_y2_for_velocity(velocity, dt, dv, x2):
    return 1.0 - (velocity * dt / dv) * (1.0 - x2)


def _departure_y1_for_velocity(velocity, dt, dv, x1):
    return (velocity * dt / dv) * x1


def _apply_c1_bone(keys, source_keys):
    if len(keys) < 3:
        return keys
    axes = ("pos_x", "pos_y", "pos_z")
    out = list(keys)
    cps = [[_bone_channel_cp(k.interpolation, c) for c in range(4)] for k in out]
    for k in range(1, len(out) - 1):
        fk = out[k].frame
        for a in range(3):
            source_velocity = (
                interp.sample(source_keys, axes[a], fk + 1)
                - interp.sample(source_keys, axes[a], fk - 1)
            ) / 2.0
            dv_l = out[k].position[a] - out[k - 1].position[a]
            dt_l = out[k].frame - out[k - 1].frame
            if abs(dv_l) > 1e-9 and dt_l > 0:
                x2 = cps[k][a][2] / 127.0
                y2 = _arrival_y2_for_velocity(source_velocity, dt_l, dv_l, x2)
                cps[k][a][3] = min(127, max(0, _round_half_up(y2 * 127.0)))
            dv_r = out[k + 1].position[a] - out[k].position[a]
            dt_r = out[k + 1].frame - out[k].frame
            if abs(dv_r) > 1e-9 and dt_r > 0:
                x1 = cps[k + 1][a][0] / 127.0
                y1 = _departure_y1_for_velocity(source_velocity, dt_r, dv_r, x1)
                cps[k + 1][a][1] = min(127, max(0, _round_half_up(y1 * 127.0)))
    for k in range(1, len(out)):
        c = cps[k]
        out[k] = dataclasses.replace(
            out[k],
            interpolation=bone_interp_bytes(tuple(c[0]), tuple(c[1]), tuple(c[2]), tuple(c[3])),
        )
    return out


def _bone_seam_interp(source_keys, a, b, tols):
    rng = range(a, b + 1)
    positions = [
        (
            interp.sample(source_keys, "pos_x", f),
            interp.sample(source_keys, "pos_y", f),
            interp.sample(source_keys, "pos_z", f),
        )
        for f in rng
    ]
    quats = [interp.sample(source_keys, "rot", f) for f in rng]
    pos_ch = EuclideanVectorChannel(a, positions, tols.bone_pos, mode="bezier")
    rot_ch = BoneRotationChannel(a, quats, tols.bone_rot, mode="bezier")
    pos_ch.label, rot_ch.label = "position", "rotation"
    cp_x, cp_y, cp_z = pos_ch.curve(a, b)
    return bone_interp_bytes(cp_x, cp_y, cp_z, rot_ch.curve(a, b))


def reduce_track(boundaries, channels, min_seg, max_seg, strict, splits=None, caps=None, progress=None):
    """channels の各要素は normalized(a, b) -> (正規化誤差, 分割候補フレーム | None) を持つこと。

    is_constant(a, b) は持たなくてよく、持たないチャンネルは常に変化があるものとして扱う。
    splits にリストを渡すと誤差で分割したフレームを {"frame", "channel", "norm_error"} で、caps に
    リストを渡すと max_seg で分割したフレームを {"frame"} で追記する。progress(処理済みフレーム数,
    総フレーム数) を渡すと処理の経過を通知する。
    """
    bounds = sorted(set(boundaries))
    keys = set(bounds)
    span0, span1 = bounds[0], bounds[-1]
    on_resolve = None
    if progress is not None:
        total = span1 - span0
        resolved = 0

        def on_resolve(span):
            nonlocal resolved
            resolved += span
            progress(resolved, total)

    for a, b in zip(bounds, bounds[1:], strict=False):
        _process_segment(a, b, channels, min_seg, max_seg, strict, keys, splits, caps, on_resolve)
    return sorted(keys)


def _span_is_constant(a, b, channels):
    for ch in channels:
        is_const = getattr(ch, "is_constant", None)
        if is_const is None or not is_const(a, b):
            return False
    return True


def _span_error_and_split(a, b, channels):
    max_norm = 0.0
    split_norm = 0.0
    split_frame = None
    split_label = None
    for ch in channels:
        ne, frame = ch.normalized(a, b)
        if ne > max_norm:
            max_norm = ne
        if frame is not None and ne > split_norm:
            split_norm = ne
            split_frame = frame
            split_label = getattr(ch, "label", None)
    return max_norm, split_frame, split_label


def _process_segment(a, b, channels, min_seg, max_seg, strict, keys, splits=None, caps=None, on_resolve=None):
    stack = [(a, b)]
    while stack:
        a, b = stack.pop()
        span = b - a
        if span <= 1:
            if on_resolve is not None:
                on_resolve(span)
            continue

        max_norm, split_frame, split_label = _span_error_and_split(a, b, channels)
        fits = max_norm <= 1.0

        if fits and (max_seg is None or span <= max_seg or _span_is_constant(a, b, channels)):
            if on_resolve is not None:
                on_resolve(span)
            continue

        if span < 2 * min_seg:
            if not fits:
                _atomic_fail(a, b, strict, keys)
            if on_resolve is not None:
                on_resolve(span)
            continue

        if fits:
            w = min(max(a + max_seg, a + min_seg), b - min_seg)
            keys.add(w)
            if caps is not None:
                caps.append({"frame": w})
        else:
            if split_frame is None:
                _atomic_fail(a, b, strict, keys)
                if on_resolve is not None:
                    on_resolve(span)
                continue
            w = min(max(split_frame, a + min_seg), b - min_seg)
            keys.add(w)
            if splits is not None:
                splits.append({"frame": w, "channel": split_label, "norm_error": max_norm})
        stack.append((a, w))
        stack.append((w, b))


def _atomic_fail(a, b, strict, keys):
    if strict:
        raise StrictError(f"許容誤差を満たせない区間: [{a}, {b}]")
    for f in range(a + 1, b):
        keys.add(f)


def _in_any_range(frame, ranges):
    return any(f0 <= frame <= f1 for f0, f1 in ranges)


def _boundary_with_predecessor(frames, f0):
    out = set(frames)
    out |= {f - 1 for f in frames if f - 1 >= f0}
    return out


def _sampled_positions(source_keys, f0, f1):
    return [
        (
            interp.sample(source_keys, "pos_x", f),
            interp.sample(source_keys, "pos_y", f),
            interp.sample(source_keys, "pos_z", f),
        )
        for f in range(f0, f1 + 1)
    ]


def reduce_camera_track(
    source_keys,
    ranges,
    tols,
    *,
    cut_thresholds,
    keep_frames,
    no_cut_detect,
    min_seg,
    max_seg,
    strict,
    curve_mode="linear",
    force_bezier=False,
    diagnostics=None,
    progress=None,
):
    """source_keys はフレーム昇順であること。ranges の外のキーも含めたキー列をフレーム順で返す。

    cut_thresholds は (位置, 回転[度], 距離)。curve_mode は "linear" か "bezier"。diagnostics に辞書を
    渡すと cuts・splits・maxspan_caps・seam_rewrites・verify・fit_counts・fit_counts_by_channel を書き込む。
    progress(処理済みフレーム数, 総フレーム数, 段階名) を渡すと処理の経過を通知する。段階名は出力後検証の
    間だけ "出力後検証" で、それ以外は空文字列。
    """
    reduced = []
    diag_cuts = set()
    diag_splits = [] if diagnostics is not None else None
    diag_caps = [] if diagnostics is not None else None
    diag_seams = set()
    diag_verify = [] if diagnostics is not None else None
    if diagnostics is not None:
        reset_fit_counters()
    total_frames = sum(f1 - f0 for f0, f1 in ranges)
    progress_base = 0
    for f0, f1 in ranges:
        positions = _sampled_positions(source_keys, f0, f1)
        distances = [interp.sample(source_keys, "distance", f) for f in range(f0, f1 + 1)]
        fovs = [interp.sample(source_keys, "fov", f) for f in range(f0, f1 + 1)]
        eulers = [interp.sample(source_keys, "rot", f) for f in range(f0, f1 + 1)]
        persp = perspective_series(source_keys, f0, f1)

        cuts = detect_cuts_camera(f0, positions, eulers, distances, cut_thresholds)
        pcuts = perspective_cut_frames(f0, persp)
        if not no_cut_detect:
            diag_cuts.update(cuts)
        diag_cuts.update(pcuts)
        bounds = assemble_boundaries(
            f0,
            f1,
            cuts=_boundary_with_predecessor(cuts, f0),
            perspective_frames=_boundary_with_predecessor(pcuts, f0),
            keep_frames=keep_frames,
            no_cut_detect=no_cut_detect,
        )
        pos_ch = EuclideanVectorChannel(f0, positions, tols.camera_pos, mode=curve_mode, force_bezier=force_bezier)
        dist_ch = LinearScalarChannel(f0, distances, tols.camera_distance, mode=curve_mode, force_bezier=force_bezier)
        fov_ch = FovChannel(f0, fovs, tols.camera_fov, mode=curve_mode, force_bezier=force_bezier)
        rot_ch = CameraRotationChannel(f0, eulers, tols.camera_rot, mode=curve_mode, force_bezier=force_bezier)
        pos_ch.label, dist_ch.label, fov_ch.label, rot_ch.label = (
            "position", "distance", "fov", "rotation",
        )
        channels = [pos_ch, dist_ch, fov_ch, rot_ch]
        range_progress = None
        if progress is not None:
            def range_progress(done, _span, note="", base=progress_base):
                progress(base + done, total_frames, note)
        range_frames = set(
            reduce_track(
                bounds, channels, min_seg, max_seg, strict,
                splits=diag_splits, caps=diag_caps, progress=range_progress,
            )
        )
        progress_base += f1 - f0

        segment_interp = None
        if curve_mode == "bezier":
            def segment_interp(a, b, pos_ch=pos_ch, dist_ch=dist_ch, fov_ch=fov_ch, rot_ch=rot_ch):
                cp_x, cp_y, cp_z = pos_ch.curve(a, b)
                return camera_interp_bytes(
                    cp_x, cp_y, cp_z, rot_ch.curve(a, b), dist_ch.curve(a, b), fov_ch.curve(a, b)
                )

        if progress is not None:
            progress(progress_base, total_frames, "出力後検証")
        record = diag_verify is not None
        bad_counts = [] if record else None
        added_counts = [] if record else None
        while True:
            keys = build_camera_keys(source_keys, sorted(range_frames), segment_interp)
            bad = verify_camera_track(source_keys, keys, [(f0, f1)], tols)
            if record:
                bad_counts.append(len(bad))
            if not bad:
                if record:
                    added_counts.append(0)
                break
            if strict:
                raise StrictError(f"出力後検証で許容を満たせない: 範囲[{f0},{f1}] フレーム{bad[:8]}")
            added = set(bad) - range_frames
            range_frames |= added
            if record:
                added_counts.append(len(added))
            if not added:
                break
        if record:
            diag_verify.append(_verify_record(f0, f1, bad_counts, added_counts))

        if curve_mode == "bezier":
            prev_src = _nearest_source_before(source_keys, f0)
            if prev_src is not None and _interval_clear_of_ranges(ranges, prev_src, f0):
                keys[0] = dataclasses.replace(
                    keys[0],
                    interpolation=_camera_seam_interp(
                        source_keys, prev_src, f0, tols, force_bezier=force_bezier
                    ),
                )
                diag_seams.add(keys[0].frame)
        reduced.extend(keys)

    outside = [k for k in source_keys if not _in_any_range(k.frame, ranges)]
    if diagnostics is not None:
        diagnostics["cuts"] = sorted(diag_cuts)
        diagnostics["splits"] = diag_splits
        diagnostics["maxspan_caps"] = diag_caps
        diagnostics["seam_rewrites"] = sorted(diag_seams)
        diagnostics["verify"] = diag_verify
        diagnostics["fit_counts"] = read_fit_counters()
        diagnostics["fit_counts_by_channel"] = read_fit_counters_by_category()
    return sorted(reduced + outside, key=lambda k: k.frame)


def reduce_bone_track(
    source_keys,
    ranges,
    tols,
    *,
    cut_thresholds,
    keep_frames,
    no_cut_detect,
    min_seg,
    max_seg,
    strict,
    curve_mode="linear",
    diagnostics=None,
):
    """source_keys はフレーム昇順であること。ranges の外のキーも含めたキー列をフレーム順で返す。

    cut_thresholds は (位置, 回転[度])。curve_mode と diagnostics の扱いは reduce_camera_track と同じ。
    """
    reduced = []
    diag_cuts = set()
    diag_splits = [] if diagnostics is not None else None
    diag_caps = [] if diagnostics is not None else None
    diag_seams = set()
    diag_verify = [] if diagnostics is not None else None
    if diagnostics is not None:
        reset_fit_counters()
    for f0, f1 in ranges:
        positions = _sampled_positions(source_keys, f0, f1)
        quats = [interp.sample(source_keys, "rot", f) for f in range(f0, f1 + 1)]

        cuts = detect_cuts_bone(f0, positions, quats, cut_thresholds)
        if not no_cut_detect:
            diag_cuts.update(cuts)
        bounds = assemble_boundaries(
            f0,
            f1,
            cuts=_boundary_with_predecessor(cuts, f0),
            perspective_frames=set(),
            keep_frames=keep_frames,
            no_cut_detect=no_cut_detect,
        )
        pos_ch = EuclideanVectorChannel(f0, positions, tols.bone_pos, mode=curve_mode)
        rot_ch = BoneRotationChannel(f0, quats, tols.bone_rot, mode=curve_mode)
        pos_ch.label, rot_ch.label = "position", "rotation"
        channels = [pos_ch, rot_ch]
        range_frames = set(
            reduce_track(bounds, channels, min_seg, max_seg, strict, splits=diag_splits, caps=diag_caps)
        )

        segment_interp = None
        if curve_mode == "bezier":
            def segment_interp(a, b, pos_ch=pos_ch, rot_ch=rot_ch):
                cp_x, cp_y, cp_z = pos_ch.curve(a, b)
                return bone_interp_bytes(cp_x, cp_y, cp_z, rot_ch.curve(a, b))

        record = diag_verify is not None
        bad_counts = [] if record else None
        added_counts = [] if record else None
        while True:
            keys = build_bone_keys(source_keys, sorted(range_frames), segment_interp)
            if curve_mode == "bezier" and _C1_SMOOTHING:
                c1_keys = _apply_c1_bone(keys, source_keys)
                if not strict or not verify_bone_track(source_keys, c1_keys, [(f0, f1)], tols):
                    keys = c1_keys
            bad = verify_bone_track(source_keys, keys, [(f0, f1)], tols)
            if record:
                bad_counts.append(len(bad))
            if not bad:
                if record:
                    added_counts.append(0)
                break
            if strict:
                raise StrictError(f"出力後検証で許容を満たせない: 範囲[{f0},{f1}] フレーム{bad[:8]}")
            added = set(bad) - range_frames
            range_frames |= added
            if record:
                added_counts.append(len(added))
            if not added:
                break
        if record:
            diag_verify.append(_verify_record(f0, f1, bad_counts, added_counts))

        if curve_mode == "bezier":
            prev_src = _nearest_source_before(source_keys, f0)
            if prev_src is not None and _interval_clear_of_ranges(ranges, prev_src, f0):
                keys[0] = dataclasses.replace(
                    keys[0], interpolation=_bone_seam_interp(source_keys, prev_src, f0, tols)
                )
                diag_seams.add(keys[0].frame)
        reduced.extend(keys)

    outside = [k for k in source_keys if not _in_any_range(k.frame, ranges)]
    if diagnostics is not None:
        diagnostics["cuts"] = sorted(diag_cuts)
        diagnostics["splits"] = diag_splits
        diagnostics["maxspan_caps"] = diag_caps
        diagnostics["seam_rewrites"] = sorted(diag_seams)
        diagnostics["verify"] = diag_verify
        diagnostics["fit_counts"] = read_fit_counters()
        diagnostics["fit_counts_by_channel"] = read_fit_counters_by_category()
    return sorted(reduced + outside, key=lambda k: k.frame)
