import math
from dataclasses import dataclass

import numpy as np

from shakevmd import cuts, motion, noise
from shakevmd.warn import ShakeWarning
from vmd import camera, interp
from vmd.types import CameraKey


class RangeOverlapError(ValueError):
    pass


def round_half_up(x) -> int:
    # 組み込みの round() は .5 を偶数側へ丸める。
    return int(math.floor(float(x) + 0.5))


def apply_gaze_shake(camera_key, rot_noise, pos_noise=(0.0, 0.0, 0.0), naive=False) -> dict:
    """rot_noise はラジアンの (rx, ry, rz)。戻り値は {"position": カメラ中心, "rotation": 角度}。"""
    new_rot = tuple(camera_key.rotation[i] + rot_noise[i] for i in range(3))
    if naive:
        center = np.array(camera_key.position, dtype=float) + np.array(pos_noise, dtype=float)
        return {"position": tuple(center), "rotation": new_rot}
    cam_pos = np.array(camera.to_world(camera_key).position, dtype=float)
    cam_pos = cam_pos + np.array(pos_noise, dtype=float)
    center_at_origin = CameraKey(
        0, camera_key.distance, (0.0, 0.0, 0.0), new_rot, bytes(24),
        camera_key.fov, camera_key.perspective,
    )
    offset = np.array(camera.to_world(center_at_origin).position, dtype=float)
    center = cam_pos - offset
    return {"position": tuple(center), "rotation": new_rot}


LINEAR_CAMERA_INTERP = bytes([20, 107, 20, 107]) * 6

FPS = 30.0

_ROT_CHANNELS = ("rot_x", "rot_y", "rot_z")
_POS_CHANNELS = ("pos_x", "pos_y", "pos_z")
_LATERAL_AXIS = 0
_VERTICAL_AXIS = 1
_PHASE_RESOLUTION = 100000


def _crossfaded_channel(seed, si, ch, t, freq, weights, octaves=noise.DEFAULT_OCTAVES):
    comps, warns = noise.octave_components(noise.derive_seed(seed, si, ch), t, freq, octaves=octaves)
    acc = np.zeros(len(t))
    for oct_i, c in enumerate(comps):
        acc = acc + weights[:, oct_i] * c
    return acc, warns


@dataclass
class BakeResult:
    """resolved は端を既存キーへスナップした適用範囲 (start, end) の昇順。"""

    camera_keys: list[CameraKey]
    warnings: list[ShakeWarning]
    resolved: list[tuple[int, int]]


def _working_view_and_dropped_count(camera_keys):
    by_frame = {}
    for k in camera_keys:
        by_frame[k.frame] = k
    wv = [by_frame[f] for f in sorted(by_frame)]
    return wv, len(camera_keys) - len(wv)


def _snap(frame, wv_frames):
    return min(wv_frames, key=lambda f: (abs(f - frame), f))


def _resolve_ranges(ranges, wv_frames):
    if ranges is None:
        resolved = [(wv_frames[0], wv_frames[-1])]
    else:
        resolved = []
        for s, e in ranges:
            a, b = _snap(s, wv_frames), _snap(e, wv_frames)
            if a > b:
                a, b = b, a
            resolved.append((a, b))
    resolved.sort()
    for i in range(1, len(resolved)):
        if resolved[i][0] <= resolved[i - 1][1]:
            raise RangeOverlapError(
                f"範囲が重複/接触している: {resolved[i - 1]} と {resolved[i]}"
            )
    return resolved


def _gradual_fov_change_spans(wv):
    frames: set = set()
    boundary_frames: set = set()
    for i in range(len(wv) - 1):
        if wv[i].fov != wv[i + 1].fov and wv[i + 1].frame - wv[i].frame > 1:
            frames.update(range(wv[i].frame, wv[i + 1].frame + 1))
            boundary_frames.add(wv[i].frame)
            boundary_frames.add(wv[i + 1].frame)
    return frames, boundary_frames


def _impulse_direction(seed: int, frame: int) -> np.ndarray:
    comps = [
        (noise.derive_seed(seed, "impulse_dir", frame, axis) % 2000) / 1000.0 - 1.0
        for axis in ("x", "y", "z")
    ]
    v = np.array(comps, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 1.0, 0.0])
    return v / n


def _governing_perspective(wv, frame):
    persp = wv[0].perspective
    for k in wv:
        if k.frame <= frame:
            persp = k.perspective
        else:
            break
    return persp


def _segment_speeds(samples, speed_ref_world, speed_ref_angle):
    world = np.array([
        camera.to_world(CameraKey(
            0, s["distance"], s["position"], s["rotation"],
            LINEAR_CAMERA_INTERP, 0, 0,
        )).position
        for s in samples
    ], dtype=float)
    angles = np.array([s["rotation"] for s in samples], dtype=float)
    angle_speed = motion.frame_speeds(angles, speed_ref_angle)
    speeds = np.maximum(motion.frame_speeds(world, speed_ref_world), angle_speed)
    return angles, angle_speed, speeds


def _breathing_offsets(seed, si, t, speeds, amp_pos):
    breath = np.zeros((len(t), 3))
    breath_amp = amp_pos * motion.BREATHING_AMP_FACTOR
    if breath_amp > 0.0:
        for axis in range(3):
            ph_seed = noise.derive_seed(seed, si, "breath", axis)
            phase_sec = (ph_seed % _PHASE_RESOLUTION) / float(_PHASE_RESOLUTION) / motion.BREATHING_HZ
            breath[:, axis] = (1.0 - speeds) * motion.breathing_drift(
                t + phase_sec, breath_amp
            )
    return breath


def _gait_offsets(seed, si, t, gait_freq, gait_amp):
    gait = np.zeros((len(t), 3))
    if gait_freq > 0.0 and gait_amp != 0.0:
        for axis, freq_multiplier in ((_LATERAL_AXIS, 1.0), (_VERTICAL_AXIS, 2.0)):
            ph_seed = noise.derive_seed(seed, si, "gait", axis)
            phase = (ph_seed % _PHASE_RESOLUTION) / float(_PHASE_RESOLUTION) * 2.0 * np.pi
            gait[:, axis] = gait_amp * np.sin(
                2.0 * np.pi * (gait_freq * freq_multiplier) * t + phase
            )
    return gait


def _settle_rotation(angles, angle_speed, settle, settle_time):
    settle_rot = np.zeros((len(angles), 3))
    if settle > 0.0:
        for stop_idx in motion.detect_stops(angle_speed):
            if stop_idx < 2:
                continue
            velocity_before_stop = angles[stop_idx - 1] - angles[stop_idx - 2]
            nrm = float(np.linalg.norm(velocity_before_stop))
            if nrm < 1e-9:
                continue
            direction = velocity_before_stop / nrm
            for j in range(stop_idx, len(angles)):
                val = motion.settle_oscillation(
                    (j - stop_idx) / FPS, math.radians(settle), settle_time=settle_time
                )
                settle_rot[j] += val * direction
    return settle_rot


def _impulse_rotation(seed, sframes, t, impulses, warnings):
    impulse_rot = np.zeros((len(sframes), 3))
    for impulse_frame, strength_deg, decay_sec in impulses:
        if strength_deg <= 0.0 or decay_sec <= 0.0:
            continue
        direction = _impulse_direction(seed, impulse_frame)
        osc, owarns = noise.band_limited_noise(
            noise.derive_seed(seed, "impulse_osc", impulse_frame), t, noise.BANDLIMIT_HZ,
            octaves=1,
        )
        warnings.extend(ShakeWarning("octave_clamped", w) for w in owarns)
        for j, f in enumerate(sframes):
            if f < impulse_frame:
                continue
            env = math.radians(strength_deg) * math.exp(-((f - impulse_frame) / FPS) / decay_sec)
            impulse_rot[j] += direction * osc[j] * env
    return impulse_rot


def _unique_in_order(warnings):
    seen, uniq = set(), []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def bake(
    camera_keys,
    ranges=None,
    *,
    seed: int = 1,
    amp_rot: float = 0.8,
    amp_pos: float = 0.05,
    rot_weights=(1.0, 1.0, 0.3),
    freq: float = 1.2,
    motion_damp: float = 1.0,
    settle: float = 0.0,
    settle_time: float = motion.DEFAULT_SETTLE_TIME_SEC,
    still_profile=motion.STILL_PROFILE,
    moving_profile=motion.MOVING_PROFILE,
    speed_ref_world: float = motion.DEFAULT_SPEED_REF_WORLD,
    speed_ref_angle: float = motion.DEFAULT_SPEED_REF_ANGLE,
    naive_rotation: bool = False,
    gait_freq: float = 0.0,
    gait_amp: float = 0.0,
    fade_sec: float = 0.7,
    cut_pos_threshold: float = 5.0,
    cut_rot_threshold: float = 20.0,
    manual_cuts_add=(),
    manual_cuts_remove=(),
    impulses=(),
    progress=None,
) -> BakeResult:
    """ranges の各要素は (start, end) のフレーム番号で、None は全範囲。impulses の各要素は
    (フレーム, 強さ, 減衰)。rot_weights は (rx, ry, rz) の順。progress は (処理済みフレーム数, 総数) で呼ぶ。

    単位: amp_rot・settle・cut_rot_threshold・impulses の強さは度、amp_pos・gait_amp・cut_pos_threshold は
    MMD の距離単位、freq・gait_freq は Hz、fade_sec・settle_time・impulses の減衰は秒、speed_ref_world は
    1フレームあたりの距離、speed_ref_angle は1フレームあたりのラジアン。
    """
    if not camera_keys:
        raise ValueError("カメラキーが空")
    if len(still_profile) != len(moving_profile):
        raise ValueError(
            f"still_profile と moving_profile の長さが不一致(オクターブ数の整合): "
            f"{len(still_profile)} != {len(moving_profile)}"
        )
    octaves = len(still_profile)

    warnings: list = []
    wv, dropped = _working_view_and_dropped_count(camera_keys)
    if dropped:
        warnings.append(ShakeWarning(
            "bake_normalize_duplicate",
            f"正規化: 同一フレーム重複 {dropped} 件を後勝ちで破棄した",
            ("camera",),
        ))
    resolved = _resolve_ranges(ranges, [k.frame for k in wv])

    detected = cuts.detect_cuts(wv, cut_pos_threshold, cut_rot_threshold)
    cut_frames = cuts.resolve_cuts(
        detected, list(manual_cuts_add), list(manual_cuts_remove)
    )

    gradual_fov_frames, gradual_fov_boundaries = _gradual_fov_change_spans(wv)

    total_baked = sum(
        1 for a, b in resolved for f in range(a, b + 1) if f not in gradual_fov_frames
    )
    done_baked = 0

    fade_frames = int(round(fade_sec * FPS))
    baked: list = []

    for a, b in resolved:
        n = b - a + 1
        fade = motion.fade_envelope(n, fade_sec, FPS)
        if n < 2 * fade_frames:
            warnings.append(ShakeWarning(
                "fade_shortened",
                f"範囲[{a},{b}](長さ{n}f)が 2×fade({2 * fade_frames}f)未満。"
                f"フェードを自動短縮した",
            ))
        for si, seg in enumerate(cuts.segment_bounds(a, b, cut_frames)):
            sframes = list(range(seg.start, seg.end + 1))
            if all(f in gradual_fov_frames for f in sframes):
                continue
            t = np.array([f / FPS for f in sframes], dtype=float)

            samples = [interp.sample_camera(wv, f) for f in sframes]
            angles, angle_speed, speeds = _segment_speeds(samples, speed_ref_world, speed_ref_angle)

            weights = motion.profile_weights(speeds, still_profile, moving_profile)
            rot_n = []
            for ch in _ROT_CHANNELS:
                vals, warns = _crossfaded_channel(seed, si, ch, t, freq, weights, octaves)
                rot_n.append(vals)
                warnings.extend(ShakeWarning("octave_clamped", w) for w in warns)
            pos_n = []
            for ch in _POS_CHANNELS:
                vals, warns = _crossfaded_channel(seed, si, ch, t, freq, weights, octaves)
                pos_n.append(vals)
                warnings.extend(ShakeWarning("octave_clamped", w) for w in warns)

            breath = _breathing_offsets(seed, si, t, speeds, amp_pos)
            gait = _gait_offsets(seed, si, t, gait_freq, gait_amp)
            settle_rot = _settle_rotation(angles, angle_speed, settle, settle_time)
            impulse_rot = _impulse_rotation(seed, sframes, t, impulses, warnings)

            for idx, f in enumerate(sframes):
                if f in gradual_fov_frames:
                    continue
                s = samples[idx]
                fade_v = fade[f - a]
                amp_factor = max(0.0, 1.0 - motion_damp * speeds[idx]) * fade_v
                rot_noise = tuple(
                    rot_n[i][idx] * math.radians(amp_rot * rot_weights[i]) * amp_factor
                    + settle_rot[idx][i] * fade_v
                    + impulse_rot[idx][i] * fade_v
                    for i in range(3)
                )
                pos_noise = tuple(
                    pos_n[i][idx] * amp_pos * amp_factor
                    + breath[idx][i] * fade_v
                    + gait[idx][i] * fade_v
                    for i in range(3)
                )
                persp = _governing_perspective(wv, f)
                fov = round_half_up(s["fov"])
                sampled = CameraKey(
                    f, s["distance"], s["position"], s["rotation"],
                    LINEAR_CAMERA_INTERP, fov, persp,
                )
                shaken = apply_gaze_shake(sampled, rot_noise, pos_noise, naive=naive_rotation)
                baked.append(CameraKey(
                    f, s["distance"], shaken["position"], shaken["rotation"],
                    LINEAR_CAMERA_INTERP, fov, persp,
                ))
                if progress is not None:
                    done_baked += 1
                    progress(done_baked, total_baked)

    def _in_range(fr):
        return any(a <= fr <= b for a, b in resolved)

    out_of_range_originals = [k for k in camera_keys if not _in_range(k.frame)]
    preserved_fov_keys = [k for k in wv if k.frame in gradual_fov_boundaries and _in_range(k.frame)]

    result_keys = sorted(baked + preserved_fov_keys + out_of_range_originals, key=lambda k: k.frame)
    return BakeResult(camera_keys=result_keys, warnings=_unique_in_order(warnings), resolved=resolved)
