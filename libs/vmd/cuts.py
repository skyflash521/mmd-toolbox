import math


def _axis_angle_deg(r1, r0):
    d = math.degrees(r1 - r0)
    d = (d + 180.0) % 360.0 - 180.0
    return abs(d)


def _quat_angle_deg(q1, q0):
    dot = abs(sum(a * b for a, b in zip(q1, q0, strict=True)))
    dot = min(1.0, dot)
    return math.degrees(2.0 * math.acos(dot))


def detect_cuts_camera(start_frame, positions, rotations, distances, thresholds):
    """rotations はラジアン。thresholds は (位置, 回転[度], 距離)。"""
    pos_t, rot_t, dist_t = thresholds
    cuts = set()
    for i in range(1, len(positions)):
        pos_d = math.dist(positions[i], positions[i - 1])
        rot_d = max(_axis_angle_deg(rotations[i][a], rotations[i - 1][a]) for a in range(3))
        dist_d = abs(distances[i] - distances[i - 1])
        if pos_d > pos_t or rot_d > rot_t or dist_d > dist_t:
            cuts.add(start_frame + i)
    return cuts


def detect_cuts_bone(start_frame, positions, rotations, thresholds):
    """thresholds は (位置, 回転[度])。"""
    pos_t, rot_t = thresholds
    cuts = set()
    for i in range(1, len(positions)):
        pos_d = math.dist(positions[i], positions[i - 1])
        rot_d = _quat_angle_deg(rotations[i], rotations[i - 1])
        if pos_d > pos_t or rot_d > rot_t:
            cuts.add(start_frame + i)
    return cuts


def perspective_cut_frames(start_frame, perspectives):
    frames = set()
    for i in range(1, len(perspectives)):
        if perspectives[i] != perspectives[i - 1]:
            frames.add(start_frame + i)
    return frames


def assemble_boundaries(f0, f1, *, cuts, perspective_frames, keep_frames, no_cut_detect):
    boundaries = {f0, f1}
    boundaries.update(k for k in keep_frames if f0 <= k <= f1)
    boundaries.update(perspective_frames)
    if not no_cut_detect:
        boundaries.update(cuts)
    return sorted(boundaries)
