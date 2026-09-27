import dataclasses
import math
import re
import statistics
from typing import Literal

from . import presets

MIN_GROUND_LEN = 4
HORIZ_VEL_THRESH = 0.08
VERT_VEL_THRESH = 0.04
LOCAL_WINDOW = 5
GROUND_Y_TOL = 0.08
RELATIVE_DELTA_THRESH = 0.08
MAX_CORRECTION = 0.5

Side = Literal["left", "right"]
Vec3 = tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class GroundSegment:
    """start と end は 0 始まりの相対サンプル番号で、end も区間に含む。"""

    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class GroundingDetection:
    candidate_frames: frozenset
    segments: tuple
    relative_rejected_frames: frozenset = frozenset()


@dataclasses.dataclass(frozen=True)
class IkPair:
    side: Side
    foot: str
    toe: str


@dataclasses.dataclass(frozen=True)
class PairingResult:
    pairs: tuple
    unpaired: tuple
    ambiguous: tuple


@dataclasses.dataclass(frozen=True)
class SegmentLock:
    segment: GroundSegment
    anchor: tuple
    max_displacement: float
    coef_scale: float
    clamped: bool


@dataclasses.dataclass(frozen=True)
class TrackStabilization:
    name: str
    category: str
    side: Side | None
    paired: bool
    grounding: GroundingDetection
    locks: tuple
    locked_positions: tuple
    max_change: float
    mean_change: float
    lock_applied_ratio: float
    warnings: tuple


def _runs(frames, min_len):
    if not frames:
        return ()
    segments = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        if prev - start + 1 >= min_len:
            segments.append(GroundSegment(start, prev))
        start = prev = f
    if prev - start + 1 >= min_len:
        segments.append(GroundSegment(start, prev))
    return tuple(segments)


def detect_grounding_segments(
    positions,
    *,
    paired_positions=None,
    min_ground_len=MIN_GROUND_LEN,
    horiz_vel_thresh=HORIZ_VEL_THRESH,
    vert_vel_thresh=VERT_VEL_THRESH,
    local_window=LOCAL_WINDOW,
    ground_y_tol=GROUND_Y_TOL,
    relative_delta_thresh=RELATIVE_DELTA_THRESH,
):
    """paired_positions は positions と同じ添字に揃え、相方のキーが無いフレームを None とする。"""
    n = len(positions)
    if n == 0:
        return GroundingDetection(frozenset(), (), frozenset())

    step_is_slow = []
    for k in range(n - 1):
        x0, _, z0 = positions[k]
        x1, _, z1 = positions[k + 1]
        horiz = math.hypot(x1 - x0, z1 - z0)
        vert = abs(positions[k + 1][1] - positions[k][1])
        step_is_slow.append(horiz <= horiz_vel_thresh and vert <= vert_vel_thresh)

    candidate = []
    for f in range(n):
        vel_ok = (f > 0 and step_is_slow[f - 1]) or (f < n - 1 and step_is_slow[f])
        if not vel_ok:
            continue
        lo = max(0, f - local_window)
        hi = min(n - 1, f + local_window)
        local_min = min(positions[j][1] for j in range(lo, hi + 1))
        if positions[f][1] <= local_min + ground_y_tol:
            candidate.append(f)

    rejected = frozenset()
    if paired_positions is not None:
        rejected = relative_rejected_frames(
            positions, paired_positions, threshold=relative_delta_thresh
        )
        candidate = [f for f in candidate if f not in rejected]

    return GroundingDetection(frozenset(candidate), _runs(candidate, min_ground_len), rejected)


def relative_rejected_frames(positions, paired_positions, *, threshold=RELATIVE_DELTA_THRESH):
    """paired_positions は positions と同じ添字に揃え、相方のキーが無いフレームを None とする。"""
    rejected = set()
    for f in range(1, len(positions)):
        p1, p0 = paired_positions[f], paired_positions[f - 1]
        if p1 is None or p0 is None:
            continue
        c1, c0 = positions[f], positions[f - 1]
        dx = (p1[0] - c1[0]) - (p0[0] - c0[0])
        dy = (p1[1] - c1[1]) - (p0[1] - c0[1])
        dz = (p1[2] - c1[2]) - (p0[2] - c0[2])
        if math.hypot(dx, dy, dz) > threshold:
            rejected.add(f)
    return frozenset(rejected)


def detect_side(name) -> Side | None:
    has_l = "左" in name
    has_r = "右" in name
    if has_l and not has_r:
        return "left"
    if has_r and not has_l:
        return "right"
    if has_l and has_r:
        return None

    tokens = re.findall(r"[a-z]+", name.lower())
    if "left" in tokens and "right" not in tokens:
        return "left"
    if "right" in tokens and "left" not in tokens:
        return "right"
    if "left" in tokens and "right" in tokens:
        return None
    if "l" in tokens and "r" not in tokens:
        return "left"
    if "r" in tokens and "l" not in tokens:
        return "right"
    return None


def pair_ik_tracks(tracks):
    foot_by_side = {}
    toe_by_side = {}
    for name, category in tracks:
        bucket = {"foot_ik": foot_by_side, "toe_ik": toe_by_side}.get(category)
        if bucket is None:
            continue
        bucket.setdefault(detect_side(name), []).append(name)

    pairs = []
    unpaired = []
    ambiguous = []
    sides = set(foot_by_side) | set(toe_by_side)
    for side in sorted(sides, key=lambda s: (s is None, s or "")):
        feet = foot_by_side.get(side, [])
        toes = toe_by_side.get(side, [])
        if side is None:
            ambiguous.extend(feet)
            ambiguous.extend(toes)
            continue
        foot_amb = len(feet) > 1
        toe_amb = len(toes) > 1
        if foot_amb:
            ambiguous.extend(feet)
        if toe_amb:
            ambiguous.extend(toes)
        if len(feet) == 1 and len(toes) == 1:
            pairs.append(IkPair(side, feet[0], toes[0]))
            continue
        if not foot_amb:
            unpaired.extend(feet)
        if not toe_amb:
            unpaired.extend(toes)

    return PairingResult(tuple(pairs), tuple(unpaired), tuple(ambiguous))


def compute_ground_anchor(positions, segment):
    rows = [positions[i] for i in range(segment.start, segment.end + 1)]
    return (
        statistics.median([p[0] for p in rows]),
        statistics.median([p[1] for p in rows]),
        statistics.median([p[2] for p in rows]),
    )


def _fade_coef(offset, length, center, edge, fade_width):
    w = min(fade_width, length // 2)
    if w <= 0:
        return center
    d = min(offset, length - 1 - offset)
    if d >= w:
        return center
    return edge + (center - edge) * (d / w)


def apply_foot_lock(positions, segments, strength, *, max_displacement=MAX_CORRECTION):
    out = [tuple(float(c) for c in p) for p in positions]
    locks = []
    xz_c, xz_e = strength["xz_center"], strength["xz_edge"]
    y_c, y_e = strength["y_center"], strength["y_edge"]
    fade_width = strength["fade_width"]

    for seg in segments:
        anchor = compute_ground_anchor(positions, seg)
        length = seg.end - seg.start + 1
        coefs_and_offsets_to_anchor = []
        raw_max = 0.0
        for i in range(seg.start, seg.end + 1):
            offset = i - seg.start
            cxz = _fade_coef(offset, length, xz_c, xz_e, fade_width)
            cy = _fade_coef(offset, length, y_c, y_e, fade_width)
            dx = anchor[0] - positions[i][0]
            dy = anchor[1] - positions[i][1]
            dz = anchor[2] - positions[i][2]
            raw_max = max(raw_max, math.hypot(cxz * dx, cy * dy, cxz * dz))
            coefs_and_offsets_to_anchor.append((cxz, cy, dx, dy, dz))

        clamped = raw_max > max_displacement
        coef_scale = max_displacement / raw_max if clamped else 1.0

        for idx, i in enumerate(range(seg.start, seg.end + 1)):
            cxz, cy, dx, dy, dz = coefs_and_offsets_to_anchor[idx]
            cxz *= coef_scale
            cy *= coef_scale
            out[i] = (
                positions[i][0] + cxz * dx,
                positions[i][1] + cy * dy,
                positions[i][2] + cxz * dz,
            )

        locks.append(SegmentLock(seg, anchor, raw_max * coef_scale, coef_scale, clamped))

    return out, locks


def stabilize_foot_ik(
    tracks: dict[str, tuple[str, list[int], list[Vec3]]], suppression
) -> dict[str, TrackStabilization]:
    """tracks の値は (種別, 昇順の絶対フレーム番号列, それに揃えた位置列)。"""
    pairing = pair_ik_tracks([(name, cat) for name, (cat, _, _) in tracks.items()])
    partner = {}
    for pair in pairing.pairs:
        partner[pair.foot] = pair.toe
        partner[pair.toe] = pair.foot

    frame_pos = {name: dict(zip(frames, positions, strict=True)) for name, (_, frames, positions) in tracks.items()}
    det = presets.resolve_foot_detection(suppression)

    result = {}
    for name, (category, frames, positions) in tracks.items():
        mate = partner.get(name)
        if mate is not None:
            mate_map = frame_pos[mate]
            paired_positions = [mate_map.get(f) for f in frames]
        else:
            paired_positions = None
        grounding = detect_grounding_segments(
            positions, paired_positions=paired_positions, horiz_vel_thresh=det["horiz_vel_thresh"]
        )
        locked, locks = apply_foot_lock(
            positions, grounding.segments, presets.resolve_foot_lock(suppression, category),
            max_displacement=det["max_displacement"],
        )

        changes = [math.dist(orig, locked_pos) for orig, locked_pos in zip(positions, locked, strict=True)]
        in_seg = sum(s.end - s.start + 1 for s in grounding.segments)
        result[name] = TrackStabilization(
            name=name,
            category=category,
            side=detect_side(name),
            paired=mate is not None,
            grounding=grounding,
            locks=tuple(locks),
            locked_positions=tuple(locked),
            max_change=max(changes) if changes else 0.0,
            mean_change=sum(changes) / len(changes) if changes else 0.0,
            lock_applied_ratio=in_seg / len(positions) if positions else 0.0,
            warnings=tuple(s for s in locks if s.clamped),
        )
    return result
