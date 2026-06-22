"""足IK接地区間検出(mocapvmd.md §4.3 / 実装計画 §4.2)。

足IK・つま先IKの密サンプル(0始まり相対インデックス、各要素 (x,y,z))から接地区間を推定する。
本モジュールは単独トラックの検出(速度・Y局所最小・継続長)を担い、検出は denoise と同様に
密入力(隣接フレーム差をそのまま速度とみなし、ギャップ正規化は行わない)を前提とする。

接地候補フレームは次をすべて満たすフレーム:
- 速度が小さい: 隣接フレーム差の水平速度 sqrt(dx^2+dz^2) が閾値以下、垂直速度 abs(dy) が閾値以下。
  あるフレームは隣接ステップの少なくとも一方が遅ければ接地候補に含める(接地開始・終了端を
  取りこぼさないため)。
- Y が局所的に低い: Y が前後 local_window フレーム窓内の局所最小に許容幅を足した値以下。

接地区間は接地候補フレームの連続 run のうち、長さが最小接地長以上のもの。接地候補と接地区間は
別々に返す(レポート §4.4 が「接地候補」と「接地区間」を別項目に挙げるため)。
"""

import dataclasses
import math

MIN_GROUND_LEN = 4       # 最小接地長(フレーム)
HORIZ_VEL_THRESH = 0.08  # 水平速度閾値(MMD単位/frame)
VERT_VEL_THRESH = 0.04   # 垂直速度閾値(MMD単位/frame)
LOCAL_WINDOW = 5         # 接地Y局所窓(前後フレーム数)
GROUND_Y_TOL = 0.08      # 接地Y許容幅(局所最小に足す MMD単位)


@dataclasses.dataclass(frozen=True)
class GroundSegment:
    """接地区間。start/end は 0始まり相対インデックスで両端を含む。"""

    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class GroundingDetection:
    """接地検出結果。candidate_frames は接地候補フレーム集合、segments は接地区間列。"""

    candidate_frames: frozenset
    segments: tuple


def _runs(frames, min_len):
    """昇順フレーム列の連続 run のうち、長さ min_len 以上を GroundSegment 列で返す。"""
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
    min_ground_len=MIN_GROUND_LEN,
    horiz_vel_thresh=HORIZ_VEL_THRESH,
    vert_vel_thresh=VERT_VEL_THRESH,
    local_window=LOCAL_WINDOW,
    ground_y_tol=GROUND_Y_TOL,
):
    """密サンプルから接地候補フレームと接地区間を検出する(§4.2)。"""
    n = len(positions)
    if n == 0:
        return GroundingDetection(frozenset(), ())

    # 各隣接ステップが「遅い」か(水平・垂直とも閾値以下)。
    slow = []
    for k in range(n - 1):
        x0, _, z0 = positions[k]
        x1, _, z1 = positions[k + 1]
        horiz = math.hypot(x1 - x0, z1 - z0)
        vert = abs(positions[k + 1][1] - positions[k][1])
        slow.append(horiz <= horiz_vel_thresh and vert <= vert_vel_thresh)

    candidate = []
    for f in range(n):
        vel_ok = (f > 0 and slow[f - 1]) or (f < n - 1 and slow[f])
        if not vel_ok:
            continue
        lo = max(0, f - local_window)
        hi = min(n - 1, f + local_window)
        local_min = min(positions[j][1] for j in range(lo, hi + 1))
        if positions[f][1] <= local_min + ground_y_tol:
            candidate.append(f)

    return GroundingDetection(frozenset(candidate), _runs(candidate, min_ground_len))
