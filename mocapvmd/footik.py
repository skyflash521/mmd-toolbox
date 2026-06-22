"""足IK接地区間検出と左右ペアリング(mocapvmd.md §4.3 / 実装計画 §4.2)。

足IK・つま先IKの密サンプル(0始まり相対インデックス、各要素 (x,y,z))から接地区間を推定する。
検出は denoise と同様に密入力(隣接フレーム差をそのまま速度とみなし、ギャップ正規化は行わない)を
前提とする。

接地候補フレームは次をすべて満たすフレーム:
- 速度が小さい: 隣接フレーム差の水平速度 sqrt(dx^2+dz^2) が閾値以下、垂直速度 abs(dy) が閾値以下。
  あるフレームは隣接ステップの少なくとも一方が遅ければ接地候補に含める(接地開始・終了端を
  取りこぼさないため)。
- Y が局所的に低い: Y が前後 local_window フレーム窓内の局所最小に許容幅を足した値以下。
- 相対位置が急変していない(対応するつま先IK等の相方トラックを与えた場合): 相方との差分ベクトルの
  1フレーム変化が閾値以下。相方トラックが無い・相方キーを欠くフレームではこの条件をスキップする。

接地区間は接地候補フレームの連続 run のうち、長さが最小接地長以上のもの。接地候補と接地区間は
別々に返す(レポート §4.4 が「接地候補」と「接地区間」を別項目に挙げるため)。

左右ペアリングは、足IK・つま先IKの名前から側を推定し、同側で各1本のときだけペアにする(§3.4)。
"""

import dataclasses
import math
import re

MIN_GROUND_LEN = 4         # 最小接地長(フレーム)
HORIZ_VEL_THRESH = 0.08    # 水平速度閾値(MMD単位/frame)
VERT_VEL_THRESH = 0.04     # 垂直速度閾値(MMD単位/frame)
LOCAL_WINDOW = 5           # 接地Y局所窓(前後フレーム数)
GROUND_Y_TOL = 0.08        # 接地Y許容幅(局所最小に足す MMD単位)
RELATIVE_DELTA_THRESH = 0.08  # 相対位置急変の閾値(差分ベクトルの1フレーム変化ノルム MMD単位)


@dataclasses.dataclass(frozen=True)
class GroundSegment:
    """接地区間。start/end は 0始まり相対インデックスで両端を含む。"""

    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class GroundingDetection:
    """接地検出結果。

    candidate_frames は接地候補フレーム集合、segments は接地区間列、relative_rejected_frames は
    相方トラックとの相対位置急変で接地候補から除外したフレーム集合(相方未指定なら空)。
    """

    candidate_frames: frozenset
    segments: tuple
    relative_rejected_frames: frozenset = frozenset()


@dataclasses.dataclass(frozen=True)
class IkPair:
    """左右が一致した足IK・つま先IKの対。side は "left"/"right"。"""

    side: str
    foot: str
    toe: str


@dataclasses.dataclass(frozen=True)
class PairingResult:
    """ペアリング結果。pairs は成立した対、unpaired は相方を欠く単独、ambiguous は曖昧で対にできない名。"""

    pairs: tuple
    unpaired: tuple
    ambiguous: tuple


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
    paired_positions=None,
    min_ground_len=MIN_GROUND_LEN,
    horiz_vel_thresh=HORIZ_VEL_THRESH,
    vert_vel_thresh=VERT_VEL_THRESH,
    local_window=LOCAL_WINDOW,
    ground_y_tol=GROUND_Y_TOL,
    relative_delta_thresh=RELATIVE_DELTA_THRESH,
):
    """密サンプルから接地候補フレームと接地区間を検出する(§4.2)。

    paired_positions(相方トラックを positions と同じ相対インデックスで整列した位置列。キーを欠く
    フレームは None)を与えると、相方との相対位置が急変するフレームを接地候補から除外する。
    None なら相対条件を使わない(単独トラック検出)。
    """
    n = len(positions)
    if n == 0:
        return GroundingDetection(frozenset(), (), frozenset())

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

    rejected = frozenset()
    if paired_positions is not None:
        rejected = relative_rejected_frames(
            positions, paired_positions, threshold=relative_delta_thresh
        )
        candidate = [f for f in candidate if f not in rejected]

    return GroundingDetection(frozenset(candidate), _runs(candidate, min_ground_len), rejected)


# --- 相対位置急変 -----------------------------------------------------------


def relative_rejected_frames(positions, paired_positions, *, threshold=RELATIVE_DELTA_THRESH):
    """相方トラックとの差分ベクトルが1フレームで急変するフレーム集合を返す(§4.2)。

    差分ベクトル r[f] = paired[f] - positions[f] の1フレーム変化 |r[f]-r[f-1]| が threshold を超える
    フレーム f を返す。差分の絶対距離でなく変化量のみを使う。相方キーを欠くフレーム(None)は判定を
    スキップする(欠損は急変ではない)。positions(検出対象トラック)は密前提でキー欠損を扱わない。
    """
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


# --- 左右ペアリング ---------------------------------------------------------


def detect_side(name):
    """ボーン名から側("left"/"right")を推定する。判定できなければ None(§3.4)。

    優先順は 名前中の 左/右(最優先)→ 独立語 left/right → 区切りに囲まれた L/R。部分文字列
    (leftover の left、leg の l 等)は側マーカーにしない。左右が競合する名は None とする。
    """
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
    """(name, category) の列から足IK・つま先IKを左右で対応付ける(§3.4)。

    同側で foot_ik と toe_ik が各1本のときだけペアにする。同側同種が複数・側不明はペアにできず
    ambiguous、側が判る単独で相方を欠くものは unpaired とする。foot_ik / toe_ik 以外は対象外。
    """
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
