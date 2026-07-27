"""足IK接地区間検出と左右ペアリング。

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
別々に返す(レポートが「接地候補」と「接地区間」を別項目に挙げるため)。

左右ペアリングは、足IK・つま先IKの名前から側を推定し、同側で各1本のときだけペアにする。
"""

import dataclasses
import math
import re
import statistics

from . import presets

MIN_GROUND_LEN = 4         # 最小接地長(フレーム)
HORIZ_VEL_THRESH = 0.08    # 水平速度閾値(MMD単位/frame)
VERT_VEL_THRESH = 0.04     # 垂直速度閾値(MMD単位/frame)
LOCAL_WINDOW = 5           # 接地Y局所窓(前後フレーム数)
GROUND_Y_TOL = 0.08        # 接地Y許容幅(局所最小に足す MMD単位)
RELATIVE_DELTA_THRESH = 0.08  # 相対位置急変の閾値(差分ベクトルの1フレーム変化ノルム MMD単位)
MAX_CORRECTION = 0.5       # 接地ロックの最大補正量(接地区間内の1フレーム最大変位 MMD単位)


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


@dataclasses.dataclass(frozen=True)
class SegmentLock:
    """接地区間ごとのロック記録。

    anchor は接地アンカー(中央値)、max_displacement はロック後の1フレーム最大変位、coef_scale は
    最大補正量クランプによる一律係数倍率(1.0 ならクランプ無し)、clamped はクランプの有無。レポートは
    clamped が立った区間を警告として出す。
    """

    segment: GroundSegment
    anchor: tuple
    max_displacement: float
    coef_scale: float
    clamped: bool


@dataclasses.dataclass(frozen=True)
class TrackStabilization:
    """1トラックの接地安定化結果と診断。

    side はトラック名から決まる側("left"/"right"/None)、paired は相方の有無、grounding は接地検出、
    locks は接地区間ごとのロック記録、locked_positions はロック後位置列。max_change/mean_change は
    ロック前後のユークリッド距離の最大・平均、lock_applied_ratio は接地区間内フレーム数の割合、
    warnings はクランプされた区間(レポートの警告対象)。
    """

    name: str
    category: str
    side: str
    paired: bool
    grounding: GroundingDetection
    locks: tuple
    locked_positions: tuple
    max_change: float
    mean_change: float
    lock_applied_ratio: float
    warnings: tuple


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
    """密サンプルから接地候補フレームと接地区間を検出する。

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
    """相方トラックとの差分ベクトルが1フレームで急変するフレーム集合を返す。

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
    """ボーン名から側("left"/"right")を推定する。判定できなければ None。

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
    """(name, category) の列から足IK・つま先IKを左右で対応付ける。

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


# --- 接地ロック適用 ---------------------------------------------------------


def compute_ground_anchor(positions, segment):
    """接地区間の各IK位置の X/Y/Z 各軸独立の中央値を接地アンカーとして返す。

    中央値は外れ値に強く、片足を置いた瞬間の1フレーム跳ねに引っ張られにくい。
    """
    rows = [positions[i] for i in range(segment.start, segment.end + 1)]
    return (
        statistics.median([p[0] for p in rows]),
        statistics.median([p[1] for p in rows]),
        statistics.median([p[2] for p in rows]),
    )


def _fade_coef(offset, length, center, edge, fade_width):
    """区間内オフセット offset(0始まり)でのロック係数。

    端は edge、端から fade_width 以上内側は center、間は線形フェード。接地長が 2*fade_width 未満
    なら接地長の半分ずつに按分する(実効フェード幅 = min(fade_width, length//2))。
    """
    w = min(fade_width, length // 2)
    if w <= 0:
        return center
    d = min(offset, length - 1 - offset)
    if d >= w:
        return center
    return edge + (center - edge) * (d / w)


def apply_foot_lock(positions, segments, strength, *, max_displacement=MAX_CORRECTION):
    """接地区間で位置を接地アンカーへ寄せ、(ロック後位置列, SegmentLock 列)を返す。

    strength は resolve_foot_lock の dict(xz_center/xz_edge/y_center/y_edge/fade_width)。各区間で
    アンカー(中央値)を求め、X/Z は xz 係数、Y は y 係数でフェードしながら new=orig+coef*(anchor-orig)
    へ寄せる。区間内の1フレーム最大変位(X/Y/Z ユークリッド)が max_displacement を超える場合は、その
    区間の係数を一律 max_displacement/max でスケールしてクランプし、SegmentLock.clamped を立てる。
    区間外のフレームは変更しない。
    """
    out = [tuple(float(c) for c in p) for p in positions]
    locks = []
    xz_c, xz_e = strength["xz_center"], strength["xz_edge"]
    y_c, y_e = strength["y_center"], strength["y_edge"]
    fade_width = strength["fade_width"]

    for seg in segments:
        anchor = compute_ground_anchor(positions, seg)
        length = seg.end - seg.start + 1
        frames = []  # (cxz, cy, dx, dy, dz)
        raw_max = 0.0
        for i in range(seg.start, seg.end + 1):
            offset = i - seg.start
            cxz = _fade_coef(offset, length, xz_c, xz_e, fade_width)
            cy = _fade_coef(offset, length, y_c, y_e, fade_width)
            dx = anchor[0] - positions[i][0]
            dy = anchor[1] - positions[i][1]
            dz = anchor[2] - positions[i][2]
            raw_max = max(raw_max, math.hypot(cxz * dx, cy * dy, cxz * dz))
            frames.append((cxz, cy, dx, dy, dz))

        clamped = raw_max > max_displacement
        coef_scale = max_displacement / raw_max if clamped else 1.0

        for idx, i in enumerate(range(seg.start, seg.end + 1)):
            cxz, cy, dx, dy, dz = frames[idx]
            cxz *= coef_scale
            cy *= coef_scale
            out[i] = (
                positions[i][0] + cxz * dx,
                positions[i][1] + cy * dy,
                positions[i][2] + cxz * dz,
            )

        locks.append(SegmentLock(seg, anchor, raw_max * coef_scale, coef_scale, clamped))

    return out, locks


# --- 統合(ペアリング→検出→ロック→診断)----------------------------------


def stabilize_foot_ik(tracks, suppression):
    """足IK・つま先IKトラック群に接地安定化を適用し、トラックごとの結果と診断を返す。

    tracks は dict name -> (category, frames, positions)。category は "foot_ik"/"toe_ik"、frames は
    昇順の絶対フレーム列、positions は frames に整列した (x,y,z) 列(foot_ik/toe_ik 以外は呼び出し側で
    除外する)。各トラックは左右ペアリング(pair_ik_tracks)で相方を決め、相方があればその位置を絶対
    フレームで整列(欠けるフレームは None)して接地検出の相対位置参照に使う。曖昧・相方欠如のトラックは
    相対参照なしで処理する。横滑り抑制 S(0〜1)は接地検出の水平速度許容・接地ロックの X/Z 強度・
    最大補正量上限を連動制御する(resolve_foot_detection / resolve_foot_lock)。返り値は
    dict name -> TrackStabilization。
    """
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

        changes = [math.dist(o, l) for o, l in zip(positions, locked, strict=True)]
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
