"""区間分割・キー削減の全体制御(sparsevmd.md §5.1, §5.5, §2.5)。

必須境界(範囲端・不連続・keep-frame。cuts.py 由来)の間を区間化し、各区間を全
チャンネルが許容誤差以内で表現できるか検査する。超過時は最大正規化誤差フレームで
再帰分割する。max-segment-frames で事前分割し、min-segment-frames を下回る tol 探索
分割はしない。非strictでは min-segment まで分割しても満たせない区間を1フレームまで
密に保持し(§1.2 の破綻回避)、strictでは StrictError(終了コード4)を送出する。

チャンネルは normalized(a, b) -> (正規化誤差, 最大誤差フレーム|None) を持つダックタイプ。
"""

import math

from mmd_toolbox.vmd import interp
from mmd_toolbox.vmd.types import BoneKey, CameraKey

from .cuts import (
    assemble_boundaries,
    detect_cuts_bone,
    detect_cuts_camera,
    perspective_cut_frames,
)
from .fit import (
    BoneRotationChannel,
    CameraRotationChannel,
    EuclideanVectorChannel,
    FovChannel,
    LinearScalarChannel,
    _quat_angle_deg,
    _round_half_up,
)
from .sample import perspective_series

# linear mode の補間ブロック(真の線形: 各チャンネル x1==y1, x2==y2)。
CAMERA_LINEAR_INTERP = bytes([20, 107, 20, 107]) * 6


def camera_interp_bytes(cp_x, cp_y, cp_z, cp_r, cp_l, cp_v):
    """6軸の制御点 (x1,y1,x2,y2) からカメラ補間24バイトを組み立てる。

    docs/specs/vmd/VMD_file_format.md のレイアウトに従い、軸順 X位置・Y位置・Z位置・
    回転・距離・視野角で、各軸を ax,bx,ay,by = (x1,x2,y1,y2) の4バイトで格納する。
    mmd_toolbox.vmd.interp.CAMERA_CHANNEL_OFFSET のオフセット(0,4,8,12,16,20)と整合する。
    """
    out = bytearray()
    for x1, y1, x2, y2 in (cp_x, cp_y, cp_z, cp_r, cp_l, cp_v):
        out += bytes([x1, x2, y1, y2])
    return bytes(out)


def bone_interp_bytes(x_cp, y_cp, z_cp, r_cp):
    """4チャンネルの制御点 (x1,y1,x2,y2) からボーン補間64バイトを組み立てる。

    docs/specs/vmd/VMD_file_format.md のレイアウトに従い、先頭16バイト
    (Byte[0]〜Byte[15])に本体を置き、Byte[16]以降は先頭シーケンスを1バイトずつ
    左シフトしたコピーを格納する。これにより Byte[2]/[3] が物理フラグで上書きされても
    シフトコピー側から制御点を復元できる(vmd-io.md §2.2)。詰めパッドは現行MMDに倣い0。
    """
    first = [
        x_cp[0], y_cp[0], z_cp[0], r_cp[0],  # x1
        x_cp[1], y_cp[1], z_cp[1], r_cp[1],  # y1
        x_cp[2], y_cp[2], z_cp[2], r_cp[2],  # x2
        x_cp[3], y_cp[3], z_cp[3], r_cp[3],  # y2
    ]
    b = bytearray(64)
    b[0:16] = bytes(first)
    b[16:31] = bytes(first[1:16])
    b[32:46] = bytes(first[2:16])
    b[48:61] = bytes(first[3:16])
    # 詰めパッド Byte[31]/[46]/[47]/[61]/[62]/[63] は0のまま(版依存・非検証)。
    return bytes(b)


_LINEAR_CP = (20, 20, 107, 107)
BONE_LINEAR_INTERP = bone_interp_bytes(_LINEAR_CP, _LINEAR_CP, _LINEAR_CP, _LINEAR_CP)


def build_camera_keys(source_keys, frames, segment_interp=None):
    """削減後フレーム列からカメラ出力キーを生成する(§3.2, §4.2)。

    各フレームで位置・回転(Euler)・距離・視野角(整数度へ四捨五入)・perspective
    (直近ホールド)をソースキーからサンプリングする。補間ブロックは既定で線形固定。
    segment_interp(前フレーム, 当フレーム) を渡すと、到達側キー(2個目以降)に
    その24バイトを格納する(curve_mode="bezier" の制御点注入。先頭キーは線形)。
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
    """削減後フレーム列からボーン出力キーを生成する(§3.2)。

    name_raw はソースの生バイトを保持する。各フレームで位置・回転(quaternion)を
    サンプリングする。補間ブロックは既定で線形固定。segment_interp(前フレーム,
    当フレーム) を渡すと、到達側キー(2個目以降)にその64バイトを格納する。
    """
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
    """--strict 指定時に許容誤差を満たせない(§2.5、終了コード4)。"""


def _angle_diff_deg(a, b):
    """2つの角度(ラジアン)の最小角度差(度、非負)。±2πのラップに不変。

    単一フレームの軸別角度誤差を unwrap 後と同値に測る(§7.2/§5.3)。
    """
    d = a - b
    return abs(math.degrees(math.atan2(math.sin(d), math.cos(d))))


def verify_camera_track(source_keys, output_keys, ranges, tols):
    """出力キーを再サンプリングし、§7.2 メトリクスで許容超過フレームを返す(§7.3)。

    各範囲を 1 フレーム間隔で元サンプルと比較する。視野角は出力に保存済みの整数度キーを
    補間した値で評価する(保存時に丸め済みなので再度丸めない)。丸め由来の差は §7.2/§7.3 の
    とおり許容(0.5度以上)内なら超過にならない。回転は軸別角度誤差(ラップ不変)の最大。
    戻り値は超過フレームの昇順リスト(空なら合格)。
    """
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
    """出力キーを再サンプリングし、§7.2 メトリクスで許容超過フレームを返す(§7.3)。

    位置はユークリッド距離、回転は quaternion 角度距離で測る。戻り値は昇順リスト。
    """
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


def reduce_track(boundaries, channels, min_seg, max_seg, strict):
    """必須境界とチャンネル群から、出力キーのフレーム列(昇順)を返す(§5.1)。"""
    bounds = sorted(set(boundaries))
    presplit = _presplit(bounds, max_seg)

    keys = set(presplit)
    for a, b in zip(presplit, presplit[1:]):
        _process_segment(a, b, channels, min_seg, strict, keys)
    return sorted(keys)


def _presplit(bounds, max_seg):
    """必須境界間を max_seg 以下に事前分割する(§5.1 step3)。"""
    out = [bounds[0]]
    for a, b in zip(bounds, bounds[1:]):
        span = b - a
        if span > max_seg:
            pieces = math.ceil(span / max_seg)
            for i in range(1, pieces):
                out.append(a + round(i * span / pieces))
        out.append(b)
    # 丸めの衝突を除去(昇順は構成上保たれる)。
    dedup = []
    for x in out:
        if not dedup or dedup[-1] != x:
            dedup.append(x)
    return dedup


def _worst_channel(a, b, channels):
    """区間 [a,b] の (採否用の最大正規化誤差, 分割フレーム) を返す。

    採否は全チャンネルの最大正規化誤差(分割フレームを持たないチャンネルも含む)で
    判定する。分割フレームは、フレームを報告するチャンネルのうち正規化誤差が最大の
    ものを採る。最大誤差が許容超過でも分割フレームが得られない場合は None を返し、
    呼び出し側で atomic 扱いにする。
    """
    max_norm = 0.0
    split_norm = 0.0
    split_frame = None
    for ch in channels:
        ne, frame = ch.normalized(a, b)
        if ne > max_norm:
            max_norm = ne
        if frame is not None and ne > split_norm:
            split_norm = ne
            split_frame = frame
    return max_norm, split_frame


def _process_segment(a, b, channels, min_seg, strict, keys):
    """区間 [a,b] を再帰的に処理し、必要なキーを keys に加える。"""
    stack = [(a, b)]
    while stack:
        a, b = stack.pop()
        if b - a <= 1:
            continue  # 隣接区間は内部点が無く受理(両端は既にキー)

        max_norm, split_frame = _worst_channel(a, b, channels)
        if max_norm <= 1.0:
            continue  # 全チャンネル許容内

        # min_seg を下回る、または両側を min_seg で割れない区間はこれ以上 tol 分割不可。
        # 分割フレームが得られない場合も同様に atomic 扱い。
        if b - a < 2 * min_seg or split_frame is None:
            _atomic_fail(a, b, strict, keys)
            continue

        # 分割候補フレームを [a+min_seg, b-min_seg] に収める(min_seg を下回る区間を作らない)。
        w = min(max(split_frame, a + min_seg), b - min_seg)
        keys.add(w)
        stack.append((a, w))
        stack.append((w, b))


def _atomic_fail(a, b, strict, keys):
    """min_seg 下限に達しても許容を満たせない区間の処理(§2.5)。"""
    if strict:
        raise StrictError(f"許容誤差を満たせない区間: [{a}, {b}]")
    # 非strict: 下限を無視し1フレームまで密に保持(§1.2 の破綻回避)。
    for f in range(a + 1, b):
        keys.add(f)


def _in_any_range(frame, ranges):
    return any(f0 <= frame <= f1 for f0, f1 in ranges)


def _boundary_with_predecessor(frames, f0):
    """不連続フレーム F に対し F-1 も境界に加える(§6.2)。

    境界は F-1 と F の間にあり、境界をまたいで補間曲線を作らない。両側を必須キーに
    することで、許容誤差が緩い場合でもジャンプが平滑化されず隣接フレームとして残る。
    """
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
):
    """カメラトラックを範囲ごとに削減し、出力キー列(昇順)を返す(§5.1, §3.2)。

    各範囲を 30fps 整数フレームでサンプリングし、不連続検出→必須境界→チャンネル評価→
    区間削減→出力キー生成→出力後検証(§7.3)する。範囲外の元キーは逐語保持する。
    curve_mode="bezier" ではチャンネルが1本のベジェ曲線で採否を判定し(より少ないキーに
    削減)、各出力区間の制御点を到達側キーの補間バイトに格納する。

    出力後検証(§7.3)は verify_camera_track で出力を再サンプリングし、許容超過があれば
    非strictは超過フレームをキーに追加して再構築(1フレーム間隔まで密化すれば元値を逐語
    保持でき必ず収束)、strictは StrictError。float32 格納差は §2.4 で量子化誤差として
    許容されるため float64 上の検証で扱う。

    スコープ外として以下は未実装:
    - §6.3 の範囲端境界キー注入・隣接曲線書き換え。範囲外キーは逐語保持する保守的方針。
    """
    reduced = []
    for f0, f1 in ranges:
        positions = _sampled_positions(source_keys, f0, f1)
        distances = [interp.sample(source_keys, "distance", f) for f in range(f0, f1 + 1)]
        fovs = [interp.sample(source_keys, "fov", f) for f in range(f0, f1 + 1)]
        eulers = [interp.sample(source_keys, "rot", f) for f in range(f0, f1 + 1)]
        persp = perspective_series(source_keys, f0, f1)

        cuts = detect_cuts_camera(f0, positions, eulers, distances, cut_thresholds)
        pcuts = perspective_cut_frames(f0, persp)
        bounds = assemble_boundaries(
            f0,
            f1,
            cuts=_boundary_with_predecessor(cuts, f0),
            perspective_frames=_boundary_with_predecessor(pcuts, f0),
            keep_frames=keep_frames,
            no_cut_detect=no_cut_detect,
        )
        pos_ch = EuclideanVectorChannel(f0, positions, tols.camera_pos, mode=curve_mode)
        dist_ch = LinearScalarChannel(f0, distances, tols.camera_distance, mode=curve_mode)
        fov_ch = FovChannel(f0, fovs, tols.camera_fov, mode=curve_mode)
        rot_ch = CameraRotationChannel(f0, eulers, tols.camera_rot, mode=curve_mode)
        channels = [pos_ch, dist_ch, fov_ch, rot_ch]
        range_frames = set(reduce_track(bounds, channels, min_seg, max_seg, strict))

        segment_interp = None
        if curve_mode == "bezier":
            def segment_interp(a, b, pos_ch=pos_ch, dist_ch=dist_ch, fov_ch=fov_ch, rot_ch=rot_ch):
                cp_x, cp_y, cp_z = pos_ch.curve(a, b)
                return camera_interp_bytes(
                    cp_x, cp_y, cp_z, rot_ch.curve(a, b), dist_ch.curve(a, b), fov_ch.curve(a, b)
                )

        # §7.3 出力後検証: 出力を再サンプリングし許容超過があれば、非strictは超過フレームを
        # キーに追加して再構築(1フレーム間隔は元値を逐語保持し必ず収束)、strictはエラー。
        while True:
            keys = build_camera_keys(source_keys, sorted(range_frames), segment_interp)
            bad = verify_camera_track(source_keys, keys, [(f0, f1)], tols)
            if not bad:
                break
            if strict:
                raise StrictError(f"出力後検証で許容を満たせない: 範囲[{f0},{f1}] フレーム{bad[:8]}")
            range_frames |= set(bad)
        reduced.extend(keys)

    outside = [k for k in source_keys if not _in_any_range(k.frame, ranges)]
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
):
    """ボーントラックを範囲ごとに削減し、出力キー列(昇順)を返す(§5.1, §3.2)。

    cut_thresholds は (POS, ROT)。範囲外の元キーは逐語保持する。curve_mode="bezier" では
    位置(軸別)と回転(slerp 係数)を1本のベジェ曲線で採否判定し、制御点を出力キーへ格納する。
    """
    reduced = []
    for f0, f1 in ranges:
        positions = _sampled_positions(source_keys, f0, f1)
        quats = [interp.sample(source_keys, "rot", f) for f in range(f0, f1 + 1)]

        cuts = detect_cuts_bone(f0, positions, quats, cut_thresholds)
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
        channels = [pos_ch, rot_ch]
        range_frames = set(reduce_track(bounds, channels, min_seg, max_seg, strict))

        segment_interp = None
        if curve_mode == "bezier":
            def segment_interp(a, b, pos_ch=pos_ch, rot_ch=rot_ch):
                cp_x, cp_y, cp_z = pos_ch.curve(a, b)
                return bone_interp_bytes(cp_x, cp_y, cp_z, rot_ch.curve(a, b))

        # §7.3 出力後検証(カメラと同様。非strictは密化で収束、strictはエラー)。
        while True:
            keys = build_bone_keys(source_keys, sorted(range_frames), segment_interp)
            bad = verify_bone_track(source_keys, keys, [(f0, f1)], tols)
            if not bad:
                break
            if strict:
                raise StrictError(f"出力後検証で許容を満たせない: 範囲[{f0},{f1}] フレーム{bad[:8]}")
            range_frames |= set(bad)
        reduced.extend(keys)

    outside = [k for k in source_keys if not _in_any_range(k.frame, ranges)]
    return sorted(reduced + outside, key=lambda k: k.frame)
