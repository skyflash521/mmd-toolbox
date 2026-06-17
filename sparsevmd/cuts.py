"""不連続検出・必須境界の管理(sparsevmd.md §2.6, §6.1, §6.2)。

隣接サンプル F-1 と F の差が閾値を超えたフレーム F を不連続境界として返す。
- camera: 中心位置(ユークリッド距離)・回転(軸別角度誤差の最大,度)・距離(絶対値)。
- bone: 位置(ユークリッド距離)・回転(quaternion角度距離,度)。
回転の角度差はラップを除いた最小角で測り、見かけ上の360度ジャンプで誤検出しない。
perspective の切り替えフレームは閾値によらず常に境界(§6.1)。
assemble_boundaries は範囲端・keep-frame・perspective境界・(自動検出が有効なら)cut を統合する。
"""

import math


def parse_cut_threshold_camera(text):
    """`POS,ROT,DIST` を (pos, rot, dist) に解析する(§2.6)。"""
    return _parse_thresholds(text, 3)


def parse_cut_threshold_bone(text):
    """`POS,ROT` を (pos, rot) に解析する(§2.6)。"""
    return _parse_thresholds(text, 2)


def _parse_thresholds(text, n):
    parts = text.split(",")
    if len(parts) != n:
        raise ValueError(f"閾値は {n} 個のカンマ区切り: {text!r}")
    vals = []
    for p in parts:
        if p == "" or p != p.strip():
            raise ValueError(f"閾値に空要素・空白は不可: {text!r}")
        # 数値変換は float() を用いる(shakevmd の _finite_float と同じ規約)。
        # 符号付き・指数表記等の float() が受理する形式は許容し、非有限・負値のみ弾く。
        # より厳格な書式制限はプロジェクト全体の数値CLIパース方針として別途扱う。
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"閾値は数値: {text!r}")
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"閾値は0以上の有限値: {text!r}")
        vals.append(v)
    return tuple(vals)


def _axis_angle_deg(r1, r0):
    """1軸のオイラー角差(ラジアン)をラップ除去した最小角(度,絶対値)で返す。"""
    d = math.degrees(r1 - r0)
    d = (d + 180.0) % 360.0 - 180.0
    return abs(d)


def _quat_angle_deg(q1, q0):
    """2つのquaternion間の角度距離(度)。"""
    dot = abs(sum(a * b for a, b in zip(q1, q0)))
    dot = min(1.0, dot)
    return math.degrees(2.0 * math.acos(dot))


def detect_cuts_camera(start_frame, positions, rotations, distances, thresholds):
    """カメラの不連続フレーム集合を返す(§6.1)。閾値を厳密に超えたら境界。"""
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
    """ボーンの不連続フレーム集合を返す(§6.1)。閾値を厳密に超えたら境界。"""
    pos_t, rot_t = thresholds
    cuts = set()
    for i in range(1, len(positions)):
        pos_d = math.dist(positions[i], positions[i - 1])
        rot_d = _quat_angle_deg(rotations[i], rotations[i - 1])
        if pos_d > pos_t or rot_d > rot_t:
            cuts.add(start_frame + i)
    return cuts


def perspective_cut_frames(start_frame, perspectives):
    """perspective が切り替わるフレーム集合を返す(§4.2, §6.1)。常に境界。"""
    frames = set()
    for i in range(1, len(perspectives)):
        if perspectives[i] != perspectives[i - 1]:
            frames.add(start_frame + i)
    return frames


def assemble_boundaries(f0, f1, *, cuts, perspective_frames, keep_frames, no_cut_detect):
    """必須境界フレームを統合してソート済みリストで返す(§2.6, §6)。

    範囲端(f0, f1)、範囲内の keep-frame、perspective境界(常時)を含める。
    no_cut_detect が False のときのみ閾値検出 cut を含める。
    """
    boundaries = {f0, f1}
    boundaries.update(k for k in keep_frames if f0 <= k <= f1)
    boundaries.update(perspective_frames)
    if not no_cut_detect:
        boundaries.update(cuts)
    return sorted(boundaries)
