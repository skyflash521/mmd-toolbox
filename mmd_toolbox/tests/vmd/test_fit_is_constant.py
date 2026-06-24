"""定数(無変化)区間判定 is_constant のテスト(flat-span-presplit-plan §4.2)。

各チャンネル種別が自分の単位で「区間 [a,b] が無変化か」を is_constant(a, b) で返す。
- 位置(EuclideanVectorChannel): 各成分 X/Y/Z の区間サンプル最大−最小がしきい値以下。
- 距離(LinearScalarChannel): サンプル最大−最小がしきい値以下。
- FOV(FovChannel): 生サンプル最大−最小がしきい値以下(出力の整数丸めは判定に使わない)。
- カメラ回転(CameraRotationChannel): unwrap 済み Euler の各軸変動幅がしきい値(rad)以下。
- ボーン回転(BoneRotationChannel): 各サンプルと先頭の角度差(度)の最大がしきい値以下。

しきい値は exact-constant(do-nothing/厳密静止)を捕らえる量子化下限相当の小さい値。
種別ごとに単位が異なるのでしきい値も種別ごと。各種別について「定数→True」「実変化→False」
「量子化下限以下の揺れ→True」「区間末尾での逸脱→False(早期打ち切りの取りこぼし防止)」を検証する。
"""

import math

from mmd_toolbox.vmd.fit import (
    BoneRotationChannel,
    CameraRotationChannel,
    EuclideanVectorChannel,
    FovChannel,
    LinearScalarChannel,
)

# 各種別で「量子化下限以下(定数扱い)」「明確な実変化(非定数)」とみなす代表値。
# しきい値(~1e-9)の正確な値に依存しないよう、下は十分小さく上は十分大きい値を使う。
_TINY = 1e-11
_REAL = 1e-3


def quat_z(deg):
    a = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(a), math.cos(a))


# --- LinearScalarChannel ----------------------------------------------------


def test_scalar_constant_true_over_long_span():
    ch = LinearScalarChannel(0, [5.0] * 200, tol=0.01)
    assert ch.is_constant(0, 199) is True


def test_scalar_varying_false():
    ch = LinearScalarChannel(0, [float(i) for i in range(11)], tol=0.01)
    assert ch.is_constant(0, 10) is False


def test_scalar_below_threshold_true_above_false():
    ch_small = LinearScalarChannel(0, [5.0, 5.0 + _TINY, 5.0], tol=0.01)
    assert ch_small.is_constant(0, 2) is True
    ch_big = LinearScalarChannel(0, [5.0, 5.0 + _REAL, 5.0], tol=0.01)
    assert ch_big.is_constant(0, 2) is False


def test_scalar_deviation_at_span_end_detected():
    vals = [3.0] * 50
    vals[-1] = 9.0  # 末尾でのみ逸脱(早期打ち切りで取りこぼさない)
    ch = LinearScalarChannel(0, vals, tol=0.01)
    assert ch.is_constant(0, 49) is False


def test_scalar_endpoint_a_included_in_subrange():
    # 区間先頭 a のサンプルも比較に含む。a だけ別値なら非定数(基準を a+1 に取らない)。
    ch = LinearScalarChannel(0, [9.0, 9.0, 0.0, 5.0, 5.0, 5.0], tol=0.01)
    assert ch.is_constant(2, 5) is False


def test_scalar_single_sample_span_is_constant():
    # 単一サンプル区間(a==b)は内部点が無く空走査 → 定数(True)。
    ch = LinearScalarChannel(0, [5.0, 999.0], tol=0.01)
    assert ch.is_constant(0, 0) is True


# --- EuclideanVectorChannel -------------------------------------------------


def test_vector_constant_true():
    vecs = [(1.0, 2.0, 3.0)] * 100
    ch = EuclideanVectorChannel(0, vecs, tol=0.01)
    assert ch.is_constant(0, 99) is True


def test_vector_single_axis_varies_false():
    vecs = [(1.0, 2.0, 3.0)] * 5
    vecs[3] = (1.0, 2.0, 3.5)  # Z だけ動く(軸別感度の確認)
    ch = EuclideanVectorChannel(0, vecs, tol=0.01)
    assert ch.is_constant(0, 4) is False


def test_vector_below_threshold_true_above_false():
    small = [(1.0, 2.0, 3.0), (1.0 + _TINY, 2.0, 3.0), (1.0, 2.0, 3.0)]
    assert EuclideanVectorChannel(0, small, tol=0.01).is_constant(0, 2) is True
    big = [(1.0, 2.0, 3.0), (1.0 + _REAL, 2.0, 3.0), (1.0, 2.0, 3.0)]
    assert EuclideanVectorChannel(0, big, tol=0.01).is_constant(0, 2) is False


def test_vector_deviation_at_span_end_detected():
    vecs = [(1.0, 2.0, 3.0)] * 50
    vecs[-1] = (1.0, 2.0, 9.0)
    ch = EuclideanVectorChannel(0, vecs, tol=0.01)
    assert ch.is_constant(0, 49) is False


def test_vector_subrange_constant_true():
    # 区間 [a,b] のみを見る。範囲外で動いても範囲内が定数なら True。
    vecs = [(0.0, 0.0, 0.0)] * 10
    vecs[0] = (9.0, 9.0, 9.0)  # 範囲外
    ch = EuclideanVectorChannel(0, vecs, tol=0.01)
    assert ch.is_constant(2, 9) is True


# --- FovChannel -------------------------------------------------------------


def test_fov_constant_true():
    ch = FovChannel(0, [30.0] * 100, tol=0.5)
    assert ch.is_constant(0, 99) is True


def test_fov_non_integer_constant_true():
    ch = FovChannel(0, [30.4] * 50, tol=0.5)
    assert ch.is_constant(0, 49) is True


def test_fov_varying_false():
    ch = FovChannel(0, [30.0, 30.0, 35.0, 30.0, 30.0], tol=0.5)
    assert ch.is_constant(0, 4) is False


def test_fov_uses_raw_samples_not_rounded():
    # 30.1 と 30.3 は共に整数30へ丸められるが、生サンプルは 0.2 動いている。
    # 丸め後で判定すると定数(True)に見えるため、生サンプルで判定して False を担保する。
    ch = FovChannel(0, [30.1, 30.3, 30.1], tol=0.5)
    assert ch.is_constant(0, 2) is False


def test_fov_below_threshold_true():
    ch = FovChannel(0, [30.0, 30.0 + _TINY, 30.0], tol=0.5)
    assert ch.is_constant(0, 2) is True


def test_fov_deviation_at_span_end_detected():
    vals = [30.0] * 50
    vals[-1] = 35.0
    ch = FovChannel(0, vals, tol=0.5)
    assert ch.is_constant(0, 49) is False


# --- CameraRotationChannel --------------------------------------------------


def test_camera_rotation_constant_true():
    eulers = [(0.0, 0.5, 0.0)] * 100
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    assert ch.is_constant(0, 99) is True


def test_camera_rotation_varying_false():
    eulers = [(0.0, math.radians(i), 0.0) for i in range(11)]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    assert ch.is_constant(0, 10) is False


def test_camera_rotation_below_threshold_true_above_false():
    small = [(0.0, 0.5, 0.0), (0.0, 0.5 + _TINY, 0.0), (0.0, 0.5, 0.0)]
    assert CameraRotationChannel(0, small, tol=0.05).is_constant(0, 2) is True
    big = [(0.0, 0.5, 0.0), (0.0, 0.5 + _REAL, 0.0), (0.0, 0.5, 0.0)]
    assert CameraRotationChannel(0, big, tol=0.05).is_constant(0, 2) is False


def test_camera_rotation_wrap_boundary_is_constant():
    # ±π をまたぐ定値オリエンテーション。raw Euler は符号反転で ~2π 振れるが、__init__ の
    # unwrap で連続化され変動幅0 → 定数。is_constant は unwrap 済み Euler で判定する(raw 判定なら
    # 2π の変動で False になるため、unwrap 語義を弁別する)。
    eulers = [(0.0, math.pi, 0.0), (0.0, -math.pi, 0.0), (0.0, math.pi, 0.0), (0.0, -math.pi, 0.0)]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    assert ch.is_constant(0, 3) is True


def test_camera_rotation_deviation_at_span_end_detected():
    eulers = [(0.0, 0.5, 0.0)] * 50
    eulers[-1] = (0.0, 1.5, 0.0)
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    assert ch.is_constant(0, 49) is False


# --- BoneRotationChannel ----------------------------------------------------


def test_bone_rotation_constant_true():
    quats = [quat_z(33.0)] * 100
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(0, 99) is True


def test_bone_rotation_sign_flip_is_constant():
    # q と -q は同一回転。__init__ で同一半球整列されるため定数とみなす。
    q = quat_z(33.0)
    quats = [q, tuple(-c for c in q), q, tuple(-c for c in q)]
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(0, 3) is True


def test_bone_rotation_varying_false():
    quats = [quat_z(float(i)) for i in range(11)]
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(0, 10) is False


def test_bone_rotation_below_threshold_true_above_false():
    small = [quat_z(33.0), quat_z(33.0 + _TINY), quat_z(33.0)]
    assert BoneRotationChannel(0, small, tol=0.1).is_constant(0, 2) is True
    big = [quat_z(33.0), quat_z(33.0 + _REAL), quat_z(33.0)]
    assert BoneRotationChannel(0, big, tol=0.1).is_constant(0, 2) is False


def test_bone_rotation_deviation_at_span_end_detected():
    quats = [quat_z(33.0)] * 50
    quats[-1] = quat_z(40.0)
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(0, 49) is False


def test_bone_rotation_endpoint_a_included_in_subrange():
    # 区間先頭 a の quaternion も比較に含む。a だけ別角度なら非定数。
    quats = [quat_z(90.0)] * 6
    quats[2] = quat_z(33.0)  # 区間先頭 a(=2)だけ別角度
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(2, 5) is False


def test_bone_rotation_subrange_constant_true():
    # 区間 [a,b] のみを見る(整列済み quaternion の索引が frame_start 基準で正しいこと)。
    quats = [quat_z(33.0)] * 10
    quats[0] = quat_z(90.0)  # 範囲外
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.is_constant(2, 9) is True
