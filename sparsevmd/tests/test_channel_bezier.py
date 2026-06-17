"""チャンネルの bezier モード採否誤差のテスト(sparsevmd.md §5.2, §5.4)。

各チャンネルは mode="bezier" のとき、区間 [a,b] を1本のベジェ曲線で表したときの最大誤差を
返す(mode="linear" は直線近似の誤差)。曲線的な動きでは bezier 誤差 << linear 誤差となり、
1本の曲線で許容内に収まる区間は分割されずに済む。分割点(worst_frame)も bezier モードでは
ベジェ誤差プロファイルから選ぶ(§5.5: 採否と分割点を同じ正規化誤差基準で揃える)。
"""

import math

import pytest

from mmd_toolbox.vmd import interp
from sparsevmd.fit import (
    BoneRotationChannel,
    CameraRotationChannel,
    EuclideanVectorChannel,
    FovChannel,
    LinearScalarChannel,
)


def _eased_scalar(v0, v1, curve, n=11):
    # 端点 v0,v1 を結ぶ、curve(=制御点)に従うイージング値列(0..n-1 フレーム)。
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*curve, f / span) for f in range(n)]


EASE = (96, 0, 96, 30)  # 強いイージング(直線から大きく外れる)


# --- スカラー(距離など) ----------------------------------------------------


def test_scalar_bezier_low_error_on_eased_curve():
    vals = _eased_scalar(0.0, 100.0, EASE)
    lin = LinearScalarChannel(0, vals, tol=1.0, mode="linear")
    bez = LinearScalarChannel(0, vals, tol=1.0, mode="bezier")
    lin_err, _ = lin.residual(0, 10)
    bez_err, _ = bez.residual(0, 10)
    assert lin_err > 5.0          # 直線では大きく外れる
    assert bez_err < 0.5          # 1本の曲線でほぼ表現できる
    assert bez_err < lin_err * 0.2


def test_scalar_bezier_worst_frame_from_bezier_profile():
    # 滑らかなイージング上に1点スパイク(frame8)。ベジェは ease を曲線で吸収するので
    # 残差ピークはスパイク(frame8)。一方、直線残差のピークは ease の偏差最大(frame7)。
    # bezier モードの worst_frame がベジェ残差プロファイルから選ばれることを反証可能にする。
    vals = _eased_scalar(0.0, 100.0, EASE)
    vals[8] += 20.0
    bez_err, bez_frame = LinearScalarChannel(0, vals, tol=1.0, mode="bezier").residual(0, 10)
    lin_err, lin_frame = LinearScalarChannel(0, vals, tol=1.0, mode="linear").residual(0, 10)
    assert bez_frame == 8
    assert lin_frame == 7
    assert bez_frame != lin_frame


def test_scalar_bezier_linear_data_zero_error():
    vals = [float(i) for i in range(11)]
    bez = LinearScalarChannel(0, vals, tol=1.0, mode="bezier")
    err, _ = bez.residual(0, 10)
    assert err < 1e-6


def test_scalar_bezier_default_mode_is_linear():
    # mode 省略時は linear(既存挙動)。
    vals = _eased_scalar(0.0, 100.0, EASE)
    ch = LinearScalarChannel(0, vals, tol=1.0)
    err, _ = ch.residual(0, 10)
    assert err > 5.0  # linear 相当


def test_scalar_bezier_endpoints_equal_uses_deviation():
    # 端点同値で内部が動く場合(正規化不能)は、平坦曲線からの最大偏差を誤差とする(§5.2)。
    vals = [0.0, 3.0, 5.0, 3.0, 0.0]
    bez = LinearScalarChannel(0, vals, tol=1.0, mode="bezier")
    err, _ = bez.residual(0, 4)
    assert err == pytest.approx(5.0)


# --- FOV ---------------------------------------------------------------------


def test_fov_bezier_low_error_on_eased_curve():
    vals = _eased_scalar(30.0, 80.0, EASE)
    lin = FovChannel(0, vals, tol=1.0, mode="linear")
    bez = FovChannel(0, vals, tol=1.0, mode="bezier")
    bez_err = bez.residual(0, 10)[0]
    assert bez_err < lin.residual(0, 10)[0]
    assert bez_err < 1.0  # 丸め込み込みでも tol(1.0)内


def test_vector_bezier_euclidean_combines_not_per_axis_max():
    # 端点同値の山(0→5→0)を x のみ / x+z に入れる。採否がユークリッド距離なら x+z は
    # 約√2倍の誤差になる(軸別最大なら両者とも5で等しくなり、この差が出ない)。
    bump = [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vx = [(bump[i], 0.0, 0.0) for i in range(11)]
    vxz = [(bump[i], 0.0, bump[i]) for i in range(11)]
    ex = EuclideanVectorChannel(0, vx, tol=1.0, mode="bezier").residual(0, 10)[0]
    exz = EuclideanVectorChannel(0, vxz, tol=1.0, mode="bezier").residual(0, 10)[0]
    assert ex == pytest.approx(5.0)
    assert exz == pytest.approx(5.0 * math.sqrt(2), abs=1e-6)
    assert exz > ex * 1.3


# --- 位置(ユークリッド、軸別曲線) -----------------------------------------


def test_vector_bezier_per_axis_low_error():
    # Y軸だけイージング、X/Z は0。bezier は軸別に曲線をフィットし、ユークリッド誤差が下がる。
    ys = _eased_scalar(0.0, 100.0, EASE)
    vecs = [(0.0, y, 0.0) for y in ys]
    lin = EuclideanVectorChannel(0, vecs, tol=1.0, mode="linear")
    bez = EuclideanVectorChannel(0, vecs, tol=1.0, mode="bezier")
    assert lin.residual(0, 10)[0] > 5.0
    assert bez.residual(0, 10)[0] < 0.5


def test_vector_bezier_combines_axes_euclidean():
    # 2軸が別々のイージング。bezier でも採否はユークリッド距離(両軸合成)。
    ay = _eased_scalar(0.0, 50.0, EASE)
    az = _eased_scalar(0.0, 50.0, (0, 30, 96, 96))
    vecs = [(0.0, ay[i], az[i]) for i in range(11)]
    bez = EuclideanVectorChannel(0, vecs, tol=1.0, mode="bezier")
    err, _ = bez.residual(0, 10)
    assert err < 1.0  # 各軸を曲線で表現でき合成誤差も小さい


# --- カメラ回転(3軸共通の1曲線) -------------------------------------------


def _eased_coeff(curve, n=11):
    span = n - 1
    return [interp._solve_factor(*curve, f / span) for f in range(n)]


def test_camera_rotation_bezier_shared_curve_low_error():
    # 3軸が同じタイミング曲線(EASE)で動く。1本の共通曲線で表せるので bezier 誤差は小さく、
    # 直線(各軸が直線進行を仮定)では大きく外れる。誤差は軸別角度の最大(度)。
    c = _eased_coeff(EASE)
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    eulers = [(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f]) for f in range(11)]
    lin = CameraRotationChannel(0, eulers, tol=0.1, mode="linear")
    bez = CameraRotationChannel(0, eulers, tol=0.1, mode="bezier")
    assert lin.residual(0, 10)[0] > 1.0
    assert bez.residual(0, 10)[0] < 0.1


def test_camera_rotation_bezier_default_mode_is_linear():
    c = _eased_coeff(EASE)
    e1 = (math.radians(30), math.radians(20), math.radians(-10))
    eulers = [(e1[0] * c[f], e1[1] * c[f], e1[2] * c[f]) for f in range(11)]
    ch = CameraRotationChannel(0, eulers, tol=0.1)
    assert ch.residual(0, 10)[0] > 1.0  # linear 相当


def test_camera_rotation_bezier_linear_data_zero_error():
    # 各軸が直線進行(共通の直線係数)なら bezier 誤差は ~0。
    eulers = [(0.01 * f, -0.02 * f, 0.005 * f) for f in range(11)]
    bez = CameraRotationChannel(0, eulers, tol=0.1, mode="bezier")
    assert bez.residual(0, 10)[0] < 1e-3


# --- ボーン回転(slerp 係数の1曲線) ---------------------------------------


def _z_quat(deg):
    h = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(h), math.cos(h))


def test_bone_rotation_bezier_slerp_coeff_low_error():
    # 90°Z 回転を EASE タイミングで進める。bezier は slerp 係数の曲線でほぼ表現でき、
    # 直線(係数が線形)では大きく外れる。誤差はクォータニオン角距離(度)。
    c = _eased_coeff(EASE)
    quats = [_z_quat(90.0 * c[f]) for f in range(11)]
    lin = BoneRotationChannel(0, quats, tol=0.1, mode="linear")
    bez = BoneRotationChannel(0, quats, tol=0.1, mode="bezier")
    assert lin.residual(0, 10)[0] > 1.0
    assert bez.residual(0, 10)[0] < 0.1


def test_bone_rotation_bezier_default_mode_is_linear():
    c = _eased_coeff(EASE)
    quats = [_z_quat(90.0 * c[f]) for f in range(11)]
    ch = BoneRotationChannel(0, quats, tol=0.1)
    assert ch.residual(0, 10)[0] > 1.0


def test_bone_rotation_bezier_linear_data_zero_error():
    # 係数が線形(slerp の等速進行)なら bezier 誤差は ~0。
    quats = [_z_quat(90.0 * (f / 10.0)) for f in range(11)]
    bez = BoneRotationChannel(0, quats, tol=0.1, mode="bezier")
    assert bez.residual(0, 10)[0] < 1e-3
