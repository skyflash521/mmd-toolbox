"""回転チャンネル評価器のテスト(sparsevmd.md §5.3, §7.2)。

- CameraRotationChannel: 3軸Euler(ラジアン)を軸ごとに線形補間。誤差はunwrap後の
  軸別角度誤差(度)の最大。360度境界のラップで誤検出しない。
- BoneRotationChannel: 端点 quaternion の slerp(線形係数)で予測し、サンプルとの
  角度距離(度)で誤差。slerp 軌道から外れたフレームで分割。

linear mode の評価のみ(ベジェ係数探索は後続)。
"""

import math

import pytest

pytest.importorskip("mmd_toolbox.vmd.fit", reason="impl pending: Step 1a fit extraction")

from mmd_toolbox.vmd.fit import BoneRotationChannel, CameraRotationChannel  # noqa: E402


def quat_z(deg):
    a = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(a), math.cos(a))


def quat_x(deg):
    a = math.radians(deg) / 2.0
    return (math.sin(a), 0.0, 0.0, math.cos(a))


# --- CameraRotationChannel --------------------------------------------------


def test_camera_rotation_linear_zero_error():
    # Y軸が 0→10度 を線形に動く。各軸線形補間と一致 → 誤差0。
    eulers = [(0.0, math.radians(i), 0.0) for i in range(11)]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0)
    assert frame is None


def test_camera_rotation_deviation_in_degrees():
    # 端点(0,0,0)。frame2 で Y軸 30度のずれ。誤差は度で30、frame2。
    eulers = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, math.radians(30), 0.0),
              (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(30.0)
    assert frame == 2


def test_camera_rotation_max_over_axes():
    # frame2 で X軸2度・Z軸30度のずれ → 最大軸(30度)。
    eulers = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
              (math.radians(2), 0.0, math.radians(30)), (0.0, 0.0, 0.0)]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    err, frame = ch.residual(0, 3)
    assert err == pytest.approx(30.0)
    assert frame == 2


def test_camera_rotation_prefers_axis_reversal_over_max_error():
    # Y軸 0,30,10,15,90度(端点0→90)。最大誤差は単調区間の frame3(52.5度)、
    # 軸別速度反転は frame1/frame2。§5.5 で分割候補は反転中の誤差最大 frame2。
    ys = [0.0, 30.0, 10.0, 15.0, 90.0]
    eulers = [(0.0, math.radians(y), 0.0) for y in ys]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(52.5)
    assert frame == 2


def test_camera_rotation_unwrap_no_false_error():
    # X軸 170→175→181度。181度は -179度として格納されるが、unwrap して滑らかに扱う。
    eulers = [
        (math.radians(170), 0.0, 0.0),
        (math.radians(175), 0.0, 0.0),
        (math.radians(-179), 0.0, 0.0),  # = 181度
    ]
    ch = CameraRotationChannel(0, eulers, tol=0.05)
    err, frame = ch.residual(0, 2)
    # unwrap 後は 170,175,181 の滑らかな動き。端点直線(170→181)と frame1(175)の差は0.5度。
    assert err == pytest.approx(0.5)
    assert frame == 1


# --- BoneRotationChannel ----------------------------------------------------


def test_bone_rotation_on_slerp_path_zero_error():
    # Z軸 0→90度の等速回転。slerp(線形係数)と一致 → 誤差0。
    quats = [quat_z(i * 9.0) for i in range(11)]
    ch = BoneRotationChannel(0, quats, tol=0.1)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0, abs=1e-6)
    assert frame is None


def test_bone_rotation_off_path_detected():
    # 端点 identity→Z90度。frame5 が identity(本来は Z45度)→ 角度距離45度。
    quats = [quat_z(i * 9.0) for i in range(11)]
    quats[5] = (0.0, 0.0, 0.0, 1.0)  # 軌道から外す
    ch = BoneRotationChannel(0, quats, tol=0.1)
    err, frame = ch.residual(0, 10)
    assert frame == 5
    assert err == pytest.approx(45.0, abs=1e-6)


def test_bone_rotation_sign_invariant():
    # q と -q は同じ回転。符号反転したサンプルでも角度距離(|dot|)は不変で誤差0。
    # §5.3 の符号不連続除去は角度距離評価では |dot| により自然に満たされる。
    quats = [quat_z(i * 9.0) for i in range(11)]
    quats[10] = tuple(-c for c in quats[10])  # 終点を符号反転(同じ回転)
    quats[3] = tuple(-c for c in quats[3])  # 内部サンプルも符号反転
    ch = BoneRotationChannel(0, quats, tol=0.1)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_bone_rotation_prefers_direction_reversal_over_max_error():
    # Z軸が 0,30,10,15,90度(端点0→90)と切り返す。最大誤差は単調区間の frame3、
    # 回転方向反転(軸の符号反転)は frame1/frame2。§5.5 で分割候補は反転中の誤差最大 frame2。
    degs = [0.0, 30.0, 10.0, 15.0, 90.0]
    quats = [quat_z(d) for d in degs]
    ch = BoneRotationChannel(0, quats, tol=0.1)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(52.5, abs=1e-4)
    assert frame == 2


def test_bone_rotation_normalized_and_zero_tol():
    quats = [quat_z(i * 9.0) for i in range(11)]
    quats[5] = (0.0, 0.0, 0.0, 1.0)
    ch = BoneRotationChannel(0, quats, tol=9.0)
    nerr, frame = ch.normalized(0, 10)
    assert nerr == pytest.approx(5.0, abs=1e-3)  # 45 / 9
    assert frame == 5
    ch0 = BoneRotationChannel(0, quats, tol=0.0)
    nerr0, frame0 = ch0.normalized(0, 10)
    assert nerr0 == math.inf and frame0 == 5


def test_bone_rotation_zero_tol_zero_error():
    # 軌道上(誤差0)で tol=0 なら (0.0, None)(§5.5)。
    quats = [quat_z(i * 9.0) for i in range(11)]
    ch = BoneRotationChannel(0, quats, tol=0.0)
    nerr, frame = ch.normalized(0, 10)
    assert nerr == 0.0 and frame is None
