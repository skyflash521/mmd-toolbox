"""回転チャンネル評価器のテスト(sparsevmd.md §5.3, §7.2)。

- CameraRotationChannel: 3軸Euler(ラジアン)を軸ごとに線形補間。誤差はunwrap後の
  軸別角度誤差(度)の最大。360度境界のラップで誤検出しない。
- BoneRotationChannel: 端点 quaternion の slerp(線形係数)で予測し、サンプルとの
  角度距離(度)で誤差。slerp 軌道から外れたフレームで分割。

linear mode の回転評価を対象とする。
"""

import math

import pytest

from mmd_toolbox.vmd.fit import BoneRotationChannel, CameraRotationChannel, _quat_angle_deg


def quat_z(deg):
    a = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(a), math.cos(a))


# --- _quat_angle_deg 数値精度 ----------------------------------------------


def test_quat_angle_deg_small_angle_precision():
    # 微小角(0.1度刻み)を acos の悪条件域(dot≈1)で正確に測れること。
    # 2·acos(|dot|) は dot≈1 で丸め誤差が増幅し、プラットフォームの libm 差で
    # 軌道上のはずの角度に ~1e-6 度の偽差が出ていた。atan2 形は安定。
    base = quat_z(0.0)
    for deg in (0.05, 0.1, 0.2, 0.5, 1.0):
        assert _quat_angle_deg(base, quat_z(deg)) == pytest.approx(deg, abs=1e-9)


def test_quat_angle_deg_sign_invariant_exact_zero():
    # q と -q は同一回転 → 角度0(符号反転でも厳密に0)。
    q = quat_z(33.0)
    assert _quat_angle_deg(q, tuple(-c for c in q)) == pytest.approx(0.0, abs=1e-12)


def test_quat_angle_deg_identical_exact_zero():
    q = quat_z(17.0)
    assert _quat_angle_deg(q, q) == 0.0


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


# --- Step B: 回転 curve() のベクトル化契約 ----------------------------------
#
# CameraRotationChannel.curve / BoneRotationChannel.curve は係数曲線フィット
# (_fit_coeff_curve)で、残差評価に配列版 _bezier_y_at_many、量子化評価に配列版
# interp._solve_factor_many を使う。単点版 _bezier_y_at / interp._solve_factor を
# curve() の内部で呼ばないこと(§4 支配要因2 の解消=性能目的の契約)。


def _spy_curve_paths(monkeypatch):
    """curve() 内のベジェ評価をスパイする。

    戻り値 (scalar_calls, bez_lens, sf_lens):
    - scalar_calls: 単点版 _bezier_y_at / interp._solve_factor の呼び出し回数(配列版なら0)。
    - bez_lens: 配列版 _bezier_y_at_many(残差経路)が受けた xs 長のリスト。
    - sf_lens: 配列版 interp._solve_factor_many(量子化経路)が受けた xs 長のリスト。
    残差側と量子化側を別リストで記録し、片側だけ配列化する不完全実装を弾く。1点ずつ呼ぶ
    偽装も、全内部点を1回で受けること(長さ=内部点数)で検出する。interp._solve_factor_many は
    未実装なら getattr で None になり差し替えない(その場合 sf_lens は空のまま)。
    """
    import mmd_toolbox.vmd.fit as fit
    import mmd_toolbox.vmd.interp as interp_mod
    import numpy as np

    scalar_calls, bez_lens, sf_lens = [], [], []
    real_bez, real_sf = fit._bezier_y_at, interp_mod._solve_factor
    real_bez_many = fit._bezier_y_at_many
    real_sf_many = getattr(interp_mod, "_solve_factor_many", None)

    monkeypatch.setattr(fit, "_bezier_y_at",
                        lambda *a, **k: (scalar_calls.append(1), real_bez(*a, **k))[1])
    monkeypatch.setattr(interp_mod, "_solve_factor",
                        lambda *a, **k: (scalar_calls.append(1), real_sf(*a, **k))[1])

    def bez_many(px1, py1, px2, py2, xs):
        bez_lens.append(np.asarray(xs).size)
        return real_bez_many(px1, py1, px2, py2, xs)

    monkeypatch.setattr(fit, "_bezier_y_at_many", bez_many)
    if real_sf_many is not None:
        def sf_many(x1, y1, x2, y2, xs):
            sf_lens.append(np.asarray(xs).size)
            return real_sf_many(x1, y1, x2, y2, xs)
        monkeypatch.setattr(interp_mod, "_solve_factor_many", sf_many)
    return scalar_calls, bez_lens, sf_lens


@pytest.mark.xfail(reason="impl pending: Step B vectorize", strict=False)
def test_camera_rotation_curve_uses_vectorized(monkeypatch):
    # 内部点9個(範囲 0..10)を、単点版でなく配列版で一括評価する(§4 支配要因2 の解消)。
    # 残差(_bezier_y_at_many)と量子化(_solve_factor_many)の両経路が全内部点を1回で受ける。
    scalar_calls, bez_lens, sf_lens = _spy_curve_paths(monkeypatch)
    degs = [0.0, 5.0, 12.0, 20.0, 30.0, 42.0, 55.0, 70.0, 82.0, 90.0, 95.0]
    eulers = [(0.0, 0.0, math.radians(d)) for d in degs]  # 曲がった Z 回転(内部点で分割不要な曲線)
    ch = CameraRotationChannel(0, eulers, tol=0.05, mode="bezier")
    ch.curve(0, 10)
    assert scalar_calls == []                 # 単点版は呼ばない
    assert bez_lens and all(n == 9 for n in bez_lens)  # 残差は全内部点を1回で
    assert sf_lens and all(n == 9 for n in sf_lens)    # 量子化も全内部点を1回で


@pytest.mark.xfail(reason="impl pending: Step B vectorize", strict=False)
def test_bone_rotation_curve_uses_vectorized(monkeypatch):
    scalar_calls, bez_lens, sf_lens = _spy_curve_paths(monkeypatch)
    degs = [0.0, 5.0, 12.0, 20.0, 30.0, 42.0, 55.0, 70.0, 82.0, 90.0, 95.0]
    quats = [quat_z(d) for d in degs]
    ch = BoneRotationChannel(0, quats, tol=0.1, mode="bezier")
    ch.curve(0, 10)
    assert scalar_calls == []
    assert bez_lens and all(n == 9 for n in bez_lens)
    assert sf_lens and all(n == 9 for n in sf_lens)
