"""カメラのスカラー系チャンネル評価器のテスト。

- EuclideanVectorChannel: カメラ中心位置。各軸を線形補間し、採否はユークリッド距離で測る。
- FovChannel: 視野角。出力は整数度保存のため、線形補間値を四捨五入(0.5切り上げ)した
  整数で再評価し、元サンプルとの差を測る。許容は0.5度以上。
"""

import math

import pytest

from vmd.fit import EuclideanVectorChannel, FovChannel, _round_half_up


# --- EuclideanVectorChannel -------------------------------------------------


def test_vector_linear_zero_error():
    vecs = [(float(i), float(2 * i), float(-i)) for i in range(11)]
    ch = EuclideanVectorChannel(0, vecs, tol=0.01)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0)
    assert frame is None


def test_vector_euclidean_combines_axes():
    # 端点(0,0,0)→(0,0,0)。frame1 で (3,4,0) のずれ → ユークリッド距離5。
    vecs = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (0.0, 0.0, 0.0)]
    ch = EuclideanVectorChannel(0, vecs, tol=1.0)
    err, frame = ch.residual(0, 2)
    assert err == pytest.approx(5.0)
    assert frame == 1


def test_vector_normalized_and_zero_tol():
    vecs = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (0.0, 0.0, 0.0)]
    ch = EuclideanVectorChannel(0, vecs, tol=2.0)
    nerr, frame = ch.normalized(0, 2)
    assert nerr == pytest.approx(2.5)
    assert frame == 1
    ch0 = EuclideanVectorChannel(0, vecs, tol=0.0)
    nerr0, frame0 = ch0.normalized(0, 2)
    assert nerr0 == math.inf and frame0 == 1


def test_vector_split_frame_at_max_distance():
    vecs = [(0.0, 0.0, 0.0)] * 5
    vecs[3] = (0.0, 0.0, 10.0)  # frame3 に大きなずれ
    ch = EuclideanVectorChannel(0, vecs, tol=1.0)
    err, frame = ch.residual(0, 4)
    assert frame == 3
    assert err == pytest.approx(10.0)


def test_vector_prefers_velocity_reversal_over_max_error():
    # Z軸が 0,10,2,4,20 と動く(端点 0→20)。最大ユークリッド誤差は単調区間の frame3、
    # 速度反転(切り返し)は frame1/frame2。分割候補は反転中の誤差最大 frame2。
    # 採否用の最大誤差は真の最大(11)を返す。
    vecs = [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0), (0.0, 0.0, 2.0), (0.0, 0.0, 4.0), (0.0, 0.0, 20.0)]
    ch = EuclideanVectorChannel(0, vecs, tol=1.0)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(11.0)
    assert frame == 2


# --- FovChannel -------------------------------------------------------------


def test_fov_integer_linear_zero_error():
    # 30→40 の整数線形。整数フレームでも整数値、丸め誤差なし。
    vals = [float(30 + i) for i in range(11)]
    ch = FovChannel(0, vals, tol=0.5)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0)


def test_fov_rounding_within_half_degree():
    # 30→31 の緩い変化。線形補間値を整数へ丸めた誤差は frame5(30.5)で最大0.5度。
    # 丸め再評価が行われていれば 0.5、丸めをしない実装なら 0.0 になるため、
    # 厳密に 0.5 を要求して丸めの実施を担保する(tol=0.5以内)。
    vals = [30.0 + 0.1 * i for i in range(11)]
    ch = FovChannel(0, vals, tol=0.5)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.5)


def test_fov_half_integer_quantization_is_half_degree():
    # 半整数(x.5)サンプルは整数へ丸めると必ず 0.5 度の量子化誤差になる。
    # (丸め方向 half-up は誤差では区別できない。)
    vals = [30.0, 30.5, 31.0, 31.5, 32.0]
    ch = FovChannel(0, vals, tol=0.5)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(0.5)


def test_fov_round_down_below_half():
    # x.4 は切り捨て側。端点30,30.8 over 0..2、f1=30.4 → 30、誤差0.4(<0.5)。
    vals = [30.0, 30.4, 30.8]
    ch = FovChannel(0, vals, tol=0.5)
    err, frame = ch.residual(0, 2)
    assert err == pytest.approx(0.4)


def test_round_half_up_is_not_bankers():
    # 四捨五入(0.5切り上げ)。Python 標準 round の偶数丸めとは異なる。
    assert _round_half_up(0.5) == 1
    assert _round_half_up(1.5) == 2  # 偶数丸めなら 2 だが…
    assert _round_half_up(2.5) == 3  # 偶数丸めなら 2。half-up は 3。
    assert _round_half_up(30.4) == 30
    assert _round_half_up(30.6) == 31


def test_fov_prefers_velocity_reversal_over_max_error():
    # FOV が 30,40,32,34,50(端点30→50)。最大誤差は単調区間の frame3、速度反転は
    # frame1/frame2。分割候補は反転中の誤差最大 frame2。採否用最大誤差は11。
    ch = FovChannel(0, [30.0, 40.0, 32.0, 34.0, 50.0], tol=1.0)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(11.0)
    assert frame == 2


def test_fov_normalized_and_zero_tol():
    vals = [30.0, 30.0, 35.0, 30.0, 30.0]
    ch = FovChannel(0, vals, tol=2.0)
    nerr, frame = ch.normalized(0, 4)
    assert nerr == pytest.approx(2.5)  # 5.0 / 2.0
    assert frame == 2
    ch0 = FovChannel(0, vals, tol=0.0)
    nerr0, frame0 = ch0.normalized(0, 4)
    assert nerr0 == math.inf and frame0 == 2


def test_fov_large_deviation_detected():
    # 端点30,30。frame2 で実サンプルが 35 → 丸め後35との差5。tol0.5 超過、frame2 分割候補。
    vals = [30.0, 30.0, 35.0, 30.0, 30.0]
    ch = FovChannel(0, vals, tol=0.5)
    err, frame = ch.residual(0, 4)
    assert frame == 2
    assert err == pytest.approx(5.0)
