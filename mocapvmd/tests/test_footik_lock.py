"""接地ロック適用のテスト(mocapvmd.md §5.4 / §5.5、実装計画 §4.3)。

接地区間ごとに、各IK自身の位置の中央値を接地アンカーとし、接地中はアンカーへブレンド係数で寄せる。
係数は接地中央値を区間内側で効かせ、区間端では端値から中央値へ線形フェードする(フェード幅は端から
fade_width、接地長が 2*fade_width 未満なら接地長の半分ずつに按分)。X/Z は xz 係数、Y は y 係数を使う。
ブレンドは new = orig + coef*(anchor - orig)。接地区間内で元値からの1フレーム最大変位(X/Y/Z の
ユークリッド距離)が最大補正量 0.5 を超える場合は、その区間の係数を一律 0.5/max でスケールし、その
クランプ事象を記録する(記録はレポート層が警告として出す)。
"""

import math

import pytest

from mocapvmd import footik
from mocapvmd.footik import GroundSegment

# テストを presets から切り離すための明示的な強度。foot_ik balanced 相当。
FOOT = {"xz_center": 0.90, "xz_edge": 0.25, "y_center": 0.50, "y_edge": 0.10, "fade_width": 3}


def _track(xs, ys=None, zs=None):
    n = len(xs)
    ys = ys if ys is not None else [0.0] * n
    zs = zs if zs is not None else [0.0] * n
    return [(xs[i], ys[i], zs[i]) for i in range(n)]


# --- 接地アンカー(中央値)-------------------------------------------------


def test_anchor_is_median_robust_to_outlier():
    # 区間内に1フレームの跳ねがあっても中央値は引っ張られない。
    pos = _track([1.0, 1.0, 5.0, 1.0, 1.0])
    a = footik.compute_ground_anchor(pos, GroundSegment(0, 4))
    assert a == pytest.approx((1.0, 0.0, 0.0))


def test_anchor_per_axis_median():
    # X/Y/Z それぞれ独立に中央値を取る。
    pos = [(0.0, 10.0, 100.0), (2.0, 12.0, 100.0), (4.0, 14.0, 130.0)]
    a = footik.compute_ground_anchor(pos, GroundSegment(0, 2))
    assert a == pytest.approx((2.0, 12.0, 100.0))


# --- ロック適用: フェード曲線 ----------------------------------------------


@pytest.mark.parametrize("offset,new_x", [
    (0, 0.15),            # 開始端: 端値 0.25 → 0.2*(1-0.25)
    (1, 0.10666667),      # 0.25 + 0.65*(1/3)
    (2, 0.06333333),      # 0.25 + 0.65*(2/3)
    (3, 0.02),            # 内側: 中央値 0.90 → 0.2*(1-0.90)
    (4, 0.06333333),      # 終了端側(対称): 端から距離2
    (5, 0.10666667),      # 終了端側: 端から距離1
    (6, 0.15),            # 終了端: 端値 0.25
])
def test_fade_curve_from_edge_to_center(offset, new_x):
    # 長さ7の区間。アンカーは中央値0。1フレームだけ X を 0.2 ずらし、その係数を適用結果から確認する。
    # 開始端・終了端の両側で端から距離に応じてフェードする(両端対称)。
    xs = [0.0] * 7
    xs[offset] = 0.2
    locked, _ = footik.apply_foot_lock(_track(xs), [GroundSegment(0, 6)], FOOT)
    assert locked[offset][0] == pytest.approx(new_x)


@pytest.mark.parametrize("offset", [1, 2])
def test_fade_width_splits_for_short_segment(offset):
    # 長さ4の区間ではフェード幅が接地長の半分(2)に縮み、中央値0.90には達しない。両端側とも
    # 端からの距離1で係数 0.25+0.65*(1/2)=0.575、new=0.2*(1-0.575)=0.085。
    xs = [0.0, 0.0, 0.0, 0.0]
    xs[offset] = 0.2
    locked, _ = footik.apply_foot_lock(_track(xs), [GroundSegment(0, 3)], FOOT)
    assert locked[offset][0] == pytest.approx(0.085)


# --- ロック適用: チャンネル分離・区間外 ------------------------------------


def test_xz_and_y_use_separate_coefficients():
    # 内側フレームで X は xz中央0.90、Y は y中央0.50 を使う。
    xs = [0.0] * 7
    ys = [0.0] * 7
    zs = [0.0] * 7
    xs[3] = 0.2
    ys[3] = 0.2
    zs[3] = 0.2
    locked, _ = footik.apply_foot_lock(_track(xs, ys, zs), [GroundSegment(0, 6)], FOOT)
    assert locked[3][0] == pytest.approx(0.02)   # X は xz中央0.90: 0.2*(1-0.90)
    assert locked[3][1] == pytest.approx(0.10)   # Y は y中央0.50: 0.2*(1-0.50)
    assert locked[3][2] == pytest.approx(0.02)   # Z は X と同じ xz中央0.90


def test_frames_outside_segments_unchanged():
    xs = [9.0, 9.0, 9.0, 0.0, 0.5, 0.0, 0.0, 7.0, 7.0, 7.0]
    pos = _track(xs)
    locked, _ = footik.apply_foot_lock(pos, [GroundSegment(3, 6)], FOOT)
    assert locked[0] == pos[0] and locked[1] == pos[1] and locked[2] == pos[2]
    assert locked[7] == pos[7] and locked[8] == pos[8] and locked[9] == pos[9]


# --- 最大補正量クランプ(§5.5)---------------------------------------------


def test_no_clamp_when_within_limit():
    xs = [0.0] * 7
    xs[3] = 0.2  # 内側変位 0.90*0.2=0.18 <= 0.5
    locked, locks = footik.apply_foot_lock(_track(xs), [GroundSegment(0, 6)], FOOT)
    assert len(locks) == 1
    assert locks[0].clamped is False
    assert locks[0].coef_scale == pytest.approx(1.0)
    assert locks[0].max_displacement == pytest.approx(0.18)


def test_clamp_is_euclidean_and_uniform_across_segment():
    # 内側 frame3 を X・Y・Z 同時に 0.6 ずらす。変位は X/Z が xz係数0.90、Y が y係数0.50 を掛けた
    # 3次元ユークリッド hypot(0.54, 0.30, 0.54)=0.8205 > 0.5(Y を無視する hypot(dx,dz) 実装を弾く)。
    # 同区間の別の内側 frame4 も同じ coef_scale でスケールされる(軸別・違反フレームのみ補正の誤実装を弾く)。
    # クランプ事象は clamped/coef_scale/max_displacement に記録し、レポート層がこれを警告として出す。
    xs = [0.0] * 9
    ys = [0.0] * 9
    zs = [0.0] * 9
    xs[3], ys[3], zs[3] = 0.6, 0.6, 0.6
    xs[4] = 0.3
    locked, locks = footik.apply_foot_lock(_track(xs, ys, zs), [GroundSegment(0, 8)], FOOT)
    raw_max = math.hypot(0.90 * 0.6, 0.50 * 0.6, 0.90 * 0.6)
    scale = 0.5 / raw_max
    assert locks[0].clamped is True
    assert locks[0].coef_scale == pytest.approx(scale)
    assert locks[0].max_displacement == pytest.approx(0.5)
    assert locked[3][0] == pytest.approx(0.6 * (1 - 0.90 * scale))
    assert locked[3][1] == pytest.approx(0.6 * (1 - 0.50 * scale))  # Y は y係数で一律スケール
    assert locked[3][2] == pytest.approx(0.6 * (1 - 0.90 * scale))
    assert locked[4][0] == pytest.approx(0.3 * (1 - 0.90 * scale))  # 一律スケール(区間全体)


def test_segment_anchor_is_per_segment():
    # 2区間それぞれが自身の中央値をアンカーにする。
    xs = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]
    _, locks = footik.apply_foot_lock(
        _track(xs), [GroundSegment(0, 3), GroundSegment(4, 7)], FOOT
    )
    assert [lk.segment for lk in locks] == [GroundSegment(0, 3), GroundSegment(4, 7)]
    assert locks[0].anchor == pytest.approx((0.0, 0.0, 0.0))
    assert locks[1].anchor == pytest.approx((5.0, 0.0, 0.0))


def test_clamp_is_independent_per_segment():
    # 片方の区間だけがクランプされ、もう片方は coef_scale=1.0 のまま(スケール漏洩しない)。
    xs = [0.0] * 14
    xs[3] = 1.0    # 区間 [0,6] の内側 → クランプ
    xs[10] = 0.2   # 区間 [7,13] の内側 → 非クランプ
    locked, locks = footik.apply_foot_lock(
        _track(xs), [GroundSegment(0, 6), GroundSegment(7, 13)], FOOT
    )
    assert locks[0].clamped is True
    assert locks[0].coef_scale == pytest.approx(0.5 / 0.90)
    assert locks[1].clamped is False
    assert locks[1].coef_scale == pytest.approx(1.0)
    assert locked[10][0] == pytest.approx(0.2 * (1 - 0.90))  # 区間2は区間1のスケールに影響されない
