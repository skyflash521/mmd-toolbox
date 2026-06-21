"""一般ノイズ軽減の検出層のテスト(mocapvmd.md §4.2 / 実装計画 §4.1)。

各ボーンの密サンプル(0始まり相対インデックス)から検出する。位置は各軸スカラー、回転は quaternion。

確定基準(スパイクとアクセントは別の量で判定):
- スパイク候補 = 窓基準からの逸脱が 位置0.3(各軸)/ 回転5度 超(ちょうどは非候補)。位置は窓中央値、
  回転は窓内正規化平均が基準。
- スパイク = 候補かつ単発の速度往復(符号反転)。位置は各軸独立、回転は相対回転ベクトルの向き反転。
- アクセント = フレーム間変化量(速度)が 位置0.3 / 回転5度 を超え、符号反転せず同方向に2本以上連続する
  差分区間。その run が張るフレーム全体(位置は該当軸)を保護。
- 一定速度の継続運動は窓基準からの逸脱が出ないため、アクセントは速度で検出する(逸脱では検出できない)。
- カットは閾値 位置1.0/回転30度、境界の両側(F-1,F)を保護。範囲端も保護。
"""

import math

import pytest

from mocapvmd import denoise

IDENT = (0.0, 0.0, 0.0, 1.0)


def quat_y(deg):
    """Y軸まわり deg 度回転の quaternion (x,y,z,w)。"""
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def still(n, pos=(0.0, 0.0, 0.0)):
    return [tuple(pos) for _ in range(n)]


def idents(n):
    return [IDENT] * n


def detect(positions, rotations, pos_window=5, rot_window=5):
    return denoise.detect_noise_events(
        positions, rotations, pos_window=pos_window, rot_window=rot_window
    )


def _axis(events, axis):
    return {f for (f, a) in events if a == axis}


# --- 退化ケース -------------------------------------------------------------


def test_empty_track():
    r = detect([], [])
    assert r.pos_spikes == set()
    assert r.rot_spikes == set()


@pytest.mark.parametrize("n", [1, 2])
def test_short_track_has_no_spikes(n):
    # 実効窓 < 3 では平滑化・スパイク検出をしない(往復の判定材料も揃わない)。
    r = detect(still(n), idents(n))
    assert r.pos_spikes == set()
    assert r.rot_spikes == set()
    assert r.pos_candidates == set()
    assert r.rot_candidates == set()


def test_all_same_value_no_events():
    r = detect(still(11), idents(11))
    assert r.pos_candidates == set()
    assert r.rot_candidates == set()
    assert r.pos_spikes == set()
    assert r.rot_spikes == set()
    assert r.pos_accent == set()
    assert r.rot_accent == set()


# --- 位置スパイク -----------------------------------------------------------


def test_position_single_axis_spike():
    # フレーム5で X が単発で 0.5 跳ねて戻る(往復)。X軸だけスパイク。
    pos = still(11)
    pos[5] = (0.5, 0.0, 0.0)
    r = detect(pos, idents(11))
    assert (5, 0) in r.pos_candidates
    assert (5, 0) in r.pos_spikes
    assert (5, 1) not in r.pos_spikes
    assert (5, 2) not in r.pos_spikes


def test_position_spike_threshold_exact_is_not_candidate():
    # 窓中央値(=0)からの逸脱がちょうど 0.3 は非候補=非スパイク(strict >)。
    pos = still(11)
    pos[5] = (0.3, 0.0, 0.0)
    r = detect(pos, idents(11))
    assert (5, 0) not in r.pos_candidates
    assert (5, 0) not in r.pos_spikes


def test_position_spike_threshold_over_is_spike():
    pos = still(11)
    pos[5] = (0.31, 0.0, 0.0)
    r = detect(pos, idents(11))
    assert (5, 0) in r.pos_spikes


def test_no_return_is_not_spike():
    # 跳ねたまま戻らない(往復でない)は候補ではあってもスパイクにしない。
    pos = still(11)
    pos[5] = (0.5, 0.0, 0.0)
    pos[6] = (0.5, 0.0, 0.0)  # 戻らない → outgoing 差分が0、往復不成立
    r = detect(pos, idents(11))
    assert (5, 0) not in r.pos_spikes


# --- 位置アクセント(速度ベース・保護) ------------------------------------


def test_position_sustained_run_is_accent():
    # 静止→frames4-7で+0.5/frame(速度0.5>0.3を同方向継続)→静止。run が張る frame3-7 を保護。
    xs = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0]
    pos = [(x, 0.0, 0.0) for x in xs]
    r = detect(pos, idents(11))
    assert _axis(r.pos_accent, 0) == {3, 4, 5, 6, 7}
    # 一定速度の継続は窓中央値=注目値で逸脱が出ず、スパイク候補にはならない。
    assert _axis(r.pos_candidates, 0) == set()
    assert _axis(r.pos_spikes, 0) == set()


def test_accent_priority_over_spike_at_run_peak():
    # 三角波の山頂(同方向run終端かつ窓逸脱+往復)はアクセント優先で保護し、スパイク補正しない。
    xs = [0.0, 0.0, 0.5, 1.0, 1.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    pos = [(x, 0.0, 0.0) for x in xs]
    r = detect(pos, idents(11))
    # 山頂 frame4 は窓中央値から逸脱した候補だが、両側の同方向 run に属するので保護されスパイクにしない。
    assert (4, 0) in r.pos_candidates
    assert (4, 0) in r.pos_accent
    assert (4, 0) not in r.pos_spikes
    # 正の run(frames1-4)と負の run(frames4-7)の和を保護。
    assert _axis(r.pos_accent, 0) == {1, 2, 3, 4, 5, 6, 7}
    assert _axis(r.pos_spikes, 0) == set()


def test_axis_independence_spike_and_accent():
    # X は frame5 単発往復スパイク、Y は frames4-7 の同方向継続アクセント。
    ys = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0]
    pos = [(0.0, y, 0.0) for y in ys]
    pos[5] = (0.6, pos[5][1], 0.0)
    r = detect(pos, idents(11))
    assert (5, 0) in r.pos_spikes        # X スパイク
    assert (5, 0) not in r.pos_accent
    assert (5, 1) in r.pos_accent        # Y アクセント
    assert (5, 1) not in r.pos_spikes


# --- カット・範囲端 ---------------------------------------------------------


def test_range_edges_protected():
    r = detect(still(11), idents(11))
    assert 0 in r.boundaries
    assert 10 in r.boundaries


def test_cut_protects_both_sides():
    # フレーム6で位置が 2.0 ジャンプ(カット閾値1.0超)。境界の両側 5,6 を保護。
    pos = [(0.0, 0.0, 0.0)] * 6 + [(2.0, 0.0, 0.0)] * 5
    r = detect(pos, idents(11))
    assert 5 in r.boundaries
    assert 6 in r.boundaries


def test_apparent_spike_at_cut_not_corrected():
    # カット直後の大きな差はカット境界として保護され、スパイク補正しない。
    pos = [(0.0, 0.0, 0.0)] * 6 + [(2.0, 0.0, 0.0)] * 5
    r = detect(pos, idents(11))
    assert (6, 0) not in r.pos_spikes


def test_cuts_reported_separately_from_boundaries():
    # 検出カットフレームを boundaries(両側に潰した集合)とは別に保持する(§4.4 検出カット数用)。
    pos = [(0.0, 0.0, 0.0)] * 6 + [(2.0, 0.0, 0.0)] * 5
    r = detect(pos, idents(11))
    assert r.cuts == {6}


# --- 回転スパイク -----------------------------------------------------------


def test_rotation_single_spike():
    # フレーム5で 10度跳ねて戻る(往復)。回転スパイク。
    rots = idents(11)
    rots[5] = quat_y(10.0)
    r = detect(still(11), rots)
    assert 5 in r.rot_candidates
    assert 5 in r.rot_spikes


def test_small_rotation_jump_is_not_spike():
    # 3度の単発跳ねは窓内平均からの逸脱が5度以下で非候補=非スパイク。
    rots = idents(11)
    rots[5] = quat_y(3.0)
    r = detect(still(11), rots)
    assert 5 not in r.rot_candidates
    assert 5 not in r.rot_spikes


def test_rotation_sign_flip_is_not_spike():
    # q と -q は同一姿勢。符号反転を見かけのスパイクにしない。
    rots = idents(11)
    rots[5] = (0.0, 0.0, 0.0, -1.0)
    r = detect(still(11), rots)
    assert 5 not in r.rot_candidates
    assert 5 not in r.rot_spikes


def test_non_unit_quaternion_is_normalized_before_detection():
    # 非単位 quaternion(一律2倍)でも内部で正規化され、10度の単発スパイクを検出できる。
    scale = lambda q: tuple(2.0 * c for c in q)  # noqa: E731
    rots = [scale(IDENT)] * 11
    rots[5] = scale(quat_y(10.0))
    r = detect(still(11), rots)
    assert 5 in r.rot_spikes


def test_rotation_sustained_run_is_accent():
    # frames4-7で+8度/frame(速度8度>5度を同方向継続)。run が張る frame3-7 を保護。
    degs = [0.0, 0.0, 0.0, 0.0, 8.0, 16.0, 24.0, 32.0, 32.0, 32.0, 32.0]
    rots = [quat_y(d) for d in degs]
    r = detect(still(11), rots)
    assert r.rot_accent == {3, 4, 5, 6, 7}
    assert r.rot_spikes == set()


# --- 入力検証 ---------------------------------------------------------------


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        detect(still(5), idents(4))


def test_non_finite_position_raises():
    pos = still(11)
    pos[5] = (float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError):
        detect(pos, idents(11))


def test_infinite_position_raises():
    pos = still(11)
    pos[5] = (float("inf"), 0.0, 0.0)
    with pytest.raises(ValueError):
        detect(pos, idents(11))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_quaternion_raises(bad):
    rots = idents(11)
    rots[5] = (bad, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        detect(still(11), rots)


def test_zero_norm_quaternion_raises():
    rots = idents(11)
    rots[5] = (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        detect(still(11), rots)


@pytest.mark.parametrize("pw,rw", [(4, 5), (5, 4)])
def test_even_window_raises(pw, rw):
    with pytest.raises(ValueError):
        denoise.detect_noise_events(still(11), idents(11), pos_window=pw, rot_window=rw)


@pytest.mark.parametrize("pw,rw", [(-3, 5), (5, -3), (0, 5)])
def test_non_positive_window_raises(pw, rw):
    # 負の奇数やゼロは検出を黙って無効化しないよう引数エラーにする。
    with pytest.raises(ValueError):
        denoise.detect_noise_events(still(11), idents(11), pos_window=pw, rot_window=rw)
