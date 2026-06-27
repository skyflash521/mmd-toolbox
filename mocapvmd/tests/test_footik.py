"""足IK接地区間検出の単独トラックテスト(mocapvmd.md §4.3)。

本テストの範囲は、左右ペアリング・相対位置急変条件を含まない単一トラックの接地区間検出に限る。
密サンプル(0始まり相対インデックス、各要素 (x,y,z))から、速度・Y局所最小・継続長で接地区間を求める。

確定基準(mocapvmd.md §4.3。本テストは低レベル detect_grounding_segments の既定値で検証する):
- 速度が小さい: 隣接フレーム差の水平速度 sqrt(dx^2+dz^2) <= 0.08(§4.3 の水平速度許容は横滑り抑制 S 連動で、
  0.08 は S=0 相当の基準端=低レベル既定。CLI 既定 S=1.0 では 1.0)かつ 垂直速度 |dy| <= 0.04(ちょうどは接地)。
  フレーム f は、隣接ステップの少なくとも一方が「遅い」とき velocity_ok とする(接地開始・終了の端フレームを含めるため)。
- Y が局所的に低い: Y[f] <= (±5 フレーム窓内の局所最小) + 0.08(ちょうどは接地)。
- 一定フレーム以上継続: candidate フレームの連続 run のうち長さ >= 4 を接地区間とする。
- candidate_frames は速度・Y を満たすフレーム集合(接地候補)、segments は長さ4以上の接地区間。
"""

import pytest

from mocapvmd import footik


def detect(positions, **kw):
    return footik.detect_grounding_segments(positions, **kw)


def segs(det):
    return [(s.start, s.end) for s in det.segments]


# --- 退化ケース -------------------------------------------------------------


def test_empty_track():
    det = detect([])
    assert det.segments == ()
    assert det.candidate_frames == frozenset()


@pytest.mark.parametrize("n", [1, 2, 3])
def test_too_short_still_track_has_no_segment(n):
    # 完全静止でも継続長が最小接地長(4)未満なら接地区間にしない。
    det = detect([(0.0, 0.0, 0.0)] * n)
    assert det.segments == ()


# --- 基本の静止接地 ---------------------------------------------------------


def test_full_still_low_track_is_one_segment():
    det = detect([(0.0, 0.0, 0.0)] * 8)
    assert segs(det) == [(0, 7)]
    assert det.candidate_frames == frozenset(range(8))


def test_min_ground_length_boundary():
    # 速い移動に挟まれた静止ブロック。長さ3は接地区間にならず、4で初めて接地区間になる。
    def build(block):
        rows = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]  # 0.5 ステップ(速い)で進入
        rows += [(1.0, 0.0, 0.0)] * block            # 静止ブロック
        rows += [(1.5, 0.0, 0.0), (2.0, 0.0, 0.0)]   # 0.5 ステップ(速い)で離脱
        return rows

    # 長さ3は接地区間にならないが、接地候補フレーム(§4.4)は区間化と独立に保持される。
    det3 = detect(build(3))
    assert det3.segments == ()
    assert det3.candidate_frames == frozenset({2, 3, 4})
    assert segs(detect(build(4))) == [(2, 5)]


# --- 速度しきい値 -----------------------------------------------------------


def test_horizontal_velocity_below_threshold_is_grounded():
    # 各ステップ 0.079(0.08直下)で一方向に滑る。全フレームが接地候補で1区間になる。
    # 0.08 直下・直上を約0.001 だけ離して挟み、閾値が 0.075 等でなく 0.08 であることを固定する
    # (差は浮動小数の丸め誤差より十分大きく、sqrt 比較・二乗比較のどちらでも安定)。
    det = detect([(round(0.079 * i, 6), 0.0, 0.0) for i in range(8)])
    assert det.candidate_frames == frozenset(range(8))
    assert segs(det) == [(0, 7)]


def test_horizontal_velocity_above_threshold_is_not_grounded():
    # 各ステップ 0.081(0.08直上)。接地候補も接地区間も生じない。
    det = detect([(round(0.081 * i, 6), 0.0, 0.0) for i in range(8)])
    assert det.candidate_frames == frozenset()
    assert det.segments == ()


def test_horizontal_velocity_is_euclidean():
    # X・Z 同時に進み、合成水平速度 sqrt(dx^2+dz^2) で判定される(軸別判定なら誤接地)。
    # v=0.05→合成約0.0707(接地)、v=0.06→合成約0.0849(非接地)。各軸単独では 0.06 も 0.08 未満。
    # 0.08 ちょうどの境界は sqrt の丸めで結果が揺れるため避け、明確に内側・外側の値で検証する。
    g = detect([(round(0.05 * i, 6), 0.0, round(0.05 * i, 6)) for i in range(8)])
    assert g.candidate_frames == frozenset(range(8))
    assert segs(g) == [(0, 7)]
    ng = detect([(round(0.06 * i, 6), 0.0, round(0.06 * i, 6)) for i in range(8)])
    assert ng.candidate_frames == frozenset()
    assert ng.segments == ()


def test_vertical_velocity_threshold():
    # Y が 0 と v を往復(|dy|=v)。垂直速度は abs(dy) で sqrt を経ないため 0.04 ちょうども安定に接地し、
    # 直上の 0.041 は非接地。0.04 と 0.041 で挟み、閾値が 0.045 等でなく 0.04 であることを固定する。
    g = detect([(0.0, 0.04 if i % 2 else 0.0, 0.0) for i in range(8)])
    assert g.candidate_frames == frozenset(range(8))
    assert segs(g) == [(0, 7)]
    ng = detect([(0.0, 0.041 if i % 2 else 0.0, 0.0) for i in range(8)])
    assert ng.candidate_frames == frozenset()
    assert ng.segments == ()


# --- Y 局所最小条件 ---------------------------------------------------------


def test_y_tolerance_boundary_is_local_min_plus_0_08():
    # 速度は常に遅い(各 |dy|<=0.04)谷。局所最小は 0.00。許容幅が 局所最小+0.08 であることを、
    # 0.08 直下の 0.079(接地に含む)と直上の 0.081(除外)で挟んで固定する。
    # frame0=0.081 は除外、frame10=0.079 は接地候補に残る。許容幅が 0.05 や 0.10 の誤実装を弾く。
    ys = [0.081, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.05, 0.079]
    det = detect([(0.0, y, 0.0) for y in ys])
    assert det.candidate_frames == frozenset(range(1, 11))
    assert segs(det) == [(1, 10)]


def test_y_local_window_radius_is_5():
    # 高い平坦(Y=0.20)に1点だけ低い谷(frame7, Y=0.00)。平坦フレームは静止で速度は遅い。
    # 谷から距離<=5 のフレームは局所最小0で除外、距離6以上は局所最小0.20で接地候補に残る。
    # これにより窓半径が ±4 でも ±6 でも全体最小でもない、ちょうど ±5 であることを固定する。
    ys = [0.20] * 15
    ys[7] = 0.00
    det = detect([(0.0, y, 0.0) for y in ys])
    # 距離6の frame1,13 は残り、距離5の frame2,12 は除外される。残る候補は短く接地区間にならない。
    assert det.candidate_frames == frozenset({0, 1, 13, 14})
    assert det.segments == ()


# --- 遊脚で2区間に分かれる --------------------------------------------------


def test_stance_swing_stance_splits_into_two_segments():
    rows = [(0.0, 0.0, 0.0)] * 6                       # 接地1(0-5)
    rows += [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
             (1.4, 0.0, 0.0), (1.7, 0.0, 0.0)]         # 遊脚(6-9)各ステップ>0.08
    rows += [(2.0, 0.0, 0.0)] * 6                       # 接地2(10-15)
    det = detect(rows)
    assert segs(det) == [(0, 5), (10, 15)]
