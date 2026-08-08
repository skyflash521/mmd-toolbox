"""足IK接地安定化の統合テスト。

stabilize_foot_ik は、足IK・つま先IKの密トラック群に対して、左右ペアリング・接地区間検出(対応する
相方があれば相対位置条件を併用)・接地ロック適用を束ね、トラックごとのロック後位置と診断
(接地候補・接地区間・最大変更量・平均変更量・接地ロック適用率・警告)を返す。

入力 tracks は dict: name -> (category, frames, positions)。frames は昇順の絶対フレーム列、positions は
frames に整列した (x,y,z) 列。foot_ik / toe_ik 以外は呼び出し側で除外しておく。第2引数は横滑り抑制
S(0〜1)で、接地検出の水平速度許容・接地ロックの X/Z 強度・最大補正量上限を連動制御する。

ここでは検出・ロックの数値そのものは前段(検出・ロック)のテストに委ね、本テストは束ね方(どの建材を
どう呼び結果へ載せるか)を、建材を直接呼んだ手計算と突き合わせて検証する。建材呼び出しには stabilize と
同じ S 由来パラメータ(水平速度許容・最大補正量)を通す。
"""

import math

import pytest

from mocapvmd import footik, presets

# 既定の横滑り抑制(規定=横滑りなし)。stabilize と建材呼び出しで同じ値を使う。
_S = 1.0
_DET = presets.resolve_foot_detection(_S)


def _detect(positions, paired_positions=None):
    return footik.detect_grounding_segments(
        positions, paired_positions=paired_positions, horiz_vel_thresh=_DET["horiz_vel_thresh"]
    )


def _lock(positions, segments, category):
    return footik.apply_foot_lock(
        positions, segments, presets.resolve_foot_lock(_S, category),
        max_displacement=_DET["max_displacement"],
    )


def _still_then_move(n_still=8, n_move=4):
    # 緩いドリフトで接地し、その後 S=1 の水平速度許容(1.0)を超える速さで動いて遊脚になる foot トラック。
    drift = [0.0, 0.03, 0.06, 0.03, 0.0, 0.03, 0.06, 0.03][:n_still]
    base = drift[-1] if drift else 0.0
    xs = drift + [base + 1.5 * (i + 1) for i in range(n_move)]
    return [(x, 0.0, 0.0) for x in xs]


def _euclid(a, b):
    return math.dist(a, b)


def test_unpaired_track_matches_building_blocks():
    # 相方の無い foot トラックは、相対条件なしの検出 + ロックを束ねた結果と一致する。
    pos = _still_then_move()
    frames = list(range(len(pos)))
    result = footik.stabilize_foot_ik({"右足ＩＫ": ("foot_ik", frames, pos)}, _S)
    ts = result["右足ＩＫ"]

    det = _detect(pos)
    locked, locks = _lock(pos, det.segments, "foot_ik")

    assert ts.category == "foot_ik"
    assert ts.side == "right"
    assert ts.paired is False
    assert ts.grounding == det
    assert tuple(ts.locked_positions) == tuple(locked)
    assert tuple(ts.locks) == tuple(locks)


def test_paired_foot_uses_toe_as_relative_reference():
    # 同側に足IKとつま先IKがあると、足IKの検出は相方つま先IKを相対位置参照に使う(paired=True)。
    foot = [(0.0, 0.0, 0.0)] * 12
    # つま先はゆるいドリフト(接地区間内でアンカーと差が出る→種別別強度が効く)+ frame6 スパイク
    # (足IKとの相対差分が急変)。
    toe_x = [1.0, 1.03, 1.06, 1.03, 1.0, 1.03, 1.3, 1.0, 1.03, 1.06, 1.03, 1.0]
    toe = [(x, 0.0, 0.0) for x in toe_x]
    frames = list(range(12))
    result = footik.stabilize_foot_ik(
        {"右足ＩＫ": ("foot_ik", frames, foot), "右つま先ＩＫ": ("toe_ik", frames, toe)},
        _S,
    )
    foot_ts = result["右足ＩＫ"]
    expected_foot = _detect(foot, paired_positions=toe)
    assert foot_ts.paired is True
    assert foot_ts.side == "right"
    assert foot_ts.grounding == expected_foot
    assert 6 in foot_ts.grounding.relative_rejected_frames

    # つま先IKも処理され、足IKを相対参照に使い toe_ik 用強度でロックされる。
    toe_ts = result["右つま先ＩＫ"]
    assert toe_ts.category == "toe_ik"
    assert toe_ts.paired is True
    assert toe_ts.side == "right"
    expected_toe = _detect(toe, paired_positions=foot)
    assert toe_ts.grounding == expected_toe
    toe_locked, _ = _lock(toe, expected_toe.segments, "toe_ik")
    assert tuple(toe_ts.locked_positions) == tuple(toe_locked)


def test_paired_alignment_uses_absolute_frames_both_directions():
    # 足IK frames [0..5]、つま先IK frames [2..7]。重なりは [2..5]。両IKとも相方が覆わない端で欠損が
    # 生じ、相方位置は絶対フレームで整列され、欠けるフレームは None(相対判定スキップ)になる。両方向を検証する。
    # 両トラックとも重なり境界付近で位置が変化する(frame5 で足IK、frame2 でつま先IK)。これにより
    # 絶対フレーム整列(欠損は None)と、位置インデックスでの誤整列とで相対棄却フレームが変わる。
    foot = [(x, 0.0, 0.0) for x in [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]]
    foot_frames = [0, 1, 2, 3, 4, 5]
    toe = [(x, 0.0, 0.0) for x in [1.5, 1.0, 1.0, 1.0, 1.0, 1.0]]
    toe_frames = [2, 3, 4, 5, 6, 7]
    result = footik.stabilize_foot_ik(
        {"右足ＩＫ": ("foot_ik", foot_frames, foot),
         "右つま先ＩＫ": ("toe_ik", toe_frames, toe)},
        _S,
    )
    # 足IK視点: つま先を foot frames [0..5] へ整列。frame 2 未満は None。
    foot_paired = [None, None, (1.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    assert result["右足ＩＫ"].grounding == _detect(foot, paired_positions=foot_paired)
    # つま先視点: 足IKを toe frames [2..7] へ整列。frame 6,7 は None(末尾欠損のスキップを固定)。
    toe_paired = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), None, None]
    assert result["右つま先ＩＫ"].grounding == _detect(toe, paired_positions=toe_paired)


def test_ambiguous_tracks_processed_without_pairing():
    # 同側に足IKが2本(曖昧)→ どのトラックもペアにせず、相対参照なしの単独検出で処理する(paired=False)。
    foot1 = _still_then_move()
    foot2 = _still_then_move()
    # つま先は足IKと別軌跡。もし誤って相方に使えば frame3 で相対差分が急変し棄却が起きる。曖昧なので
    # 使われず棄却は起きない、という形で「相対参照なし」を固定する。
    toe = [(1.0, 0.0, 0.0)] * len(foot1)
    toe[3] = (1.5, 0.0, 0.0)
    frames = list(range(len(foot1)))
    result = footik.stabilize_foot_ik({
        "右足ＩＫ": ("foot_ik", frames, foot1),
        "右足ＩＫ先": ("foot_ik", frames, foot2),
        "右つま先ＩＫ": ("toe_ik", frames, toe),
    }, _S)
    for name in ("右足ＩＫ", "右足ＩＫ先", "右つま先ＩＫ"):
        assert result[name].paired is False
    # 相方を任意に選ばず単独検出していること(相対棄却が起きないこと)を確認する。
    assert result["右足ＩＫ"].grounding == _detect(foot1)
    assert result["右足ＩＫ"].grounding.relative_rejected_frames == frozenset()


def test_diagnostics_change_and_ratio():
    # 最大・平均変更量はロック前後のユークリッド距離、接地ロック適用率は接地区間内フレーム数の割合。
    pos = _still_then_move()
    frames = list(range(len(pos)))
    ts = footik.stabilize_foot_ik({"右足ＩＫ": ("foot_ik", frames, pos)}, _S)["右足ＩＫ"]

    changes = [_euclid(p, q) for p, q in zip(pos, ts.locked_positions, strict=True)]
    in_seg = sum(s.end - s.start + 1 for s in ts.grounding.segments)
    assert ts.max_change == pytest.approx(max(changes))
    assert ts.mean_change == pytest.approx(sum(changes) / len(changes))
    assert ts.lock_applied_ratio == pytest.approx(in_seg / len(pos))


def test_warning_lists_clamped_segments():
    # 長く滑る接地区間。アンカー(中央値)から端が大きく離れ、内側フレームの補正が S=1 の最大補正量
    # 2.0 を超えてクランプされる。警告はクランプされた区間そのものと一致する(件数だけでない)。
    pos = [(round(0.3 * i, 6), 0.0, 0.0) for i in range(24)]  # 全ステップ0.3 で接地・1区間
    frames = list(range(24))
    ts = footik.stabilize_foot_ik({"右足ＩＫ": ("foot_ik", frames, pos)}, _S)["右足ＩＫ"]
    clamped = tuple(s for s in ts.locks if s.clamped)
    assert len(clamped) >= 1
    assert tuple(ts.warnings) == clamped


def test_non_grounding_track_is_unchanged():
    # 終始 S=1 の水平速度許容(1.0)を超えて速く動くトラックは接地区間が無く、位置は不変・適用率0。
    pos = [(1.5 * i, 0.0, 0.0) for i in range(10)]
    frames = list(range(10))
    ts = footik.stabilize_foot_ik({"右足ＩＫ": ("foot_ik", frames, pos)}, _S)["右足ＩＫ"]
    assert ts.grounding.segments == ()
    assert tuple(ts.locked_positions) == tuple(pos)
    assert ts.lock_applied_ratio == pytest.approx(0.0)
    assert ts.max_change == pytest.approx(0.0)
