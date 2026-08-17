"""bake_bone_track(ボーントラック密ベイク)のテスト。

密キーでの端値一致、不等間隔トラック全フレームでの一致、ファストパス2種、
境界(単一キー・値ジャンプ・範囲端)を検証する。評価意味論の基準は同一トラックへの
interp.sample(境界規約・到達側キーの補間曲線・回転の正規化)との一致に置く。
"""

import numpy as np
import pytest

from vmd import interp
from vmd.types import BoneKey

pytestmark = pytest.mark.xfail(reason="impl pending: vmd.interp ベイクAPI", strict=True)

# 制御点 (x1,y1,x2,y2)。LINEAR は MMD デフォルトの線形。
LINEAR = (20, 20, 107, 107)


def bone_interp(x=LINEAR, y=LINEAR, z=LINEAR, r=LINEAR) -> bytes:
    """チャンネル別制御点 (x1,y1,x2,y2) からボーン補間ブロック64バイトを組む。

    types.control_points() の読み出し位置に対応: X=(b0,b4,b8,b12),
    Y=(b1,b5,b9,b13), Z=(b17,b6,b10,b14), R=(b18,b7,b11,b15)。
    """
    b = bytearray(64)
    b[0], b[4], b[8], b[12] = x
    b[1], b[5], b[9], b[13] = y
    b[17], b[6], b[10], b[14] = z
    b[18], b[7], b[11], b[15] = r
    return bytes(b)


def bkey(frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), curves=None):
    return BoneKey(b"\x00" * 15, frame, tuple(pos), tuple(rot),
                   bone_interp(**(curves or {})))


def assert_matches_sample(keys, frame_start, frame_end, abs_tol=1e-8):
    """[frame_start, frame_end] の全フレームで bake と sample の一致を検証する。"""
    positions, rotations = interp.bake_bone_track(keys, frame_start, frame_end)
    n = frame_end - frame_start + 1
    assert len(positions) == n
    assert len(rotations) == n
    for i, f in enumerate(range(frame_start, frame_end + 1)):
        expected_pos = (interp.sample(keys, "pos_x", f),
                        interp.sample(keys, "pos_y", f),
                        interp.sample(keys, "pos_z", f))
        assert tuple(positions[i]) == pytest.approx(expected_pos, abs=abs_tol), f"pos at {f}"
        expected_rot = np.array(interp.sample(keys, "rot", f), dtype=float)
        got_rot = np.array(rotations[i], dtype=float)
        if float(np.dot(got_rot, expected_rot)) < 0:  # クォータニオン符号の自由度を吸収
            expected_rot = -expected_rot
        assert got_rot == pytest.approx(expected_rot, abs=abs_tol), f"rot at {f}"


# ---------------------------------------------------------------------------
# テスト要件7: gap=1 の密キー列で位置は厳密一致・回転は正規化差を除き一致
# ---------------------------------------------------------------------------


class TestDenseIdentity:
    def test_positions_exact_on_dense_keys(self):
        keys = [bkey(f, pos=(f * 1.5, -float(f), f * f * 0.25)) for f in range(6)]
        positions, _ = interp.bake_bone_track(keys, 0, 5)
        assert len(positions) == 6
        for f in range(6):
            assert tuple(positions[f]) == keys[f].position  # 厳密一致

    def test_rotations_exact_with_unit_norm_data(self):
        # ノルムが浮動小数で厳密に1になるクォータニオンのみ使う
        quats = [(0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 0.5),
                 (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)]
        keys = [bkey(f, rot=quats[f]) for f in range(4)]
        _, rotations = interp.bake_bone_track(keys, 0, 3)
        for f in range(4):
            assert tuple(rotations[f]) == quats[f]  # 単位ノルムのデータでは厳密一致

    def test_rotations_normalized_for_non_unit_data(self):
        # 非単位ノルムの格納値は正規化されて返り、sample と一致する
        keys = [bkey(0, rot=(0.0, 0.0, 0.0, 2.0)), bkey(1, rot=(0.0, 0.0, 2.0, 0.0))]
        _, rotations = interp.bake_bone_track(keys, 0, 1)
        for f in (0, 1):
            assert np.linalg.norm(rotations[f]) == pytest.approx(1.0, abs=1e-12)
        assert_matches_sample(keys, 0, 1, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# テスト要件8: 不等間隔トラックの全フレームで sample と一致(線形・非線形)
# ---------------------------------------------------------------------------

# 同一トラック内に gap 1・2・10・60・2 を混在させるフレーム列
MIXED_FRAMES = [0, 1, 3, 13, 73, 75]


def _mixed_keys(curves_list):
    """フレームごとに位置・回転を変えた不等間隔トラックを組む。"""
    keys = []
    for i, f in enumerate(MIXED_FRAMES):
        c = curves_list[i % len(curves_list)]
        q = (0.0, 0.0, np.sin(0.2 * i), np.cos(0.2 * i))
        keys.append(bkey(f, pos=(f * 0.1, 5.0 - i, (i % 3) * 2.0), rot=q, curves=c))
    return keys


class TestUnequalGaps:
    def test_all_frames_match_sample_linear(self):
        keys = _mixed_keys([None])
        assert_matches_sample(keys, MIXED_FRAMES[0], MIXED_FRAMES[-1])

    def test_all_frames_match_sample_nonlinear(self):
        # チャンネルごとに異なる非線形曲線(取り違えが値差として現れる)。
        # 許容差は 1e-6: px1=0 の曲線は区間始端で dX/ds=0 となり、ソルバの
        # 収束条件 |X(s)-x| < 1e-9 を満たす別実装どうしでも勾配増幅×値スパンで
        # 1e-8 を超える差が許容内に生じうるため、係数比較より緩くとる。
        curves = [
            {"x": (0, 127, 127, 0), "y": (5, 122, 122, 5), "z": (0, 64, 127, 64), "r": (10, 117, 117, 10)},
            {"x": (64, 0, 127, 64), "y": (0, 0, 127, 127), "z": (40, 10, 80, 120), "r": (5, 122, 122, 5)},
        ]
        keys = _mixed_keys(curves)
        assert_matches_sample(keys, MIXED_FRAMES[0], MIXED_FRAMES[-1], abs_tol=1e-6)


# ---------------------------------------------------------------------------
# テスト要件9: 線形ファストパスと一般解(ニュートン法)の一致
# ---------------------------------------------------------------------------


class TestLinearFastpath:
    @pytest.mark.parametrize("cp", [LINEAR, (30, 30, 90, 90), (0, 0, 127, 127)])
    def test_fastpath_matches_newton(self, cp):
        # X側と Y側の制御点が一致する曲線は y=x。値スパンを1にして
        # 補間係数の差がそのまま位置の差になる形で比較する(許容差 1e-8)。
        keys = [bkey(0, pos=(0.0, 0.0, 0.0)),
                bkey(16, pos=(1.0, 1.0, 1.0), curves={"x": cp, "y": cp, "z": cp})]
        positions, _ = interp.bake_bone_track(keys, 0, 16)
        for f in range(17):
            expected = interp.sample(keys, "pos_x", f)  # 一般解(ニュートン法)の経路
            assert positions[f][0] == pytest.approx(expected, abs=1e-8)
            # ファストパス対象の曲線では解析解 y=x とも一致する
            assert positions[f][0] == pytest.approx(f / 16.0, abs=1e-8)


# ---------------------------------------------------------------------------
# 定数ファストパス: 区間両端の値が一致するチャンネルは端値の評価値
# ---------------------------------------------------------------------------


class TestConstantFastpath:
    def test_constant_channel_with_nonlinear_curve(self):
        # 両端の値が一致するチャンネルは曲線によらず端値(x のみ変化する)
        cp = (0, 127, 127, 0)
        keys = [bkey(0, pos=(0.0, 4.0, -2.0)),
                bkey(10, pos=(10.0, 4.0, -2.0),
                     curves={"x": cp, "y": cp, "z": cp})]
        positions, _ = interp.bake_bone_track(keys, 0, 10)
        for f in range(11):
            assert positions[f][1] == 4.0
            assert positions[f][2] == -2.0

    def test_constant_rotation_returns_normalized_value(self):
        # 両端で一致する非単位ノルムの回転は、正規化後の端値で充填される
        q = (0.0, 0.0, 0.0, 2.0)
        keys = [bkey(0, rot=q), bkey(10, rot=q, curves={"r": (0, 127, 127, 0)})]
        _, rotations = interp.bake_bone_track(keys, 0, 10)
        for f in range(11):
            assert tuple(rotations[f]) == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)


# ---------------------------------------------------------------------------
# テスト要件10: 境界(単一キー・フレーム差1の値ジャンプ・範囲端)
# ---------------------------------------------------------------------------


class TestBoundaries:
    def test_single_key_constant_fill(self):
        keys = [bkey(7, pos=(1.0, 2.0, 3.0), rot=(0.5, 0.5, 0.5, 0.5))]
        positions, rotations = interp.bake_bone_track(keys, 0, 10)
        assert len(positions) == 11
        for i in range(11):
            assert tuple(positions[i]) == (1.0, 2.0, 3.0)
            assert tuple(rotations[i]) == (0.5, 0.5, 0.5, 0.5)

    def test_gap1_value_jump_preserved(self):
        # フレーム差1の区間は中間点が存在せず、両端キーの値がそのまま並ぶ
        cp = (0, 127, 127, 0)
        keys = [bkey(10, pos=(0.0, 0.0, 0.0)),
                bkey(11, pos=(5.0, -5.0, 1.0), curves={"x": cp, "y": cp, "z": cp})]
        positions, _ = interp.bake_bone_track(keys, 10, 11)
        assert tuple(positions[0]) == (0.0, 0.0, 0.0)
        assert tuple(positions[1]) == (5.0, -5.0, 1.0)

    def test_range_extends_beyond_keys(self):
        # 範囲端: 最初のキー以前・最後のキー以後は端キーの値で一定
        keys = [bkey(5, pos=(1.0, 0.0, 0.0)), bkey(8, pos=(4.0, 0.0, 0.0))]
        assert_matches_sample(keys, 0, 12)

    def test_partial_range_inside_segment(self):
        # 区間の途中から始まる範囲でも sample と一致する
        keys = [bkey(0, pos=(0.0, 0.0, 0.0)),
                bkey(20, pos=(10.0, 0.0, 0.0), curves={"x": (5, 122, 122, 5)})]
        assert_matches_sample(keys, 6, 13)
