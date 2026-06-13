"""vmd-camera のテスト(vmd-camera.md §6 テスト要件1〜4)。

要件4(独立実装との数値クロスバリデーション)の基準として、§2 の式を別実装で
書き直した順次回転適用版 _fk を末尾近くに置く——回転を1軸ずつ順に適用する素朴な
実装で、読めば正しさが確認できる「テストの底」として機能する。

絶対的なMMD一致(規約そのものの正しさ)は §4 の視覚A/Bスモークが最終ゲート。
本ファイルのテストは to_world/from_world の自己整合性と、式の実装ミス検出を担う。
"""

import math

import numpy as np
import pytest

from mmd_toolbox.vmd import camera, normalize, read
from mmd_toolbox.vmd.types import CameraKey

# ---------------------------------------------------------------------------
# 独立実装(クロスバリデーションの基準): §2 の式を順次回転で書き直す
# ---------------------------------------------------------------------------


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _matrix(rx, ry, rz):
    # R = Ry(-ry) · Rx(-rx) · Rz(-rz)(vmd-camera.md §2)
    return _ry(-ry) @ _rx(-rx) @ _rz(-rz)


def _fk(center, distance, rotation):
    """順次回転で前方運動学を計算する独立実装。"""
    rx, ry, rz = rotation
    # (0,0,distance) に Rz(-rz) → Rx(-rx) → Ry(-ry) を順に適用
    v = np.array([0.0, 0.0, distance])
    v = _rz(-rz) @ v
    v = _rx(-rx) @ v
    v = _ry(-ry) @ v
    pos = np.array(center, dtype=float) + v
    R = _matrix(rx, ry, rz)
    forward = R @ np.array([0.0, 0.0, 1.0])
    up = R @ np.array([0.0, 1.0, 0.0])
    return pos, forward, up


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def cam_key(center=(0.0, 0.0, 0.0), distance=-30.0, rotation=(0.0, 0.0, 0.0),
            fov=30, perspective=0):
    return CameraKey(0, distance, center, rotation, bytes(24), fov, perspective)


def _approx(a, b, abs=1e-6):
    return np.asarray(a) == pytest.approx(np.asarray(b), abs=abs)


# (center, distance, rotation) のテストグリッド(ジンバル近傍は別テスト)
GRID = [
    ((0.0, 0.0, 0.0), -30.0, (0.0, 0.0, 0.0)),
    ((1.0, 2.0, 3.0), -45.0, (0.3, 0.0, 0.0)),
    ((-5.0, 10.0, 2.0), -20.0, (0.0, 0.7, 0.0)),
    ((0.0, 8.0, 0.0), -50.0, (0.0, 0.0, 0.5)),
    ((3.0, -2.0, 7.0), -15.0, (0.2, -0.6, 0.4)),
    ((0.0, 0.0, 0.0), -100.0, (-0.4, 1.2, -0.3)),
]


# ---------------------------------------------------------------------------
# テスト4 + 基盤: to_world が §2 の式(独立実装)と一致
# ---------------------------------------------------------------------------


class TestToWorldMatchesReference:
    @pytest.mark.parametrize("center,distance,rotation", GRID)
    def test_synthetic(self, center, distance, rotation):
        pose = camera.to_world(cam_key(center, distance, rotation))
        pos, fwd, up = _fk(center, distance, rotation)
        assert _approx(pose.position, pos)
        assert _approx(pose.forward, fwd)
        assert _approx(pose.up, up)

    def test_forward_points_toward_center(self):
        # distance<0 のとき前方軸はカメラ位置→カメラ中心の向きと一致(§2)
        center = np.array([1.0, 2.0, 3.0])
        key = cam_key(tuple(center), -25.0, (0.3, 0.5, 0.0))
        pose = camera.to_world(key)
        to_center = center - np.array(pose.position)
        to_center /= np.linalg.norm(to_center)
        assert _approx(pose.forward, to_center, abs=1e-6)

    def test_real_file(self, camera_basic_bytes):
        doc, _ = read(camera_basic_bytes)
        doc, _ = normalize(doc, ["camera"])
        for k in doc.camera:
            pose = camera.to_world(k)
            pos, fwd, up = _fk(k.position, k.distance, k.rotation)
            assert _approx(pose.position, pos, abs=1e-4)
            assert _approx(pose.forward, fwd, abs=1e-6)
            assert pose.fov == k.fov
            assert pose.perspective == k.perspective


# ---------------------------------------------------------------------------
# テスト1: 往復一致 to_world → from_world
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize("center,distance,rotation", GRID)
    def test_recovers_center_and_rotation(self, center, distance, rotation):
        pose = camera.to_world(cam_key(center, distance, rotation))
        got = camera.from_world(pose, distance, prev_rotation=rotation)
        assert _approx(got["position"], center, abs=1e-5)
        # 角度は prev_rotation 付きでアンラップされ、元の角度に一致する
        assert _approx(got["rotation"], rotation, abs=1e-5)

    def test_recovered_orientation_matches(self):
        # 角度の表現に依らず、復元した姿勢(前方・上)は元と一致する
        center, distance, rotation = (2.0, 1.0, -3.0), -40.0, (0.5, -0.8, 0.6)
        pose = camera.to_world(cam_key(center, distance, rotation))
        got = camera.from_world(pose, distance, prev_rotation=rotation)
        pos2, fwd2, up2 = _fk(got["position"], distance, got["rotation"])
        assert _approx(pose.position, pos2, abs=1e-5)
        assert _approx(pose.forward, fwd2, abs=1e-5)
        assert _approx(pose.up, up2, abs=1e-5)


# ---------------------------------------------------------------------------
# テスト: distance = 0(カメラ位置=カメラ中心)
# ---------------------------------------------------------------------------


class TestDistanceZero:
    def test_position_equals_center(self):
        center = (1.0, 2.0, 3.0)
        pose = camera.to_world(cam_key(center, 0.0, (0.3, 0.5, 0.7)))
        assert _approx(pose.position, center, abs=1e-9)

    def test_from_world_recovers_center(self):
        center = (4.0, -1.0, 2.0)
        rotation = (0.2, 0.4, -0.1)
        pose = camera.to_world(cam_key(center, 0.0, rotation))
        got = camera.from_world(pose, 0.0, prev_rotation=rotation)
        assert _approx(got["position"], center, abs=1e-9)


# ---------------------------------------------------------------------------
# テスト2: 角度連続性(逆変換で±180°ジャンプを起こさない)
# ---------------------------------------------------------------------------


class TestAngleContinuity:
    def test_yaw_sweep_across_pi_is_continuous(self):
        # ヨーを π をまたいで滑らかに動かす。生の atan2 は±2πの跳びを生むが、
        # prev_rotation 付きの from_world は連続な系列を返すべき(§5)。
        prev = None
        recovered = []
        for ry in np.linspace(2.9, 3.4, 12):  # π≈3.14159 をまたぐ
            pose = camera.to_world(cam_key((0.0, 0.0, 0.0), -30.0, (0.0, float(ry), 0.0)))
            got = camera.from_world(pose, -30.0, prev_rotation=prev)
            recovered.append(got["rotation"][1])
            prev = got["rotation"]
        diffs = np.diff(recovered)
        assert np.all(np.abs(diffs) < math.pi), f"不連続なジャンプ: {diffs}"


# ---------------------------------------------------------------------------
# テスト3: ジンバル近傍(ピッチ±90°付近)の安定性
# ---------------------------------------------------------------------------


class TestGimbal:
    @pytest.mark.parametrize("pitch", [math.pi / 2 - 1e-4, -math.pi / 2 + 1e-4,
                                       math.pi / 2, -math.pi / 2])
    def test_roundtrip_orientation_stable_near_gimbal(self, pitch):
        center, distance, rotation = (1.0, 0.0, 0.0), -30.0, (pitch, 0.6, 0.3)
        pose = camera.to_world(cam_key(center, distance, rotation))
        got = camera.from_world(pose, distance, prev_rotation=rotation)
        # ジンバル位置ではオイラー角は一意でないため、姿勢(前方・上)と中心で検証
        pos2, fwd2, up2 = _fk(got["position"], distance, got["rotation"])
        assert _approx(pose.position, pos2, abs=1e-4)
        assert _approx(pose.forward, fwd2, abs=1e-4)
        assert _approx(pose.up, up2, abs=1e-4)
