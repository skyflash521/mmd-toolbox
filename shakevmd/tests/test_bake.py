"""bake の視線揺れ変換のテスト(shakevmd.md §4.2)。

サブステップA: apply_gaze_shake のみ。ベイクループ本体 bake() は後続サブステップ。
§4.2 の実現は「揺れ角度 = 元角度 + ノイズ(オイラー加算)、カメラ位置固定になるよう
中心を逆算」。距離0では素朴な角度加算と一致する。
"""

import numpy as np
import pytest

from shakevmd import bake
from mmd_toolbox.vmd import camera
from mmd_toolbox.vmd.types import CameraKey


def cam(distance=-30.0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), fov=30, perspective=0):
    return CameraKey(0, distance, center, rotation, bytes(24), fov, perspective)


def rebuilt(key, res):
    """apply_gaze_shake の結果(中心・角度)から CameraKey を再構成する。"""
    return CameraKey(0, key.distance, res["position"], res["rotation"],
                     bytes(24), key.fov, key.perspective)


class TestApplyGazeShake:
    def test_zero_noise_is_identity(self):
        key = cam(distance=-30.0, center=(1.0, 2.0, 3.0), rotation=(0.1, 0.2, 0.3))
        res = bake.apply_gaze_shake(key, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert res["position"] == pytest.approx(key.position, abs=1e-6)
        assert res["rotation"] == pytest.approx(key.rotation, abs=1e-6)

    def test_rotation_is_naive_euler_addition(self):
        # 揺れ角度は成分ごとのオイラー加算(順序・符号・軸の取り違えを完全に固定)
        key = cam(distance=-30.0, center=(1.0, 2.0, 3.0), rotation=(0.1, -0.2, 0.3))
        noise = (0.05, 0.07, -0.04)
        res = bake.apply_gaze_shake(key, noise, (0.0, 0.0, 0.0))
        expected = tuple(key.rotation[i] + noise[i] for i in range(3))
        assert res["rotation"] == pytest.approx(expected, abs=1e-9)

    def test_rotation_addition_does_not_wrap(self):
        # 揺れ角度は raw sum(元角度+ノイズ)。π をまたいでもラップ・正規化しない。
        key = cam(distance=-30.0, center=(0.0, 0.0, 0.0), rotation=(0.0, 3.10, 0.0))
        noise = (0.0, 0.1, 0.0)  # ry: 3.10 + 0.1 = 3.20 > π(≈3.14159)
        res = bake.apply_gaze_shake(key, noise, (0.0, 0.0, 0.0))
        assert res["rotation"][1] == pytest.approx(3.20, abs=1e-9)  # 3.20-2π にならない

    def test_rotation_only_keeps_camera_world_position(self):
        # 回転ノイズのみ → カメラワールド位置は不変(視線だけ揺れる。§4.2)
        key = cam(distance=-30.0, center=(1.0, 2.0, 3.0), rotation=(0.1, 0.2, 0.0))
        res = bake.apply_gaze_shake(key, (0.05, -0.03, 0.02), (0.0, 0.0, 0.0))
        before = camera.to_world(key).position
        after = camera.to_world(rebuilt(key, res)).position
        assert after == pytest.approx(before, abs=1e-5)

    def test_position_noise_is_world_add_with_orientation(self):
        # 位置ノイズはワールド座標への加算(非ゼロ姿勢でも局所軸加算でない)。
        # 多軸ノイズで camera.to_world(...).position == before + pos_noise。
        key = cam(distance=-30.0, center=(1.0, 2.0, 3.0), rotation=(0.3, -0.4, 0.2))
        pn = np.array([5.0, -2.0, 1.0])
        res = bake.apply_gaze_shake(key, (0.0, 0.0, 0.0), tuple(pn))
        p0 = np.array(camera.to_world(key).position)
        p1 = np.array(camera.to_world(rebuilt(key, res)).position)
        assert p1 == pytest.approx(p0 + pn, abs=1e-5)
        # 回転ノイズ0なので角度は不変
        assert res["rotation"] == pytest.approx(key.rotation, abs=1e-9)

    def test_distance_zero_matches_naive_addition(self):
        # distance=0: カメラ位置=中心。回転ノイズは中心を動かさず、角度は素朴加算と一致(§4.2)
        key = cam(distance=0.0, center=(1.0, 2.0, 3.0), rotation=(0.2, 0.1, -0.1))
        noise = (0.1, -0.05, 0.03)
        res = bake.apply_gaze_shake(key, noise, (0.0, 0.0, 0.0))
        assert res["position"] == pytest.approx((1.0, 2.0, 3.0), abs=1e-6)
        expected = tuple(key.rotation[i] + noise[i] for i in range(3))
        assert res["rotation"] == pytest.approx(expected, abs=1e-9)

    def test_distance_zero_position_noise(self):
        # distance=0 では位置ノイズが中心へ直接乗る
        key = cam(distance=0.0, center=(1.0, 2.0, 3.0), rotation=(0.2, 0.1, 0.0))
        res = bake.apply_gaze_shake(key, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        assert res["position"] == pytest.approx((3.0, 2.0, 3.0), abs=1e-5)

    def test_rotation_and_position_combined(self):
        # 回転+位置ノイズ併用: 角度=オイラー加算、カメラ位置=元+pos_noise
        key = cam(distance=-40.0, center=(2.0, 1.0, -3.0), rotation=(0.2, 0.3, 0.1))
        noise, pn = (0.04, -0.06, 0.02), np.array([1.0, 2.0, -1.0])
        res = bake.apply_gaze_shake(key, noise, tuple(pn))
        assert res["rotation"] == pytest.approx(
            tuple(key.rotation[i] + noise[i] for i in range(3)), abs=1e-9
        )
        p0 = np.array(camera.to_world(key).position)
        p1 = np.array(camera.to_world(rebuilt(key, res)).position)
        assert p1 == pytest.approx(p0 + pn, abs=1e-5)

    def test_deterministic(self):
        key = cam(rotation=(0.1, 0.2, 0.3))
        a = bake.apply_gaze_shake(key, (0.05, 0.05, 0.05), (1.0, 0.0, 0.0))
        b = bake.apply_gaze_shake(key, (0.05, 0.05, 0.05), (1.0, 0.0, 0.0))
        assert a["position"] == pytest.approx(b["position"], abs=1e-12)
        assert a["rotation"] == pytest.approx(b["rotation"], abs=1e-12)
