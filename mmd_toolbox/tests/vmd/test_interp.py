"""vmd-interp のテスト(vmd-interp.md §6 テスト要件1〜6)。

要件6(独立実装との数値クロスバリデーション)のため、本ファイル末尾近くに
仕様(§2)から独立に書いた素朴なベジェ評価器 _ref_factor を置く。これは
二分法のみで X(s)=x を解き、ベルンスタイン基底で Y(s) を直接計算する——
読めば正しさが確認できる「テストの底」として機能する(tests/data/README.md の方針)。
"""

import math

import numpy as np
import pytest

from mmd_toolbox.vmd import interp, normalize, read
from mmd_toolbox.vmd.types import BoneKey, CameraKey

# ---------------------------------------------------------------------------
# 独立実装(クロスバリデーションの基準)
# ---------------------------------------------------------------------------


def _bernstein(t: float, c1: float, c2: float) -> float:
    """端点 0,1 固定の3次ベジェの1成分。制御点 c1,c2 は [0,1] 正規化済み。"""
    u = 1.0 - t
    return 3 * u * u * t * c1 + 3 * u * t * t * c2 + t * t * t


def _ref_factor(x1: int, y1: int, x2: int, y2: int, x: float) -> float:
    """制御点 (x1,y1),(x2,y2)(0..127)・正規化時間 x∈[0,1] → 補間係数 y∈[0,1]。

    X(s)=x を二分法で解き(Xは単調増加前提)、y=Y(s) を返す。
    """
    px1, py1, px2, py2 = x1 / 127.0, y1 / 127.0, x2 / 127.0, y2 / 127.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _bernstein(mid, px1, px2) < x:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2.0
    return _bernstein(s, py1, py2)


def _ref_slerp(q0, q1, t):
    a = np.array(q0, dtype=float)
    b = np.array(q1, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < 0.0:
        b = -b
        d = -d
    if d > 0.9995:
        r = a + t * (b - a)
        return tuple(r / np.linalg.norm(r))
    th0 = math.acos(d)
    th = th0 * t
    s0 = math.sin(th0 - th) / math.sin(th0)
    s1 = math.sin(th) / math.sin(th0)
    return tuple(s0 * a + s1 * b)


# ---------------------------------------------------------------------------
# フィクスチャ用ヘルパー(宣言的にキーを組み立てる)
# ---------------------------------------------------------------------------

LINEAR = (20, 20, 107, 107)  # MMDデフォルト。制御点 (20,20),(107,107) → y=x


def cam_interp(per_channel: dict) -> bytes:
    """チャンネル -> (x1,y1,x2,y2) からカメラ補間ブロック24バイトを組む。

    格納順は ax,bx,ay,by(始点X,終点X,始点Y,終点Y)= x1,x2,y1,y2。
    未指定チャンネルは線形。
    """
    b = bytearray(24)
    for ch, off in interp.CAMERA_CHANNEL_OFFSET.items():
        x1, y1, x2, y2 = per_channel.get(ch, LINEAR)
        b[off : off + 4] = bytes([x1, x2, y1, y2])
    return bytes(b)


def cam_key(
    frame,
    *,
    distance=0.0,
    position=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
    fov=30,
    curves=None,
    perspective=0,
):
    return CameraKey(
        frame, distance, position, rotation, cam_interp(curves or {}), fov, perspective
    )


def bone_interp_linear_rot() -> bytes:
    """R(回転)チャンネルが線形のボーン補間ブロック64バイト。

    types.control_points() の R は (b[18], b[7], b[11], b[15]) = (x1,y1,x2,y2)。
    回転スラープ検証専用(他チャンネルは参照しない)。
    """
    b = bytearray(64)
    b[18], b[7], b[11], b[15] = 20, 20, 107, 107
    return bytes(b)


def bone_key(frame, *, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(b"\x00" * 15, frame, position, rotation, bone_interp_linear_rot())


# ---------------------------------------------------------------------------
# テスト1: 線形の補間曲線 = 解析解(線形補間)
# ---------------------------------------------------------------------------


class TestLinearCurve:
    def test_distance_is_linear(self):
        keys = [cam_key(0, distance=0.0), cam_key(10, distance=10.0)]
        for f in range(0, 11):
            assert interp.sample(keys, "distance", f) == pytest.approx(float(f), abs=1e-6)

    def test_position_component_is_linear(self):
        keys = [cam_key(0, position=(0.0, 0.0, 0.0)), cam_key(8, position=(8.0, 0.0, 0.0))]
        assert interp.sample(keys, "pos_x", 4) == pytest.approx(4.0, abs=1e-6)

    def test_endpoints_are_exact(self):
        keys = [cam_key(5, distance=2.0), cam_key(25, distance=9.0)]
        assert interp.sample(keys, "distance", 5) == pytest.approx(2.0, abs=1e-9)
        assert interp.sample(keys, "distance", 25) == pytest.approx(9.0, abs=1e-9)


# ---------------------------------------------------------------------------
# テスト2: 極端な制御点でソルバが正しく動作する
# ---------------------------------------------------------------------------


class TestExtremeControlPoints:
    @pytest.mark.parametrize(
        "cp",
        [
            (0, 0, 127, 127),    # 端点まで張り付き
            (5, 122, 122, 5),    # 急峻なS字
            (0, 127, 127, 0),    # 強い非単調Y
            (0, 64, 127, 64),    # 片側が平坦
        ],
    )
    def test_matches_reference(self, cp):
        x1, y1, x2, y2 = cp
        keys = [cam_key(0, distance=0.0, curves={"distance": cp}),
                cam_key(12, distance=12.0, curves={"distance": cp})]
        for f in range(1, 12):
            expected = 12.0 * _ref_factor(x1, y1, x2, y2, f / 12.0)
            assert interp.sample(keys, "distance", f) == pytest.approx(expected, abs=1e-4)

    def test_monotonic_in_frame(self):
        cp = (5, 122, 122, 5)
        keys = [cam_key(0, distance=0.0, curves={"distance": cp}),
                cam_key(20, distance=20.0, curves={"distance": cp})]
        vals = [interp.sample(keys, "distance", f) for f in range(0, 21)]
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# テスト3: 区間の補間に後側キーのパラメーターが使われる
# ---------------------------------------------------------------------------


class TestLaterKeyParams:
    def test_segment_uses_arriving_key_curve(self):
        front = (0, 127, 127, 0)   # key0 に持たせる(誤って使われたら検出される)
        mid = (5, 122, 122, 5)     # key1 の曲線。区間 [0,10] はこれを使うべき
        keys = [
            cam_key(0, distance=0.0, curves={"distance": front}),
            cam_key(10, distance=10.0, curves={"distance": mid}),
            cam_key(20, distance=20.0),  # 線形
        ]
        # 区間 [0,10]: 後側キー(frame10, mid曲線)を使う
        for f in range(1, 10):
            expected = 10.0 * _ref_factor(*mid, f / 10.0)
            got = interp.sample(keys, "distance", f)
            assert got == pytest.approx(expected, abs=1e-4)
            # 前側キーの曲線とは異なる値になる(取り違え検出)
            wrong = 10.0 * _ref_factor(*front, f / 10.0)
            if abs(expected - wrong) > 1e-3:
                assert abs(got - wrong) > 1e-4

    def test_second_segment_uses_its_own_arriving_key(self):
        keys = [
            cam_key(0, distance=0.0),
            cam_key(10, distance=10.0, curves={"distance": (5, 122, 122, 5)}),
            cam_key(20, distance=20.0),  # 線形 → 区間[10,20]は線形
        ]
        assert interp.sample(keys, "distance", 15) == pytest.approx(15.0, abs=1e-6)


# ---------------------------------------------------------------------------
# テスト4: カメラ回転は3軸独立・共通の補間曲線(クォータニオン化しない)
# ---------------------------------------------------------------------------


class TestCameraRotation:
    def test_three_axes_share_one_curve(self):
        rot_cp = (10, 117, 117, 10)  # 非線形
        keys = [
            cam_key(0, rotation=(0.0, 0.0, 0.0), curves={"rot": rot_cp}),
            cam_key(10, rotation=(0.3, -0.6, 1.2), curves={"rot": rot_cp}),
        ]
        y = _ref_factor(*rot_cp, 0.5)
        result = interp.sample(keys, "rot", 5)
        assert len(result) == 3  # オイラー角3軸(クォータニオンではない)
        assert result[0] == pytest.approx(0.3 * y, abs=1e-4)
        assert result[1] == pytest.approx(-0.6 * y, abs=1e-4)
        assert result[2] == pytest.approx(1.2 * y, abs=1e-4)

    def test_euler_lerp_not_slerp(self):
        # オイラー線形補間では各軸が独立に比例する。共通yなので比が保たれる。
        keys = [
            cam_key(0, rotation=(0.0, 0.0, 0.0)),
            cam_key(10, rotation=(1.0, 2.0, 4.0)),
        ]
        rx, ry, rz = interp.sample(keys, "rot", 5)
        assert ry == pytest.approx(2.0 * rx, abs=1e-6)
        assert rz == pytest.approx(4.0 * rx, abs=1e-6)


# ---------------------------------------------------------------------------
# テスト5: 範囲外フレームは端キーの値
# ---------------------------------------------------------------------------


class TestOutOfRange:
    def test_before_first_returns_first(self):
        keys = [cam_key(5, distance=2.0), cam_key(15, distance=8.0)]
        assert interp.sample(keys, "distance", 0) == pytest.approx(2.0, abs=1e-9)

    def test_after_last_returns_last(self):
        keys = [cam_key(5, distance=2.0), cam_key(15, distance=8.0)]
        assert interp.sample(keys, "distance", 100) == pytest.approx(8.0, abs=1e-9)

    def test_single_key_constant(self):
        keys = [cam_key(7, distance=3.5)]
        for f in (0, 7, 50):
            assert interp.sample(keys, "distance", f) == pytest.approx(3.5, abs=1e-9)

    def test_adjacent_frame_segment_keeps_endpoints(self):
        # フレーム差1の区間は中間点が現れず、両端の値のみ(値ジャンプ保持)
        keys = [cam_key(10, distance=1.0), cam_key(11, distance=9.0)]
        assert interp.sample(keys, "distance", 10) == pytest.approx(1.0, abs=1e-9)
        assert interp.sample(keys, "distance", 11) == pytest.approx(9.0, abs=1e-9)


# ---------------------------------------------------------------------------
# ボーン回転スラープ(§3。カメラ回転=オイラー線形 と異なる値パス)
# ---------------------------------------------------------------------------


class TestBoneRotationSlerp:
    def test_slerp_midpoint(self):
        q0 = (0.0, 0.0, 0.0, 1.0)  # 単位
        q1 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # Z軸90°
        keys = [bone_key(0, rotation=q0), bone_key(10, rotation=q1)]
        got = interp.sample(keys, "rot", 5)  # 線形R → t=0.5
        expected = _ref_slerp(q0, q1, 0.5)
        assert len(got) == 4
        # 符号の自由度を吸収して比較
        gv = np.array(got)
        ev = np.array(expected)
        if np.dot(gv, ev) < 0:
            ev = -ev
        assert gv == pytest.approx(ev, abs=1e-4)


# ---------------------------------------------------------------------------
# sample_range(ベイク用途の一括評価)
# ---------------------------------------------------------------------------


class TestSampleRange:
    def test_inclusive_length_and_values(self):
        keys = [cam_key(0, distance=0.0), cam_key(10, distance=10.0)]
        arr = interp.sample_range(keys, "distance", 0, 10)
        assert len(arr) == 11  # 両端含む
        for f, v in enumerate(arr):
            assert float(v) == pytest.approx(interp.sample(keys, "distance", f), abs=1e-9)


# ---------------------------------------------------------------------------
# テスト6: 独立実装との数値クロスバリデーション
# ---------------------------------------------------------------------------

# (x1,y1,x2,y2)。Xが単調増加(x1<=x2)になる構成のみ。
GRID = [
    (20, 20, 107, 107),
    (0, 0, 127, 127),
    (10, 117, 117, 10),
    (0, 127, 64, 64),
    (64, 0, 127, 64),
    (5, 122, 122, 5),
    (40, 10, 80, 120),
]


class TestCrossValidationSynthetic:
    @pytest.mark.parametrize("cp", GRID)
    def test_distance_matches_reference(self, cp):
        keys = [cam_key(0, distance=0.0, curves={"distance": cp}),
                cam_key(30, distance=30.0, curves={"distance": cp})]
        for f in range(1, 30):
            expected = 30.0 * _ref_factor(*cp, f / 30.0)
            assert interp.sample(keys, "distance", f) == pytest.approx(expected, abs=1e-4)


def _cam_cp(interp_bytes: bytes, channel: str):
    """カメラ補間ブロックからチャンネルの制御点 (x1,y1,x2,y2) を独立に取り出す。"""
    off = interp.CAMERA_CHANNEL_OFFSET[channel]
    ax, bx, ay, by = interp_bytes[off : off + 4]
    return (ax, ay, bx, by)


class TestCrossValidationRealFile:
    def test_camera_basic_all_channels(self, camera_basic_bytes):
        doc, _ = read(camera_basic_bytes)
        # sample() はフレーム昇順を前提とする(vmd-interp.md §4)。
        # 実ファイルのカメラキーはファイル内で順不同なので正規化してから渡す。
        doc, _ = normalize(doc, ["camera"])
        keys = doc.camera
        assert len(keys) >= 3
        scalar = {
            "distance": lambda k: k.distance,
            "pos_x": lambda k: k.position[0],
            "pos_y": lambda k: k.position[1],
            "pos_z": lambda k: k.position[2],
            "fov": lambda k: float(k.fov),
        }
        checked = 0
        for k0, k1 in zip(keys, keys[1:]):
            span = k1.frame - k0.frame
            if span <= 1:
                continue
            frames = sorted({k0.frame + 1, k0.frame + span // 2, k1.frame - 1})
            for f in frames:
                x = (f - k0.frame) / span
                # スカラー5チャンネル
                for ch, get in scalar.items():
                    fac = _ref_factor(*_cam_cp(k1.interpolation, ch), x)
                    expected = get(k0) + (get(k1) - get(k0)) * fac
                    assert interp.sample(keys, ch, f) == pytest.approx(expected, abs=1e-4), (
                        f"{ch} at frame {f}"
                    )
                # 回転(共通曲線で3軸)
                rfac = _ref_factor(*_cam_cp(k1.interpolation, "rot"), x)
                got_rot = interp.sample(keys, "rot", f)
                for i in range(3):
                    expected = k0.rotation[i] + (k1.rotation[i] - k0.rotation[i]) * rfac
                    assert got_rot[i] == pytest.approx(expected, abs=1e-4), (
                        f"rot[{i}] at frame {f}"
                    )
                checked += 1
        assert checked > 0, "検証対象の区間がない(テストデータの作り直しが必要)"


# ---------------------------------------------------------------------------
# sample_camera(全チャンネル一括評価)
# ---------------------------------------------------------------------------


class TestSampleCamera:
    def test_matches_per_channel_sample(self):
        keys = [
            cam_key(0, distance=0.0, position=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0), fov=20),
            cam_key(10, distance=10.0, position=(1.0, 2.0, 3.0),
                    rotation=(0.1, 0.2, 0.3), fov=60,
                    curves={"distance": (5, 122, 122, 5)}),
        ]
        state = interp.sample_camera(keys, 4)
        assert state["distance"] == pytest.approx(interp.sample(keys, "distance", 4), abs=1e-9)
        assert state["position"][0] == pytest.approx(interp.sample(keys, "pos_x", 4), abs=1e-9)
        assert state["position"][1] == pytest.approx(interp.sample(keys, "pos_y", 4), abs=1e-9)
        assert state["position"][2] == pytest.approx(interp.sample(keys, "pos_z", 4), abs=1e-9)
        assert state["fov"] == pytest.approx(interp.sample(keys, "fov", 4), abs=1e-9)
        rot = interp.sample(keys, "rot", 4)
        for i in range(3):
            assert state["rotation"][i] == pytest.approx(rot[i], abs=1e-9)


# --- Step B: _solve_factor のベクトル化 ------------------------------------
#
# interp._solve_factor_many(x1,y1,x2,y2,xs) は単点版 _solve_factor を numpy 配列 xs で
# 一括評価したもの。同一の Newton 法+二分法フォールバックなので結果が一致する。

_SOLVE_MANY_CASES = [
    (20, 20, 107, 107),  # 線形
    (0, 0, 127, 64),     # ease(x1=0)
    (40, 10, 90, 118),   # 非対称
    (0, 64, 127, 64),    # x1=0, x2=127(端点)
    (10, 118, 20, 120),  # 急峻
]


@pytest.mark.xfail(reason="impl pending: Step B vectorize", strict=False)
@pytest.mark.parametrize("cp", _SOLVE_MANY_CASES)
def test_solve_factor_many_matches_scalar(cp):
    xs = np.array([0.0, 1e-6, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0 - 1e-6, 1.0])
    many = interp._solve_factor_many(*cp, xs)
    scalar = np.array([interp._solve_factor(*cp, float(x)) for x in xs])
    assert many.shape == xs.shape
    np.testing.assert_allclose(many, scalar, atol=1e-9, rtol=0.0)


@pytest.mark.xfail(reason="impl pending: Step B vectorize", strict=False)
def test_solve_factor_many_random_and_clamp():
    rng = np.random.default_rng(987654)
    xs = rng.random(150)
    for cp in _SOLVE_MANY_CASES:
        many = interp._solve_factor_many(*cp, xs)
        scalar = np.array([interp._solve_factor(*cp, float(x)) for x in xs])
        np.testing.assert_allclose(many, scalar, atol=1e-9, rtol=0.0)
    # x<=0 は 0.0、x>=1 は 1.0(クランプ)。戻り値は ndarray で同形。
    out = interp._solve_factor_many(40, 10, 90, 118, np.array([-1.0, 0.0, 1.0, 2.0]))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.array([0.0, 0.0, 1.0, 1.0]))
