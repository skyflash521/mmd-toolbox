"""bake の視線揺れ変換のテスト(shakevmd.md §4.2)。

サブステップA: apply_gaze_shake のみ。ベイクループ本体 bake() は後続サブステップ。
§4.2 の実現は「揺れ角度 = 元角度 + ノイズ(オイラー加算)、カメラ位置固定になるよう
中心を逆算」。距離0では素朴な角度加算と一致する。
"""

import inspect

import numpy as np
import pytest

from shakevmd import bake, motion
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


# ---------------------------------------------------------------------------
# bake() ループ統合(§3/§4/§5、§7テスト契約)
#
# 本サブステップの範囲: ベイクコアループ(範囲解決・スナップ、正規化作業ビュー、
# カット分割と位相独立、毎フレームのサンプリング→ノイズ→視線揺れ変換、視野角丸め・
# パースホールド、範囲外原本バイト保持、再現性)。
# 後続サブステップ(本テストの対象外): モーション適応の静止/移動プロファイル
# クロスフェード・呼吸ドリフト・settle(§6.2 の高度部分)。基本の adaptive_amplitude
# (motion_scale 連動)は実装に含むが、専用契約テストは後続でまとめる。
# ---------------------------------------------------------------------------

from mmd_toolbox.vmd import interp

# 非線形(非デフォルト)の補間ブロック。入力がこれでも、ベイク出力は常に線形に
# なること(原本補間の誤透過を検出)を確かめるために使う。
NONLINEAR_INTERP = bytes([54, 12, 91, 110]) * 6


def kf(frame, distance=-30.0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
       fov=30, perspective=0, interp_block=NONLINEAR_INTERP):
    """フレーム指定のカメラキー(既定で非線形補間ブロック)。"""
    return CameraKey(frame, distance, center, rotation, interp_block, fov, perspective)


# 動きのある原本カメラ列(ソート済み・一意)。距離・中心・角度・視野角・パースが変化。
SEQ = [
    kf(0, distance=-30.0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), fov=30, perspective=0),
    kf(30, distance=-25.0, center=(10.0, 5.0, 2.0), rotation=(0.2, 0.1, 0.0), fov=45, perspective=1),
    kf(60, distance=-20.0, center=(20.0, 0.0, -3.0), rotation=(-0.1, 0.3, 0.05), fov=30, perspective=1),
]

# カット入り原本列: frame30→31(差1)でカメラ中心が大きく跳ぶ → frame31 がカット。
SEQ_CUT = [
    kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)),
    kf(30, center=(10.0, 5.0, 2.0), rotation=(0.2, 0.1, 0.0)),
    kf(31, center=(40.0, 5.0, 2.0), rotation=(0.2, 0.1, 0.0)),   # 中心ジャンプ30 > 既定閾値5
    kf(60, center=(50.0, 0.0, -3.0), rotation=(-0.1, 0.3, 0.05)),
]

# settle 用フィクスチャは**線形補間**(等速パン)を使う。非線形だとパン途中で速度が揺らぎ
# frame_speeds 正規化で誤った停止が検出され settle が早発するため、停止を遷移点に限定する。
_LIN = bake.LINEAR_CAMERA_INTERP


# パン→停止: ry を [0,30] で 0→0.5 に等速パンし、[30,60] はホールド(frame30 で停止)。
PAN_STOP = [
    kf(0, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(30, rotation=(0.0, 0.5, 0.0), interp_block=_LIN),
    kf(60, rotation=(0.0, 0.5, 0.0), interp_block=_LIN),
]

# カット直前まで角速度大・カット直後は角速度ゼロ。カットをまたいで停止検出すると
# frame30 で誤発動するが、セグメント単位なら発動しない(§5.3-3)の検証用。
CUT_THEN_STOP = [
    kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(29, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.5, 0.0), interp_block=_LIN),   # [0,29] 等速パン
    kf(30, center=(40.0, 0.0, 0.0), rotation=(0.0, 0.5, 0.0), interp_block=_LIN),  # frame30: 中心40ジャンプ=カット
    kf(60, center=(40.0, 0.0, 0.0), rotation=(0.0, 0.5, 0.0), interp_block=_LIN),  # [30,60] 角度ホールド(角速度0)
]

# 位置のみパン→停止(角度は不変)。settle は角速度ベース(§6.2)なので発動しないことの検証用。
POS_PAN_STOP = [
    kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(30, center=(20.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=_LIN),  # 位置パン
    kf(60, center=(20.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=_LIN),  # 停止
]

# 範囲端近く(frame50)で停止。settle が範囲端でフェード(§5.1)で0になることの検証用。
LATE_STOP = [
    kf(0, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(50, rotation=(0.0, 0.5, 0.0), interp_block=_LIN),   # [0,50] 等速パン
    kf(60, rotation=(0.0, 0.5, 0.0), interp_block=_LIN),   # frame50 で停止(範囲端に近い)
]

# 無動(完全静止)。impulse(§6.3)成分だけを観測するための土台(base揺れ・settleを切る)。
STATIC = [
    kf(0, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(60, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
]

# 無動だが中間キーを持つ(範囲 [20,40] を既存キーへスナップできる)。
# F が範囲外のときの impulse の挙動検証用。
STATIC4 = [
    kf(0, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(20, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(40, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
    kf(60, rotation=(0.0, 0.0, 0.0), interp_block=_LIN),
]


def by_frame(result):
    return {k.frame: k for k in result.camera_keys}


def key_tuple(k):
    """CameraKey をバイナリ一致相当の比較用タプルへ(全チャンネル+補間+パース)。"""
    return (k.frame, k.distance, tuple(k.position), tuple(k.rotation),
            bytes(k.interpolation), k.fov, k.perspective)


def rot_dev(res_map, frame):
    """ベイク角度と原本サンプリング角度のずれ(ラジアン・ユークリッド)。"""
    s = interp.sample_camera(SEQ, frame)
    return float(np.linalg.norm(np.array(res_map[frame].rotation) - np.array(s["rotation"])))


class TestBake:
    # --- 基本構造 -----------------------------------------------------------
    def test_dense_one_frame_interval(self):
        # 全範囲ベイク → 範囲内は1フレーム間隔の高密度キー(0..60 の全61フレーム)
        res = bake.bake(SEQ, seed=1)
        frames = sorted(k.frame for k in res.camera_keys)
        assert frames == list(range(0, 61))

    def test_all_baked_keys_use_linear_interp(self):
        # 入力が非線形でも、生成キーは全フレーム MMD 線形デフォルト(端も含む。§3.2)
        res = bake.bake(SEQ, seed=1)
        for k in res.camera_keys:
            assert k.interpolation == bake.LINEAR_CAMERA_INTERP

    def test_empty_camera_keys_raises(self):
        # カメラキー0件はエラー(§3.1。CLIでは終了コード1。コアAPIでは例外)
        with pytest.raises(ValueError):
            bake.bake([], seed=1)

    # --- 入力正規化(§3.1 / §7.8) -----------------------------------------
    def test_unsorted_input_uses_normalized_view(self):
        # 順不同入力でも、サンプリング・範囲解決は正規化作業ビュー(ソート済み)を使う。
        # フレーム集合だけでなく全レコードが整列入力と一致することで、正規化ビュー
        # 経由であることを固定する(§3.1, §7.8)。
        shuffled = [SEQ[2], SEQ[0], SEQ[1]]
        a = [key_tuple(k) for k in bake.bake(SEQ, seed=1, amp_rot=8.0, amp_pos=1.0).camera_keys]
        b = [key_tuple(k) for k in bake.bake(shuffled, seed=1, amp_rot=8.0, amp_pos=1.0).camera_keys]
        assert a == b

    def test_duplicate_frame_last_wins(self):
        # 同一フレーム重複は後勝ち(作業ビュー)。後の key の全チャンネルでサンプリングされる。
        # SEQ[1]=frame30(persp=1) を後発 dup_late(persp=0)が上書きする。
        dup_late = kf(30, distance=-22.0, center=(99.0, 99.0, 99.0),
                      rotation=(0.5, -0.5, 0.2), fov=20, perspective=0)
        keys = [SEQ[0], SEQ[1], dup_late, SEQ[2]]   # frame30 が2つ。後者=dup_late が勝つ
        res = by_frame(bake.bake(keys, seed=1, amp_rot=0.0, amp_pos=0.0))
        assert res[30].position == pytest.approx((99.0, 99.0, 99.0), abs=1e-6)
        assert res[30].rotation == pytest.approx((0.5, -0.5, 0.2), abs=1e-6)
        assert res[30].distance == pytest.approx(-22.0, abs=1e-6)
        assert res[30].fov == 20
        assert res[30].perspective == 0   # パースも後勝ち(SEQ[1]の1ではなくdup_lateの0)

    def test_duplicate_frame_emits_normalization_warning(self):
        # 同一フレーム重複の後勝ち破棄はデータ欠落なので正規化警告を出す(§3.1)。
        # (順不同は実カメラVMDで常態=警告対象外。重複破棄のみ警告する)
        dup = kf(30, center=(99.0, 99.0, 99.0))
        res_dup = bake.bake([SEQ[0], SEQ[1], dup, SEQ[2]], seed=1)
        res_clean = bake.bake(SEQ, seed=1)
        assert res_dup.warnings           # 重複あり → 警告が出る
        assert not res_clean.warnings     # 重複なし(全範囲) → 余計な警告は出さない

    # --- 再現性(§7.5: バイナリ一致相当) -----------------------------------
    def test_reproducible_full_record(self):
        a = bake.bake(SEQ, seed=7, amp_rot=8.0, amp_pos=1.0)
        b = bake.bake(SEQ, seed=7, amp_rot=8.0, amp_pos=1.0)
        assert [key_tuple(k) for k in a.camera_keys] == [key_tuple(k) for k in b.camera_keys]
        assert a.warnings == b.warnings

    def test_different_seed_differs(self):
        r1 = by_frame(bake.bake(SEQ, seed=1, fade_sec=0.1))
        r2 = by_frame(bake.bake(SEQ, seed=2, fade_sec=0.1))
        # 中盤フレームで揺れ位相が異なる → 値が変わる
        assert r1[30].rotation != pytest.approx(r2[30].rotation, abs=1e-9)

    # --- 揺れ0忠実性(§7.2 全チャンネル) -----------------------------------
    def test_zero_amplitude_matches_sampled_all_channels(self):
        # 全揺れ0(amp_rot=amp_pos=settle=0)→ ベイク結果は原本の再サンプリングと一致
        # (全チャンネル+補間線形+パース)。settle は別成分なので明示的に0にする。
        res = by_frame(bake.bake(SEQ, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0))
        for f in range(0, 61):
            s = interp.sample_camera(SEQ, f)
            assert res[f].position == pytest.approx(s["position"], abs=1e-6)
            assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-6)
            assert res[f].distance == pytest.approx(s["distance"], abs=1e-6)
            assert res[f].fov == bake.round_half_up(s["fov"])   # 四捨五入(§3)
            assert res[f].interpolation == bake.LINEAR_CAMERA_INTERP

    # --- 範囲端フェード・境界連続(§5.1 / §7.4) ----------------------------
    def test_fade_zero_at_range_ends(self):
        # 範囲端では揺れ強度0 → 端フレームは原本サンプリングと一致(大振幅でも)
        res = by_frame(bake.bake(SEQ, seed=1, amp_rot=30.0, amp_pos=5.0, fade_sec=0.7))
        for f in (0, 60):
            s = interp.sample_camera(SEQ, f)
            assert res[f].position == pytest.approx(s["position"], abs=1e-4)
            assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-4)

    def test_fade_ramps_in_smoothly_no_jump(self):
        # 端付近は揺れが滑らかに立ち上がる(C1フェード)。§5.1/§7.4 は範囲端前後の
        # 値と差分(速度)の両方の連続性を要求する。
        # 値: 端のずれ(揺れ量)が中盤よりはるかに小さい。
        # 速度: 揺れのフレーム間差分(=揺れ速度。元曲線を引いた残差なので揺れ由来の
        #       速度成分そのもの)が端で中盤よりはるかに小さい(揺れ速度の飛びがない)。
        res = by_frame(bake.bake(SEQ, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.7))
        devs = [rot_dev(res, f) for f in range(0, 61)]
        vel = [abs(devs[f + 1] - devs[f]) for f in range(0, 60)]  # 揺れ速度(一次差分)
        mid_max = max(devs[20:41])
        mid_vel_max = max(vel[20:40])
        # 値の連続
        assert mid_max > 1e-3                        # 中盤は確かに揺れている
        assert max(devs[0:3]) < 0.2 * mid_max        # 始端の立ち上がりは緩やか
        assert max(devs[58:61]) < 0.2 * mid_max      # 終端の収束も緩やか
        # 速度の連続(端で揺れ速度が中盤よりはるかに小さい=速度の飛びがない)
        assert mid_vel_max > 1e-4
        assert max(vel[0:2]) < 0.25 * mid_vel_max    # 始端の揺れ速度はほぼ0から
        assert max(vel[58:60]) < 0.25 * mid_vel_max  # 終端の揺れ速度もほぼ0へ

    def test_midrange_actually_shakes(self):
        # 中盤では揺れが乗る(端フェードで全消ししていないことの確認)
        res = by_frame(bake.bake(SEQ, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.3))
        s = interp.sample_camera(SEQ, 30)
        assert res[30].rotation != pytest.approx(s["rotation"], abs=1e-3)

    def test_short_range_shortens_fade_with_warning(self):
        # 範囲長 < 2×fade 秒 → フェードを自動短縮し警告(§5.1)。端は依然0。
        # fade_sec=0.7 → 2×fade=42フレーム。範囲[0,30]=31フレーム<42 で短縮が起きる。
        res = bake.bake(SEQ, ranges=[(0, 30)], seed=1, amp_rot=20.0, amp_pos=3.0, fade_sec=0.7)
        rm = by_frame(res)
        assert res.warnings   # 自動短縮の警告が出る
        for f in (0, 30):   # 短縮後も範囲端は揺れ0
            s = interp.sample_camera(SEQ, f)
            assert rm[f].rotation == pytest.approx(s["rotation"], abs=1e-4)
            assert rm[f].position == pytest.approx(s["position"], abs=1e-4)

    # --- 強度の単位(amp_rot は度 → ラジアン換算) --------------------------
    def test_amp_rot_is_degrees_not_radians(self):
        # amp_rot=10 は「10度」(≈0.175rad)。ラジアン誤用なら桁違いに大きくなる。
        # 中盤の角度ずれは radians(10) スケールに収まり、1.0rad を超えない。
        res = by_frame(bake.bake(SEQ, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.3))
        peak = max(rot_dev(res, f) for f in range(10, 51))
        assert 5e-3 < peak < 1.0   # 度として乗っている(ラジアン誤用なら peak≫1)

    # --- 視線揺れの幾何(§4.2 / §7.6) --------------------------------------
    def test_rotation_only_keeps_camera_path(self):
        # 位置振幅0 → 各フレームのカメラワールド位置は原本サンプリングと一致(視線だけ揺れる)
        res = by_frame(bake.bake(SEQ, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.3))
        for f in range(1, 60):
            s = interp.sample_camera(SEQ, f)
            orig = kf(f, distance=s["distance"], center=s["position"], rotation=s["rotation"])
            p_orig = np.array(camera.to_world(orig).position)
            p_baked = np.array(camera.to_world(res[f]).position)
            assert p_baked == pytest.approx(p_orig, abs=1e-4)

    def test_position_noise_shifts_camera_world(self):
        # amp_pos>0(回転0)→ 位置ノイズがカメラワールド位置をずらす(§4.2 手順4)。
        # 端ではフェードで0。位置揺れを無視する実装はここで落ちる。
        res = by_frame(bake.bake(SEQ, seed=1, amp_rot=0.0, amp_pos=3.0, fade_sec=0.3))

        def world_dev(f):
            s = interp.sample_camera(SEQ, f)
            orig = kf(f, distance=s["distance"], center=s["position"], rotation=s["rotation"])
            return float(np.linalg.norm(
                np.array(camera.to_world(res[f]).position)
                - np.array(camera.to_world(orig).position)))

        assert max(world_dev(f) for f in range(10, 51)) > 1e-2   # 位置揺れが効いている
        assert world_dev(0) < 1e-4 and world_dev(60) < 1e-4      # 端は0(フェード)

    # --- モーション適応(§6.2 角速度＋移動速度) ---------------------------
    def test_motion_adaptation_responds_to_rotation_only(self):
        # その場回転(中心固定・角度のみ変化)でも motion_scale が効く(§6.2 角速度)。
        # カメラ中心位置だけで速度を測る実装は静止扱いになり、ここで落ちる。
        keys = [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), distance=-30.0),
                kf(60, center=(0.0, 0.0, 0.0), rotation=(0.0, 1.0, 0.0), distance=-30.0)]
        a = by_frame(bake.bake(keys, seed=1, amp_rot=5.0, amp_pos=0.0, motion_scale=0.0, fade_sec=0.2))
        b = by_frame(bake.bake(keys, seed=1, amp_rot=5.0, amp_pos=0.0, motion_scale=10.0, fade_sec=0.2))
        assert a[30].rotation != pytest.approx(b[30].rotation, abs=1e-4)

    def test_motion_adaptation_responds_to_distance_only(self):
        # ズーム(中心・角度固定・距離のみ変化)でも motion_scale が効く(§6.2 移動速度)。
        keys = [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), distance=-50.0),
                kf(60, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), distance=-10.0)]
        a = by_frame(bake.bake(keys, seed=1, amp_rot=5.0, amp_pos=0.0, motion_scale=0.0, fade_sec=0.2))
        b = by_frame(bake.bake(keys, seed=1, amp_rot=5.0, amp_pos=0.0, motion_scale=10.0, fade_sec=0.2))
        assert a[30].rotation != pytest.approx(b[30].rotation, abs=1e-4)

    # --- settle 停止後の減衰振動(§6.2) -----------------------------------
    def _settle_rdev(self, fixture, res, f):
        s = interp.sample_camera(fixture, f)
        return np.array(res[f].rotation) - np.array(s["rotation"])   # 回転残差ベクトル(rad)

    def test_settle_adds_decaying_oscillation_after_stop(self):
        # パン(ry)→停止後、settle が直前の回転方向(ry)に減衰振動を加える(§6.2)。
        # base ノイズを切る(amp_rot=0/amp_pos=0)ことで settle 成分だけを観測する。
        res = by_frame(bake.bake(PAN_STOP, seed=1, amp_rot=0.0, amp_pos=0.0,
                                 settle=5.0, fade_sec=0.1))
        dev = {f: self._settle_rdev(PAN_STOP, res, f) for f in range(0, 61)}

        def mag(rng):
            return max(float(np.linalg.norm(dev[f])) for f in rng)

        # 停止後に揺れが出る(settle 発動)
        assert mag(range(33, 50)) > 1e-3
        # 直前の回転方向(ry=index1)が主成分: ry 残差が rx/rz よりはるかに大きい
        peak_f = max(range(33, 50), key=lambda f: abs(dev[f][1]))
        assert abs(dev[peak_f][1]) > 5 * max(abs(dev[peak_f][0]), abs(dev[peak_f][2]))
        # 方向(符号): 停止直後の最初の山は +ry(直前のパン方向と同符号。逆方向なら落ちる)
        ry = [dev[f][1] for f in range(31, 56)]
        early_peak_f = max(range(31, 38), key=lambda f: abs(dev[f][1]))
        assert dev[early_peak_f][1] > 0
        # 「振動」: ry 残差は両符号をとる(片側減衰でなく sin 振動。逆方向/片側なら落ちる)
        assert max(ry) > 1e-3 and min(ry) < -1e-3
        # 初期振幅は settle 度のスケール(≈radians(5)=0.087)。度/ラジアン取り違えや桁違いを排除
        assert 0.01 < mag(range(31, 45)) < 0.3
        # 収束: 後半(フェード手前 50-55、fade≈1)は停止直後よりはるかに小さく、ほぼ0へ
        assert mag(range(33, 43)) > mag(range(50, 56))
        assert mag(range(50, 56)) < 0.02
        # 停止前(パン中)は settle 非発動 → base ノイズも0なのでほぼ0
        assert mag(range(1, 25)) < 1e-4

    def test_settle_zero_disables_only_settle(self):
        # settle=0 は settle 成分だけを無効化する(base ノイズには影響しない)。
        # (a) settle=0 + base ノイズ0 → 全フレーム原本一致(settle 無し)
        res0 = by_frame(bake.bake(PAN_STOP, seed=1, amp_rot=0.0, amp_pos=0.0,
                                  settle=0.0, fade_sec=0.1))
        for f in range(0, 61):
            s = interp.sample_camera(PAN_STOP, f)
            assert res0[f].rotation == pytest.approx(s["rotation"], abs=1e-6)
        # (b) settle=0 + base ノイズ有り → 通常の揺れは残る(settle=0 が base を殺さない)
        resn = by_frame(bake.bake(PAN_STOP, seed=1, amp_rot=10.0, amp_pos=0.0,
                                  settle=0.0, fade_sec=0.1))
        s30 = interp.sample_camera(PAN_STOP, 30)
        assert resn[30].rotation != pytest.approx(s30["rotation"], abs=1e-3)

    def test_settle_is_additive_to_base_noise(self):
        # settle は通常の揺れに「加算」される(置換しない。§6.2)。回転は sampled+rot_noise で
        # 線形なので、回転残差は base 成分と settle 成分の和に厳密一致する。
        base = by_frame(bake.bake(PAN_STOP, seed=1, amp_rot=8.0, amp_pos=0.0, settle=0.0, fade_sec=0.1))
        only = by_frame(bake.bake(PAN_STOP, seed=1, amp_rot=0.0, amp_pos=0.0, settle=5.0, fade_sec=0.1))
        both = by_frame(bake.bake(PAN_STOP, seed=1, amp_rot=8.0, amp_pos=0.0, settle=5.0, fade_sec=0.1))
        settle_seen = False
        for f in range(33, 50):
            s = np.array(interp.sample_camera(PAN_STOP, f)["rotation"])
            dev_base = np.array(base[f].rotation) - s
            dev_only = np.array(only[f].rotation) - s
            dev_both = np.array(both[f].rotation) - s
            assert dev_both == pytest.approx(dev_base + dev_only, abs=1e-9)  # 加算(置換でない)
            if float(np.linalg.norm(dev_only)) > 1e-3:
                settle_seen = True
        assert settle_seen   # settle 成分が実際に効いている(恒真でない)

    def test_settle_faded_at_range_end(self):
        # settle も範囲端フェードの対象(§5.1: 範囲端で揺れ強度は必ず0)。停止が範囲端近く
        # (frame50)でも、範囲端(frame60)では fade=0 により settle が0になる。
        res = by_frame(bake.bake(LATE_STOP, seed=1, amp_rot=0.0, amp_pos=0.0,
                                 settle=10.0, fade_sec=0.2))

        def mag(rng):
            return max(
                float(np.linalg.norm(
                    np.array(res[f].rotation) - np.array(interp.sample_camera(LATE_STOP, f)["rotation"])))
                for f in rng)

        # 停止直後(フェード手前)は settle が乗っている
        assert mag(range(51, 55)) > 1e-3
        # 範囲端(frame60)は fade=0 で settle も0(原本一致)。settle がまだ減衰しきる前でも端は0
        s60 = interp.sample_camera(LATE_STOP, 60)
        assert res[60].rotation == pytest.approx(s60["rotation"], abs=1e-4)

    def test_settle_is_angular_not_positional(self):
        # settle は角速度ベース(§6.2)。位置のみパン→停止(角度不変)では発動しない。
        res = by_frame(bake.bake(POS_PAN_STOP, seed=1, amp_rot=0.0, amp_pos=0.0,
                                 settle=10.0, fade_sec=0.1))
        for f in range(0, 61):
            s = interp.sample_camera(POS_PAN_STOP, f)
            assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-6)

    def test_settle_not_triggered_at_cut(self):
        # カット点では settle を発動しない(§5.3-3)。CUT_THEN_STOP は frame30 のカット直前まで
        # 角速度大・直後ゼロ。カットをまたいで停止検出すると frame30 で誤発動するが、
        # セグメント単位なら発動しない。base ノイズ0・settle 大でも frame30 以降は原本一致。
        res = by_frame(bake.bake(CUT_THEN_STOP, seed=1, amp_rot=0.0, amp_pos=0.0,
                                 settle=10.0, fade_sec=0.1))
        for f in range(30, 45):
            s = interp.sample_camera(CUT_THEN_STOP, f)
            assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-4)

    # --- impulse 衝撃(§6.3) ----------------------------------------------
    def _imp_dev(self, res, f):
        s = interp.sample_camera(STATIC, f)
        return np.array(res[f].rotation) - np.array(s["rotation"])   # 回転残差ベクトル(rad)

    def _imp_mag(self, res, f):
        return float(np.linalg.norm(self._imp_dev(res, f)))

    def test_impulse_fires_from_frame_and_decays(self):
        # フレームF以降に S·exp(-t/D) 包絡の高周波揺れ(§6.3)。base/settle を切り impulse だけ観測。
        res = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                                 impulses=[(30, 10.0, 0.5)], fade_sec=0.1))
        # 発火前(F=30 より前。F-1=29 まで含める=off-by-one検出)はほぼ0
        assert max(self._imp_mag(res, f) for f in range(5, 30)) < 1e-4
        # F以降に揺れが出る
        assert max(self._imp_mag(res, f) for f in range(30, 45)) > 1e-3
        # 振幅は S 度スケール(≈radians(10)=0.17)。度/ラジアン取り違え・桁違いを排除
        peak = max(self._imp_mag(res, f) for f in range(30, 35))
        assert 0.02 < peak < 0.6
        # 指数減衰 S·exp(-t/D): F付近とF+D(D=0.5s=15f→frame45)付近の比は ≈e。2〜5倍に収まる
        near_F = max(self._imp_mag(res, f) for f in range(30, 35))
        near_FD = max(self._imp_mag(res, f) for f in range(44, 49))
        assert 2.0 < near_F / max(near_FD, 1e-9) < 5.0
        # 高周波性: 残差(主軸)が窓内で何度も符号反転する(低周波なら反転は1回程度)
        ax = int(np.argmax(np.abs(self._imp_dev(res, 31))))
        comp = [self._imp_dev(res, f)[ax] for f in range(30, 45)]
        signs = [c > 0 for c in comp if abs(c) > 1e-4]
        flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        assert flips >= 3

    def test_no_impulse_no_effect(self):
        # impulses 未指定(既定)でも明示の()でも impulse 成分なし(base/settle も0なら原本一致)
        for kw in ({}, {"impulses": ()}):
            res = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0,
                                     settle=0.0, fade_sec=0.1, **kw))
            for f in range(0, 61):
                s = interp.sample_camera(STATIC, f)
                assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-9)

    def test_impulse_seed_derived_direction(self):
        # 方向・揺れはシードから決定(§6.3)。同一シードで再現、異なるシードで変わる。
        a = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                               impulses=[(30, 10.0, 0.5)], fade_sec=0.1))
        b = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                               impulses=[(30, 10.0, 0.5)], fade_sec=0.1))
        c = by_frame(bake.bake(STATIC, seed=2, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                               impulses=[(30, 10.0, 0.5)], fade_sec=0.1))
        for f in range(30, 45):
            assert a[f].rotation == pytest.approx(b[f].rotation, abs=1e-12)   # 再現性
        assert any(a[f].rotation != pytest.approx(c[f].rotation, abs=1e-6) for f in range(30, 45))  # シード依存

    def test_impulses_additive(self):
        # 複数 impulse は加算合成(§6.3)。重なる2衝撃で残差が各単独の和に一致(max/last-wins を排除)。
        # 回転は sampled+rot_noise で線形なので和が厳密に成立する。impulse のシードは F 由来
        # (index 非依存)なので、単独/複合で同一 F の衝撃は同じ揺れになる。
        A = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                               impulses=[(30, 6.0, 1.0)], fade_sec=0.1))
        B = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                               impulses=[(36, 6.0, 1.0)], fade_sec=0.1))
        AB = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                                impulses=[(30, 6.0, 1.0), (36, 6.0, 1.0)], fade_sec=0.1))
        overlap_seen = False
        for f in range(36, 50):   # 両衝撃が重なる領域
            assert self._imp_dev(AB, f) == pytest.approx(
                self._imp_dev(A, f) + self._imp_dev(B, f), abs=1e-9)   # 加算(max/置換でない)
            if self._imp_mag(A, f) > 1e-3 and self._imp_mag(B, f) > 1e-3:
                overlap_seen = True
        assert overlap_seen   # 実際に重なっている(恒真でない)

    def test_impulse_faded_at_range_end(self):
        # impulse も範囲端フェード(§5.1)の対象。範囲端近く(F=55)で発火しても frame60 では0。
        res = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                                 impulses=[(55, 10.0, 0.5)], fade_sec=0.2))
        assert max(self._imp_mag(res, f) for f in range(55, 58)) > 1e-3   # 発火直後は乗る
        s60 = interp.sample_camera(STATIC, 60)
        assert res[60].rotation == pytest.approx(s60["rotation"], abs=1e-4)  # 範囲端は0

    def test_impulse_frame_outside_range(self):
        # impulse の F は絶対フレーム。範囲[20,40]に対し F>b は寄与なし、F<a は減衰した尾を出す。
        def mags(impulses):
            res = by_frame(bake.bake(STATIC4, ranges=[(20, 40)], seed=1, amp_rot=0.0,
                                     amp_pos=0.0, settle=0.0, impulses=impulses, fade_sec=0.1))
            return [float(np.linalg.norm(
                np.array(res[f].rotation) - np.array(interp.sample_camera(STATIC4, f)["rotation"])))
                for f in range(21, 40)]   # 端フェードを避けた範囲内側
        # F=50 > b=40 → 範囲内に寄与なし
        assert max(mags([(50, 10.0, 1.0)])) < 1e-4
        # F=10 < a=20 → 範囲内に減衰した尾が出る
        assert max(mags([(10, 10.0, 1.0)])) > 1e-3

    def test_impulse_nonpositive_params_no_effect(self):
        # S<=0 または D<=0 は無効(ガード。div-by-zero回避)
        for imp in [(30, 0.0, 0.5), (30, 10.0, 0.0), (30, -5.0, 0.5)]:
            res = by_frame(bake.bake(STATIC, seed=1, amp_rot=0.0, amp_pos=0.0, settle=0.0,
                                     impulses=[imp], fade_sec=0.1))
            for f in range(0, 61):
                s = interp.sample_camera(STATIC, f)
                assert res[f].rotation == pytest.approx(s["rotation"], abs=1e-9)

    # --- 視野角・パース(§3.1 / §7.3) --------------------------------------
    def test_fov_equals_rounded_sample_even_with_shake(self):
        # 視野角は揺れ対象外。非ゼロ揺れでも各フレームの視野角=元曲線サンプルの四捨五入
        res = by_frame(bake.bake(SEQ, seed=1, amp_rot=20.0, amp_pos=3.0, fade_sec=0.3))
        for f in range(0, 61):
            s = interp.sample_camera(SEQ, f)
            assert isinstance(res[f].fov, int)
            assert res[f].fov == bake.round_half_up(s["fov"])

    def test_fov_rounds_half_up_not_bankers(self):
        # 視野角の丸めは四捨五入(round half up)で、Python round() の銀行丸めではない
        # (vmd-interp.md §3/§5)。線形補間で frame1 がちょうど 36.5 になる構成。
        # 36.5 → 四捨五入=37、銀行丸め(Python round)=36 で区別できる。
        keys = [
            kf(0, fov=36, interp_block=bake.LINEAR_CAMERA_INTERP),
            kf(2, fov=37, interp_block=bake.LINEAR_CAMERA_INTERP),
        ]
        assert interp.sample(keys, "fov", 1) == pytest.approx(36.5, abs=1e-9)  # 前提
        res = by_frame(bake.bake(keys, seed=1, amp_rot=0.0, amp_pos=0.0))
        assert res[1].fov == 37   # 四捨五入(Python round(36.5) は 36 になる)

    def test_perspective_holds_governing_key(self):
        # パースは当該フレーム以前で最も近いキーの値をホールド
        # SEQ: frame0 persp=0, frame30 persp=1, frame60 persp=1
        res = by_frame(bake.bake(SEQ, seed=1))
        assert res[10].perspective == 0    # key0(persp0)区間
        assert res[29].perspective == 0
        assert res[30].perspective == 1    # key1 以降は persp1
        assert res[45].perspective == 1

    # --- 範囲指定・範囲外保持(§3.2 / §5.2 / §7.1) -------------------------
    def test_default_range_is_full_span(self):
        res = bake.bake(SEQ, ranges=None, seed=1)
        frames = sorted(k.frame for k in res.camera_keys)
        assert frames[0] == 0 and frames[-1] == 60

    def test_range_bakes_only_inside(self):
        res = bake.bake(SEQ, ranges=[(30, 60)], seed=1)
        baked_frames = sorted(k.frame for k in res.camera_keys if k.frame >= 30)
        assert baked_frames == list(range(30, 61))

    def test_out_of_range_leading_key_preserved(self):
        # 範囲[30,60] → frame0 は範囲外。原本レコードがバイト相当(全フィールド)で保持(§7.1)。
        # 非線形補間ブロックのまま透過する(線形化されない)ことを含む。
        res = bake.bake(SEQ, ranges=[(30, 60)], seed=1)
        out = {k.frame: k for k in res.camera_keys if k.frame < 30}
        assert list(out) == [0]
        assert key_tuple(out[0]) == key_tuple(SEQ[0])

    def test_out_of_range_trailing_key_preserved(self):
        # 範囲[0,30] → frame60 は範囲外。原本レコードがバイト相当で保持(§7.1)。
        res = bake.bake(SEQ, ranges=[(0, 30)], seed=1)
        out = {k.frame: k for k in res.camera_keys if k.frame > 30}
        assert list(out) == [60]
        assert key_tuple(out[60]) == key_tuple(SEQ[2])

    def test_output_is_frame_ordered(self):
        # 出力はフレーム昇順。範囲[30,60]の先頭 frame0(範囲外原本)がベイク群の後ろに
        # 紛れず先頭に来る(§3.2 の出力。dict/sorted では順序崩れを検出できないため明示)。
        res = bake.bake(SEQ, ranges=[(30, 60)], seed=1)
        frames = [k.frame for k in res.camera_keys]
        assert frames == sorted(frames)          # 出力リスト自体が昇順
        assert frames[0] == 0                    # 先頭の範囲外原本が先頭に来る

    def test_range_snaps_to_nearest_existing_key(self):
        # 既存キー上にない範囲端は最近接の既存キーへスナップ(§5.2)。
        # (25,55) → 25は30へ(|5|<|25|)、55は60へ(|5|<|25|) → 実効[30,60]。
        res = bake.bake(SEQ, ranges=[(25, 55)], seed=1)
        frames = sorted(k.frame for k in res.camera_keys)
        # frame0 は範囲外で残り、ベイクは [30,60] の高密度
        assert frames == [0] + list(range(30, 61))
        out0 = next(k for k in res.camera_keys if k.frame == 0)
        assert key_tuple(out0) == key_tuple(SEQ[0])

    def test_overlapping_ranges_raise(self):
        # スナップ後に範囲が重複したらエラー(§5.2)
        with pytest.raises(ValueError):
            bake.bake(SEQ, ranges=[(0, 60), (30, 60)], seed=1)

    def test_multiple_nonoverlapping_ranges(self):
        # 非重複の複数 range は両方とも有効にベイクされ、間隙の原本キーは保持される(§2.2/§5.2)
        seq5 = [
            kf(0, center=(0.0, 0.0, 0.0)),
            kf(15, center=(5.0, 1.0, 0.0)),
            kf(30, center=(10.0, 5.0, 2.0), fov=45, perspective=1),
            kf(45, center=(15.0, 2.0, -1.0)),
            kf(60, center=(20.0, 0.0, -3.0)),
        ]
        res = bake.bake(seq5, ranges=[(0, 15), (45, 60)], seed=1)
        frames = sorted(k.frame for k in res.camera_keys)
        # 2範囲が高密度ベイク + 間隙の原本キー(frame30)のみ
        assert frames == list(range(0, 16)) + [30] + list(range(45, 61))
        # 間隙(16..29, 31..44)にはキーが生えない
        assert not (set(range(16, 30)) & set(frames))
        assert not (set(range(31, 45)) & set(frames))
        # frame30 は原本そのまま(バイト相当)
        out30 = next(k for k in res.camera_keys if k.frame == 30)
        assert key_tuple(out30) == key_tuple(seq5[2])

    # --- カット波及(§5.3 / §7.7) ------------------------------------------
    def test_cut_boundary_is_not_faded(self):
        # 内部カット境界(frame31)ではフェードしない → 揺れは非ゼロのまま(§5.3-4)
        res = bake.bake(SEQ_CUT, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.2)
        rm = by_frame(res)
        s = interp.sample_camera(SEQ_CUT, 31)
        dev = float(np.linalg.norm(np.array(rm[31].rotation) - np.array(s["rotation"])))
        assert dev > 1e-3   # カット直後フレームでも揺れが乗っている(0に落ちない)

    def test_cut_makes_segment_phase_independent(self):
        # カットの有無で後半セグメントの揺れ位相が変わる(セグメント別シード派生。§5.3-1)
        with_cut = by_frame(bake.bake(SEQ_CUT, seed=1, amp_rot=10.0, amp_pos=0.0, fade_sec=0.2))
        no_cut = by_frame(bake.bake(SEQ_CUT, seed=1, amp_rot=10.0, amp_pos=0.0,
                                    fade_sec=0.2, manual_cuts_remove=[31]))
        # 後半[31,60]内のフレームで角度が一致しない(位相が独立化している)
        assert with_cut[45].rotation != pytest.approx(no_cut[45].rotation, abs=1e-9)


class TestProfileCrossfadeAndBreathing:
    """§6.2 高度部分の bake 統合: 静止/移動プロファイルのオクターブ重みクロスフェードと
    完全静止区間の呼吸ドリフト(0.3Hz)。"""

    # N+1=100 サンプル → rfftfreq(100, 1/30) のビン間隔=0.3Hz、bin1 がちょうど 0.3Hz。
    # かつ 0.3Hz は窓(100/30秒)で整数1周期=リーケージなし。
    N = 99
    _LIN = bytes([20, 107, 20, 107]) * 6   # 線形補間ブロック(等速にして速度を一定にする)

    def _static_input(self):
        # 同一キーの繰り返し → セグメント全域で速度0(完全静止 → 静止プロファイル)。
        return [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.N, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN)]

    def _moving_input(self):
        # 線形補間の大きなパン → 等速 → 速度≈1(移動プロファイル)。補間形状由来の速度変動を排除。
        return [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.N, center=(60.0, 0.0, 0.0), rotation=(0.0, 1.5, 0.0), interp_block=self._LIN)]

    def _rot_shake(self, res, src, axis):
        return np.array([res[f].rotation[axis] - interp.sample_camera(src, f)["rotation"][axis]
                         for f in range(1, self.N)])

    def _pos_shake_axis(self, res, src, axis):
        return np.array([res[f].position[axis] - interp.sample_camera(src, f)["position"][axis]
                         for f in range(self.N + 1)])

    def _shake_series(self, res, src, kind, axis, end):
        # 絶対フレーム 0..end の残差(ベイク−サンプリング)。kind は "rot" / "pos"。
        key = "rotation" if kind == "rot" else "position"
        return np.array([getattr(res[f], key)[axis] - interp.sample_camera(src, f)[key][axis]
                         for f in range(end + 1)])

    @staticmethod
    def _hf_ratio(series):
        # 相対高周波 = std(隣接差)/std(全体)。振幅差に不変な高周波成分の指標。
        return float(np.std(np.diff(series)) / (np.std(series) + 1e-12))

    def test_moving_has_more_high_freq_than_still(self):
        # プロファイルクロスフェード(§6.2): 移動セグメントは静止より高周波成分が多い。
        # 回転3軸で確認する。クロスフェードの hf 検証は回転チャンネルで行う:
        # 位置チャンネルは呼吸ドリフト(0.3Hz、位置のみ)が静止窓の hf を下げ、クロスフェード
        # 無しでも移動>静止を満たしうるため(round6 指摘#1)。位置のクロスフェードは呼吸帯域を
        # 除いた band-ratio で別途検証する(test_crossfade_applies_to_position_via_octave_band_ratio)。
        src_s, src_m = self._static_input(), self._moving_input()
        rst = by_frame(bake.bake(src_s, seed=1, amp_rot=5.0, amp_pos=0.0, settle=0.0, fade_sec=0.3))
        rmv = by_frame(bake.bake(src_m, seed=1, amp_rot=5.0, amp_pos=0.0, settle=0.0, fade_sec=0.3))
        hf = self._hf_ratio
        for ax in range(3):
            assert (hf(self._rot_shake(rmv, src_m, ax))
                    > hf(self._rot_shake(rst, src_s, ax))), f"rot{ax}"

    def test_per_frame_crossfade_within_single_segment(self):
        # 速度クロスフェードは「フレーム毎の速度」で効く(§6.2)。同一ノイズ実現(同一 seed)で
        # 「同じ時間窓」を移動側/静止側に切り替えて比較し、マジックマージン(round4 指摘#1)を排す。
        #   decel: 0..50 移動 → 50..99 静止。 accel: 0..50 静止 → 50..99 移動。
        # キーは 0/50/99 で隣接1フレーム差なし → カット無し → 各々単一セグメント。
        # 後半窓[58:88]: decel=静止プロファイル、accel=移動プロファイルを「同一ノイズ」に適用する
        # ので、クロスフェードがあれば accel の高周波が多い(前半窓[12:42]は逆)。
        # 固定スペクトルなら両者同一重み → 合成完全一致 → strict > で落ちる(マージン不要)。
        # セグメント単位の二値実装も、accel/decel は速度分布が対称で同一プロファイルを選ぶため落ちる。
        # 速度はワールド位置パンで駆動されるので回転チャンネルも切り替わる(回転で観測 → 呼吸の
        # 汚染なし。round6 指摘#1)。3回転軸で確認する。
        L = self._LIN
        decel = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(50, center=(60.0, 0.0, 0.0), interp_block=L),
                 kf(self.N, center=(60.0, 0.0, 0.0), interp_block=L)]
        accel = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(50, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(self.N, center=(60.0, 0.0, 0.0), interp_block=L)]
        LATE, EARLY = slice(58, 88), slice(12, 42)  # フェード端と境目(frame50)を避けた内部窓
        for ax in range(3):
            late_a, late_d, early_a, early_d = [], [], [], []
            for seed in range(3):
                # settle 切(停止過渡の高周波混入を排除)。同一 seed で decel/accel は同一ノイズ。
                d = by_frame(bake.bake(decel, seed=seed, amp_rot=5.0, amp_pos=0.0,
                                       settle=0.0, fade_sec=0.3))
                a = by_frame(bake.bake(accel, seed=seed, amp_rot=5.0, amp_pos=0.0,
                                       settle=0.0, fade_sec=0.3))
                ds = self._shake_series(d, decel, "rot", ax, self.N)
                as_ = self._shake_series(a, accel, "rot", ax, self.N)
                late_a.append(self._hf_ratio(as_[LATE]))   # accel: 後半は移動
                late_d.append(self._hf_ratio(ds[LATE]))    # decel: 後半は静止
                early_a.append(self._hf_ratio(as_[EARLY]))  # accel: 前半は静止
                early_d.append(self._hf_ratio(ds[EARLY]))   # decel: 前半は移動
            assert np.mean(late_a) > np.mean(late_d), f"late rot{ax}"
            assert np.mean(early_d) > np.mean(early_a), f"early rot{ax}"

    def test_crossfade_is_speed_proportional_not_binary(self):
        # §6.2 のクロスフェードは速度比例 (1-s)·still + s·moving。s=0/1 だけでなく中間 s=0.5 が
        # 「両端の間」に入ること(厳密単調)を確認し、フレーム毎の二値スイッチ(round5 指摘#1)を排除する。
        # 同一 seed・同一窓[58:88]に s=0/0.5/1 を与える3経路(同一ノイズの再重み付け = マージン不要):
        #   s0  : 窓は静止(decel)。 sfull: 窓は全速(accel、唯一の運動 → 正規化1)。
        #   shalf: 窓の前(0..25)に2倍速のピークを置き、窓は半速 → 正規化0.5。
        # 二値スイッチは中間が端へ張り付き、固定スペクトルは同一重みで3者一致 → どちらも落ちる。
        # 回転チャンネルで観測する(呼吸の汚染なし。round6 指摘#1)。
        L = self._LIN
        s0 = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
              kf(50, center=(60.0, 0.0, 0.0), interp_block=L),
              kf(self.N, center=(60.0, 0.0, 0.0), interp_block=L)]
        shalf = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(25, center=(60.0, 0.0, 0.0), interp_block=L),     # 0..25: 2.4/f がピーク
                 kf(self.N, center=(148.8, 0.0, 0.0), interp_block=L)]  # 25..99: 1.2/f = 半速
        sfull = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(50, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(self.N, center=(60.0, 0.0, 0.0), interp_block=L)]
        W = slice(58, 88)

        def hf_rot(src, seed):
            res = by_frame(bake.bake(src, seed=seed, amp_rot=5.0, amp_pos=0.0,
                                     settle=0.0, fade_sec=0.3))
            return self._hf_ratio(self._shake_series(res, src, "rot", 0, self.N)[W])

        h0 = [hf_rot(s0, s) for s in range(3)]
        hh = [hf_rot(shalf, s) for s in range(3)]
        hful = [hf_rot(sfull, s) for s in range(3)]
        assert np.mean(hful) > np.mean(hh) > np.mean(h0)

    # 呼吸ドリフト(§6.2「完全静止区間: 長周期ドリフト 0.3Hz 相当」)。位置の長周期ドリフトとして
    # フレーム毎に (1-speed) スケールで加算する(回転=向きの揺れには載せない)。仕様は軸分布を
    # 規定しないので「位置全体の 0.3Hz」で判別する(軸独立性・等振幅は実装裁量、テストで縛らない)。
    def _e03(self, res, src, kind, axis, end, win=slice(None)):
        # 残差の 0.3Hz ビン振幅(win 適用後の長さで 0.3Hz が整数1周期になる窓を渡す)。
        series = self._shake_series(res, src, kind, axis, end)[win]
        freqs = np.fft.rfftfreq(series.size, d=1.0 / 30.0)
        k = int(np.argmin(np.abs(freqs - 0.3)))
        return float(np.abs(np.fft.rfft(series))[k])

    def _e03_sum(self, res, src, kind, end, win=slice(None)):
        return float(sum(self._e03(res, src, kind, ax, end, win) for ax in range(3)))

    def _band_ratio(self, res, src, axis, end, kind="pos"):
        # 残差の「高オクターブ(>1.8Hz)/低オクターブ(0.6–1.8Hz)」振幅比。
        # 呼吸(0.3Hz)は両帯域外なので影響しない。freq=1.2 でオクターブは 1.2/2.4/4.8Hz。
        series = self._shake_series(res, src, kind, axis, end)
        spec = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(series.size, d=1.0 / 30.0)
        low = spec[(freqs > 0.6) & (freqs <= 1.8)].sum()
        high = spec[freqs > 1.8].sum()
        return float(high / (low + 1e-12))

    def test_crossfade_applies_to_position_via_octave_band_ratio(self):
        # 位置の hf は呼吸(0.3Hz)で汚染されるため(round6 指摘#1)、呼吸帯域を除いた高/低オクターブ
        # エネルギー比で位置のクロスフェードを直接検証する。移動は MOVING_PROFILE で高オクターブ重みが
        # 大きい → 比が大きい。同一 seed → 同一ノイズ → 重み差のみ(固定スペクトルは比一致 → red)。
        src_s, src_m = self._static_input(), self._moving_input()
        st = by_frame(bake.bake(src_s, seed=1, amp_rot=0.0, amp_pos=1.0, freq=1.2, settle=0.0, fade_sec=0.3))
        mv = by_frame(bake.bake(src_m, seed=1, amp_rot=0.0, amp_pos=1.0, freq=1.2, settle=0.0, fade_sec=0.3))
        for ax in range(3):
            assert self._band_ratio(mv, src_m, ax, self.N) > self._band_ratio(st, src_s, ax, self.N), f"pos{ax}"

    def test_breathing_present_in_static_absent_on_rotation(self):
        # 完全静止セグメントの位置揺れに 0.3Hz 成分が現れ、移動セグメントでは (1-speed)≈0 で消える。
        # 位置全体の 0.3Hz が移動を大きく(>3×)上回る。回転には呼吸を載せない(round4 指摘#2)ので
        # 回転は静止が移動を大きく上回らない(<2×)→ 回転にドリフトを載せる実装はここで落ちる。
        # N=99 → 100サンプルで 0.3Hz が整数1周期。
        src_s, src_m = self._static_input(), self._moving_input()
        pst = by_frame(bake.bake(src_s, seed=1, amp_rot=0.0, amp_pos=1.0, settle=0.0, fade_sec=0.3))
        pmv = by_frame(bake.bake(src_m, seed=1, amp_rot=0.0, amp_pos=1.0, settle=0.0, fade_sec=0.3))
        rst = by_frame(bake.bake(src_s, seed=1, amp_rot=5.0, amp_pos=0.0, freq=1.2, settle=0.0, fade_sec=0.3))
        rmv = by_frame(bake.bake(src_m, seed=1, amp_rot=5.0, amp_pos=0.0, freq=1.2, settle=0.0, fade_sec=0.3))
        assert self._e03_sum(pst, src_s, "pos", self.N) > 3.0 * self._e03_sum(pmv, src_m, "pos", self.N)
        assert self._e03_sum(rst, src_s, "rot", self.N) < 2.0 * self._e03_sum(rmv, src_m, "rot", self.N)
        # §6.2「完全静止区間: 高周波微動+長周期ドリフト」: 静止でも高周波微動が残る(round7 指摘)。
        # 静止区間の高オクターブを消して呼吸ドリフトだけにする実装はここで落ちる(回転で観測、freq=1.2)。
        for ax in range(3):
            assert self._band_ratio(rst, src_s, ax, self.N, "rot") > 0.1, f"tremor rot{ax}"

    def test_breathing_scales_with_inverse_speed_in_single_segment(self):
        # 呼吸は (1-speed) 比例(§6.2)。同一 seed・同一窓[20:120](100サンプル=0.3Hz整数1周期)に
        # s=0/0.5/1 を与える3経路で、0.3Hz エネルギー合計が s について厳密単調減少することを確認する。
        # セグメント単位の静止ゲート実装(混合セグメントで呼吸ゼロ)も二値ゲート(中間で張り付く)も
        # ここで落ちる。motion_scale=0 でノイズ側の速度依存振幅ブーストを切り、呼吸のみ分離する。
        END = 119
        L = self._LIN
        s0 = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
              kf(19, center=(60.0, 0.0, 0.0), interp_block=L),     # 0..19 移動 → 19..119 静止
              kf(END, center=(60.0, 0.0, 0.0), interp_block=L)]
        shalf = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(10, center=(60.0, 0.0, 0.0), interp_block=L),  # 0..10: 6/f がピーク
                 kf(END, center=(387.0, 0.0, 0.0), interp_block=L)]  # 10..119: 3/f = 半速
        sfull = [kf(0, center=(0.0, 0.0, 0.0), interp_block=L),
                 kf(END, center=(238.0, 0.0, 0.0), interp_block=L)]  # 全域 2/f 等速 = 正規化1
        win = slice(20, END + 1)  # frame20.. が静止(s0)/半速(shalf)/全速(sfull)、100サンプル

        def e03_sum(src):
            res = by_frame(bake.bake(src, seed=5, amp_rot=0.0, amp_pos=2.0,
                                     motion_scale=0.0, settle=0.0, fade_sec=0.3))
            return self._e03_sum(res, src, "pos", END, win)

        assert e03_sum(s0) > e03_sum(shalf) > e03_sum(sfull)


class TestWalkingGait:
    """walking プリセットの歩調周期成分(§2.7/§95): 乱数ノイズに加え、位置の左右(pos_x)へ
    歩調周波数 f、上下(pos_y)へ 2f の周期成分を混合する。コア引数 gait_freq/gait_amp で制御
    (既定 0=無効)。奥行(pos_z)・回転には載せない。"""

    N = 99
    _LIN = bytes([20, 107, 20, 107]) * 6

    def _static(self):
        # 完全静止の土台。位置揺れ・回転揺れを切り、歩調成分だけを観測する。
        return [kf(0, center=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.N, center=(0.0, 0.0, 0.0), interp_block=self._LIN)]

    def _pos(self, res, src, axis):
        return np.array([res[f].position[axis] - interp.sample_camera(src, f)["position"][axis]
                         for f in range(self.N + 1)])

    def _rot(self, res, src, axis):
        return np.array([res[f].rotation[axis] - interp.sample_camera(src, f)["rotation"][axis]
                         for f in range(self.N + 1)])

    @staticmethod
    def _dom_freq(series):
        # DC を除く最大振幅ビンの周波数。
        spec = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(series.size, d=1.0 / 30.0)
        return float(freqs[1 + int(np.argmax(spec[1:]))])

    @staticmethod
    def _bin_amp(series, freq):
        # freq 最近傍ビンの振幅。
        spec = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(series.size, d=1.0 / 30.0)
        return float(spec[int(np.argmin(np.abs(freqs - freq)))])

    @staticmethod
    def _maxabs(series):
        # 最大絶対値。定数オフセット(DC)も検出する(std だと見逃す。round4 指摘)。
        return float(np.max(np.abs(series)))

    def test_gait_tracks_freq_lr_f_updown_2f(self):
        # 左右(x)=歩調f、上下(y)=2f、奥行(z)=なし(§95)。gait_freq を無視して固定周波数を
        # 出力する実装を排除するため、2つの f(1.8/2.4Hz、100サンプルで整数周期)で確認する。
        # 支配周波数だけでなく軸別成分分離も課す: x は f が 2f を大きく上回り、y は 2f が f を
        # 大きく上回る(両軸に両周波数を載せる実装を排除)。amp_pos=0/amp_rot=0 で歩調のみ分離。
        src = self._static()
        for f in (1.8, 2.4):
            res = by_frame(bake.bake(src, seed=1, amp_rot=0.0, amp_pos=0.0,
                                     gait_freq=f, gait_amp=1.0, settle=0.0, fade_sec=0.1))
            x, y, z = (self._pos(res, src, ax) for ax in range(3))
            assert abs(self._dom_freq(x) - f) < 0.16, (f, "x")
            assert abs(self._dom_freq(y) - 2 * f) < 0.16, (f, "y")
            assert self._bin_amp(x, f) > 5.0 * self._bin_amp(x, 2 * f), (f, "x-sep")
            assert self._bin_amp(y, 2 * f) > 5.0 * self._bin_amp(y, f), (f, "y-sep")
            assert self._maxabs(z) < 1e-6, (f, "z")   # 奥行に歩調なし(定数オフセットも不可)

    def test_gait_amplitude_scales_with_gait_amp(self):
        # 振幅は gait_amp(position units)に比例(§95)。gait_amp を無視して固定振幅にする実装を
        # 排除する。gait_amp=0 は歩調なし、gait_amp=2 は =1 の概ね2倍。左右(x)・上下(y)両軸で確認。
        src = self._static()

        def res_for(gait_amp):
            return by_frame(bake.bake(src, seed=1, amp_rot=0.0, amp_pos=0.0,
                                      gait_freq=1.8, gait_amp=gait_amp, settle=0.0, fade_sec=0.1))

        r0, r1, r2 = res_for(0.0), res_for(1.0), res_for(2.0)
        for axis in (0, 1):
            assert self._maxabs(self._pos(r0, src, axis)) < 1e-9, f"axis{axis} zero"  # gait_amp=0→無
            s1 = float(np.std(self._pos(r1, src, axis)))
            s2 = float(np.std(self._pos(r2, src, axis)))
            assert s1 > 1e-6 and abs(s2 - 2.0 * s1) < 0.005 * s1, f"axis{axis} scale"  # 線形

    def test_gait_not_applied_to_rotation(self):
        # 歩調は位置のみ(§95 左右/上下)。回転には載せない。amp_rot=0 で回転の乱数も切り、
        # 歩調 ON でも回転揺れがゼロ(定数オフセット含め)のままであることを確認する。
        src = self._static()
        res = by_frame(bake.bake(src, seed=1, amp_rot=0.0, amp_pos=0.0,
                                 gait_freq=1.8, gait_amp=1.0, settle=0.0, fade_sec=0.1))
        for ax in range(3):
            assert self._maxabs(self._rot(res, src, ax)) < 1e-9, f"rot{ax}"

    def test_gait_mixed_with_random_noise_not_replacing(self):
        # 「乱数ノイズに歩調成分を混合」(§95): 歩調は乱数へ加算(置換ではない)。
        # (1) 差分が左右=f・上下=2f に集中(乱数が相殺=同一 seed で共有 → 混合の証拠)。
        # (2) 歩調 ON/OFF の時間波形の相関が高い(乱数が保持される。FFT リーケージに依らない)。
        #     置換実装は on=歩調のみ → 相関≈0 で落ちる。乱数 ≥ 歩調 になるよう振幅を選ぶ。
        # (3) 歩調を載せない z 位置・回転は ON/OFF で完全一致(歩調が他チャンネルを変えない)。
        src = self._static()
        kw = dict(amp_rot=2.0, amp_pos=1.0, settle=0.0, fade_sec=0.1)
        on = by_frame(bake.bake(src, seed=2, gait_freq=1.8, gait_amp=0.2, **kw))
        off = by_frame(bake.bake(src, seed=2, gait_freq=0.0, gait_amp=0.2, **kw))
        on_x, off_x = self._pos(on, src, 0), self._pos(off, src, 0)
        on_y, off_y = self._pos(on, src, 1), self._pos(off, src, 1)
        assert abs(self._dom_freq(on_x - off_x) - 1.8) < 0.16, ("x", self._dom_freq(on_x - off_x))
        assert abs(self._dom_freq(on_y - off_y) - 3.6) < 0.16, ("y", self._dom_freq(on_y - off_y))
        assert float(np.corrcoef(on_x, off_x)[0, 1]) > 0.5   # x 乱数保持(置換でない)
        assert float(np.corrcoef(on_y, off_y)[0, 1]) > 0.5   # y 乱数保持
        # z 位置・回転3軸は歩調対象外 → ON/OFF で完全一致(歩調が抑制/改変しない)
        assert np.allclose(self._pos(on, src, 2), self._pos(off, src, 2), atol=1e-12)
        for ax in range(3):
            assert np.allclose(self._rot(on, src, ax), self._rot(off, src, ax), atol=1e-12), f"rot{ax}"

    def test_gait_off_when_freq_zero(self):
        # gait_freq=0 → 歩調成分なし(§95「0=off」)。既定省略時と明示 0 の両方で、gait_amp>0 でも
        # 位置揺れが消える(定数オフセットも不可)。amp_pos=0 で乱数も切り絶対ゼロを検証する。
        src = self._static()
        # (a) gait_freq 省略(既定 0)
        r_default = by_frame(bake.bake(src, seed=1, amp_rot=0.0, amp_pos=0.0,
                                       gait_amp=1.0, settle=0.0, fade_sec=0.1))
        # (b) gait_freq=0 を明示
        r_explicit = by_frame(bake.bake(src, seed=1, amp_rot=0.0, amp_pos=0.0,
                                        gait_freq=0.0, gait_amp=1.0, settle=0.0, fade_sec=0.1))
        for res in (r_default, r_explicit):
            assert self._maxabs(self._pos(res, src, 0)) < 1e-9
            assert self._maxabs(self._pos(res, src, 1)) < 1e-9


class TestCoreApiTuning:
    """§8/§2.5: 内蔵パラメータ(静止/移動プロファイル=オクターブ重み構成、settle 収束時間)を
    bake() 引数で調整できる(coreAPI)。既定は従来の内蔵定数で挙動不変。
    オクターブ数はプロファイル長で決まる(別引数を設けず連動を一本化)。"""

    N = 99
    _LIN = bytes([20, 107, 20, 107]) * 6

    def _static(self):
        return [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.N, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN)]

    def _moving(self):
        return [kf(0, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), interp_block=self._LIN),
                kf(self.N, center=(60.0, 0.0, 0.0), rotation=(0.0, 1.5, 0.0), interp_block=self._LIN)]

    def _rot_hf(self, res, src, axis=0):
        # 回転チャンネルの相対高周波(呼吸=位置のみなので汚染なし)。内部窓で端のフェードを避ける。
        r = np.array([res[f].rotation[axis] - interp.sample_camera(src, f)["rotation"][axis]
                      for f in range(12, self.N - 11)])
        return float(np.std(np.diff(r)) / (np.std(r) + 1e-12))

    def _pos_hf(self, res, src, axis=0):
        # 位置残差の相対高周波。移動レジーム(呼吸≈0)で使う(静止は呼吸が std を汚染するため不可)。
        p = np.array([res[f].position[axis] - interp.sample_camera(src, f)["position"][axis]
                      for f in range(12, self.N - 11)])
        return float(np.std(np.diff(p)) / (np.std(p) + 1e-12))

    def test_still_profile_tunes_static_octave_weights(self):
        # 静止セグメントのオクターブ重み = still_profile。moving_profile は固定し still のみ変える。
        # 静止で moving_profile を使う誤実装は hi==lo になり落ちる(round1 指摘)。
        src = self._static()
        common = dict(seed=1, amp_rot=5.0, amp_pos=0.0, settle=0.0, fade_sec=0.1,
                      moving_profile=motion.MOVING_PROFILE)
        hi = by_frame(bake.bake(src, still_profile=(1.0, 1.0, 1.0), **common))
        lo = by_frame(bake.bake(src, still_profile=(1.0, 0.05, 0.01), **common))
        for ax in range(3):   # 全回転軸(x のみ効かせる実装を排除。round6)
            assert self._rot_hf(hi, src, ax) > self._rot_hf(lo, src, ax), ax

    def test_moving_profile_tunes_moving_octave_weights(self):
        # 移動セグメントのオクターブ重み = moving_profile。still_profile は固定し moving のみ変える。
        # 移動で still_profile を使う誤実装は hi==lo になり落ちる(round1 指摘)。
        src = self._moving()
        common = dict(seed=1, amp_rot=5.0, amp_pos=0.0, settle=0.0, fade_sec=0.1,
                      still_profile=motion.STILL_PROFILE)
        hi = by_frame(bake.bake(src, moving_profile=(1.0, 1.0, 1.0), **common))
        lo = by_frame(bake.bake(src, moving_profile=(1.0, 0.05, 0.01), **common))
        for ax in range(3):   # 全回転軸
            assert self._rot_hf(hi, src, ax) > self._rot_hf(lo, src, ax), ax

    def test_profiles_apply_to_position_channels(self, monkeypatch):
        # プロファイルは回転だけでなく位置チャンネルにも適用される(§2.5 は generic。round2#1)。
        # 静止(still_profile)・移動(moving_profile)両レジームで確認する(位置を片側に固定する実装を
        # 排除。round5)。呼吸を切って(BREATHING_AMP_FACTOR=0)位置 std の汚染を除き、相対高周波で判別。
        monkeypatch.setattr(motion, "BREATHING_AMP_FACTOR", 0.0)
        for src, vary in ((self._static(), "still_profile"), (self._moving(), "moving_profile")):
            common = dict(seed=1, amp_rot=0.0, amp_pos=1.0, settle=0.0, fade_sec=0.1)
            hi_kw = {"still_profile": motion.STILL_PROFILE, "moving_profile": motion.MOVING_PROFILE}
            lo_kw = dict(hi_kw)
            hi_kw[vary] = (1.0, 1.0, 1.0)
            lo_kw[vary] = (1.0, 0.05, 0.01)
            hi = by_frame(bake.bake(src, **common, **hi_kw))
            lo = by_frame(bake.bake(src, **common, **lo_kw))
            for ax in range(3):   # 全位置軸(x のみ効かせる実装を排除。round6)
                assert self._pos_hf(hi, src, ax) > self._pos_hf(lo, src, ax), (vary, ax)

    def test_octave_count_follows_profile_length(self):
        # オクターブ数 = プロファイル長。長さ3(オクターブ 1.2/2.4/4.8Hz)は長さ2(1.2/2.4Hz)より
        # 高周波成分が多い。相対高周波(std(diff)/std、勾配ノイズの分散に頑健)で判別する。
        # オクターブ数をプロファイル長に追従させる実装でのみ成立(DEFAULT_OCTAVES=3 固定だと長さ2の
        # 重み列(2列)と不整合でクラッシュする)。静止(still 長)・移動(moving 長)の両レジームで確認。
        common = dict(seed=1, amp_rot=5.0, amp_pos=0.0, freq=1.2, settle=0.0, fade_sec=0.1)
        for src in (self._static(), self._moving()):
            o2 = by_frame(bake.bake(src, still_profile=(1.0, 1.0), moving_profile=(1.0, 1.0), **common))
            o3 = by_frame(bake.bake(src, still_profile=(1.0, 1.0, 1.0),
                                    moving_profile=(1.0, 1.0, 1.0), **common))
            for ax in range(3):   # 全回転軸で 3オクターブ > 2オクターブ
                assert self._rot_hf(o3, src, ax) > self._rot_hf(o2, src, ax), ax

    def test_settle_time_tunes_decay_rate_not_gain(self):
        # settle 収束時間: 長いほど減衰が遅い。後半/前半エネルギー比で判別する(振幅ゲインだと比は
        # 不変=落ちる。round1 指摘#3)。amp_rot=0 で基本ノイズを切り settle 成分(回転)だけ観測。
        # PAN_STOP は frame30 で停止。
        def late_over_early(settle_time):
            res = by_frame(bake.bake(PAN_STOP, seed=1, amp_rot=0.0, amp_pos=0.0,
                                     settle=2.0, settle_time=settle_time, fade_sec=0.1))
            dev = [res[f].rotation[1] - interp.sample_camera(PAN_STOP, f)["rotation"][1]
                   for f in range(31, 58)]
            early = float(np.sum(np.array(dev[:13]) ** 2))   # frames 31..43(停止直後)
            late = float(np.sum(np.array(dev[14:]) ** 2))    # frames 45..57(後半)
            return late / (early + 1e-15)
        assert late_over_early(2.0) > late_over_early(0.3)

    def test_defaults_match_internal_constants(self):
        # 既定省略 == 内蔵定数を明示、で出力一致(引数化が挙動を変えない保証)。PAN_STOP は移動区間・
        # 静止区間・停止を全て含むので still_profile/moving_profile/settle_time 既定をすべて行使する。
        a = bake.bake(PAN_STOP, seed=1, settle=2.0).camera_keys
        b = bake.bake(PAN_STOP, seed=1, settle=2.0, still_profile=motion.STILL_PROFILE,
                      moving_profile=motion.MOVING_PROFILE,
                      settle_time=motion.DEFAULT_SETTLE_TIME_SEC).camera_keys
        assert a == b
        # 既定は内蔵定数そのものに束縛されている(別値を既定にする誤配線を排除。round3#2)。
        sig = inspect.signature(bake.bake).parameters
        assert sig["still_profile"].default is motion.STILL_PROFILE
        assert sig["moving_profile"].default is motion.MOVING_PROFILE
        assert sig["settle_time"].default == motion.DEFAULT_SETTLE_TIME_SEC

    def test_profile_length_mismatch_raises(self):
        # still/moving のプロファイル長は一致が必要(オクターブ数の整合)。両方向の不一致で ValueError
        # (片方向だけ検証する非対称バリデータを排除。round2 指摘#3)。
        with pytest.raises(ValueError):
            bake.bake(self._static(), still_profile=(1.0, 0.5), moving_profile=(1.0, 0.5, 0.25))
        with pytest.raises(ValueError):
            bake.bake(self._static(), still_profile=(1.0, 0.5, 0.25), moving_profile=(1.0, 0.5))
