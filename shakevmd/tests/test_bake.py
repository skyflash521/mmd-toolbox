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
        # 振幅0 → ベイク結果は原本の再サンプリングと一致(全チャンネル+補間線形+パース)
        res = by_frame(bake.bake(SEQ, seed=1, amp_rot=0.0, amp_pos=0.0))
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
