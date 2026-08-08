"""motion の確定的コアのテスト(フェード, 速度適応)。

このファイルは motion の確定的コア(フェード・フレーム速度・停止検出・適応振幅・settle 過渡・
インパルス包絡・プロファイルクロスフェード・呼吸ドリフト)を対象とする。
"""

import numpy as np
import pytest

from shakevmd import motion


def _ref_smoothstep(t):
    # 仕様から独立に書いた参照(3t^2-2t^3)
    t = np.asarray(t, dtype=float)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# fade_envelope
# ---------------------------------------------------------------------------


class TestFadeEnvelope:
    def test_length_and_zero_ends(self):
        env = motion.fade_envelope(100, fade_sec=0.5, fps=30.0)
        assert env.shape == (100,)
        assert env[0] == pytest.approx(0.0, abs=1e-9)
        assert env[-1] == pytest.approx(0.0, abs=1e-9)

    def test_plateau_is_one(self):
        # 中央(フェード区間の外)は 1
        env = motion.fade_envelope(100, fade_sec=0.3, fps=30.0)
        assert env[50] == pytest.approx(1.0, abs=1e-9)
        assert np.max(env) == pytest.approx(1.0, abs=1e-9)

    def test_monotonic_in_out(self):
        env = motion.fade_envelope(100, fade_sec=0.5, fps=30.0)
        half = len(env) // 2
        assert np.all(np.diff(env[:half]) >= -1e-12)   # 立ち上がり: 非減少
        assert np.all(np.diff(env[half:]) <= 1e-12)    # 立ち下がり: 非増加

    def test_bounded_0_1(self):
        env = motion.fade_envelope(80, fade_sec=0.6, fps=30.0)
        assert np.all(env >= -1e-12) and np.all(env <= 1.0 + 1e-12)

    def test_ramp_follows_smoothstep(self):
        # 立ち上がり f フレームは smoothstep(linspace(0,1,f)) 形状、立ち下がりはその反転
        n, f = 120, 15  # fade 0.5s × 30fps = 15、2×15=30 <= 120 なので短縮なし
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        ref = _ref_smoothstep(np.linspace(0.0, 1.0, f))
        assert np.allclose(env[:f], ref, atol=1e-9)
        assert np.allclose(env[n - f:], ref[::-1], atol=1e-9)

    def test_velocity_continuous_at_both_ends(self):
        # smoothstep の端微分0 により、両端の初差分(速度)が小さい(0接続で速度連続)
        env = motion.fade_envelope(120, fade_sec=0.5, fps=30.0)
        assert abs(env[1] - env[0]) < 0.02
        assert abs(env[-1] - env[-2]) < 0.02

    def test_short_segment_shortens_fade_to_half(self):
        # 2×fade(=30フレーム)未満のセグメントはフェードを n//2 に短縮。
        # n=20 → f=10。立ち上がりは smoothstep(linspace(0,1,10))、中央(index 9,10)で1。
        n = 20
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        ref = _ref_smoothstep(np.linspace(0.0, 1.0, 10))
        assert np.allclose(env[:10], ref, atol=1e-9)         # 立ち上がり
        assert np.allclose(env[10:], ref[::-1], atol=1e-9)   # 立ち下がり(反転)
        assert env[9] == pytest.approx(1.0, abs=1e-9)
        assert env[10] == pytest.approx(1.0, abs=1e-9)
        assert env[0] == pytest.approx(0.0, abs=1e-9)
        assert env[-1] == pytest.approx(0.0, abs=1e-9)

    def test_odd_short_segment_structure(self):
        # 奇数短尺 n=21 → f=10。立ち上がり env[:10]=ramp(末尾1)、index10 は untouched の1、
        # 立ち下がり env[11:]=ramp 反転(先頭1)。よって index 9,10,11 が連続して1になる。
        n = 21
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        ref = _ref_smoothstep(np.linspace(0.0, 1.0, 10))
        assert np.allclose(env[:10], ref, atol=1e-9)
        assert env[10] == pytest.approx(1.0, abs=1e-9)        # untouched plateau
        assert np.allclose(env[11:], ref[::-1], atol=1e-9)
        assert np.allclose(env[9:12], 1.0, atol=1e-9)         # 連続して1の領域
        assert env[0] == pytest.approx(0.0, abs=1e-9)
        assert env[-1] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, []),
            (1, [0.0]),
            (2, [0.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
        ],
    )
    def test_tiny_segments(self, n, expected):
        # 極短セグメントの挙動を定義: 端は常に0、間があれば1
        env = motion.fade_envelope(n, fade_sec=0.5, fps=30.0)
        assert env.shape == (n,)
        assert np.allclose(env, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# frame_speeds
# ---------------------------------------------------------------------------


class TestFrameSpeeds:
    # frame_speeds(values, ref): 正規化速度 = clamp(|フレーム間差分| / ref, 0, 1)。
    # セグメント内ピークでは割らない(絶対基準)。speed[0]=0。
    def test_static_is_zero(self):
        speeds = motion.frame_speeds(np.full(50, 3.0), 1.0)
        assert np.allclose(speeds, 0.0)

    def test_absolute_not_peak_normalized(self):
        # 基準より十分遅い一定運動は、ピーク正規化のように1へ増幅されず小さいまま。
        # (これがバグ修正の核: 同セグメントに速い動きがあっても遅い動きは過大評価されない)
        speeds = motion.frame_speeds(np.arange(0, 5, 1.0), ref=10.0)  # 生速度=1.0/フレーム
        assert np.allclose(speeds[1:], 0.1)        # 1.0/10.0、ピークでも1にならない
        assert np.max(speeds) == pytest.approx(0.1)

    def test_proportional_to_ref(self):
        # 基準未満では絶対速度に比例(差分2.0、ref=8 → 0.25)。
        speeds = motion.frame_speeds(np.arange(0, 12, 2.0), ref=8.0)
        assert np.allclose(speeds[1:], 0.25)

    def test_clamped_at_ref(self):
        # 基準以上の速度は1でクランプ(それ以上速くしても1)。
        speeds = motion.frame_speeds(np.array([0.0, 20.0, 60.0]), ref=5.0)
        assert np.allclose(speeds, [0.0, 1.0, 1.0])

    def test_relative_magnitude_preserved_below_ref(self):
        # 基準未満では後半の2倍速が正規化後も約2倍を保つ。
        vals = np.concatenate([np.arange(0, 10, 1.0), np.arange(10, 30, 2.0)])
        speeds = motion.frame_speeds(vals, ref=8.0)   # 速度1と2は基準未満
        slow = np.median(speeds[2:9])
        fast = np.median(speeds[12:19])
        assert fast == pytest.approx(2.0 * slow, rel=0.2)

    def test_length_matches(self):
        speeds = motion.frame_speeds(np.zeros(37), 1.0)
        assert speeds.shape == (37,)

    def test_frame_alignment_explicit(self):
        # speed[i] = |x[i]-x[i-1]| / ref、speed[0]=0。値が変化したフレームに速度が現れる。
        vals = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        speeds = motion.frame_speeds(vals, ref=10.0)
        assert np.allclose(speeds, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def test_scalar_speed_is_absolute(self):
        # 速度は差の「大きさ」。減少列でも絶対値で扱う(差分1.0、ref=1 → 1.0)。
        vals = np.array([3.0, 2.0, 1.0])
        speeds = motion.frame_speeds(vals, ref=1.0)
        assert np.allclose(speeds, [0.0, 1.0, 1.0])

    def test_vector_norm_is_euclidean(self):
        # ベクトル差の大きさはユークリッドノルム([3,4,0] の差 → 5、ref=5 → 1)。
        vals = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
        speeds = motion.frame_speeds(vals, ref=5.0)
        assert np.allclose(speeds, [0.0, 1.0, 0.0])

    def test_nonpositive_ref_is_zero(self):
        # ref<=0 は退避: 全0(モーション適応無効)。0 も負も同様(abs(speed/ref) 等の誤実装を排除)。
        for ref in (0.0, -3.0, -1e-9):
            speeds = motion.frame_speeds(np.arange(0, 5, 1.0), ref=ref)
            assert np.allclose(speeds, 0.0), ref


# ---------------------------------------------------------------------------
# adaptive_amplitude
# ---------------------------------------------------------------------------


class TestAdaptiveAmplitude:
    def test_zero_speed_is_base(self):
        # 静止(速度0)は減衰なし=基本振幅のまま
        assert motion.adaptive_amplitude(0.8, 0.0, motion_damp=1.0) == pytest.approx(0.8)

    def test_full_speed_damps_by_motion_damp(self):
        # 速度1・motion_damp=0.5 → base×(1-0.5)=0.4(動くほど減衰)
        assert motion.adaptive_amplitude(0.8, 1.0, motion_damp=0.5) == pytest.approx(0.4)

    def test_motion_damp_zero_disables(self):
        # motion_damp=0 は減衰なし=基本振幅(速度に依らず一定)
        assert motion.adaptive_amplitude(0.8, 1.0, motion_damp=0.0) == pytest.approx(0.8)

    def test_clamped_to_zero(self):
        # motion_damp×速度 > 1 でも負にならず 0 でクランプ(振幅は 0 まで)
        assert motion.adaptive_amplitude(0.8, 1.0, motion_damp=2.0) == pytest.approx(0.0)

    def test_array_input(self):
        out = motion.adaptive_amplitude(1.0, np.array([0.0, 0.5, 1.0]), motion_damp=0.4)
        assert np.allclose(out, [1.0, 0.8, 0.6])


# ---------------------------------------------------------------------------
# detect_stops(停止検出)
# ---------------------------------------------------------------------------


class TestDetectStops:
    def test_crossing_below_is_stop(self):
        # 0.5,0.5 (動) → 0.0,0.0 (停止): index2 で閾値0.1を上から下へ横切る
        speeds = np.array([0.5, 0.5, 0.0, 0.0])
        assert motion.detect_stops(speeds, threshold=0.1) == [2]

    def test_no_stop_when_always_moving(self):
        assert motion.detect_stops(np.full(10, 0.5), threshold=0.1) == []

    def test_no_stop_when_always_stopped(self):
        assert motion.detect_stops(np.zeros(10), threshold=0.1) == []

    def test_multiple_stops(self):
        # 動→停→動→停
        speeds = np.array([0.5, 0.0, 0.0, 0.5, 0.5, 0.0])
        assert motion.detect_stops(speeds, threshold=0.1) == [1, 5]

    def test_only_downward_crossing(self):
        # 停→動(下から上)は停止点ではない
        speeds = np.array([0.0, 0.0, 0.5, 0.5])
        assert motion.detect_stops(speeds, threshold=0.1) == []

    def test_threshold_boundary(self):
        # 境界: speeds[i-1] >= threshold かつ speeds[i] < threshold で停止
        # [0.1, 0.0] (th=0.1): 0.1>=0.1 かつ 0.0<0.1 → 停止 @1
        assert motion.detect_stops(np.array([0.1, 0.0]), threshold=0.1) == [1]
        # [0.5, 0.1] (th=0.1): 0.1<0.1 が偽 → 停止でない
        assert motion.detect_stops(np.array([0.5, 0.1]), threshold=0.1) == []


# ---------------------------------------------------------------------------
# settle_oscillation(停止過渡)
# ---------------------------------------------------------------------------


class TestSettleOscillation:
    def test_zero_at_start(self):
        assert motion.settle_oscillation(0.0, amp=0.3) == pytest.approx(0.0, abs=1e-9)

    def test_matches_formula(self):
        # 設計式 amp·exp(-t/(settle_time/4))·sin(2π·freq·t) を既知点で固定。
        # freq は明示指定(暫定デフォルトに依存しない)。
        amp, st, freq = 0.3, 1.0, 3.0
        t = np.linspace(0.0, 2.0, 257)
        vals = np.asarray(motion.settle_oscillation(t, amp=amp, settle_time=st, freq=freq))
        expected = amp * np.exp(-t / (st / 4.0)) * np.sin(2 * np.pi * freq * t)
        assert np.allclose(vals, expected, atol=1e-9)

    def test_envelope_upper_bound(self):
        # |値| <= 包絡 amp·exp(-t/(settle_time/4))(freq に依存しない上限)
        amp, st, freq = 0.3, 1.0, 3.0
        t = np.linspace(0.0, 2.0, 400)
        vals = np.asarray(motion.settle_oscillation(t, amp=amp, settle_time=st, freq=freq))
        env = amp * np.exp(-t / (st / 4.0))
        assert np.all(np.abs(vals) <= env + 1e-9)

    def test_converged_after_settle_time(self):
        # settle_time 以降は包絡(exp(-4)≈1.8%)で抑えられ、全点で初期振幅の2%未満。
        # 位相に依存しないよう [settle_time, 2·settle_time] の範囲で評価する。
        amp, st, freq = 0.3, 1.0, 3.0
        t = np.linspace(st, 2 * st, 200)
        vals = np.asarray(motion.settle_oscillation(t, amp=amp, settle_time=st, freq=freq))
        assert np.all(np.abs(vals) < 0.02 * amp)

    def test_oscillates(self):
        # 振動する(符号変化が複数回)。freq は明示指定。
        amp, st, freq = 0.3, 1.0, 3.0
        t = np.linspace(0.0, 1.0, 600)
        vals = np.asarray(motion.settle_oscillation(t, amp=amp, settle_time=st, freq=freq))
        nonzero = vals[np.abs(vals) > 1e-6]
        sign_changes = int(np.sum(np.diff(np.sign(nonzero)) != 0))
        assert sign_changes >= 2

    def test_deterministic(self):
        a = motion.settle_oscillation(0.3, amp=0.3, freq=3.0)
        b = motion.settle_oscillation(0.3, amp=0.3, freq=3.0)
        assert a == b


# ---------------------------------------------------------------------------
# impulse_envelope
# ---------------------------------------------------------------------------


class TestImpulseEnvelope:
    def test_zero_before_frame(self):
        env = motion.impulse_envelope(60, frame=30, strength=2.0, decay_sec=0.5, fps=30.0)
        assert env.shape == (60,)
        assert np.allclose(env[:30], 0.0)

    def test_peak_at_frame(self):
        env = motion.impulse_envelope(60, frame=30, strength=2.0, decay_sec=0.5, fps=30.0)
        assert env[30] == pytest.approx(2.0, abs=1e-9)

    def test_exponential_decay(self):
        # decay_sec 経過(= D×fps フレーム後)で strength/e
        env = motion.impulse_envelope(120, frame=10, strength=2.0, decay_sec=1.0, fps=30.0)
        assert env[10 + 30] == pytest.approx(2.0 / np.e, rel=1e-6)

    def test_monotonic_decay_after_frame(self):
        env = motion.impulse_envelope(120, frame=10, strength=2.0, decay_sec=1.0, fps=30.0)
        assert np.all(np.diff(env[10:]) <= 1e-12)

    def test_matches_exponential_formula(self):
        # frame 以降の複数点で strength·exp(-Δt/decay_sec) と一致(線形減衰等を排除)
        F, S, D, fps = 10, 2.0, 0.8, 30.0
        env = motion.impulse_envelope(120, frame=F, strength=S, decay_sec=D, fps=fps)
        idx = np.arange(F, 120)
        dt = (idx - F) / fps
        assert np.allclose(env[F:], S * np.exp(-dt / D), atol=1e-9)


class TestProfileCrossfade:
    """静止/移動プロファイルのオクターブ重みクロスフェード。"""

    def test_endpoints_and_midpoint(self):
        # s=0→still、s=1→moving、s=0.5→中点(線形ブレンド)。
        assert list(motion.profile_weights(0.0)) == pytest.approx(list(motion.STILL_PROFILE))
        assert list(motion.profile_weights(1.0)) == pytest.approx(list(motion.MOVING_PROFILE))
        mid = motion.profile_weights(0.5)
        expect = [(motion.STILL_PROFILE[i] + motion.MOVING_PROFILE[i]) / 2
                  for i in range(len(motion.STILL_PROFILE))]
        assert list(mid) == pytest.approx(expect)
        # 非対称点(s=0.25)で線形ブレンドを確認(smoothstep 等の非線形対称を排除)。
        q = motion.profile_weights(0.25)
        expect_q = [0.75 * motion.STILL_PROFILE[i] + 0.25 * motion.MOVING_PROFILE[i]
                    for i in range(len(motion.STILL_PROFILE))]
        assert list(q) == pytest.approx(expect_q)

    def test_array_input_per_frame(self):
        # 配列入力 → (n_frames, n_oct)。各行が対応速度のブレンド(中間 s=0.5 も式どおり)。
        s = np.array([0.0, 0.5, 1.0])
        w = np.asarray(motion.profile_weights(s))
        assert w.shape == (3, len(motion.STILL_PROFILE))
        assert list(w[0]) == pytest.approx(list(motion.STILL_PROFILE))
        assert list(w[-1]) == pytest.approx(list(motion.MOVING_PROFILE))
        mid = [(motion.STILL_PROFILE[i] + motion.MOVING_PROFILE[i]) / 2
               for i in range(len(motion.STILL_PROFILE))]
        assert list(w[1]) == pytest.approx(mid)

    def test_profiles_differ_moving_has_more_high_freq(self):
        # 2プロファイルは異なり(クロスフェードが意味を持つ)、移動は高オクターブ重みが大きい
        # (移動時はより細かい=高周波の揺れ)。
        assert tuple(motion.STILL_PROFILE) != tuple(motion.MOVING_PROFILE)
        assert motion.MOVING_PROFILE[-1] > motion.STILL_PROFILE[-1]
        # 完全静止区間も「高周波微動」を残すため、静止プロファイルの高オクターブ重みは非ゼロ。
        assert motion.STILL_PROFILE[-1] > 0.0


class TestBreathingDrift:
    """完全静止区間の長周期ドリフト(呼吸 0.3Hz)。"""

    def test_frequency_and_amplitude(self):
        amp = 0.5
        assert motion.BREATHING_HZ == pytest.approx(0.3)   # 呼吸0.3Hz相当(被検定数を固定)
        period = 1.0 / 0.3                                 # spec 値の literal(定数由来にしない)
        t = np.linspace(0.0, 2.0 * period, 2001)
        x = np.asarray(motion.breathing_drift(t, amp))
        assert np.max(np.abs(x)) <= amp + 1e-9                 # 振幅有界
        q = float(np.asarray(motion.breathing_drift(np.array([period / 4.0]), amp))[0])
        assert q == pytest.approx(amp, abs=1e-3)               # 1/4周期で最大=amp(sin)
        z = float(np.asarray(motion.breathing_drift(np.array([period]), amp))[0])
        assert abs(z) < 1e-3                                   # 1周期で約0
        assert float(np.asarray(motion.breathing_drift(np.array([0.0]), amp))[0]) == pytest.approx(0.0, abs=1e-9)
        # 3/4周期で sin=-1 → -amp(負の半周期。abs(sin)/半波整流を排除)。
        neg = float(np.asarray(motion.breathing_drift(np.array([3.0 * period / 4.0]), amp))[0])
        assert neg == pytest.approx(-amp, abs=1e-3)
