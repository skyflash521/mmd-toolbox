"""モーション適応・境界フェード・停止過渡・インパルス(shakevmd.md §5.1, §6.2, §6.3)。

このモジュールの数値パラメータ(既定値)は実装後に調整する前提の暫定値。
速度解析・正規化・プロファイルブレンドはカットで分割されたセグメント単位で行う
(カットをまたぐ差分は含めない。§5.3)。
"""

import numpy as np

FPS = 30.0

# 既定値(暫定。実装完了後に調整)
DEFAULT_MOTION_SCALE = 0.5   # 実効振幅 = 基本振幅 ×(1 + motion_scale × 正規化速度)
DEFAULT_FADE_SEC = 0.7       # 範囲端の自動フェード時間
DEFAULT_SETTLE_TIME_SEC = 1.0       # settle 減衰振動の収束時間(内蔵)
DEFAULT_SETTLE_FREQ_HZ = 2.5        # settle 振動の周波数(内蔵)
DEFAULT_STOP_SPEED_THRESHOLD = 0.1  # 正規化速度がこれを下回ると停止とみなす(内蔵)

# 静止/移動プロファイル(§6.2)。オクターブ重みベクトル(長さ=noise.DEFAULT_OCTAVES=3)。
# 静止: 低周波寄り(急減衰=落ち着いた揺れ)。移動: 高周波寄り(緩減衰=細かい揺れ)。暫定値。
STILL_PROFILE = (1.0, 0.25, 0.0625)
MOVING_PROFILE = (1.0, 0.6, 0.36)
BREATHING_HZ = 0.3                  # 完全静止区間の長周期ドリフト周波数(§6.2 呼吸相当、内蔵)
BREATHING_AMP_FACTOR = 0.5          # 呼吸ドリフト振幅 = amp_pos × この係数(内蔵、暫定)


def profile_weights(normalized_speed, still=STILL_PROFILE, moving=MOVING_PROFILE):
    """正規化速度に応じて静止/移動プロファイルのオクターブ重みをクロスフェードする(§6.2)。

    weight_i = (1 - s)·still_i + s·moving_i。s はスカラーまたは配列([0,1] 想定)。
    戻り値: スカラー入力なら (n_oct,) の np.ndarray、配列入力なら (n_frames, n_oct)。
    """
    s = np.asarray(normalized_speed, dtype=float)
    still = np.asarray(still, dtype=float)
    moving = np.asarray(moving, dtype=float)
    if s.ndim == 0:
        return (1.0 - s) * still + s * moving
    # 配列入力: 各フレーム s[k] で全オクターブをブレンド → (n_frames, n_oct)
    return (1.0 - s)[:, None] * still[None, :] + s[:, None] * moving[None, :]


def breathing_drift(t_sec, amp: float, freq: float = BREATHING_HZ):
    """完全静止区間の長周期ドリフト(呼吸相当、§6.2)。amp·sin(2π·freq·t)。t_sec は秒。"""
    t = np.asarray(t_sec, dtype=float)
    return amp * np.sin(2.0 * np.pi * freq * t)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """3t^2-2t^3。端で値0/1かつ微分0(位置と速度の連続性。§5.1)。"""
    return t * t * (3.0 - 2.0 * t)


def fade_envelope(n_frames: int, fade_sec: float = DEFAULT_FADE_SEC, fps: float = FPS) -> np.ndarray:
    """セグメント長 n_frames のフェード包絡 [0,1] を返す(§5.1)。

    両端は厳密に0、中央は1。立ち上がり/立ち下がりはイーズイン/アウト(smoothstep)で
    位置・速度を連続接続する。セグメントが 2×fade 秒に満たない場合はフェードを
    その半分(n_frames//2)に自動短縮する。
    """
    if n_frames <= 0:
        return np.zeros(0)
    f0 = int(round(fade_sec * fps))
    f = f0 if 2 * f0 <= n_frames else n_frames // 2
    env = np.ones(n_frames)
    if f >= 1:
        ramp = _smoothstep(np.linspace(0.0, 1.0, f))  # f==1 なら [0.0]
        env[:f] = ramp
        env[n_frames - f:] = ramp[::-1]
    # 端は常に厳密0(f==0 の極短セグメントでも保証)
    env[0] = 0.0
    env[-1] = 0.0
    return env


def frame_speeds(values) -> np.ndarray:
    """フレーム毎の値列から、正規化速度 [0,1] をフレーム毎に返す(§6.2)。

    速度[i] = 隣接フレーム差の大きさ(スカラーは絶対値、ベクトルはユークリッドノルム)、
    速度[0]=0。セグメント内の最大値で正規化する(完全静止なら全0)。
    values は1チャンネルのスカラー列、または各行がベクトルの2次元配列。
    """
    v = np.asarray(values, dtype=float)
    n = v.shape[0]
    speeds = np.zeros(n)
    if n >= 2:
        d = v[1:] - v[:-1]
        speeds[1:] = np.abs(d) if v.ndim == 1 else np.linalg.norm(d, axis=1)
    peak = speeds.max() if n else 0.0
    if peak > 1e-12:
        speeds = speeds / peak
    else:
        # 数値ノイズ級(<=1e-12)の微小運動は静止扱いで全0にする。
        # フルスケールへ増幅しない/中途半端な極小値を残さない。
        speeds = np.zeros(n)
    return speeds


def adaptive_amplitude(
    base_amp: float, normalized_speed, motion_scale: float = DEFAULT_MOTION_SCALE
):
    """実効振幅 = 基本振幅 ×(1 + motion_scale × 正規化速度)(§6.2)。

    normalized_speed はスカラーまたは配列([0,1] 想定)。
    """
    return base_amp * (1.0 + motion_scale * normalized_speed)


def detect_stops(
    normalized_speeds, threshold: float = DEFAULT_STOP_SPEED_THRESHOLD
) -> list[int]:
    """速度が threshold を上から下へ横切るフレーム(停止検出点)を昇順で返す(§6.2)。

    speeds[i-1] >= threshold かつ speeds[i] < threshold となる i を停止点とする。
    """
    s = np.asarray(normalized_speeds, dtype=float)
    return [
        i for i in range(1, s.shape[0])
        if s[i - 1] >= threshold and s[i] < threshold
    ]


def settle_oscillation(
    t_sec,
    amp: float,
    settle_time: float = DEFAULT_SETTLE_TIME_SEC,
    freq: float = DEFAULT_SETTLE_FREQ_HZ,
):
    """停止後の減衰振動(オーバーシュート)。t_sec>=0(停止からの経過秒)。

    amp · exp(-t / (settle_time/4)) · sin(2π·freq·t)。
    包絡初期値が amp(初期振幅)、settle_time 経過で概ね収束(exp(-4)≈0.018)。
    t=0 では 0(sin)から立ち上がってオーバーシュートし減衰する。
    """
    t = np.asarray(t_sec, dtype=float)
    tau = settle_time / 4.0
    return amp * np.exp(-t / tau) * np.sin(2.0 * np.pi * freq * t)


def impulse_envelope(
    n_frames: int, frame: int, strength: float, decay_sec: float, fps: float = FPS
) -> np.ndarray:
    """フレーム frame 以降に strength·exp(-Δt/decay_sec) の包絡を返す(§6.3)。

    frame より前は 0。Δt = (i - frame)/fps 秒。長さ n_frames。
    実際の揺れはこの包絡 × 高周波ノイズ(§6.1の帯域制限に従う)で、本関数は包絡のみ。
    """
    env = np.zeros(n_frames)
    idx = np.arange(n_frames)
    mask = idx >= frame
    dt = (idx[mask] - frame) / fps
    env[mask] = strength * np.exp(-dt / decay_sec)
    return env
