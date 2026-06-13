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
