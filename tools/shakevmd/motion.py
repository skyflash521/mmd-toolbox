import numpy as np

FPS = 30.0

DEFAULT_MOTION_DAMP = 1.0
DEFAULT_FADE_SEC = 0.7
DEFAULT_SETTLE_TIME_SEC = 1.0
DEFAULT_SETTLE_FREQ_HZ = 2.5
DEFAULT_STOP_SPEED_THRESHOLD = 0.1
DEFAULT_SPEED_REF_WORLD = 1.0
DEFAULT_SPEED_REF_ANGLE = 0.02

STILL_PROFILE = (1.0, 0.25, 0.0625)
MOVING_PROFILE = (1.0, 0.6, 0.36)
BREATHING_HZ = 0.3
BREATHING_AMP_FACTOR = 0.5


def profile_weights(normalized_speed, still=STILL_PROFILE, moving=MOVING_PROFILE):
    """戻り値の形はスカラー入力なら (オクターブ数,)、配列入力なら (フレーム数, オクターブ数)。"""
    s = np.asarray(normalized_speed, dtype=float)
    still = np.asarray(still, dtype=float)
    moving = np.asarray(moving, dtype=float)
    if s.ndim == 0:
        return (1.0 - s) * still + s * moving
    return (1.0 - s)[:, None] * still[None, :] + s[:, None] * moving[None, :]


def breathing_drift(t_sec, amp: float, freq: float = BREATHING_HZ):
    t = np.asarray(t_sec, dtype=float)
    return amp * np.sin(2.0 * np.pi * freq * t)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def fade_envelope(n_frames: int, fade_sec: float = DEFAULT_FADE_SEC, fps: float = FPS) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros(0)
    f0 = int(round(fade_sec * fps))
    f = f0 if 2 * f0 <= n_frames else n_frames // 2
    env = np.ones(n_frames)
    if f >= 1:
        ramp = _smoothstep(np.linspace(0.0, 1.0, f))
        env[:f] = ramp
        env[n_frames - f:] = ramp[::-1]
    env[0] = 0.0
    env[-1] = 0.0
    return env


def frame_speeds(values, ref) -> np.ndarray:
    """values はスカラーの列か、各行が1フレームのベクトルである2次元配列。"""
    v = np.asarray(values, dtype=float)
    n = v.shape[0]
    speeds = np.zeros(n)
    if ref <= 0:
        return speeds
    if n >= 2:
        d = v[1:] - v[:-1]
        speeds[1:] = np.abs(d) if v.ndim == 1 else np.linalg.norm(d, axis=1)
    return np.clip(speeds / ref, 0.0, 1.0)


def adaptive_amplitude(
    base_amp: float, normalized_speed, motion_damp: float = DEFAULT_MOTION_DAMP
):
    return base_amp * np.maximum(0.0, 1.0 - motion_damp * normalized_speed)


def detect_stops(
    normalized_speeds, threshold: float = DEFAULT_STOP_SPEED_THRESHOLD
) -> list[int]:
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
    """t_sec は停止からの経過秒で、0 以上。"""
    t = np.asarray(t_sec, dtype=float)
    tau = settle_time / 4.0
    return amp * np.exp(-t / tau) * np.sin(2.0 * np.pi * freq * t)


def impulse_envelope(
    n_frames: int, frame: int, strength: float, decay_sec: float, fps: float = FPS
) -> np.ndarray:
    env = np.zeros(n_frames)
    idx = np.arange(n_frames)
    mask = idx >= frame
    dt = (idx[mask] - frame) / fps
    env[mask] = strength * np.exp(-dt / decay_sec)
    return env
