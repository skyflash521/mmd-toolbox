"""ベイクループ・視線揺れ変換(shakevmd.md §3, §4, §5)。

§4.2 の視線揺れ変換 apply_gaze_shake と、ベイクループ本体 bake() を実装する。

§4.2 の実現方針: 揺れ角度は「元の角度 + ノイズ」のオイラー加算とし、カメラの
ワールド位置が固定される(位置ノイズ分だけシフトする)ように新しいカメラ中心を
逆算する。これにより距離0では自動的に素朴な角度加算と一致し、姿勢の再分解
(camera.from_world)やジンバル対策は不要になる。
"""

import math
from dataclasses import dataclass

import numpy as np

from vmd import camera, interp
from vmd.types import CameraKey
from shakevmd import cuts, motion, noise


def round_half_up(x) -> int:
    """整数度への丸め(四捨五入)。視野角の丸め既定は四捨五入(vmd-interp.md §3/§5)。

    Python 組み込み round() は銀行丸め(round half to even)で .5 境界が
    四捨五入と食い違うため使わない。視野角は正なので floor(x+0.5) で四捨五入になる。
    """
    return int(math.floor(float(x) + 0.5))


def apply_gaze_shake(camera_key, rot_noise, pos_noise=(0.0, 0.0, 0.0), naive=False) -> dict:
    """視線揺れ変換(§4.2)。

    - 揺れ角度 = camera_key.rotation + rot_noise(成分ごとのオイラー加算)
    - カメラのワールド位置は固定(pos_noise 指定時はその分だけワールドでシフト)
    - 新しい角度・距離からカメラ中心を逆算: center = cam_pos - R(新角度)·(0,0,distance)
    naive=True(素朴な角度加算モード、§8)では中心の逆算を行わず、元のカメラ中心に
    位置ノイズだけ加える(角度はオイラー加算、ワールド位置は固定しない)。距離0では
    既定と一致する。
    rot_noise=(drx,dry,drz) ラジアン、pos_noise=(dx,dy,dz)。
    戻り値: {"position": (cx,cy,cz), "rotation": (rx,ry,rz)}
    """
    new_rot = tuple(camera_key.rotation[i] + rot_noise[i] for i in range(3))
    if naive:
        center = np.array(camera_key.position, dtype=float) + np.array(pos_noise, dtype=float)
        return {"position": tuple(center), "rotation": new_rot}
    # カメラワールド位置(位置ノイズ分シフト)
    cam_pos = np.array(camera.to_world(camera_key).position, dtype=float)
    cam_pos = cam_pos + np.array(pos_noise, dtype=float)
    # offset = R(new_rot)·(0,0,distance) を中心0のプローブから取得(camera 規約を再利用)
    probe = CameraKey(
        0, camera_key.distance, (0.0, 0.0, 0.0), new_rot, bytes(24),
        camera_key.fov, camera_key.perspective,
    )
    offset = np.array(camera.to_world(probe).position, dtype=float)
    center = cam_pos - offset
    return {"position": tuple(center), "rotation": new_rot}


# MMDデフォルトの線形補間ブロック(カメラ24バイト = 6チャンネル × (20,107,20,107))
LINEAR_CAMERA_INTERP = bytes([20, 107, 20, 107]) * 6

# ベイクは30fps・1フレーム間隔固定(§4.1)。間隔変更オプションは持たない。
FPS = 30.0

# 視線揺れノイズの6チャンネル(セグメント別にシード派生する)
_ROT_CHANNELS = ("rot_x", "rot_y", "rot_z")
_POS_CHANNELS = ("pos_x", "pos_y", "pos_z")


def _crossfaded_channel(seed, si, ch, t, freq, weights, octaves=noise.DEFAULT_OCTAVES):
    """1チャンネルの帯域制限ノイズを、フレーム毎の静止/移動プロファイル重みで合成する(§6.2)。

    band_limited_noise の固定 persistence に代えて、octave_components(合成前のオクターブ成分)を
    weights[:, oct](= motion.profile_weights が速度から作るフレーム毎のオクターブ重み)で重み付け
    合成する。octaves は要求オクターブ数(=プロファイル長)。帯域制限でクランプされ成分数が
    weights の列数より少ない場合は先頭から対応分のみ使う。
    戻り値: (合成ノイズ (len(t),), 警告コード列)。
    """
    comps, warns = noise.octave_components(noise.derive_seed(seed, si, ch), t, freq, octaves=octaves)
    acc = np.zeros(len(t))
    for oct_i, c in enumerate(comps):
        acc = acc + weights[:, oct_i] * c
    return acc, warns


@dataclass
class BakeResult:
    """bake() の結果。"""

    camera_keys: list   # ベイク後の全カメラキー(範囲内=高密度、範囲外=原本)
    warnings: list
    resolved: list      # 適用範囲(端を最近接キーへスナップ済み、昇順・非接触)


def _working_view(camera_keys):
    """正規化作業ビュー(フレーム昇順・同一フレーム重複は後勝ち)と破棄件数を返す(§3.1)。"""
    by_frame = {}
    for k in camera_keys:
        by_frame[k.frame] = k   # 同一フレームは後勝ち(入力順で後のものが残る)
    wv = [by_frame[f] for f in sorted(by_frame)]
    return wv, len(camera_keys) - len(wv)


def _snap(frame, wv_frames):
    """frame を最近接の既存キーフレームへスナップする(§5.2)。等距離は小さい側。"""
    return min(wv_frames, key=lambda f: (abs(f - frame), f))


def _impulse_direction(seed: int, frame: int) -> np.ndarray:
    """衝撃(§6.3)の方向をシード(と発火フレーム)から決定する固定単位3ベクトル。

    同一シード・同一フレームで再現、異なるシードで変わる。フレーム由来なので、
    同じ衝撃が単独でも複合でも同じ方向になる(加算合成の整合)。
    """
    comps = [
        (noise.derive_seed(seed, "impulse_dir", frame, axis) % 2000) / 1000.0 - 1.0
        for axis in ("x", "y", "z")
    ]
    v = np.array(comps, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 1.0, 0.0])   # 退避(ほぼ起きない)
    return v / n


def _governing_perspective(wv, frame):
    """当該フレーム以前で最も近いキーのパースペクティブをホールドする(§3.1)。"""
    persp = wv[0].perspective
    for k in wv:
        if k.frame <= frame:
            persp = k.perspective
        else:
            break
    return persp


def bake(
    camera_keys,
    ranges=None,
    *,
    seed: int = 1,
    amp_rot: float = 0.8,        # 度
    amp_pos: float = 0.05,       # MMD距離単位
    rot_weights=(1.0, 1.0, 0.3),  # Pitch/Yaw/Roll = rx/ry/rz 個別重み
    freq: float = 1.2,
    motion_damp: float = 1.0,
    settle: float = 0.0,         # 度。停止後の減衰振動の初期振幅(§6.2)。0で無効(既定=無効)
    settle_time: float = motion.DEFAULT_SETTLE_TIME_SEC,  # 秒。settle減衰振動の収束時間(§2.5/§8 内蔵)
    # 静止/移動プロファイル(オクターブ重み構成、§6.2/§8 内蔵)。オクターブ数=プロファイル長。
    still_profile=motion.STILL_PROFILE,
    moving_profile=motion.MOVING_PROFILE,
    # 速度正規化の絶対基準(§6.2 内蔵)。移動(ワールド位置)と回転(角度)で別個。
    speed_ref_world: float = motion.DEFAULT_SPEED_REF_WORLD,
    speed_ref_angle: float = motion.DEFAULT_SPEED_REF_ANGLE,
    naive_rotation: bool = False,  # 素朴な角度加算モード(§8 内蔵)。中心逆算を行わない
    gait_freq: float = 0.0,      # Hz。歩調周期成分の周波数(§2.7/§95 walking)。0で無効
    gait_amp: float = 0.0,       # 歩調成分の振幅(MMD距離単位)。左右=gait_freq、上下=2×gait_freq
    fade_sec: float = 0.7,
    cut_pos_threshold: float = 5.0,
    cut_rot_threshold: float = 20.0,
    manual_cuts_add=(),
    manual_cuts_remove=(),
    impulses=(),                 # (F,S,D) の並び。フレームFに強さS度・減衰D秒の衝撃(§6.3)
) -> BakeResult:
    """ベイクループ本体(§3, §4, §5)。

    camera_keys(原本、順不同可)を受け取り、ranges(各 (start,end)、None=全範囲)の
    範囲を30fps・1フレーム間隔でベイクした高密度キーを生成し、範囲外は原本を
    バイト保持して結合した全カメラキー列を返す。
    - 正規化作業ビュー(ソート・重複後勝ち)でサンプリング/範囲解決/カット検出
    - フェードは範囲レベル(範囲端で揺れ0、内部カット境界では非ゼロ)
    - ノイズ位相はセグメントごとに独立(derive_seed)
    - 視野角は四捨五入、パースは直前キーをホールド
    """
    if not camera_keys:
        raise ValueError("カメラキーが空(§3.1: 終了コード1相当)")
    if len(still_profile) != len(moving_profile):
        raise ValueError(
            f"still_profile と moving_profile の長さが不一致(オクターブ数の整合): "
            f"{len(still_profile)} != {len(moving_profile)}"
        )
    octaves = len(still_profile)

    warnings: list = []
    wv, dropped = _working_view(camera_keys)
    if dropped:
        warnings.append(f"正規化: 同一フレーム重複 {dropped} 件を後勝ちで破棄した(§3.1)")
    wv_frames = [k.frame for k in wv]
    first, last = wv_frames[0], wv_frames[-1]

    # --- 範囲解決(既定=全範囲、端は最近接キーへスナップ、重複/接触はエラー) ---
    if ranges is None:
        resolved = [(first, last)]
    else:
        resolved = []
        for s, e in ranges:
            a, b = _snap(s, wv_frames), _snap(e, wv_frames)
            if a > b:
                a, b = b, a
            resolved.append((a, b))
    resolved.sort()
    for i in range(1, len(resolved)):
        if resolved[i][0] <= resolved[i - 1][1]:
            raise ValueError(
                f"範囲が重複/接触している: {resolved[i - 1]} と {resolved[i]}(§5.2)"
            )

    # --- カット検出(作業ビュー全体)+ 手動指定 ---
    detected = cuts.detect_cuts(wv, cut_pos_threshold, cut_rot_threshold)
    cut_frames = cuts.resolve_cuts(
        detected, list(manual_cuts_add), list(manual_cuts_remove)
    )

    # 視野角が複数フレームにわたって変化する区間(隣接する作業ビューキーで視野角が異なり、かつ
    # フレーム間隔が2以上=ゆっくりズーム)は密キーを焼かず、その区間の元キーを温存して MMD の実数
    # 補間に委ねる(§3.1)。整数 FOV を毎フレーム焼くとゆっくりズームが1°刻みの階段になるため。
    # 隣接フレーム(間隔1)の瞬間 FOV ジャンプ(カット等のショット切替)は階段にならないので除外し、
    # 通常どおり密ベイクする(温存すると短いショット内で揺れが急にオン/オフして不連続になるため)。
    fov_ramp_frames: set = set()
    ramp_boundary: set = set()
    for i in range(len(wv) - 1):
        if wv[i].fov != wv[i + 1].fov and wv[i + 1].frame - wv[i].frame > 1:
            fov_ramp_frames.update(range(wv[i].frame, wv[i + 1].frame + 1))
            ramp_boundary.add(wv[i].frame)
            ramp_boundary.add(wv[i + 1].frame)

    fade_frames = int(round(fade_sec * FPS))
    baked: list = []

    for a, b in resolved:
        n = b - a + 1
        fade = motion.fade_envelope(n, fade_sec, FPS)  # 範囲レベル(端で0)
        if n < 2 * fade_frames:
            warnings.append(
                f"範囲[{a},{b}](長さ{n}f)が 2×fade({2 * fade_frames}f)未満。"
                f"フェードを自動短縮した(§5.1)"
            )
        for si, seg in enumerate(cuts.segment_bounds(a, b, cut_frames)):
            sframes = list(range(seg.start, seg.end + 1))
            if all(f in fov_ramp_frames for f in sframes):
                continue            # 全フレームが FOV 変化区間 → 密ベイクせず、ノイズ計算・警告も行わない
            t = np.array([f / FPS for f in sframes], dtype=float)

            # 速度解析(セグメント単位。カットをまたがない。§6.2 角速度＋移動速度)。
            # クロスフェード/呼吸ドリフトがフレーム毎の速度に依存するため、ノイズ合成より先に算出する。
            # カメラ中心位置だけでは、その場回転(パン/チルト/ロール)や距離のみのズームを
            # 静止と誤判定する。カメラのワールド位置(中心+R·(0,0,距離))は回転・ズーム・移動を
            # すべて反映し、角度差分が角速度を補う。両者の正規化速度の大きい方を採る。
            samples = [interp.sample_camera(wv, f) for f in sframes]
            world = np.array([
                camera.to_world(CameraKey(
                    0, s["distance"], s["position"], s["rotation"],
                    LINEAR_CAMERA_INTERP, 0, 0,
                )).position
                for s in samples
            ], dtype=float)
            angles = np.array([s["rotation"] for s in samples], dtype=float)
            angle_speed = motion.frame_speeds(angles, speed_ref_angle)
            speeds = np.maximum(motion.frame_speeds(world, speed_ref_world), angle_speed)

            # チャンネル別ノイズ(セグメント別シード派生=位相独立。§5.3-1)。
            # 静止/移動プロファイルのオクターブ重みをフレーム毎の速度でクロスフェード(§6.2)。
            weights = motion.profile_weights(speeds, still_profile, moving_profile)  # (n_frames, n_oct)
            rot_n = []
            for ch in _ROT_CHANNELS:
                vals, warns = _crossfaded_channel(seed, si, ch, t, freq, weights, octaves)
                rot_n.append(vals)
                warnings.extend(warns)
            pos_n = []
            for ch in _POS_CHANNELS:
                vals, warns = _crossfaded_channel(seed, si, ch, t, freq, weights, octaves)
                pos_n.append(vals)
                warnings.extend(warns)

            # 呼吸ドリフト(§6.2「完全静止区間: 長周期ドリフト 0.3Hz 相当」)。位置のみに、
            # フレーム毎の (1-speed) スケールで加算する(回転=向きには載せない)。軸ごとに独立位相。
            breath = np.zeros((len(sframes), 3))
            breath_amp = amp_pos * motion.BREATHING_AMP_FACTOR
            if breath_amp > 0.0:
                for axis in range(3):
                    ph_seed = noise.derive_seed(seed, si, "breath", axis)
                    phase_sec = (ph_seed % 100000) / 100000.0 / motion.BREATHING_HZ
                    breath[:, axis] = (1.0 - speeds) * motion.breathing_drift(
                        t + phase_sec, breath_amp
                    )

            # 歩調周期成分(§2.7/§95 walking): 乱数ノイズに加算で混合する。位置のみ、左右(x)=
            # gait_freq、上下(y)=2×gait_freq の正弦波。奥行(z)・回転には載せない。位相は軸別に
            # シード派生(決定論的)。gait_freq=0 または gait_amp=0 で無効。
            gait = np.zeros((len(sframes), 3))
            if gait_freq > 0.0 and gait_amp != 0.0:
                for axis, mult in ((0, 1.0), (1, 2.0)):
                    ph_seed = noise.derive_seed(seed, si, "gait", axis)
                    phase = (ph_seed % 100000) / 100000.0 * 2.0 * np.pi
                    gait[:, axis] = gait_amp * np.sin(
                        2.0 * np.pi * (gait_freq * mult) * t + phase
                    )

            # settle(§6.2): 角速度の停止点で、停止直前の回転移動方向へ減衰振動を加算する。
            # 停止時刻のΔ角度は0なので「直前の角速度ベクトル」angles[i-1]-angles[i-2] を方向に使う。
            # detect_stops/settle_oscillation はセグメントの angle_speed に対して行うため、
            # カットをまたがず(セグメント分割)、カット点では発動しない(§5.3-3)。
            settle_rot = np.zeros((len(sframes), 3))
            if settle > 0.0:
                for stop_idx in motion.detect_stops(angle_speed):
                    if stop_idx < 2:
                        continue
                    d = angles[stop_idx - 1] - angles[stop_idx - 2]   # 停止直前の角速度ベクトル
                    nrm = float(np.linalg.norm(d))
                    if nrm < 1e-9:
                        continue
                    direction = d / nrm
                    for j in range(stop_idx, len(sframes)):
                        val = motion.settle_oscillation(
                            (j - stop_idx) / FPS, math.radians(settle), settle_time=settle_time
                        )
                        settle_rot[j] += val * direction

            # impulse(§6.3): 各衝撃 (F,S,D) を、フレームF以降に
            # radians(S)·exp(-Δt/D)(Δt=(f-F)/FPS 秒)の包絡 × 8Hz帯域制限の高周波ノイズ ×
            # F由来シードの固定方向、で回転ノイズに加算する。絶対フレーム基準でセグメント
            # 非依存(カットをまたいでよい)。複数指定は加算合成。F以前は0。
            impulse_rot = np.zeros((len(sframes), 3))
            for imp in impulses:
                F, S, D = imp
                if S <= 0.0 or D <= 0.0:
                    continue
                direction = _impulse_direction(seed, F)
                # 高周波ノイズ。基本周波数=帯域上限8Hz、単一オクターブ(octaves=1)にして
                # 上位オクターブのクランプを起こさない(§6.1 のクランプ警告が出ないようにする)。
                # シードは F 由来(index 非依存=加算性)。警告が出れば伝播する(握りつぶさない)。
                osc, owarns = noise.band_limited_noise(
                    noise.derive_seed(seed, "impulse_osc", F), t, noise.BANDLIMIT_HZ,
                    octaves=1,
                )
                warnings.extend(owarns)
                for j, f in enumerate(sframes):
                    if f < F:
                        continue
                    env = math.radians(S) * math.exp(-((f - F) / FPS) / D)
                    impulse_rot[j] += direction * osc[j] * env

            for idx, f in enumerate(sframes):
                if f in fov_ramp_frames:
                    continue            # FOV 変化区間は密キーを焼かない(§3.1。元キーを後で温存)
                s = samples[idx]
                fade_v = fade[f - a]
                # 振幅 = 基本 × 適応 clamp(1 - motion_damp×速度, 0, 1) × 範囲フェード。
                # 動くほど揺れを減衰させる(§6.2)。settle/impulse は別成分(自前の振幅)で、
                # 範囲フェードのみ掛けて加算する(§6.2/§6.3)。
                amp_factor = max(0.0, 1.0 - motion_damp * speeds[idx]) * fade_v
                rot_noise = tuple(
                    rot_n[i][idx] * math.radians(amp_rot * rot_weights[i]) * amp_factor
                    + settle_rot[idx][i] * fade_v
                    + impulse_rot[idx][i] * fade_v
                    for i in range(3)
                )
                # 位置 = クロスフェードノイズ × 適応振幅 + 呼吸ドリフト + 歩調成分(範囲フェードのみ)
                pos_noise = tuple(
                    pos_n[i][idx] * amp_pos * amp_factor
                    + breath[idx][i] * fade_v
                    + gait[idx][i] * fade_v
                    for i in range(3)
                )
                persp = _governing_perspective(wv, f)
                fov = round_half_up(s["fov"])
                probe = CameraKey(
                    f, s["distance"], s["position"], s["rotation"],
                    LINEAR_CAMERA_INTERP, fov, persp,
                )
                shaken = apply_gaze_shake(probe, rot_noise, pos_noise, naive=naive_rotation)
                baked.append(CameraKey(
                    f, s["distance"], shaken["position"], shaken["rotation"],
                    LINEAR_CAMERA_INTERP, fov, persp,
                ))

    # --- 範囲外は原本レコードをそのまま透過(バイト保持。§3.2) ---
    # 原本キーはそのまま(同一オブジェクト=バイト同一)。出力はフレーム昇順に整列して、
    # 範囲外キー(例: 範囲[30,60]に対する先頭 frame0)がベイク群の後ろに紛れないようにする。
    def _in_range(fr):
        return any(a <= fr <= b for a, b in resolved)

    out_keys = [k for k in camera_keys if not _in_range(k.frame)]

    # FOV 変化区間の境界キーは、範囲内でも密ベイクせず元キー(作業ビュー)を温存する(§3.1)。
    # 範囲外の境界キーは out_keys が原本バイトで保持するため、ここでは範囲内のみを対象にする。
    fov_preserved = [k for k in wv if k.frame in ramp_boundary and _in_range(k.frame)]

    # 警告の重複を畳む(同一帯域制限警告がチャンネル/セグメントで繰り返されるため)
    seen, uniq = set(), []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            uniq.append(w)

    result_keys = sorted(baked + fov_preserved + out_keys, key=lambda k: k.frame)
    return BakeResult(camera_keys=result_keys, warnings=uniq, resolved=resolved)
