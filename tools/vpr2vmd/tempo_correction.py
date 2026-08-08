"""テンポ→GenerationParams 補正。

代表BPMが基準テンポ `ref_bpm` より速いほど、保持・アタック・リリースのフレーム数を縮める。
高速テンポで短い母音が吸収閾値を下回って消え、口が開きっぱなしに見えるのを防ぐ。`lipsync` は
フレーム基準でテンポ概念を持たないため、テンポ→パラメータ補正は `vpr2vmd` が担う。
"""

from dataclasses import replace

from lipsync import GenerationParams


def _shrunk(value: int, s: float, floor: int) -> int:
    """value を s 倍に縮める。floor 未満には縮めないが、元値を超えて増やすこともしない。

    補正は縮小方向のみ。元値が既に floor 未満(プリセットが意図的に小さく取った場合)なら、その
    値を保つ(floor まで引き上げない)。floor は補正で生じる過小化(onset 急変など)を防ぐための
    下限で、プリセットの選択を上書きするものではない。
    """
    return min(value, max(floor, round(value * s)))


def apply_tempo_correction(
    params: GenerationParams,
    representative_bpm: float,
    *,
    ref_bpm: float = 120.0,
    s_min: float = 0.5,
) -> GenerationParams:
    """代表BPMに応じて保持・アタック・リリースを縮めた GenerationParams を返す。

    1拍のフレーム数 `fpb=(60/bpm)×30` と基準テンポの `fpb_ref=(60/ref_bpm)×30` の比
    `s=clamp(fpb/fpb_ref, s_min, 1.0)`(=`clamp(ref_bpm/代表BPM, s_min, 1.0)`)を min_hold・attack・
    release に掛ける。基準より遅い曲(比>1)は 1.0 にクランプして伸ばさない。下限は min_hold≥1、
    attack/release≥2(縮めすぎて 0→全開が1フレームになる onset 急変を避ける)。下限はあくまで補正で
    生じる過小化を防ぐもので、プリセットが元から下限未満に取った値は増やさない(補正は縮小のみで値を
    増やさない)。補正対象は保持・アタック・リリースのみで、他のパラメータ(先行準備・協調調音重なり・
    谷係数・伸び表現など)は変えない。
    """
    fpb = (60.0 / representative_bpm) * 30.0
    fpb_ref = (60.0 / ref_bpm) * 30.0
    s = max(s_min, min(fpb / fpb_ref, 1.0))
    return replace(
        params,
        min_hold_frames=_shrunk(params.min_hold_frames, s, 1),
        attack_frames=_shrunk(params.attack_frames, s, 2),
        release_frames=_shrunk(params.release_frames, s, 2),
    )
