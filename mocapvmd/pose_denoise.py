"""表現空間ノイズ除去のオーケストレーション。

入力ボーンキーを、モデルプロファイルのFK・マーカー平滑化・姿勢フィットを通して
密キー列へ変換する。モデルが扱い入力にあるボーンを密キー化(線形補間)し、モデル外
ボーン(指など)は原キーのまま通す。後段の足IK安定化・疎化へ同じ密キー形式で渡せる。
"""

from mmd_toolbox.pmx.pose import evaluate_fk, sample_local_poses
from mmd_toolbox.vmd.reduce import BONE_LINEAR_INTERP
from mmd_toolbox.vmd.types import BoneKey

from . import marker_denoise, pose_fit
from .markers import extract_markers
from .model_profile import load_mocap_profile


def apply_pose_denoise(bone_keys, *, pmx_path=None, preset=None, fit_params=None):
    """ボーンキー列に表現空間ノイズ除去を適用し、新しいボーンキー列を返す。"""
    if not bone_keys:
        return []

    profile = load_mocap_profile(pmx_path)
    model = profile.model

    tracks = {}
    for k in bone_keys:
        tracks.setdefault(k.name, []).append(k)
    for keys in tracks.values():
        keys.sort(key=lambda k: k.frame)

    f0 = min(k.frame for k in bone_keys)
    f1 = max(k.frame for k in bone_keys)
    frames = range(f0, f1 + 1)

    # 密ローカル姿勢は一度だけ作り、FK・姿勢フィットで再利用する(再サンプルしない)。
    dense = [sample_local_poses(model, tracks, f) for f in frames]
    world = [evaluate_fk(model, lp) for lp in dense]

    traj = extract_markers(profile, world)
    categories = {m: mb.category for m, mb in profile.marker_bindings.items()}
    smoothed = marker_denoise.smooth(traj.markers, categories, preset=preset)
    fit_kwargs = {} if fit_params is None else {"params": fit_params}
    fitted = pose_fit.fit(profile, dense, smoothed.markers, **fit_kwargs)

    # モデルが扱い、かつ入力にあるボーンだけを密キー化する。
    processed = {b.name for b in model.bones} & set(tracks)
    name_raw = {name: tracks[name][0].name_raw for name in processed}

    out = []
    for f_idx, frame in enumerate(frames):
        poses = fitted.poses[f_idx]
        for bi, bone in enumerate(model.bones):
            if bone.name in processed:
                lp = poses[bi]
                out.append(
                    BoneKey(
                        name_raw=name_raw[bone.name],
                        frame=frame,
                        position=lp.position,
                        rotation=lp.rotation,
                        interpolation=BONE_LINEAR_INTERP,
                    )
                )

    # モデル外ボーン(指など)は原キーをそのまま通す。
    for k in bone_keys:
        if k.name not in processed:
            out.append(k)

    return out
