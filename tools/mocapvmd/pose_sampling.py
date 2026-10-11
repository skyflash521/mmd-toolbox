from pmx.pose import LocalBonePose, evaluate_fk
from vmd.interp import sample

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def sample_local_poses(model, bone_tracks, frame):
    """戻り値は model.bones と同じ並び。"""
    poses = []
    for bone in model.bones:
        keys = bone_tracks.get(bone.name)
        if keys:
            position = (
                sample(keys, "pos_x", frame),
                sample(keys, "pos_y", frame),
                sample(keys, "pos_z", frame),
            )
            rotation = tuple(sample(keys, "rot", frame))
        else:
            position = (0.0, 0.0, 0.0)
            rotation = _IDENTITY_QUAT
        poses.append(LocalBonePose(position=position, rotation=rotation))
    return tuple(poses)


def evaluate_fk_range(model, bone_tracks, frames):
    return [
        evaluate_fk(model, sample_local_poses(model, bone_tracks, f))
        for f in frames
    ]
