from dataclasses import dataclass

from pmx.pose import evaluate_fk_range

Vec3 = tuple[float, float, float]


@dataclass
class MarkerTrajectories:
    markers: dict[str, list[Vec3]]
    features: dict[str, list[Vec3]]


def evaluate_world_poses(profile, bone_tracks, frames):
    return evaluate_fk_range(profile.model, bone_tracks, frames)


def extract_markers(profile, world_poses) -> MarkerTrajectories:
    markers: dict[str, list[Vec3]] = {}
    for name, mb in profile.marker_bindings.items():
        ox, oy, oz = mb.offset
        markers[name] = [
            (
                per_frame[mb.bone].position[0] + ox,
                per_frame[mb.bone].position[1] + oy,
                per_frame[mb.bone].position[2] + oz,
            )
            for per_frame in world_poses
        ]

    features: dict[str, list[Vec3]] = {}
    for name, fb in profile.feature_bindings.items():
        head_index = profile.required_bones[fb.a]
        tail_index = profile.required_bones[fb.b]
        series = []
        for per_frame in world_poses:
            head = per_frame[head_index].position
            tail = per_frame[tail_index].position
            series.append((head[0] - tail[0], head[1] - tail[1], head[2] - tail[2]))
        features[name] = series

    return MarkerTrajectories(markers=markers, features=features)
