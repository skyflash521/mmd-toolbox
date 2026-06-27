"""マーカー抽出と共通FK接続。

profile.model のワールド姿勢を共通FK(mmd_toolbox.pmx.pose)で評価し、そこから
標準マーカーのワールド軌跡と派生特徴ベクトル列を作る。mocap側に独自FKは持たない。
"""

from dataclasses import dataclass

from mmd_toolbox.pmx.pose import evaluate_fk_range


@dataclass
class MarkerTrajectories:
    # マーカー名 -> フレーム毎ワールド座標 (x, y, z)
    markers: dict[str, list[tuple[float, float, float]]]
    # 派生特徴名 -> フレーム毎ベクトル (x, y, z)
    features: dict[str, list[tuple[float, float, float]]]


def evaluate_world_poses(profile, bone_tracks, frames):
    """profile.model の各ボーンのフレーム毎ワールド姿勢を共通FKで評価する。"""
    return evaluate_fk_range(profile.model, bone_tracks, frames)


def extract_markers(profile, world_poses) -> MarkerTrajectories:
    """marker_bindings/feature_bindings からワールド軌跡と特徴ベクトル列を作る。"""
    markers: dict[str, list[tuple[float, float, float]]] = {}
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

    features: dict[str, list[tuple[float, float, float]]] = {}
    for name, fb in profile.feature_bindings.items():
        ai = profile.required_bones[fb.a]
        bi = profile.required_bones[fb.b]
        series = []
        for per_frame in world_poses:
            a = per_frame[ai].position
            b = per_frame[bi].position
            series.append((a[0] - b[0], a[1] - b[1], a[2] - b[2]))
        features[name] = series

    return MarkerTrajectories(markers=markers, features=features)
