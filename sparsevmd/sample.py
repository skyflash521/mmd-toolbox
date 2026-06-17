"""対象トラックの正規化・分割・サンプリング(sparsevmd.md §3.1, §4.1, §5.1)。

入力VMDの camera/bone セクションを内部作業ビューで正規化(フレームソート・同一キー
後勝ち)し、camera を1トラック、bone をボーン名ごとのトラックに分割する。
サンプリングは mmd_toolbox.vmd.interp に委譲する(スカラーは numpy 配列、回転は
値のリスト)。perspective は補間しない離散値で、直近キーの値を保持する(§4.2)。
"""

from dataclasses import dataclass

from mmd_toolbox.vmd import interp, io

# perspective_series は mmd_toolbox.vmd.sample へ移送(refactor-plan-direct-reduce.md Step 1)。
# 旧 import パス(sparsevmd.sample.perspective_series)維持のため再公開する。
from mmd_toolbox.vmd.sample import perspective_series  # noqa: F401


@dataclass
class Track:
    """削減対象トラック。camera は name=None、bone はボーン表示名。"""

    kind: str  # "camera" | "bone"
    name: str | None
    keys: list
    first: int
    last: int


def build_tracks(doc, target):
    """対象セクションを正規化し、トラック列に分割する(§3.1, §4.1)。

    target は "camera" / "bone" / "all"。

    対象セクションにキーが無ければ、そのセクションのトラックは生成しない。
    したがって target="all" でカメラ0件・ボーンありなら bone のみのトラック列を返す
    (§3.1 の片側フォールバックは自動的に成立する)。全対象でトラックが0件(空リスト)
    の場合を「対象キー無し=終了コード1」(§3.1)に対応づけるのは呼び出し側(CLI)の責務で、
    本関数はトラック分割に専念しエラーは送出しない。

    正規化警告(重複ドロップ・並べ替え)は現状この層では破棄する。ユーザーへの提示
    (verbose/レポート。§2.7)が必要な場合は、レポート統合(Phase9/Step11)で
    io.normalize を別途呼ぶか本関数の警告返却拡張で対応する。
    """
    sections = {"camera": ["camera"], "bone": ["bone"], "all": ["camera", "bone"]}[target]
    normalized, _warnings = io.normalize(doc, sections=sections)

    tracks = []
    if "camera" in sections and normalized.camera:
        keys = list(normalized.camera)
        tracks.append(
            Track("camera", None, keys, keys[0].frame, keys[-1].frame)
        )
    if "bone" in sections:
        tracks.extend(_split_bone_tracks(normalized.bone))
    return tracks


def _split_bone_tracks(bone_keys):
    """正規化済みボーンキー列(name_raw,frame 順)をボーン名ごとに分割する。"""
    tracks = []
    group = []
    current_raw = None
    for k in bone_keys:
        if current_raw is None or k.name_raw == current_raw:
            group.append(k)
            current_raw = k.name_raw
        else:
            tracks.append(_bone_track(group))
            group = [k]
            current_raw = k.name_raw
    if group:
        tracks.append(_bone_track(group))
    return tracks


def _bone_track(group):
    return Track("bone", group[0].name, group, group[0].frame, group[-1].frame)


def sample_scalar(keys, channel, frame_start, frame_end):
    """スカラーチャンネルを [frame_start, frame_end] で評価する(numpy 配列)。"""
    return interp.sample_range(keys, channel, frame_start, frame_end)


def sample_rotation(keys, frame_start, frame_end):
    """回転チャンネルを評価する。camera は Euler 3要素、bone は quaternion のリスト。"""
    return interp.sample_range(keys, "rot", frame_start, frame_end)
