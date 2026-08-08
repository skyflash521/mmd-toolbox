"""表現空間ノイズ除去オーケストレーションのテスト。

入力ボーンキーを、既定モデルプロファイル(または指定PMX)のFK・マーカー平滑化・
姿勢フィットを通して密キー列へ変換する。モデルが扱うボーンは密キー化し、扱わない
ボーン(指など)は原キーのまま通す。
"""

import math

import pytest

from mocapvmd.markers import extract_markers
from mocapvmd.model_profile import STANDARD_BONE_NAMES, load_mocap_profile
from mocapvmd.pose_denoise import apply_pose_denoise
from pmx.pose import evaluate_fk, sample_local_poses
from vmd.reduce import BONE_LINEAR_INTERP
from vmd.types import BoneKey

from .helpers import BONE_NONLINEAR, bone, build_standard_pmx


def _quat_y(deg):
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def _marker_trajectory(profile, bone_keys, marker):
    """ボーンキー列から指定マーカーのワールド軌跡を、パイプラインと同じFKで再計算する。"""
    tracks = {}
    for k in bone_keys:
        tracks.setdefault(k.name, []).append(k)
    for v in tracks.values():
        v.sort(key=lambda k: k.frame)
    f0 = min(k.frame for k in bone_keys)
    f1 = max(k.frame for k in bone_keys)
    traj = []
    for f in range(f0, f1 + 1):
        world = evaluate_fk(profile.model, sample_local_poses(profile.model, tracks, f))
        traj.append(extract_markers(profile, [world]).markers[marker][0])
    return traj


def test_empty_input_returns_empty():
    assert apply_pose_denoise([], pmx_path=None) == []


def test_dense_keys_for_model_bones():
    # センターは 2..5、頭は 0..10 に入力。全体フレーム範囲(0..10)で密化される。
    keys = [
        bone("センター", 2, pos=(0.2, 0.0, 0.0)),
        bone("センター", 5, pos=(0.5, 0.0, 0.0)),
        bone("頭", 0),
        bone("頭", 10, rot=(0.0, 0.0, 0.05, 0.99875)),
    ]
    out = apply_pose_denoise(keys, pmx_path=None)
    # 入力tracksにあるモデルボーンだけを出す(余計なモデルボーンを密キー化しない)。
    assert {k.name for k in out} == {"センター", "頭"}
    # 全体フレーム範囲で密化(ボーンごとの min..max ではない)。
    center_frames = sorted(k.frame for k in out if k.name == "センター")
    assert center_frames == list(range(0, 11))
    head_frames = sorted(k.frame for k in out if k.name == "頭")
    assert head_frames == list(range(0, 11))


def test_dense_keys_use_linear_interpolation():
    keys = [bone("センター", 0), bone("センター", 6, pos=(0.5, 0.0, 0.0))]
    out = apply_pose_denoise(keys, pmx_path=None)
    center = [k for k in out if k.name == "センター"]
    assert center
    for k in center:
        assert k.interpolation == BONE_LINEAR_INTERP


def test_non_model_bone_passes_through():
    # モデル外ボーンは name_raw・position・rotation・補間まで原キーのまま通す。
    thumb_key = bone(
        "左親指１", 0, pos=(0.2, 0.1, 0.0), rot=(0.0, 0.0, 0.1, 0.995), interp=BONE_NONLINEAR
    )
    keys = [bone("センター", 0), bone("センター", 5, pos=(0.3, 0.0, 0.0)), thumb_key]
    out = apply_pose_denoise(keys, pmx_path=None)
    thumb = [k for k in out if k.name == "左親指１"]
    assert thumb == [thumb_key]  # 完全一致(再構築されていない)


def test_static_motion_is_preserved():
    # 動きの無い静止モーションは平滑化・フィットでほぼ変わらない。
    keys = [bone("センター", 0), bone("センター", 8)]
    out = apply_pose_denoise(keys, pmx_path=None)
    for k in out:
        if k.name == "センター":
            assert k.position == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
            assert k.rotation == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-6)


def test_clean_moving_motion_passes_through_unchanged():
    # ノイズの無い滑らかな動き(静止でない)は壊さない(非破壊=オプトイン設計の要)。
    # センターを0→0.5へ直線的に動かすと全マーカーが並進する。平滑化が除くべき高周波が
    # 無いので、表現空間ノイズ除去を通しても頭マーカー軌跡は変わらないことを確認する。
    profile = load_mocap_profile(None)
    keys = [bone("センター", f, pos=(0.5 * f / 20.0, 0.0, 0.0)) for f in range(21)]
    before = _marker_trajectory(profile, keys, "head")
    after = _marker_trajectory(profile, apply_pose_denoise(keys, pmx_path=None), "head")
    # 明確に動いている(静止テストと別物であることを担保)。
    assert math.dist(before[0], before[-1]) > 0.1
    for a, b in zip(before, after, strict=True):
        assert a == pytest.approx(b, abs=1e-6)


def test_output_keys_are_valid_bonekeys():
    keys = [bone("頭", 0), bone("頭", 4, rot=(0.0, 0.0, 0.05, 0.9987))]
    out = apply_pose_denoise(keys, pmx_path=None)
    assert out
    for k in out:
        assert isinstance(k, BoneKey)
        assert isinstance(k.frame, int)
        assert len(k.position) == 3
        assert len(k.rotation) == 4
        assert isinstance(k.interpolation, bytes)
        assert len(k.interpolation) == 64


# --- 診断レポート素データ -----------------------------------------------


def test_diagnostics_out_populated():
    # diagnostics_out を渡すとマーカー数・必須ボーン検証・変位・fit診断を埋める。
    keys = [
        bone("センター", 0, pos=(0.2, 0.0, 0.0)),
        bone("センター", 5, pos=(0.5, 0.0, 0.0)),
        bone("頭", 0),
        bone("頭", 10, rot=(0.0, 0.0, 0.05, 0.99875)),
    ]
    diag = {}
    out = apply_pose_denoise(keys, pmx_path=None, diagnostics_out=diag)
    assert out  # 診断要求時も従来どおり密キー列を返す
    assert diag["enabled"] is True
    assert diag["pmx"] is None  # 既定モデルプロファイル使用時は None
    assert diag["frames"] == 11  # 全体フレーム範囲 0..10
    assert diag["markers"]["available"] > 0
    assert diag["markers"]["required_bones_ok"] is True
    md = diag["marker_displacement"]
    assert md["max"] >= md["mean"] >= 0.0
    assert isinstance(md["by_marker"], dict) and md["by_marker"]
    for stats in md["by_marker"].values():
        assert stats["max"] >= stats["mean"] >= 0.0
    fit = diag["fit"]
    assert fit["frames"] == 11
    assert fit["fallback_frames"] >= 0
    # フィットはマーカー誤差を悪化させない(フォールバック含め before 以下)。
    assert fit["mean_error_after"] <= fit["mean_error_before"] + 1e-9
    assert fit["max_bone_delta_deg"] >= 0.0
    assert fit["max_center_delta"] >= 0.0


def test_diagnostics_records_pmx_path(tmp_path):
    # --pmx 指定時は診断の pmx フィールドにそのパスを記録する。
    pmx = tmp_path / "model.pmx"
    pmx.write_bytes(build_standard_pmx(list(STANDARD_BONE_NAMES.values())))
    keys = [bone("センター", 0), bone("センター", 5, pos=(0.3, 0.0, 0.0))]
    diag = {}
    apply_pose_denoise(keys, pmx_path=str(pmx), diagnostics_out=diag)
    assert diag["pmx"] == str(pmx)


def test_no_diagnostics_arg_returns_list():
    # diagnostics_out 省略(既定)は従来どおり密キー列だけを返す。
    out = apply_pose_denoise([bone("頭", 0), bone("頭", 4, rot=(0.0, 0.0, 0.05, 0.9987))])
    assert isinstance(out, list)
