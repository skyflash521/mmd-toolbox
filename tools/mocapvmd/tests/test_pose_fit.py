"""姿勢フィット(最小実装)のテスト。

平滑化マーカーへ近づくよう限定ボーンの姿勢を小さく補正する。マーカー変位が
微小なら元姿勢を保持し、補正量は上限内、改善しなければ元姿勢へフォールバックする。
"""

import math

import pytest

from pmx.pose import evaluate_fk, sample_local_poses
from vmd.types import BoneKey

from mocapvmd.markers import evaluate_world_poses, extract_markers
from mocapvmd.model_profile import load_mocap_profile

from mocapvmd.pose_fit import (
    DEFAULT_FIT_PARAMS,
    FitParams,
    FitResult,
    fit,
)


def _key(name, *, rotation=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(
        name_raw=name.encode("cp932").ljust(15, b"\x00")[:15],
        frame=0,
        position=(0.0, 0.0, 0.0),
        rotation=rotation,
        interpolation=bytes(64),
    )


def _dense(profile, tracks, frames):
    return [sample_local_poses(profile.model, tracks, f) for f in frames]


def _fk_markers(profile, dense):
    world = [evaluate_fk(profile.model, lp) for lp in dense]
    return extract_markers(profile, world).markers


def _marker_error(profile, dense, target):
    got = _fk_markers(profile, dense)
    total = 0.0
    for name, series in target.items():
        for g, t in zip(got[name], series):
            total += sum((a - b) ** 2 for a, b in zip(g, t)) ** 0.5
    return total


# ---------------------------------------------------------------------------
# 変位なし・微小変位
# ---------------------------------------------------------------------------


def test_no_displacement_returns_original():
    profile = load_mocap_profile(None)
    frames = range(0, 2)
    dense = _dense(profile, {}, frames)
    target = _fk_markers(profile, dense)  # FKそのまま=変位なし
    res = fit(profile, dense, target)
    assert isinstance(res, FitResult)
    for fitted_frame, orig_frame in zip(res.poses, dense):
        for fb, ob in zip(fitted_frame, orig_frame):
            assert fb.position == pytest.approx(ob.position, abs=1e-9)
            assert fb.rotation == pytest.approx(ob.rotation, abs=1e-9)


def test_micro_displacement_preserves_original():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _fk_markers(profile, dense)
    target = {n: list(s) for n, s in target.items()}
    # 1マーカーだけを skip_threshold 未満だけ動かす(フレーム誤差も閾値未満)。
    eps = DEFAULT_FIT_PARAMS.skip_threshold * 0.5
    hx = target["head"][0]
    target["head"][0] = (hx[0] + eps, hx[1], hx[2])
    res = fit(profile, dense, target)
    for fitted_frame, orig_frame in zip(res.poses, dense):
        for fb, ob in zip(fitted_frame, orig_frame):
            assert fb.position == pytest.approx(ob.position, abs=1e-9)
            assert fb.rotation == pytest.approx(ob.rotation, abs=1e-9)


# ---------------------------------------------------------------------------
# 近づく・悪化しない・上限
# ---------------------------------------------------------------------------


def test_fit_reduces_marker_error():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _fk_markers(profile, dense)
    target = {n: list(s) for n, s in target.items()}
    # head は縮約骨格の葉なので、親(首)の回転で head マーカーだけを動かせる
    # (他マーカーを巻き込まず到達可能)。
    hx = target["head"][0]
    target["head"][0] = (hx[0] + 0.05, hx[1], hx[2])
    before = _marker_error(profile, dense, target)
    res = fit(profile, dense, target)
    after = _marker_error(profile, res.poses, target)
    assert after < before


def test_fit_never_worsens():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _fk_markers(profile, dense)
    target = {n: list(s) for n, s in target.items()}
    # 過大(到達不能)な目標を与える。
    hx = target["head"][0]
    target["head"][0] = (hx[0] + 100.0, hx[1], hx[2])
    before = _marker_error(profile, dense, target)
    res = fit(profile, dense, target)
    after = _marker_error(profile, res.poses, target)
    assert after <= before + 1e-9  # フォールバックで悪化しない


def test_corrections_within_limits():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _fk_markers(profile, dense)
    target = {n: list(s) for n, s in target.items()}
    # 大きく動かして補正を誘発し、上限クランプを確認。
    for n in target:
        p = target[n][0]
        target[n][0] = (p[0] + 5.0, p[1], p[2])
    res = fit(profile, dense, target)
    max_rot = math.radians(DEFAULT_FIT_PARAMS.max_rot_deg)
    for fitted_frame, orig_frame in zip(res.poses, dense):
        for fb, ob in zip(fitted_frame, orig_frame):
            # 位置補正
            dp = sum((a - b) ** 2 for a, b in zip(fb.position, ob.position)) ** 0.5
            assert dp <= DEFAULT_FIT_PARAMS.max_pos + 1e-6
            # 回転補正角(クォータニオン間角度)
            dot = abs(sum(a * b for a, b in zip(fb.rotation, ob.rotation)))
            dot = min(1.0, dot)
            angle = 2.0 * math.acos(dot)
            assert angle <= max_rot + 1e-6


def test_proportional_correction_limit():
    # 補正量上限はマーカー変位量に比例(絶対上限と小さい方)。
    # 小変位では比例上限(k*d)が効き、絶対上限いっぱいまで補正しない。
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _fk_markers(profile, dense)
    target = {n: list(s) for n, s in target.items()}
    d = DEFAULT_FIT_PARAMS.skip_threshold * 3.0  # skip は超えるが小さい変位
    # この d で比例上限が絶対上限より小さい(=比例上限が効く)ことを前提として固定する。
    assert DEFAULT_FIT_PARAMS.k_pos * d < DEFAULT_FIT_PARAMS.max_pos
    assert DEFAULT_FIT_PARAMS.k_rot * d < math.radians(DEFAULT_FIT_PARAMS.max_rot_deg)
    # 葉マーカー head を動かし、首の回転補正に比例上限を効かせる。
    hx = target["head"][0]
    target["head"][0] = (hx[0] + d, hx[1], hx[2])
    before = _marker_error(profile, dense, target)
    res = fit(profile, dense, target)
    after = _marker_error(profile, res.poses, target)
    # フィットが実際に補正している(ゼロ補正で素通りしていない)。
    assert after < before
    pos_limit = DEFAULT_FIT_PARAMS.k_pos * d
    rot_limit = DEFAULT_FIT_PARAMS.k_rot * d
    max_rot = 0.0
    for fitted_frame, orig_frame in zip(res.poses, dense):
        for fb, ob in zip(fitted_frame, orig_frame):
            dp = sum((a - b) ** 2 for a, b in zip(fb.position, ob.position)) ** 0.5
            assert dp <= pos_limit + 1e-6
            dot = min(1.0, abs(sum(a * b for a, b in zip(fb.rotation, ob.rotation))))
            angle = 2.0 * math.acos(dot)
            assert angle <= rot_limit + 1e-6
            max_rot = max(max_rot, angle)
    # 比例上限がクランプとして実際に効いている(回転補正が入っている)。
    assert max_rot > 0.0


def test_default_fit_params_are_reasonable():
    assert isinstance(DEFAULT_FIT_PARAMS, FitParams)
    assert DEFAULT_FIT_PARAMS.skip_threshold > 0.0
    assert DEFAULT_FIT_PARAMS.max_rot_deg > 0.0
    assert DEFAULT_FIT_PARAMS.max_pos > 0.0
    assert DEFAULT_FIT_PARAMS.k_rot > 0.0
    assert DEFAULT_FIT_PARAMS.k_pos > 0.0


def test_fallback_count_reported():
    profile = load_mocap_profile(None)
    dense = _dense(profile, {}, range(0, 1))
    target = _fk_markers(profile, dense)
    res = fit(profile, dense, target)
    assert isinstance(res.fallback_frames, int)
    assert res.fallback_frames >= 0
