"""出力後検証のテスト(sparsevmd.md §7.3, §7.2)。

書き出し直前に対象トラック全体を再サンプリングし、全フレームが許容誤差以内かを
§7.2 のメトリクスで検証する。verify_camera_track / verify_bone_track は許容超過
フレームの昇順リストを返す(空なら合格)。reduce_*_track はこれを使い、非 strict では
超過フレームを必須キーに追加して再構築し(1フレーム間隔まで密にすれば元値を逐語保持
できるため必ず収束)、strict では StrictError(終了コード4)を送出する。

視野角の整数丸め誤差は分割不能な量子化誤差として扱い、許容(0.5度以上)を満たす限り
超過扱いにしない(§7.3)。
"""

import pytest

from mmd_toolbox.vmd import interp
from mmd_toolbox.vmd.types import BoneKey, CameraKey
from mmd_toolbox.vmd.reduce import (
    StrictError,
    Tolerances,
    reduce_camera_track,
    verify_bone_track,
    verify_camera_track,
)

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
EASE = (96, 0, 96, 30)


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()
# balanced プリセット相当の許容(sparsevmd.presets の balanced 値)。
TOLS = Tolerances(
    bone_pos=0.01,
    bone_rot=0.10,
    camera_pos=0.02,
    camera_rot=0.05,
    camera_distance=0.02,
    camera_fov=0.50,
)
CAM_CUT = (5.0, 20.0, 5.0)


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, persp)


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, rot, BL)


def _eased(v0, v1, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*EASE, f / span) for f in range(n)]


# --- verify_camera_track ----------------------------------------------------


def test_verify_camera_faithful_linear_no_violation():
    # 線形移動を2キー線形補間で表せば、再サンプリングは元と一致する。
    source = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(11)]
    output = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(10.0, 0.0, 0.0))]
    assert verify_camera_track(source, output, [(0, 10)], TOLS) == []


def test_verify_camera_detects_position_violation():
    # ソースは曲線、出力は2キー線形 → 中間フレームで位置が大きくずれる。
    xs = _eased(0.0, 100.0)
    source = [cam(f, center=(xs[f], 0.0, 0.0)) for f in range(11)]
    output = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(100.0, 0.0, 0.0))]
    bad = verify_camera_track(source, output, [(0, 10)], TOLS)
    assert bad  # 超過フレームが検出される
    assert all(0 < f < 10 for f in bad)
    assert bad == sorted(bad)


def test_verify_camera_fov_rounding_within_tol_not_flagged():
    # ソースは frame0-4 が 30 度・frame5-10 が 31 度(段)。出力2キー(30@0, 31@10)の線形
    # 補間は中間で 30.x を返し元サンプルとの差は最大 0.5 度 → FOV 許容(0.5度)内で超過なし。
    # 出力とソースは別の再構築(段 対 ランプ)なので、生値一致で通る作りにはなっていない。
    source = [cam(f, fov=30 if f < 5 else 31) for f in range(11)]
    output = [cam(0, fov=30), cam(10, fov=31)]
    assert verify_camera_track(source, output, [(0, 10)], TOLS) == []


def test_verify_camera_fov_large_error_flagged():
    # ソースは fov が曲線(密)、出力は2キー線形 → 中間で許容(0.5度)を大きく超える。
    fovs = [round(30 + 30 * interp._solve_factor(*EASE, f / 10)) for f in range(11)]
    source = [cam(f, fov=fovs[f]) for f in range(11)]
    output = [cam(0, fov=fovs[0]), cam(10, fov=fovs[10])]
    bad = verify_camera_track(source, output, [(0, 10)], TOLS)
    assert bad
    assert all(0 < f < 10 for f in bad)


# --- verify_bone_track ------------------------------------------------------


def test_verify_bone_faithful_no_violation():
    source = [bone("c", f, pos=(float(f), 0.0, 0.0)) for f in range(11)]
    output = [bone("c", 0, pos=(0.0, 0.0, 0.0)), bone("c", 10, pos=(10.0, 0.0, 0.0))]
    assert verify_bone_track(source, output, [(0, 10)], TOLS) == []


def test_verify_bone_detects_position_violation():
    xs = _eased(0.0, 100.0)
    source = [bone("c", f, pos=(xs[f], 0.0, 0.0)) for f in range(11)]
    output = [bone("c", 0, pos=(0.0, 0.0, 0.0)), bone("c", 10, pos=(100.0, 0.0, 0.0))]
    bad = verify_bone_track(source, output, [(0, 10)], TOLS)
    assert bad
    assert all(0 < f < 10 for f in bad)
    assert bad == sorted(bad)


# --- reduce 統合: strict / 非strict ----------------------------------------


def _bad_curve(monkeypatch):
    # 全チャンネルの curve を線形固定にし、採否(bezier)が曲線で受理した区間でも
    # 出力には線形を格納させて、出力段で必ず誤差を発生させる(§7.3 の検査経路を励起)。
    import mmd_toolbox.vmd.fit as fit

    linear_cp = (20, 20, 107, 107)
    for cls in (
        fit.LinearScalarChannel,
        fit.FovChannel,
        fit.CameraRotationChannel,
        fit.BoneRotationChannel,
    ):
        monkeypatch.setattr(cls, "curve", lambda self, a, b: linear_cp)
    monkeypatch.setattr(
        fit.EuclideanVectorChannel, "curve", lambda self, a, b: (linear_cp,) * 3
    )


def _opts(**kw):
    o = dict(
        cut_thresholds=CAM_CUT,
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
    )
    o.update(kw)
    return o


def test_reduce_camera_strict_raises_on_output_violation(monkeypatch):
    # 採否は曲線で2キーに受理するが、出力曲線を線形に壊すと再サンプリングが許容超過。
    # strict なので StrictError。
    xs = _eased(-10.0, -110.0)
    source = [cam(f, dist=xs[f]) for f in range(11)]
    _bad_curve(monkeypatch)
    with pytest.raises(StrictError):
        reduce_camera_track(source, [(0, 10)], TOLS, **_opts(strict=True))


def test_reduce_camera_nonstrict_densifies_on_output_violation(monkeypatch):
    # 非 strict は超過フレームを密に保持して収束する(1フレーム間隔は元値を逐語保持)。
    xs = _eased(-10.0, -110.0)
    source = [cam(f, dist=xs[f]) for f in range(11)]
    _bad_curve(monkeypatch)
    keys = reduce_camera_track(source, [(0, 10)], TOLS, **_opts(strict=False))
    # 線形では表せないので 2 キーより増える(密化で許容内に収める)。
    assert len(keys) > 2
    # 収束後、出力は許容内(検証が空)。
    assert verify_camera_track(source, keys, [(0, 10)], TOLS) == []


def test_reduce_camera_normal_passes_verification():
    # 正常な bezier 削減は検証を素通りし、出力が許容内。
    xs = _eased(-10.0, -110.0)
    source = [cam(f, dist=xs[f]) for f in range(11)]
    keys = reduce_camera_track(source, [(0, 10)], TOLS, **_opts())
    assert verify_camera_track(source, keys, [(0, 10)], TOLS) == []
