"""フィット計測カウンタのテスト。

ベジェフィットは疎化時間の支配項なので、`least_squares` 呼び出し回数と線形ファストパス採用数を
計測できると、フィット高速化の効果を測れる。fit.py はモジュールレベルのカウンタ
(fit_calls / lsq_calls / fastpath_linear)を持ち、reduce_bone_track / reduce_camera_track は
diagnostics を渡されたとき1トラック分の集計を diagnostics["fit_counts"] に出す。通常実行
(diagnostics 未指定)では出力を変えない。
"""

from mmd_toolbox.vmd import fit, interp
from mmd_toolbox.vmd.reduce import (
    BONE_LINEAR_INTERP,
    CAMERA_LINEAR_INTERP,
    Tolerances,
    build_bone_tolerances,
    reduce_bone_track,
    reduce_camera_track,
)
from mmd_toolbox.vmd.types import BoneKey, CameraKey

EASE = (96, 0, 96, 30)  # 強いイージング(線形ファストパスで収まらない曲線)


def _bone(frame, pos):
    return BoneKey(b"bone".ljust(15, b"\x00"), frame, pos, (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)


def _eased_track(n=21, v1=10.0):
    """pos_x だけを強いイージングで 0→v1 に動かす密キー列(他軸・回転は定数)。"""
    span = n - 1
    return [
        _bone(f, (v1 * interp._solve_factor(*EASE, f / span), 0.0, 0.0)) for f in range(n)
    ]


def _linear_track(n=21, v1=10.0):
    """pos_x を線形に動かす密キー列(線形ファストパスで収まる)。"""
    span = n - 1
    return [_bone(f, (v1 * f / span, 0.0, 0.0)) for f in range(n)]


def _cam(frame, pos):
    return CameraKey(frame, -30.0, pos, (0.0, 0.0, 0.0), CAMERA_LINEAR_INTERP, 30, 0)


def _eased_camera_track(n=21, v1=10.0):
    """カメラ中心 X を強いイージングで 0→v1 に動かす密キー列(他は定数)。"""
    span = n - 1
    return [_cam(f, (v1 * interp._solve_factor(*EASE, f / span), 0.0, 0.0)) for f in range(n)]


def _cam_tols():
    return Tolerances(
        bone_pos=0.0, bone_rot=0.0, camera_pos=0.5, camera_rot=5.0,
        camera_distance=0.5, camera_fov=1.0,
    )


# --- fit レベルのカウンタ ----------------------------------------------------


def test_reset_zeroes_counters():
    fit.reset_fit_counters()
    c = fit.read_fit_counters()
    assert c == {"fit_calls": 0, "lsq_calls": 0, "fastpath_linear": 0}


def test_nonlinear_increments_lsq_calls():
    # 強いイージングは線形制御点で許容内に収まらないので least_squares を回す。
    fit.reset_fit_counters()
    xs = [(i + 1) / 12 for i in range(11)]
    ys = [interp._solve_factor(*EASE, x) for x in xs]
    fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    c = fit.read_fit_counters()
    assert c["fit_calls"] == 1
    assert c["lsq_calls"] >= 1
    assert c["fastpath_linear"] == 0


def test_linear_increments_fastpath_not_lsq():
    # 線形サンプルは線形ファストパスで即採用し least_squares を呼ばない。
    fit.reset_fit_counters()
    xs = [(i + 1) / 12 for i in range(11)]
    ys = list(xs)
    fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    c = fit.read_fit_counters()
    assert c["fit_calls"] == 1
    assert c["lsq_calls"] == 0
    assert c["fastpath_linear"] == 1


def test_coeff_curve_linear_increments_fastpath():
    # 回転の係数曲線も線形なら least_squares を呼ばずファストパス採用。
    fit.reset_fit_counters()
    xs = [(i + 1) / 6 for i in range(5)]

    def resid_at(coeff):
        return [coeff(x) - x for x in xs]

    fit._fit_coeff_curve(xs, resid_at, early_exit_err=0.01)
    c = fit.read_fit_counters()
    assert c["fit_calls"] == 1
    assert c["lsq_calls"] == 0
    assert c["fastpath_linear"] == 1


def test_coeff_curve_nonlinear_increments_lsq_calls():
    # 係数が強いイージングなら線形制御点では許容外で、係数曲線側も least_squares を回す。
    fit.reset_fit_counters()
    xs = [(i + 1) / 12 for i in range(11)]

    def resid_at(coeff):
        return [coeff(x) - interp._solve_factor(*EASE, x) for x in xs]

    fit._fit_coeff_curve(xs, resid_at, early_exit_err=0.01)
    c = fit.read_fit_counters()
    assert c["fit_calls"] == 1
    assert c["lsq_calls"] >= 1
    assert c["fastpath_linear"] == 0


# --- reduce_*_track の diagnostics 連携 --------------------------------------


def _tols():
    return build_bone_tolerances(bone_pos=0.5, bone_rot=5.0)


def test_diagnostics_records_fit_counts_for_curved_track():
    src = _eased_track()
    diag = {}
    reduce_bone_track(
        src,
        [(0, 20)],
        _tols(),
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
        diagnostics=diag,
    )
    fc = diag["fit_counts"]
    assert set(fc) == {"fit_calls", "lsq_calls", "fastpath_linear"}
    assert all(isinstance(v, int) for v in fc.values())
    assert fc["fit_calls"] >= 1
    # 曲がった位置軸があるので least_squares を少なくとも1回は回す。
    assert fc["lsq_calls"] >= 1


def test_diagnostics_records_fit_counts_for_camera_track():
    src = _eased_camera_track()
    diag = {}
    reduce_camera_track(
        src,
        [(0, 20)],
        _cam_tols(),
        cut_thresholds=(5.0, 20.0, 5.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
        diagnostics=diag,
    )
    fc = diag["fit_counts"]
    assert set(fc) == {"fit_calls", "lsq_calls", "fastpath_linear"}
    assert all(isinstance(v, int) for v in fc.values())
    assert fc["fit_calls"] >= 1
    # 曲がったカメラ中心軸があるので least_squares を少なくとも1回は回す。
    assert fc["lsq_calls"] >= 1


def test_diagnostics_linear_track_prefers_fastpath():
    src = _linear_track()
    diag = {}
    reduce_bone_track(
        src,
        [(0, 20)],
        _tols(),
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
        diagnostics=diag,
    )
    fc = diag["fit_counts"]
    # 線形トラックは全軸がファストパス採用で least_squares を回さない。
    assert fc["fastpath_linear"] >= 1
    assert fc["lsq_calls"] == 0


# --- チャンネル種別別カウント(どのチャンネルのフィットが重いかの帰属) --------


def test_by_category_fit_level_buckets():
    # category を渡すとそのチャンネル種別のバケットに計上し、渡さなければバケットを作らない。
    fit.reset_fit_counters()
    xs = [(i + 1) / 12 for i in range(11)]
    ys = [interp._solve_factor(*EASE, x) for x in xs]
    fit.fit_bezier_curve(xs, ys, early_exit_err=0.01, category="position")
    by = fit.read_fit_counters_by_category()
    assert by["position"]["fit_calls"] == 1
    assert by["position"]["lsq_calls"] >= 1

    # 係数曲線(回転系の計測入口)も category を受けて種別バケットへ計上する。
    fit.reset_fit_counters()

    def resid_at(coeff):
        return [coeff(x) - interp._solve_factor(*EASE, x) for x in xs]

    fit._fit_coeff_curve(xs, resid_at, early_exit_err=0.01, category="rotation")
    by = fit.read_fit_counters_by_category()
    assert by["rotation"]["fit_calls"] == 1
    assert by["rotation"]["lsq_calls"] >= 1

    fit.reset_fit_counters()
    fit.fit_bezier_curve(xs, ys, early_exit_err=0.01)
    assert fit.read_fit_counters_by_category() == {}


def test_by_channel_attributes_position_vs_rotation():
    # 位置が曲がり回転は定数のトラックでは、フィット費用は position に帰属する。
    src = _eased_track()
    diag = {}
    reduce_bone_track(
        src,
        [(0, 20)],
        _tols(),
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
        diagnostics=diag,
    )
    by = diag["fit_counts_by_channel"]
    assert by["position"]["lsq_calls"] >= 1
    assert by["rotation"]["lsq_calls"] == 0


def test_by_channel_reconciles_with_total():
    # 全フィットがチャンネルへ帰属するので、チャンネル別の総和は合算カウントに一致する。
    src = _eased_track()
    diag = {}
    reduce_bone_track(
        src,
        [(0, 20)],
        _tols(),
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
        diagnostics=diag,
    )
    total = diag["fit_counts"]
    by = diag["fit_counts_by_channel"]
    assert sum(c["fit_calls"] for c in by.values()) == total["fit_calls"]
    assert sum(c["lsq_calls"] for c in by.values()) == total["lsq_calls"]
    assert sum(c["fastpath_linear"] for c in by.values()) == total["fastpath_linear"]


def test_by_channel_camera_has_position():
    src = _eased_camera_track()
    diag = {}
    reduce_camera_track(
        src,
        [(0, 20)],
        _cam_tols(),
        cut_thresholds=(5.0, 20.0, 5.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
        diagnostics=diag,
    )
    by = diag["fit_counts_by_channel"]
    assert by["position"]["lsq_calls"] >= 1
    assert all(isinstance(v, int) for c in by.values() for v in c.values())


def test_diagnostics_omitted_does_not_error_and_has_no_fit_counts():
    # diagnostics 未指定でも通常どおり動き、出力は変わらない(fit_counts は付かない)。
    src = _eased_track()
    out = reduce_bone_track(
        src,
        [(0, 20)],
        _tols(),
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
        diagnostics=None,
    )
    assert out  # 出力キーが得られる
