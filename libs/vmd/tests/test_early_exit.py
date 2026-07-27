"""ベジェフィットの早期終了テスト。

fit_bezier_curve / _fit_coeff_curve は初期値4種(線形/ease-in/ease-out/ease-in-out)から
最適化し最良(コスト最小)を採る。early_exit_err を渡すと、現在の最良の量子化誤差がそれ以下に
なった時点で残りの初期値を試さず打ち切る。閾値は呼び出し側(チャンネル)が許容誤差から算出して
渡す(区間が許容内にフィットできた時点で打ち切る)ため、採否(誤差 <= 許容)は変わらず、出力は
全初期値試行と許容内一致になる。early_exit_err=None なら早期終了せず全初期値を試す。

検証は2系統:
- 発火と過剰発火しないこと(least_squares 呼び出し回数)。
- 品質保存(早期終了の出力が全初期値試行と同一、または劣化が閾値以内)。early_exit_err なし
  (全試行)と あり(早期終了)の出力を直接照合する。
"""

import math

import pytest

from vmd import fit, interp

EASE = (96, 0, 96, 30)
# 表現可能データに対して確実に発火する緩めの閾値(正規化y単位)。
EE = 0.01


def _count_least_squares(monkeypatch):
    counter = {"n": 0}
    orig = fit.least_squares

    def wrapper(*args, **kwargs):
        counter["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(fit, "least_squares", wrapper)
    return counter


def _curve_ys(curve, n=5):
    # 既知の量子化曲線 curve に正規化時間で沿う y 列(内部点が4自由度のベジェで厳密表現可)。
    xs = [(i + 1) / (n + 1) for i in range(n)]
    return xs, [interp._solve_factor(*curve, x) for x in xs]


# --- 発火 / 過剰発火しないこと -----------------------------------------------


def test_representable_curve_exits_after_first_init(monkeypatch):
    # 既知曲線に厳密に沿うデータは、線形初期値からの最適化が許容内(ほぼ厳密)に収束するため
    # 最初の1初期値で確定し、least_squares は1回だけ呼ばれる。
    calls = _count_least_squares(monkeypatch)
    xs, ys = _curve_ys(EASE)
    cp, err = fit.fit_bezier_curve(xs, ys, early_exit_err=EE)
    assert err < 1e-3
    assert calls["n"] == 1


def test_unrepresentable_data_does_not_exit(monkeypatch):
    # 1本のベジェでは表現できない非単調データ(山)では、どの初期値も閾値内に収まらないため
    # 早期終了せず4初期値すべて試す(早期終了が過剰発火しないことのガード)。
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 10 for i in range(9)]
    ys = [math.sin(math.pi * x) for x in xs]  # 0.31→1→0.31 の山(単一曲線で不可)
    fit.fit_bezier_curve(xs, ys, early_exit_err=EE)
    assert calls["n"] == 4


def test_coeff_curve_exits_after_first_init(monkeypatch):
    # 回転の係数曲線フィットも、係数が既知曲線に沿うなら早期終了する。
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 6 for i in range(5)]
    targets = [interp._solve_factor(*EASE, x) for x in xs]

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets, strict=True)]

    fit._fit_coeff_curve(xs, resid_at, early_exit_err=EE)
    assert calls["n"] == 1


# --- 品質(全初期値試行=早期終了なしの基本挙動は不変) -----------------------


def test_full_search_preserves_known_curve_cp():
    # early_exit_err なし(全初期値試行)では既知曲線の制御点を安定に復元する。
    xs, ys = _curve_ys(EASE, n=9)
    cp, err = fit.fit_bezier_curve(xs, ys)
    assert err < 0.01
    assert cp == EASE


@pytest.mark.parametrize(
    "curve",
    [(20, 40, 107, 100), (0, 0, 127, 64), (40, 10, 90, 118), (96, 0, 96, 30)],
)
def test_full_search_quality_multiple_shapes(curve):
    xs, ys = _curve_ys(curve, n=9)
    cp, err = fit.fit_bezier_curve(xs, ys)
    assert err < 0.05


# --- 品質保存の直接照合(早期終了 vs 全初期値試行) -------------------------


def test_scalar_early_exit_exact_for_ease():
    # 代表ケース(EASE)では最初の初期値が大域最適へ収束するため、早期終了の出力は全初期値試行と
    # 完全一致する(制御点・誤差ともに)。一般の曲線では一致まで保証されず「閾値以内」になる
    # (下の within_threshold テストで担保。早期終了は許容内で発火し採否は不変なので品質は許容内一致)。
    xs, ys = _curve_ys(EASE, n=9)
    cp_full, err_full = fit.fit_bezier_curve(xs, ys)
    cp_fast, err_fast = fit.fit_bezier_curve(xs, ys, early_exit_err=EE)
    assert cp_fast == cp_full
    assert err_fast == err_full


@pytest.mark.parametrize(
    "curve",
    [(20, 40, 107, 100), (0, 0, 127, 64), (40, 10, 90, 118), (96, 0, 96, 30)],
)
def test_scalar_early_exit_within_threshold_clean(curve):
    # 表現可能データでも、最初の初期値が必ずしも大域最適に収束するとは限らない。早期終了の出力誤差は
    # 全初期値試行の最良誤差を early_exit_err 以上に上回らない(品質劣化は閾値=許容由来以内)。
    xs, ys = _curve_ys(curve, n=9)
    _cp_full, err_full = fit.fit_bezier_curve(xs, ys)
    _cp_fast, err_fast = fit.fit_bezier_curve(xs, ys, early_exit_err=EE)
    assert err_fast <= err_full + EE + 1e-12


def test_noisy_data_early_exit_matches_full_search():
    # 非表現データ(山型)では早期終了は発火せず(閾値内に収まらない)、出力は全試行と完全一致。
    xs = [(i + 1) / 10 for i in range(9)]
    ys = [math.sin(math.pi * x) for x in xs]
    cp_full, err_full = fit.fit_bezier_curve(xs, ys)
    cp_fast, err_fast = fit.fit_bezier_curve(xs, ys, early_exit_err=EE)
    assert cp_fast == cp_full
    assert err_fast == err_full


@pytest.mark.parametrize(
    "curve",
    [(20, 40, 107, 100), (0, 0, 127, 64), (40, 10, 90, 118), (96, 0, 96, 30)],
)
def test_early_exit_err_within_threshold_of_full_search(curve):
    # 一般保証(境界ケースを含む全入力): 早期終了の出力誤差は全初期値試行の最良誤差を early_exit_err
    # 以上に上回らない=品質劣化は閾値以内。発火時 err_fast <= early_exit_err(かつ err_full <= err_fast)で
    # 差 <= early_exit_err、非発火時 err_fast == err_full。既知曲線に微小ノイズを載せて境界化する。
    ee = 0.01
    xs = [(i + 1) / 10 for i in range(9)]
    base = [interp._solve_factor(*curve, x) for x in xs]
    ys = [b + 0.003 * math.sin(31.0 * (i + 1)) for i, b in enumerate(base)]
    _cp_full, err_full = fit.fit_bezier_curve(xs, ys)
    _cp_fast, err_fast = fit.fit_bezier_curve(xs, ys, early_exit_err=ee)
    assert err_fast <= err_full + ee + 1e-12


@pytest.mark.parametrize("curve", [(96, 0, 96, 30), (40, 10, 90, 118)])
def test_coeff_early_exit_output_identical_to_full_search(curve):
    # 回転の係数曲線フィットも、表現可能データでは早期終了の有無で制御点が完全一致する。
    xs = [(i + 1) / 6 for i in range(5)]
    targets = [interp._solve_factor(*curve, x) for x in xs]

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets, strict=True)]

    cp_full = fit._fit_coeff_curve(xs, resid_at)
    cp_fast = fit._fit_coeff_curve(xs, resid_at, early_exit_err=EE)
    assert cp_fast == cp_full


def test_coeff_noisy_data_early_exit_matches_full_search():
    # 回転の係数曲線も、単一曲線で表現できない残差(山型)では早期終了は発火せず制御点が完全一致。
    xs = [(i + 1) / 10 for i in range(9)]
    targets = [math.sin(math.pi * x) for x in xs]

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets, strict=True)]

    cp_full = fit._fit_coeff_curve(xs, resid_at)
    cp_fast = fit._fit_coeff_curve(xs, resid_at, early_exit_err=EE)
    assert cp_fast == cp_full


# --- 統合: 早期終了有効の bezier reduce が全チャンネル許容内 ----------------


def _camera_reduce_source():
    """FOV(整数丸め)・位置(3軸ユークリッド)・回転を同時に動かす reduce 用ソースと許容。"""
    from vmd.reduce import Tolerances
    from vmd.types import CameraKey

    lin = bytes([20, 107, 20, 107]) * 6
    tols = Tolerances(
        bone_pos=0.01, bone_rot=0.10,
        camera_pos=0.02, camera_rot=0.05, camera_distance=0.02, camera_fov=0.50,
    )
    n = 11

    def eased(v0, v1):
        return [v0 + (v1 - v0) * interp._solve_factor(*EASE, i / (n - 1)) for i in range(n)]

    posy = eased(0.0, 5.0)
    dist = eased(-30.0, -20.0)
    fov = eased(30.0, 50.0)            # 連続変化→FOV は整数丸めを経る
    ry = eased(0.0, math.radians(15))  # 回転(係数曲線経路)
    source = [
        CameraKey(f, dist[f], (0.0, posy[f], 0.0), (0.0, ry[f], 0.0), lin, fov[f], 0)
        for f in range(n)
    ]
    return source, tols, n


def _camera_track(source, tols, n):
    from vmd.reduce import reduce_camera_track

    return reduce_camera_track(
        source, [(0, n - 1)], tols,
        cut_thresholds=(5.0, 20.0, 5.0), keep_frames=[], no_cut_detect=True,
        min_seg=1, max_seg=180, strict=False, curve_mode="bezier",
    )


def test_reduce_bezier_within_tol_fov_position_rotation():
    # 早期終了は常にチャンネル経由で有効。FOV(整数丸め)・位置(ユークリッド3軸)・回転を
    # 同時に動かす入力を bezier で削減し、出力後検証 verify_camera_track が空(全チャンネルが
    # 実許容内)になることを確認する。FOV 丸め経路と euclidean の √3 閾値経路を reduce+verify
    # 込みで実際に踏み、早期終了が手ぶれ品質を許容超で損なわないことを統合レベルでガードする。
    from vmd.reduce import verify_camera_track

    source, tols, n = _camera_reduce_source()
    out = _camera_track(source, tols, n)
    assert verify_camera_track(source, out, [(0, n - 1)], tols) == []


def test_early_exit_fires_during_reduce(monkeypatch):
    # 統合レベルで早期終了が実際に発火することのガード: 同じ bezier reduce を「早期終了あり
    # (チャンネルが許容由来の閾値を渡す)」と「早期終了なし(fit を early_exit_err=None で強制)」
    # で実行し、early_exit 版の least_squares 呼び出し回数が少ないことを確認する。発火しなければ
    # 両者は同数になるため、早期終了が reduce 経路で実際に効いていることを示す。
    source, tols, n = _camera_reduce_source()

    counter = {"n": 0}
    orig_ls = fit.least_squares

    def counting_ls(*args, **kwargs):
        counter["n"] += 1
        return orig_ls(*args, **kwargs)

    monkeypatch.setattr(fit, "least_squares", counting_ls)

    _camera_track(source, tols, n)
    ee_calls = counter["n"]

    # 早期終了を無効化(fit を early_exit_err=None で強制)して全初期値試行にする。
    counter["n"] = 0
    orig_fbc = fit.fit_bezier_curve
    orig_fcc = fit._fit_coeff_curve
    monkeypatch.setattr(fit, "fit_bezier_curve", lambda xs, ys, early_exit_err=None, category=None, skip_fastpath=False: orig_fbc(xs, ys, early_exit_err=None, category=category, skip_fastpath=skip_fastpath))
    monkeypatch.setattr(fit, "_fit_coeff_curve", lambda xs, r, early_exit_err=None, category=None, skip_fastpath=False: orig_fcc(xs, r, early_exit_err=None, category=category, skip_fastpath=skip_fastpath))

    _camera_track(source, tols, n)
    full_calls = counter["n"]

    assert ee_calls < full_calls  # 早期終了で least_squares 呼び出しが減った=reduce 経路で発火
