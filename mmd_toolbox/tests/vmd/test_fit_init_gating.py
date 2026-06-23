"""初期値数のサンプル数連動テスト(mmd_toolbox.md §6.3)。

least_squares の試行初期値を、サンプル数(区間の内部点数)が閾値以上の高コスト区間に限って
部分集合(線形+ease-in-out)へ絞り、呼び出し回数を減らす。少数サンプルの小区間は全初期値を試す
(初期値を削っても速度効果が乏しく圧縮劣化だけ招くため)。出力(制御点)は絞る高コスト区間で
変わりうるが、采否の不変条件(出力後の最大正規化誤差 ≤ 1.0)は保つ。
"""

import math

from mmd_toolbox.vmd import fit
from mmd_toolbox.vmd import reduce as vreduce
from mmd_toolbox.vmd.reduce import build_bone_tolerances, reduce_bone_track, verify_bone_track
from mmd_toolbox.vmd.types import BoneKey

# 境界(閾値そのもの)を固定する: 閾値ちょうどは絞り(n >= 閾値で削減)、閾値直下は全初期値。
_N_LARGE = fit._LSQ_LOOSEN_MIN_SAMPLES      # 絞る下限ちょうど(=閾値)
_N_SMALL = fit._LSQ_LOOSEN_MIN_SAMPLES - 1  # 全初期値の上限ちょうど(=閾値直下)


def _capture_least_squares(monkeypatch):
    captured = []
    orig = fit.least_squares

    def cap(fun, x0, **kwargs):
        captured.append(kwargs)
        return orig(fun, x0, **kwargs)

    monkeypatch.setattr(fit, "least_squares", cap)
    return captured


def _hill_xy(n):
    # 山型=1本のベジェで表現不可。early_exit_err 無しなら早期終了せず初期値集合をすべて試す。
    xs = [(i + 1) / (n + 2) for i in range(n)]
    return xs, [math.sin(math.pi * x) for x in xs]


def test_fit_bezier_reduces_inits_for_large_segment(monkeypatch):
    # 高コスト区間(≥閾値サンプル)は部分集合の初期値だけ試す(least_squares 呼び出し数が減る)。
    cap = _capture_least_squares(monkeypatch)
    xs, ys = _hill_xy(_N_LARGE)
    fit.fit_bezier_curve(xs, ys)  # early_exit_err=None: 早期終了せず初期値集合を全て試す
    assert len(cap) == len(fit._BEZIER_INITS_LARGE)


def test_fit_bezier_uses_all_inits_for_small_segment(monkeypatch):
    # 小区間(<閾値)は全初期値を試す(絞らない)。
    cap = _capture_least_squares(monkeypatch)
    xs, ys = _hill_xy(_N_SMALL)
    fit.fit_bezier_curve(xs, ys)
    assert len(cap) == len(fit._BEZIER_INITS)


def test_fit_coeff_reduces_inits_for_large_segment(monkeypatch):
    # 回転の係数曲線も、高コスト区間では初期値を絞る。
    cap = _capture_least_squares(monkeypatch)
    n = _N_LARGE
    xs = [(i + 1) / (n + 2) for i in range(n)]
    targets = [math.sin(math.pi * x) for x in xs]

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets)]

    fit._fit_coeff_curve(xs, resid_at)
    assert len(cap) == len(fit._BEZIER_INITS_LARGE)


def _curvy_bone_keys():
    # 絞り対象(≥閾値サンプル)の区間を含む曲線的なボーン(40フレーム=最初の区間は38内部点)。
    keys = []
    for f in range(40):
        t = f / 10.0
        pos = (math.sin(t) * 3.0, math.cos(t * 0.7) * 2.0, math.sin(t * 1.3) * 1.5)
        ang = math.radians(30.0 * math.sin(t * 0.5))
        quat = (0.0, 0.0, math.sin(ang / 2), math.cos(ang / 2))
        keys.append(BoneKey(b"\x00" * 15, f, pos, quat, vreduce.BONE_LINEAR_INTERP))
    return keys


_REDUCE_ARGS = dict(
    cut_thresholds=(1.0, 30.0), keep_frames=[], no_cut_detect=True,
    min_seg=1, max_seg=180, strict=False, curve_mode="bezier",
)


def test_init_reduction_no_compression_regression_vs_full(monkeypatch):
    # 絞り対象区間を含むボーンを疎化し、采否不変(verify が空)かつ、初期値を絞った出力キー数が
    # 全初期値(絞らない)を上回らない=圧縮を劣化させないことを確認する。verify==[] は reduce が
    # 超過時に密化するため常に成立しうるので、全初期値とのキー数比較で「絞りが圧縮を悪化させない」
    # ことまで検出する。
    keys = _curvy_bone_keys()
    tols = build_bone_tolerances(0.02, 0.20)

    reduced = reduce_bone_track(keys, [(0, 39)], tols, **_REDUCE_ARGS)
    assert verify_bone_track(keys, reduced, [(0, 39)], tols) == []

    # 全区間で全初期値を試す(絞らない)よう強制して比較対象を作る。
    monkeypatch.setattr(fit, "_bezier_inits", lambda n_samples: fit._BEZIER_INITS)
    full = reduce_bone_track(keys, [(0, 39)], tols, **_REDUCE_ARGS)
    assert verify_bone_track(keys, full, [(0, 39)], tols) == []
    assert len(reduced) <= len(full)  # 初期値削減で圧縮(キー数)が劣化していない
