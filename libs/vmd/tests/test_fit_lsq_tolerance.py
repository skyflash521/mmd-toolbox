"""least_squares 収束許容のサンプル数連動テスト。

采否は量子化後誤差で判定するので scipy 既定精度(~1e-8)まで詰める必要はない。ただし緩めると
フィット精度が落ちて分割が増えうるので、緩和の速度効果が大きい「サンプル数の多い高コスト区間」だけ
ftol/xtol/gtol を緩め、少数サンプルの小区間(緩めても速度効果が乏しく圧縮劣化だけ招く)は締めたまま
にする。出力(制御点)は緩める高コスト区間で変わりうるが、采否の不変条件(出力後の最大正規化誤差
≤ 1.0)は保つ。
"""

import math

from vmd import fit, interp
from vmd import reduce as vreduce
from vmd.reduce import build_bone_tolerances, reduce_bone_track, verify_bone_track
from vmd.types import BoneKey

_LOOSE = fit._LSQ_LOOSE_TOL
# 境界(閾値そのもの)を固定する: 閾値ちょうどは緩め(n >= 閾値で緩和)、閾値直下は据え置き。
# これで実装が誤って「閾値超」へ変わると小区間側テストが落ちる。
_N_LARGE = fit._LSQ_LOOSEN_MIN_SAMPLES      # 緩める下限ちょうど(=閾値)
_N_SMALL = fit._LSQ_LOOSEN_MIN_SAMPLES - 1  # 据え置きの上限ちょうど(=閾値直下)


def _capture_least_squares(monkeypatch):
    captured = []
    orig = fit.least_squares

    def cap(fun, x0, **kwargs):
        captured.append(kwargs)
        return orig(fun, x0, **kwargs)

    monkeypatch.setattr(fit, "least_squares", cap)
    return captured


def _eased_xy(n):
    # 線形で収まらないイージング列(ファストパスを取らず least_squares を回す)。
    xs = [(i + 1) / (n + 2) for i in range(n)]
    return xs, [interp._solve_factor(96, 0, 96, 30, x) for x in xs]


def test_fit_bezier_loosens_tolerance_for_large_segment(monkeypatch):
    # サンプル数が閾値以上の高コスト区間は least_squares の収束許容を緩める(反復削減)。
    cap = _capture_least_squares(monkeypatch)
    xs, ys = _eased_xy(_N_LARGE)
    fit.fit_bezier_curve(xs, ys)  # early_exit_err=None: ファストパスを取らず least_squares を回す
    assert cap, "least_squares が呼ばれていない"
    assert all(
        kw.get("ftol") == _LOOSE and kw.get("xtol") == _LOOSE and kw.get("gtol") == _LOOSE
        for kw in cap
    )


def test_fit_bezier_keeps_tight_tolerance_for_small_segment(monkeypatch):
    # サンプル数が閾値未満の小区間は緩めない(緩めても速度効果が乏しく圧縮劣化を招くため)。
    cap = _capture_least_squares(monkeypatch)
    xs, ys = _eased_xy(_N_SMALL)
    fit.fit_bezier_curve(xs, ys)
    assert cap, "least_squares が呼ばれていない"
    assert all("ftol" not in kw and "xtol" not in kw and "gtol" not in kw for kw in cap)


def test_fit_coeff_loosens_tolerance_for_large_segment(monkeypatch):
    # 回転の係数曲線も、高コスト区間では収束許容を緩める。
    cap = _capture_least_squares(monkeypatch)
    n = _N_LARGE
    xs = [(i + 1) / (n + 2) for i in range(n)]
    targets = [math.sin(math.pi * x) for x in xs]  # 山型=1本のベジェで表現不可

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets)]

    fit._fit_coeff_curve(xs, resid_at)
    assert cap, "least_squares が呼ばれていない"
    assert all(
        kw.get("ftol") == _LOOSE and kw.get("xtol") == _LOOSE and kw.get("gtol") == _LOOSE
        for kw in cap
    )


def _curvy_bone_keys():
    # 緩和対象(≥閾値サンプル)の区間を含む曲線的なボーン(40フレーム=最初の区間は38内部点)。
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


def test_loosening_no_compression_regression_vs_tight(monkeypatch):
    # 緩和対象区間を含むボーンを疎化し、(1)緩和が実際に発火し、(2)采否不変(verify が空)、
    # (3)緩和ありの出力キー数が既定精度(緩和なし)を上回らない=圧縮を劣化させないことを確認する。
    # verify==[] は reduce が超過時に密化するため常に成立しうるので、(1)発火と(3)キー数比較で
    # 「緩和が実際に通り、かつ圧縮を悪化させない」ことまで検出する。
    keys = _curvy_bone_keys()
    tols = build_bone_tolerances(0.02, 0.20)

    cap = _capture_least_squares(monkeypatch)
    loosened = reduce_bone_track(keys, [(0, 39)], tols, **_REDUCE_ARGS)
    assert any(kw.get("ftol") == _LOOSE for kw in cap)  # ≥閾値区間を通り緩和が発火した
    assert verify_bone_track(keys, loosened, [(0, 39)], tols) == []

    # 全区間を既定精度に強制(緩和なし)して比較対象を作る。
    monkeypatch.setattr(fit, "_lsq_kwargs", lambda n_samples: {})
    tight = reduce_bone_track(keys, [(0, 39)], tols, **_REDUCE_ARGS)
    assert verify_bone_track(keys, tight, [(0, 39)], tols) == []
    assert len(loosened) <= len(tight)  # 緩和で圧縮(キー数)が劣化していない
