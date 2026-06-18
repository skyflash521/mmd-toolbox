"""ベジェフィットの早期終了テスト(性能改修 / performance-fix-plan.md Step 3 の品質保存版)。

fit_bezier_curve / _fit_coeff_curve は初期値4種(線形/ease-in/ease-out/ease-in-out)から
最適化し最良を採る。区間が4自由度のベジェでほぼ厳密に表現できる場合、最初の初期値からの
最適化が量子化下限近くまで収束するため、残りの初期値を試す必要がない。早期終了は
「得られた曲線が近似的に厳密(err < 量子化下限の小ε)」なときだけ打ち切る。単に許容内、では
打ち切らない(曲がった区間を線形へ劣化させない)。表現できないデータは従来どおり4種を試す。

本ファイルのテストファースト段は「早期終了が表現可能データで発火し、表現不能データでは発火しない」
ことと「フィット品質が保たれる」ことを検証する。早期終了の出力が全初期値試行と同一(または劣化が
ε以内)であることの直接照合は、早期終了の実装が無いと両実行とも全探索になり自明一致するため、
実装フェーズ(早期終了を入れるコミット)で追加する。
"""

import math

import pytest

from mmd_toolbox.vmd import fit, interp

EASE = (96, 0, 96, 30)


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


@pytest.mark.xfail(reason="impl pending: Step 2 early-exit", strict=True)
def test_representable_curve_exits_after_first_init(monkeypatch):
    # 既知曲線に厳密に沿うデータは、線形初期値からの最適化が近似厳密に収束するため
    # 最初の1初期値で確定し、least_squares は1回だけ呼ばれる。
    calls = _count_least_squares(monkeypatch)
    xs, ys = _curve_ys(EASE)
    cp, err = fit.fit_bezier_curve(xs, ys)
    assert err < 1e-3
    assert calls["n"] == 1


def test_unrepresentable_data_tries_all_inits(monkeypatch):
    # 1本のベジェでは表現できない非単調データ(山)では早期終了せず4初期値すべて試す。
    # 早期終了が過剰発火しない(非表現データを誤って打ち切らない)ことのガード。
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 10 for i in range(9)]
    ys = [math.sin(math.pi * x) for x in xs]  # 0.31→1→0.31 の山(単一曲線で不可)
    fit.fit_bezier_curve(xs, ys)
    assert calls["n"] == 4


@pytest.mark.xfail(reason="impl pending: Step 2 early-exit", strict=True)
def test_coeff_curve_exits_after_first_init(monkeypatch):
    # 回転の係数曲線フィットも、係数が既知曲線に沿うなら早期終了する。
    calls = _count_least_squares(monkeypatch)
    xs = [(i + 1) / 6 for i in range(5)]
    targets = [interp._solve_factor(*EASE, x) for x in xs]

    def resid_at(coeff):
        return [coeff(x) - t for x, t in zip(xs, targets)]

    fit._fit_coeff_curve(xs, resid_at)
    assert calls["n"] == 1


def test_early_exit_preserves_known_curve_cp():
    # 早期終了しても既知曲線の制御点復元は不変(品質保存)。
    xs, ys = _curve_ys(EASE, n=9)
    cp, err = fit.fit_bezier_curve(xs, ys)
    assert err < 0.01
    assert cp == EASE


@pytest.mark.parametrize(
    "curve",
    [(20, 40, 107, 100), (0, 0, 127, 64), (40, 10, 90, 118), (96, 0, 96, 30)],
)
def test_fit_quality_preserved_multiple_shapes(curve):
    # 早期終了が「非最適な初期値で打ち切る」(=eps が緩すぎる)と、得られる制御点が
    # 大域最適から外れフィット誤差が増える。複数の曲線形状で良好なフィット(誤差小)が
    # 保たれることを確認し、早期終了が品質を損なわないことをガードする。
    xs, ys = _curve_ys(curve, n=9)
    cp, err = fit.fit_bezier_curve(xs, ys)
    assert err < 0.05
