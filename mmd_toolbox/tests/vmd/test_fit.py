"""線形チャンネル評価器のテスト(sparsevmd.md §5.2 線形部分, §7.2)。

LinearScalarChannel は、区間 [a,b] の両端を結ぶ線形補間で内部フレームを予測し、
元サンプルとの最大絶対誤差と最大誤差フレーム(内部)を返す。normalized は誤差を
許容誤差で割った無次元量で、許容0は誤差0なら0・正なら無限大とする(§5.5)。
"""

import math

import pytest

pytest.importorskip("mmd_toolbox.vmd.fit", reason="impl pending: Step 1a fit extraction")

from mmd_toolbox.vmd import fit  # noqa: E402
from mmd_toolbox.vmd.fit import LinearScalarChannel  # noqa: E402


def test_linear_values_zero_error():
    # 完全な線形列は両端の線形補間で誤差0、内部分割点なし。
    ch = LinearScalarChannel(0, [float(i) for i in range(11)], tol=0.01)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(0.0)
    assert frame is None


def test_tent_peak_detected():
    # 山(0..5..0)。両端0を結ぶ線形=0、ピーク5で誤差5、最大誤差フレーム=5。
    vals = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ch = LinearScalarChannel(0, [float(v) for v in vals], tol=1.0)
    err, frame = ch.residual(0, 10)
    assert err == pytest.approx(5.0)
    assert frame == 5


def test_residual_adjacent_no_interior():
    ch = LinearScalarChannel(0, [0.0, 9.0], tol=1.0)
    err, frame = ch.residual(0, 1)
    assert err == pytest.approx(0.0)
    assert frame is None


def test_residual_subsegment_uses_endpoints():
    # 部分区間 [5,10] の両端(5,10)を結ぶ線形での誤差。
    vals = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ch = LinearScalarChannel(0, [float(v) for v in vals], tol=1.0)
    err, frame = ch.residual(5, 10)
    # 5→0 の直線。内部は線形に乗っているので誤差0、内部分割点なし。
    assert err == pytest.approx(0.0)
    assert frame is None


def test_residual_frame_offset_start():
    # frame_start=100。区間も絶対フレームで指定。
    vals = [0, 5, 0]
    ch = LinearScalarChannel(100, [float(v) for v in vals], tol=1.0)
    err, frame = ch.residual(100, 102)
    assert err == pytest.approx(5.0)
    assert frame == 101


def test_residual_prefers_velocity_reversal_over_max_error():
    # 端点 0,20 の直線に対し、最大誤差は単調区間の frame3(誤差11)だが、
    # 速度反転(極値)は frame1/frame2。§5.5 に従い分割候補は極値中の誤差最大 frame2。
    # 採否用の最大絶対誤差は真の最大(11.0)を返す。
    ch = LinearScalarChannel(0, [0.0, 10.0, 2.0, 4.0, 20.0], tol=1.0)
    err, frame = ch.residual(0, 4)
    assert err == pytest.approx(11.0)
    assert frame == 2


def test_normalized_divides_by_tol():
    vals = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ch = LinearScalarChannel(0, [float(v) for v in vals], tol=2.0)
    nerr, frame = ch.normalized(0, 10)
    assert nerr == pytest.approx(2.5)  # 5.0 / 2.0
    assert frame == 5


def test_normalized_zero_tol_positive_error_is_inf():
    ch = LinearScalarChannel(0, [0.0, 5.0, 0.0], tol=0.0)
    nerr, frame = ch.normalized(0, 2)
    assert nerr == math.inf
    assert frame == 1


def test_normalized_zero_tol_zero_error_is_zero():
    ch = LinearScalarChannel(0, [0.0, 1.0, 2.0], tol=0.0)
    nerr, frame = ch.normalized(0, 2)
    assert nerr == 0.0
    assert frame is None
