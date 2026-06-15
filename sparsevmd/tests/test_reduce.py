"""区間削減器(linear mode)のテスト(sparsevmd.md §5.1, §5.5, §2.5)。

reduce_track は必須境界の間を区間化し、各区間を全チャンネルが許容誤差以内で
線形表現できるか検査する。超過時は最大正規化誤差フレームで再帰分割する。
max-segment-frames で事前分割し、min-segment-frames を下回る tol 探索分割はしない。
非strictでは min-segment まで分割しても満たせない区間を1フレームまで密に保持し、
strictでは終了コード4につながる StrictError を送出する。
"""

import pytest

from sparsevmd import reduce as reducer
from sparsevmd.fit import LinearScalarChannel
from sparsevmd.reduce import StrictError, reduce_track


def lin(frame_start, values, tol):
    return LinearScalarChannel(frame_start, [float(v) for v in values], tol)


def test_linear_segment_no_split():
    ch = lin(0, range(11), tol=0.01)
    keys = reduce_track([0, 10], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 10]


def test_peak_splits_at_extremum():
    vals = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ch = lin(0, vals, tol=1.0)
    keys = reduce_track([0, 10], [ch], min_seg=1, max_seg=180, strict=False)
    # ピーク5で分割。両半分は線形なのでそれ以上分割不要。
    assert keys == [0, 5, 10]


def test_valley_splits_at_extremum():
    # 谷(0..-5..0)も極値で分割される(§5.5 山・谷・切り返し)。
    vals = [0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0]
    ch = lin(0, vals, tol=1.0)
    keys = reduce_track([0, 10], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 5, 10]


def test_mandatory_boundaries_kept():
    # 線形でも与えた必須境界(30)は保持される。
    ch = lin(0, [float(i) for i in range(61)], tol=0.01)
    keys = reduce_track([0, 30, 60], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 30, 60]


def test_unsorted_duplicate_boundaries_normalized():
    # 未ソート・重複の境界入力でも昇順・重複除去して扱う。
    ch = lin(0, [float(i) for i in range(61)], tol=0.01)
    keys = reduce_track([60, 0, 30, 30, 0], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 30, 60]


def test_max_segment_pre_split():
    # 線形だが max_seg=40 で事前分割。tol探索では分割されない(線形)ので、
    # 中間キーは事前分割由来。区間数は ceil(100/40)=3、全区間 <=40、端点保持。
    ch = lin(0, [float(i) for i in range(101)], tol=1.0)
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)
    assert keys[0] == 0 and keys[-1] == 100
    gaps = [b - a for a, b in zip(keys, keys[1:])]
    assert all(g <= 40 for g in gaps)
    assert len(gaps) == 3  # ceil(100/40) の最小区間数


def test_dense_fallback_non_strict():
    # 表現不能なジグザグ。min_seg=1 なので最終的に全フレームがキー。
    vals = [0, 5, 0, 5, 0]
    ch = lin(0, vals, tol=0.5)
    keys = reduce_track([0, 4], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 1, 2, 3, 4]


def test_strict_raises_when_min_seg_blocks():
    # min_seg=4 で区間[0,4]がジグザグで表現不能。分割下限に達し strict で StrictError。
    vals = [0, 5, 0, 5, 0]
    ch = lin(0, vals, tol=0.5)
    with pytest.raises(StrictError):
        reduce_track([0, 4], [ch], min_seg=4, max_seg=180, strict=True)


def test_min_seg_blocks_then_dense_non_strict():
    # 同じケースで非strict: min_seg を無視して1フレームまで密に保持(§2.5)。
    vals = [0, 5, 0, 5, 0]
    ch = lin(0, vals, tol=0.5)
    keys = reduce_track([0, 4], [ch], min_seg=4, max_seg=180, strict=False)
    assert keys == [0, 1, 2, 3, 4]


def test_multichannel_respects_each_channel_tolerance():
    # chA は大振幅だが緩い tol で誤差0(線形)。chB は微小な 0.05 のバンプだが tol=0.01。
    # 各チャンネル固有の許容で判定するため、絶対誤差が小さくても chB のピーク5で分割する
    # (グローバルな生誤差閾値では見逃すケース。§5.5 の正規化誤差)。
    ch_a = lin(0, [float(i) for i in range(11)], tol=1.0)
    ch_b = lin(0, [0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0], tol=0.01)
    keys = reduce_track([0, 10], [ch_a, ch_b], min_seg=1, max_seg=180, strict=False)
    # スパイクを隔離するまで全区間が tol 内になるよう再帰分割される。
    # [0,5] は frame4 で最大誤差0.04(>0.01)→分割、[5,10] は frame6 で同様。
    # 平坦部はそれ以上分割しない。
    assert keys == [0, 4, 5, 6, 10]


def test_split_prefers_extremum_not_max_error_frame():
    # 端点0,20。最大誤差は frame3 だが速度反転の極値は frame2。tol=9 で1回分割。
    # 極値優先なら [0,2,4]、生の最大誤差優先なら [0,3,4] になる。§5.5。
    ch = lin(0, [0, 10, 2, 4, 20], tol=9.0)
    keys = reduce_track([0, 4], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 2, 4]


def test_adjacent_segment_accepted():
    # 隣接フレーム区間は内部点が無く常に受理(分割不能)。
    ch = lin(0, [0.0, 100.0], tol=0.001)
    keys = reduce_track([0, 1], [ch], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 1]


def test_reduce_uses_only_normalized_contract():
    # reduce_track はチャンネルの normalized(a,b) のみに依存する(ダックタイプ)。
    # 余計な属性を持たない最小スタブでも動作すること。
    class StubChannel:
        def __init__(self, worst):
            self._worst = worst  # {(a,b): (norm_err, frame)}

        def normalized(self, a, b):
            return self._worst.get((a, b), (0.0, None))

    # [0,10] は frame5 で超過、分割後の [0,5]/[5,10] は誤差0。
    stub = StubChannel({(0, 10): (3.0, 5), (0, 5): (0.0, None), (5, 10): (0.0, None)})
    keys = reduce_track([0, 10], [stub], min_seg=1, max_seg=180, strict=False)
    assert keys == [0, 5, 10]
