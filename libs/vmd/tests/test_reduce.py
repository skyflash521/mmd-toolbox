"""区間削減器(linear mode)のテスト(sparsevmd.md §5.1, §5.5, §2.5)。

reduce_track は必須境界の間を区間化し、各区間を全チャンネルが許容誤差以内で
線形表現できるか検査する。超過時は最大正規化誤差フレームで再帰分割する(error-split)。
max-segment-frames は出力キー間隔の sliding 上限で、許容内でも非定数区間が上限を超える場合だけ
上限位置で分割する(maxspan-cap)。min-segment-frames を下回る tol 探索分割はしない。
非strictでは min-segment まで分割しても満たせない区間を1フレームまで密に保持し、
strictでは終了コード4につながる StrictError を送出する。
"""

import pytest

from vmd import reduce as reducer  # noqa: F401
from vmd.fit import LinearScalarChannel
from vmd.reduce import (
    BONE_LINEAR_INTERP,
    StrictError,
    build_bone_tolerances,
    reduce_bone_track,
    reduce_track,
)
from vmd.types import BoneKey


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


def test_max_seg_none_disables_cap():
    # max_seg=None は上限なし(無制限)。許容内なら非定数の長区間でも maxspan-cap せず分割しない。
    ch = lin(0, range(101), tol=0.01)   # 完全線形 0..100(非定数)
    assert reduce_track([0, 100], [ch], min_seg=1, max_seg=None, strict=False) == [0, 100]
    # 対比: 有限上限なら同じ長区間を上限位置で maxspan-cap して中間キーを足す。
    capped = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)
    assert capped != [0, 100] and len(capped) > 2


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


def test_max_segment_sliding_cap():
    # 線形だが max_seg=40 で sliding 上限超過。tol探索では分割されない(線形)ので、
    # 中間キーは maxspan-cap 由来。区間数は ceil(100/40)=3、全区間 <=40、端点保持。
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


def test_constant_span_not_capped():
    # 全フレーム同値(定数)の span は max_seg を超えても分割されず両端2キーになる
    # (定数区間には編集すべき曲がりが無いため maxspan-cap の対象外。§5.1 step3)。
    ch = lin(0, [5.0] * 101, tol=0.01)
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)
    assert keys == [0, 100]


def test_loose_non_constant_span_capped():
    # 緩い線形(定数でない)長い span は maxspan-cap で上限以下に保たれる(編集容易性の回帰防止)。
    ch = lin(0, [float(i) for i in range(101)], tol=1.0)
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)
    gaps = [b - a for a, b in zip(keys, keys[1:])]
    assert all(g <= 40 for g in gaps)
    assert len(gaps) == 3  # 定数判定が誤発火せず非定数として maxspan-cap が効く


def test_mixed_constant_and_varying_span_no_grid_in_flat():
    # 前半定数＋後半上昇が1 span に混在。sliding 上限では定数前半に機械的な等分 grid キーを撒かず、
    # 曲がり(定数→上昇の境界)付近で誤差駆動分割する。定数前半は maxspan-cap 対象外なので
    # 上限を超える1区間のまま残る。
    vals = [5.0] * 61 + [5.0 + float(i + 1) for i in range(40)]  # 0..60 定数, 61..100 上昇
    ch = lin(0, vals, tol=1.0)
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)
    assert keys[0] == 0 and keys[-1] == 100
    assert not any(0 < k < 55 for k in keys)  # 定数前半に等分 grid キーが出ない
    assert any(55 <= k <= 65 for k in keys)   # 曲がり付近にキー


def test_maxspan_cap_requires_all_channels_constant():
    # 片方のチャンネルが変化していれば span 全体は定数でない → maxspan-cap が効く(全チャンネル定数が条件)。
    const_ch = lin(0, [5.0] * 101, tol=0.01)
    vary_ch = lin(0, [float(i) for i in range(101)], tol=1.0)
    keys = reduce_track([0, 100], [const_ch, vary_ch], min_seg=1, max_seg=40, strict=False)
    gaps = [b - a for a, b in zip(keys, keys[1:])]
    assert all(g <= 40 for g in gaps)
    assert len(gaps) == 3


def test_non_constant_fallback_when_channel_lacks_is_constant():
    # is_constant を持たないチャンネル(StubChannel 等)は非定数扱いになり maxspan-cap の対象になる。
    class StubChannel:
        def normalized(self, a, b):
            return (0.0, None)  # 常に許容内(tol 分割しない)

    keys = reduce_track([0, 100], [StubChannel()], min_seg=1, max_seg=40, strict=False)
    gaps = [b - a for a, b in zip(keys, keys[1:])]
    assert all(g <= 40 for g in gaps)
    assert len(gaps) == 3  # is_constant 不在 → 非定数扱いで maxspan-cap


def test_constant_span_reduce_deterministic():
    # 同入力・同引数で同出力(決定論)。
    def run():
        ch = lin(0, [5.0] * 101, tol=0.01)
        return reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False)

    assert run() == run()


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


# --- sliding max_seg / maxspan-cap (§5.1 step3/step6, §5.5) -------------------


def test_fitting_long_span_capped_and_recorded():
    # 許容内に収まる非定数の長い span(>max_seg)は maxspan-cap で上限以下に保ち、診断 caps に
    # 別理由として記録する(誤差駆動の splits とは別)。
    ch = lin(0, [float(i) for i in range(101)], tol=1.0)  # 線形=1本で表現可
    caps = []
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False, caps=caps)
    gaps = [b - a for a, b in zip(keys, keys[1:])]
    assert all(g <= 40 for g in gaps)
    assert caps and all(0 < c["frame"] < 100 for c in caps)


def test_constant_long_span_not_capped():
    # 定数の長い span は maxspan-cap 対象外。caps 空・両端のみ・上限超過を許す。
    ch = lin(0, [5.0] * 101, tol=0.01)
    caps = []
    keys = reduce_track([0, 100], [ch], min_seg=1, max_seg=40, strict=False, caps=caps)
    assert keys == [0, 100]
    assert caps == []


def test_error_split_separate_from_cap():
    # 誤差超過の分割は splits(error-split)に入り、cap が不要(max_seg 大)なら caps は空。
    ch = lin(0, [0, 10, 2, 4, 20], tol=9.0)
    splits, caps = [], []
    reduce_track([0, 4], [ch], min_seg=1, max_seg=180, strict=False, splits=splits, caps=caps)
    assert splits  # 誤差駆動の分割が記録される
    assert caps == []


def test_no_mechanical_grid_when_span_within_max_seg():
    # span が max_seg 以下なら、フィット可の滑らかな区間に機械的な内部キーを撒かない。
    ch = lin(0, [float(i) for i in range(31)], tol=1.0)  # 線形・span 30
    caps = []
    keys = reduce_track([0, 30], [ch], min_seg=1, max_seg=180, strict=False, caps=caps)
    assert keys == [0, 30]  # 等分 grid キーが出ない
    assert caps == []


def test_reduce_bone_track_diagnostics_has_maxspan_caps():
    # diagnostics に maxspan_caps フィールドが入り、長い線形位置で cap が発火する。
    name = b"bone".ljust(15, b"\x00")
    src = [
        BoneKey(name, f, (float(f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)
        for f in range(101)
    ]
    diag = {}
    reduce_bone_track(
        src,
        [(0, 100)],
        build_bone_tolerances(bone_pos=0.5, bone_rot=5.0),
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=40,
        strict=False,
        curve_mode="bezier",
        diagnostics=diag,
    )
    assert "maxspan_caps" in diag
    assert diag["maxspan_caps"]
