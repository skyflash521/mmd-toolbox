"""L-1 母音合成プロファイルのテスト(implementation-plan.md §4.1, lipsync.md §3/§4)。

各母音が §4.1 のプロファイルで複数の標準口モーフへ重みを置き、誇張係数・母音別倍率・
保持値クランプ・合成後総量の比例縮小が確定算法どおり適用されることを既知値で検証する。
"""

import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape


def _peak_weights(events, params=None):
    """単一区間のイベント列を変換し、モーフ名→ピーク(保持)ウェイトの辞書を返す。

    形状(§4.9)で各モーフは 0→保持値→保持値→0 のエンベロープになるため、各モーフの最大重み
    (=プラトーの保持値=L-1 の合成重み)を取り出す。単一キーのみのときは恒等。
    """
    params = params or GenerationParams()
    keys = lipsync.generate_morph_keys(events, params)
    weights: dict[str, float] = {}
    for k in keys:
        weights[k.name] = max(weights.get(k.name, 0.0), k.weight)
    return weights


def _vowel(shape, open_amount, start=0.0, end=10.0):
    return [MouthEvent(shape, start, end, open_amount)]


def test_single_morph_vowel_a():
    # あ: プロファイル {あ:1.0}。hold=0.5、総量0.5<cap でクランプ無し。
    assert _peak_weights(_vowel(MouthShape.A, 0.5)) == pytest.approx({"あ": 0.5})


def test_composite_vowel_i_multiple_morphs():
    # い: {あ:0.1, い:1.0}。hold=0.5 → {あ:0.05, い:0.5}、総量0.55<cap。
    w = _peak_weights(_vowel(MouthShape.I, 0.5))
    assert set(w) == {"あ", "い"}
    assert w["あ"] == pytest.approx(0.05)
    assert w["い"] == pytest.approx(0.5)


def test_hold_clamped_to_open_cap():
    # あ: open_amount=1.0 でも hold=clamp(1.0, 0, 0.8)=0.8。
    assert _peak_weights(_vowel(MouthShape.A, 1.0)) == pytest.approx({"あ": 0.8})


def test_vowel_scale_applied_then_clamped():
    # あ: open_amount=0.5, vowel_scale[a]=2 → hold=clamp(1.0, 0, 0.8)=0.8。
    p = GenerationParams(vowel_scale=(2.0, 1.0, 1.0, 1.0, 1.0))
    assert _peak_weights(_vowel(MouthShape.A, 0.5), p) == pytest.approx({"あ": 0.8})


def test_total_clamp_proportional_scaling():
    # え: {あ:0.2,い:0.2,え:1.0}, open_amount=1.0 → hold=0.8。
    # raw={あ:0.16,い:0.16,え:0.8}, 総量1.12>cap → 全体に factor=0.8/1.12。
    w = _peak_weights(_vowel(MouthShape.E, 1.0))
    factor = 0.8 / 1.12
    assert w["あ"] == pytest.approx(0.16 * factor)
    assert w["い"] == pytest.approx(0.16 * factor)
    assert w["え"] == pytest.approx(0.8 * factor)
    assert sum(w.values()) == pytest.approx(0.8)


def test_open_cap_param_used_for_hold_clamp():
    # open_cap=0.6 を渡すと保持値クランプの上限が 0.6 になる(0.8 ハードコードでないこと)。
    # あ: open_amount=1.0 → hold=clamp(1.0, 0, 0.6)=0.6 → {あ:0.6}。
    p = GenerationParams(open_cap=0.6)
    assert _peak_weights(_vowel(MouthShape.A, 1.0), p) == pytest.approx({"あ": 0.6})


def test_open_cap_param_used_for_total_clamp():
    # open_cap=0.6 が合成後総量の比例縮小の上限にも使われる。
    # え: hold=clamp(1.0,0,0.6)=0.6 → raw={あ:0.12,い:0.12,え:0.6}, 総量0.84>0.6 → factor=0.6/0.84。
    p = GenerationParams(open_cap=0.6)
    w = _peak_weights(_vowel(MouthShape.E, 1.0), p)
    factor = 0.6 / 0.84
    assert w["あ"] == pytest.approx(0.12 * factor)
    assert w["い"] == pytest.approx(0.12 * factor)
    assert w["え"] == pytest.approx(0.6 * factor)
    assert sum(w.values()) == pytest.approx(0.6)


def test_exaggeration_scales_aux_only():
    # い: exaggeration=2.0 → 補助 あ:0.1×2=0.2、主 い:1.0 は不変。
    # hold=0.5 → {あ:0.1, い:0.5}、総量0.6<cap。
    p = GenerationParams(exaggeration=2.0)
    w = _peak_weights(_vowel(MouthShape.I, 0.5), p)
    assert w["あ"] == pytest.approx(0.1)  # 補助は誇張で 0.05→0.1
    assert w["い"] == pytest.approx(0.5)  # 主モーフは誇張対象外


def test_exaggerated_aux_included_in_total_clamp():
    # え: exaggeration=2.0 → 有効 {あ:0.4,い:0.4,え:1.0}, open_amount=1.0 hold=0.8。
    # raw={あ:0.32,い:0.32,え:0.8}, 総量1.44>cap → factor=0.8/1.44。
    p = GenerationParams(exaggeration=2.0)
    w = _peak_weights(_vowel(MouthShape.E, 1.0), p)
    factor = 0.8 / 1.44
    assert w["あ"] == pytest.approx(0.32 * factor)
    assert w["い"] == pytest.approx(0.32 * factor)
    assert w["え"] == pytest.approx(0.8 * factor)
    assert sum(w.values()) == pytest.approx(0.8)


@pytest.mark.parametrize(
    "shape,index,expected",
    [
        (MouthShape.A, 0, {"あ": 0.45}),
        (MouthShape.I, 1, {"あ": 0.045, "い": 0.45}),
        (MouthShape.U, 2, {"う": 0.45, "お": 0.09}),
        (MouthShape.E, 3, {"あ": 0.09, "い": 0.09, "え": 0.45}),
        (MouthShape.O, 4, {"う": 0.09, "お": 0.45}),
    ],
)
def test_vowel_scale_indexed_per_vowel(shape, index, expected):
    # 母音別倍率 vowel_scale=(a,i,u,e,o) が母音ごとに正しい添字で、かつ hold 全体(主・補助とも)に
    # 効くか。当該母音の倍率だけ 1.5、open_amount=0.3 で hold=clamp(0.45, 0, 0.8)=0.45(cap未満で
    # 縮小なし)。各モーフ重み = 相対重み × hold で、補助も含め固定する。
    scale = [1.0, 1.0, 1.0, 1.0, 1.0]
    scale[index] = 1.5
    p = GenerationParams(vowel_scale=tuple(scale))
    assert _peak_weights(_vowel(shape, 0.3), p) == pytest.approx(expected)


@pytest.mark.parametrize(
    "shape,expected",
    [
        (MouthShape.A, {"あ": 0.4}),
        (MouthShape.I, {"あ": 0.04, "い": 0.4}),
        (MouthShape.U, {"う": 0.4, "お": 0.08}),
        (MouthShape.E, {"あ": 0.08, "い": 0.08, "え": 0.4}),
        (MouthShape.O, {"う": 0.08, "お": 0.4}),
    ],
)
def test_profile_known_weights(shape, expected):
    # 各母音が §4.1 の非ゼロプロファイルのモーフだけに、相対重み×hold の既知値を置く
    # (0.0 のモーフは出さない)。open_amount=0.4 はどの母音でも総量 < cap で縮小なし。
    assert _peak_weights(_vowel(shape, 0.4)) == pytest.approx(expected)
