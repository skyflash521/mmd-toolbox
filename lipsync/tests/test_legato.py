"""レガート間隙(LEGATO_GAP)の谷橋渡しのテスト(lipsync.md §3/§4.12)。

母音グループ間に挟まる `MouthShape.LEGATO_GAP` 区間(あ→閉じかけ→う)は、完全閉口(0.0)でなく前後母音の
境界保持値を谷で繋ぐ。前後母音の通常の 0.0 リリース/アタックキーはこの span で抑制し、谷キー
(gap_start=w_a、gap_mid=d·(w_a+w_b)、gap_end=w_b)を線形補間で置く。中央 gap_mid では前後母音の口形を加算で
重ね(オーバーラップ)、谷係数 d が総開き量を部分的な閉じ(閉じかけ)へ抑える。d は間隙長の線形関数
`clamp(shallow − slope·gap_len, deep, shallow)` で、間隙が長いほど深く(d 小)なる。数値(shallow/deep/
slope)は GenerationParams の初期目安で、視覚で詰める。

注: LEGATO_GAP/SILENCE のどちらを渡すか(間隙分類)は呼び出し側の責務であり、lipsync は確定入力に従う
(lipsync.md §6)。本テストは確定入力としての LEGATO_GAP に対する谷生成のみを検証する。
"""

import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape


def _envelope(events, params):
    by_morph: dict[str, list[tuple[int, float]]] = {}
    for k in lipsync.generate_morph_keys(events, params):
        by_morph.setdefault(k.name, []).append((k.frame, k.weight))
    for name in by_morph:
        by_morph[name].sort()
    return by_morph


def _approx_envelope(actual, expected):
    assert [f for f, _ in actual] == [f for f, _ in expected]
    for (_, aw), (_, ew) in zip(actual, expected):
        assert aw == pytest.approx(ew)


def test_legato_gap_bridges_with_valley():
    # あ[0,10]・LEGATO_GAP[10,14]・い[14,24]。間隙長4 → d = 0.4 − 0.025·4 = 0.3。
    # 前接は SILENCE でないため先行準備は効かず(A_eff=0)、谷が閉口を置換する。
    # w_a(あの境界保持)={あ:0.5}、w_b(いの境界保持)={あ:0.05, い:0.5}。
    # 谷 mid=12(両母音を加算で重ねるオーバーラップ): あ 0.3·(0.5+0.05)=0.165、い 0.3·(0.0+0.5)=0.15。
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.LEGATO_GAP, 10.0, 14.0),
            MouthEvent(MouthShape.I, 14.0, 24.0, 0.5),
        ],
        GenerationParams(),
    )
    assert set(env) == {"あ", "い"}
    # あ: アタックで0.5、保持、谷の境界10で0.5を維持し mid12で0.165へ、gap_end14で w_b=0.05、保持後リリース。
    _approx_envelope(
        env["あ"],
        [(0, 0.0), (2, 0.5), (10, 0.5), (12, 0.165), (14, 0.05), (22, 0.05), (24, 0.0)],
    )
    # い: 谷の境界10で0(w_a に無い)、mid12で0.15、gap_end14で w_b=0.5に達し保持、リリースで0。
    _approx_envelope(
        env["い"],
        [(10, 0.0), (12, 0.15), (14, 0.5), (22, 0.5), (24, 0.0)],
    )


def test_legato_valley_deepens_with_longer_gap():
    # 谷の深さ d は間隙長に対し線形で深くなる(d 小=谷深い)。同じ前後母音で間隙長だけ変える。
    # 短間隙(長さ2): d=0.4−0.025·2=0.35、mid=11、あ谷値=0.35·(0.5+0.05)=0.192500。
    short = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.LEGATO_GAP, 10.0, 12.0),
            MouthEvent(MouthShape.I, 12.0, 22.0, 0.5),
        ],
        GenerationParams(),
    )
    # 長間隙(長さ8): d=0.4−0.025·8=0.2、mid=14、あ谷値=0.2·0.55=0.110000。
    long = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.LEGATO_GAP, 10.0, 18.0),
            MouthEvent(MouthShape.I, 18.0, 28.0, 0.5),
        ],
        GenerationParams(),
    )
    short_mid = {f: w for f, w in short["あ"]}[11]
    long_mid = {f: w for f, w in long["あ"]}[14]
    assert short_mid == pytest.approx(0.1925)
    assert long_mid == pytest.approx(0.11)
    assert long_mid < short_mid  # 間隙が長いほど谷は深い


def test_legato_gap_differs_from_silence_closure():
    # 同じ配置で SILENCE は間隙内で完全閉口、LEGATO_GAP は谷で繋ぐ。間隙中央12で両者が区別されること
    # (境界10は SILENCE 側も後行残しで保持値0.5のため、区別は間隙中央で見る)。
    base = [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), None, MouthEvent(MouthShape.I, 14.0, 24.0, 0.5)]
    silence = base[:]
    silence[1] = MouthEvent(MouthShape.SILENCE, 10.0, 14.0)
    legato = base[:]
    legato[1] = MouthEvent(MouthShape.LEGATO_GAP, 10.0, 14.0)
    s_a = {f: w for f, w in _envelope(silence, GenerationParams())["あ"]}
    l_a = {f: w for f, w in _envelope(legato, GenerationParams())["あ"]}
    # SILENCE: あ は後行残しで音符終了10まで保持し11で0へ閉じ、間隙中央12は閉口(キー無し)。
    assert s_a.get(11) == pytest.approx(0.0)
    assert 12 not in s_a
    # LEGATO_GAP: あ は谷で繋ぎ、間隙中央12に谷キー(オーバーラップ)0.165 を持つ。
    assert l_a.get(12) == pytest.approx(0.165)
