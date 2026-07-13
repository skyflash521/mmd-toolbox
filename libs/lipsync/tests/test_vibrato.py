"""任意の伸び表現のテスト。

長く伸ばす母音の実効保持プラトーに、保持値を中心とした正弦波の揺らぎを極値のみ疎に乗せる。位相は
プラトー始端基準、節点は t_k = plateau_start + period·(1/4 + k/2) の厳密内側、open を正弦波で変調して
同じ合成で重みを出し、最後に量子化へ渡す。プラトー長が閾値以下のとき・揺らぎ無効のときは
揺らぎを置かない。
"""

import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape


def _envelope(events, params=None):
    params = params or GenerationParams()
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


def test_long_vowel_vibrato_known_values():
    # あ[0,40]op0.5、既定(threshold18・amp0.05・period15・attack2・release2)。
    # プラトー[2,38]、長さ36>18。hold=0.5、amp_eff=min(0.05,0.5)=0.05。
    # 極値 t_k=2+15·(1/4+k/2)=5.75+7.5k、厳密内側(<38)の k=0..4 で sin=+1,-1,+1,-1,+1。
    # open_v=0.55/0.45/0.55/0.45/0.55、半上げ量子化で 6/13/21/28/36。公称(0,2,38,40)と衝突なし。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 40.0, 0.5)], GenerationParams())
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),
            (6, 0.55),
            (13, 0.45),
            (21, 0.55),
            (28, 0.45),
            (36, 0.55),
            (38, 0.5),
            (40, 0.0),
        ],
    )


def test_short_plateau_no_vibrato():
    # あ[0,20]op0.5、プラトー[2,18]長さ16<=threshold18。揺らぎを置かず公称のみ(非適用時と不変)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 20.0, 0.5)], GenerationParams())
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (18, 0.5), (20, 0.0)])


def test_vibrato_disabled_when_amp_zero():
    # 長い母音でも vibrato_amp=0 なら揺らぎ無効。公称のみ(非適用時と不変)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 40.0, 0.5)], GenerationParams(vibrato_amp=0.0))
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (38, 0.5), (40, 0.0)])
