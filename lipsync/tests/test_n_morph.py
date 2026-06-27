"""撥音「ん」(母音的口形)の生成テスト(lipsync.md §2.1/§3/§4)。

「ん」は閉口でなく母音と同じ機構を通る母音的口形で、合成プロファイル {"ん": 1.0}(補助なし)を
持つ。単独「ん」がアタック/保持/リリースのエンベロープを「ん」モーフへ決定論的に出すこと、
vowel_scale の「ん」倍率(添字5)が hold に効くこと、そして「ん」を含まない母音入力の出力が
「ん」モーフを一切出さない(既存母音回帰の不変性)ことを検証する。
"""

import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape


def _envelope(events, params=None):
    """イベント列を変換し、モーフ名→(frame, weight) の時間順リストの辞書を返す。"""
    params = params or GenerationParams()
    keys = lipsync.generate_morph_keys(events, params)
    by_morph: dict[str, list[tuple[int, float]]] = {}
    for k in keys:
        by_morph.setdefault(k.name, []).append((k.frame, k.weight))
    for name in by_morph:
        by_morph[name].sort()
    return by_morph


def _peak_weights(events, params=None):
    """モーフ名→ピーク(保持)ウェイトの辞書。"""
    params = params or GenerationParams()
    keys = lipsync.generate_morph_keys(events, params)
    weights: dict[str, float] = {}
    for k in keys:
        weights[k.name] = max(weights.get(k.name, 0.0), k.weight)
    return weights


@pytest.mark.xfail(reason="impl pending: ん合成プロファイル", strict=True)
def test_single_n_four_key_envelope():
    # ん [0,10] open=0.5、プロファイル {ん:1.0} で hold=0.5、attack=2/release=2。
    # 「ん」モーフへ4点: (0,0)・(2,0.5)・(8,0.5)・(10,0)。母音と同じ単一区間形状。
    env = _envelope([MouthEvent(MouthShape.N, 0.0, 10.0, 0.5)])
    assert set(env) == {"ん"}
    keys = env["ん"]
    assert [f for f, _ in keys] == [0, 2, 8, 10]
    for (_, w), expected in zip(keys, [0.0, 0.5, 0.5, 0.0]):
        assert w == pytest.approx(expected)


@pytest.mark.xfail(reason="impl pending: ん合成プロファイル", strict=True)
def test_single_n_profile_is_n_only():
    # ん のプロファイルは主モーフ「ん」のみ(補助なし)。hold=0.4 → {ん:0.4}、他モーフは出ない。
    assert _peak_weights([MouthEvent(MouthShape.N, 0.0, 10.0, 0.4)]) == pytest.approx({"ん": 0.4})


@pytest.mark.xfail(reason="impl pending: ん合成プロファイル", strict=True)
def test_n_vowel_scale_indexed_at_five():
    # vowel_scale の「ん」倍率は添字5。倍率2.0・open=0.3 → hold=clamp(0.6,0,0.8)=0.6 → {ん:0.6}。
    p = GenerationParams(vowel_scale=(1.0, 1.0, 1.0, 1.0, 1.0, 2.0))
    assert _peak_weights([MouthEvent(MouthShape.N, 0.0, 10.0, 0.3)], p) == pytest.approx({"ん": 0.6})


@pytest.mark.parametrize(
    "events",
    [
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)],
        [MouthEvent(MouthShape.I, 0.0, 10.0, 0.5)],
        [MouthEvent(MouthShape.E, 0.0, 10.0, 1.0)],
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.I, 10.0, 20.0, 0.5),
        ],
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.SILENCE, 10.0, 14.0, 0.0),
            MouthEvent(MouthShape.O, 14.0, 24.0, 0.6),
        ],
        [
            MouthEvent(MouthShape.U, 0.0, 8.0, 0.4),
            MouthEvent(MouthShape.BILABIAL, 8.0, 11.0, 0.0),
            MouthEvent(MouthShape.A, 11.0, 21.0, 0.7),
        ],
    ],
)
def test_vowel_inputs_emit_no_n_morph(events):
    # 既存母音回帰の不変性: MouthShape.N を含まない入力からは「ん」モーフキーが一切出ない
    # (「ん」を母音的口形へ加えても母音5種の合成・協調調音・量子化の出力が変わらない)。
    keys = lipsync.generate_morph_keys(events, GenerationParams())
    assert "ん" not in {k.name for k in keys}
