"""撥音「ん」(母音的口形)の生成テスト。

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


def test_single_n_four_key_envelope():
    # ん [0,10] open=0.5、プロファイル {ん:1.0} で hold=0.5、attack=2/release=2。
    # 「ん」モーフへ4点: (0,0)・(2,0.5)・(8,0.5)・(10,0)。母音と同じ単一区間形状。
    env = _envelope([MouthEvent(MouthShape.N, 0.0, 10.0, 0.5)])
    assert set(env) == {"ん"}
    keys = env["ん"]
    assert [f for f, _ in keys] == [0, 2, 8, 10]
    for (_, w), expected in zip(keys, [0.0, 0.5, 0.5, 0.0], strict=True):
        assert w == pytest.approx(expected)


def test_single_n_profile_is_n_only():
    # ん のプロファイルは主モーフ「ん」のみ(補助なし)。hold=0.4 → {ん:0.4}、他モーフは出ない。
    assert _peak_weights([MouthEvent(MouthShape.N, 0.0, 10.0, 0.4)]) == pytest.approx({"ん": 0.4})


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


# --- 「ん」が母音と同じ機構(連結・吸収・先行準備・協調調音・伸び表現)を通ることの確認 ---
#
# 「ん」のプロファイルは主モーフ「ん」のみ {ん:1.0} で、母音「あ」の {あ:1.0} と同型。
# よって他の母音との協調調音を含まない場面(単独・連続「ん」・無音隣接)では、「ん」の出力は
# 「あ」の出力とモーフ名を「ん」↔「あ」で読み替えた結果に一致する。これを使い、各母音機構が
# 「ん」にも同経路で働くことを確認する(母音↔「ん」の協調調音だけは口形差が異なるため既知値で別途確認)。


def _assert_n_mirrors_a(n_events, a_events, params=None):
    """「ん」シナリオの出力が、対応する「あ」シナリオの出力と「ん」↔「あ」読み替えで一致する。"""
    params = params or GenerationParams()
    nk = lipsync.generate_morph_keys(n_events, params)
    ak = lipsync.generate_morph_keys(a_events, params)
    n_mapped = sorted(
        (("あ" if k.name == "ん" else k.name), k.frame, k.weight) for k in nk
    )
    a_sorted = sorted((k.name, k.frame, k.weight) for k in ak)
    assert [(n, f) for n, f, _ in n_mapped] == [(n, f) for n, f, _ in a_sorted]
    for (_, _, nw), (_, _, aw) in zip(n_mapped, a_sorted, strict=True):
        assert nw == pytest.approx(aw)


def test_consecutive_n_connected_like_vowel():
    # 連続する同一「ん」は1保持区間へ連結する(内部境界に再アタック・閉口を挟まない)。
    # 母音「あ」の連結と同経路: 出力は「あ+あ」連結とモーフ名読み替えで一致する。
    n_events = [
        MouthEvent(MouthShape.N, 0.0, 10.0, 0.5),
        MouthEvent(MouthShape.N, 10.0, 20.0, 0.5),
    ]
    a_events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
        MouthEvent(MouthShape.A, 10.0, 20.0, 0.5),
    ]
    _assert_n_mirrors_a(n_events, a_events)
    # 連結の確認: 「ん」モーフのみで、weight 0.0 は外端(0, 20)だけ。内部境界(10)に閉口を挟まない。
    keys = lipsync.generate_morph_keys(n_events, GenerationParams())
    assert {k.name for k in keys} == {"ん"}
    zero_frames = {k.frame for k in keys if k.weight == pytest.approx(0.0)}
    assert zero_frames == {0, 20}


def test_long_n_gets_vibrato_like_vowel():
    # 長く伸ばす「ん」の保持プラトーにも、母音と同じ伸び表現の揺らぎ節点が付く。
    # プラトー長 36 > vibrato_threshold(18)。出力は長い「あ」とモーフ名読み替えで一致する。
    n_events = [MouthEvent(MouthShape.N, 0.0, 40.0, 0.5)]
    a_events = [MouthEvent(MouthShape.A, 0.0, 40.0, 0.5)]
    _assert_n_mirrors_a(n_events, a_events)
    # 揺らぎが付く確認: 単一区間4点(0/attack/hold_end/end)より多くの「ん」キーがある。
    keys = lipsync.generate_morph_keys(n_events, GenerationParams())
    assert len([k for k in keys if k.name == "ん"]) > 4


def test_n_anticipation_after_silence_like_vowel():
    # 直前が無音の「ん」は、母音と同じ先行準備でアタックを前倒す。
    # 出力は「無音→あ」とモーフ名読み替えで一致する(先行量・形状が同一)。
    n_events = [
        MouthEvent(MouthShape.SILENCE, 0.0, 6.0, 0.0),
        MouthEvent(MouthShape.N, 6.0, 16.0, 0.5),
    ]
    a_events = [
        MouthEvent(MouthShape.SILENCE, 0.0, 6.0, 0.0),
        MouthEvent(MouthShape.A, 6.0, 16.0, 0.5),
    ]
    _assert_n_mirrors_a(n_events, a_events)


def test_short_n_absorbed_into_neighbor():
    # 吸収対象(L<triangle_min)の短い「ん」は、母音と同じく隣接母音的口形へ吸収され独立キーを残さない。
    # 両隣の「あ」(開き量同値・長さ同値→前側へ吸収)が span を貫き、吸収後に同母音連結して一区間化。
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
        MouthEvent(MouthShape.N, 10.0, 11.0, 0.5),  # 長さ1 < triangle_min=2、吸収対象
        MouthEvent(MouthShape.A, 11.0, 21.0, 0.5),
    ]
    keys = lipsync.generate_morph_keys(events, GenerationParams())
    # 「ん」は吸収され出力に現れない。連結後は単一「あ」区間で、weight 0.0 は外端(0, 21)だけ。
    assert {k.name for k in keys} == {"あ"}
    zero_frames = {k.frame for k in keys if k.weight == pytest.approx(0.0)}
    assert zero_frames == {0, 21}


def test_n_vowel_boundary_short_coarticulation():
    # 「ん」↔母音の境界も母音同士と同じく協調調音(口形差で短縮しないので基準長2で T=min(2, 5)=2)。
    # 閉口を挟まず中間口形へクロスフェードする。N[0,10]+A[10,20] open=0.5、窓[9,11] の既知値:
    #   ん: (0,0)・(2,0.5)・(9,0.5)・(10,0.25)・(11,0)、 あ: (9,0)・(10,0.25)・(11,0.5)・(18,0.5)・(20,0)。
    # 境界フレーム10で両モーフが 0.25(中間口形)=閉口(全モーフ0)を挟まない遷移。
    keys = lipsync.generate_morph_keys(
        [
            MouthEvent(MouthShape.N, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.A, 10.0, 20.0, 0.5),
        ],
        GenerationParams(),
    )
    by_morph: dict[str, list[tuple[int, float]]] = {}
    for k in keys:
        by_morph.setdefault(k.name, []).append((k.frame, k.weight))
    for name in by_morph:
        by_morph[name].sort()
    assert set(by_morph) == {"ん", "あ"}
    assert [f for f, _ in by_morph["ん"]] == [0, 2, 9, 10, 11]
    for (_, w), e in zip(by_morph["ん"], [0.0, 0.5, 0.5, 0.25, 0.0], strict=True):
        assert w == pytest.approx(e)
    assert [f for f, _ in by_morph["あ"]] == [9, 10, 11, 18, 20]
    for (_, w), e in zip(by_morph["あ"], [0.0, 0.25, 0.5, 0.5, 0.0], strict=True):
        assert w == pytest.approx(e)
