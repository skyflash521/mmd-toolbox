"""音符内の音素→口形イベント写像のテスト(vpr2vmd.md §3、音符内の時間配分)。

1つの採用音符(フレーム区間 [s, e))の音素列を、口形イベント(両唇閉鎖・母音・撥音「ん」)へ
写像する。継続「-」は直前母音に依存するため本段では扱わず、組み立て段で処理する。時間配分:
語頭両唇閉鎖は音符先頭に取り分 d_b = min(公称長, (e-s)×取り分上限)、残り区間の母音 k 個を等分。
母音が得られない音符は既定母音「あ」を1つ置く(語頭両唇音があれば閉鎖は保持)。撥音は音符全体を
「ん」にする。
"""

from lipsync import MouthShape

from vpr2vmd import mapping


def _shapes_spans(eventlist):
    return [(e.shape, e.start, e.end) for e in eventlist]


def test_single_vowel_fills_note():
    assert _shapes_spans(mapping.note_mouth_events(["a"], 0.0, 30.0)) == [
        (MouthShape.A, 0.0, 30.0),
    ]


def test_leading_bilabial_takes_nominal_share():
    # [m, a] over [0,100): d_b = min(公称長3, 100×0.5=50) = 3。残り [3,100) を母音 a へ。
    assert _shapes_spans(mapping.note_mouth_events(["m", "a"], 0.0, 100.0)) == [
        (MouthShape.BILABIAL, 0.0, 3.0),
        (MouthShape.A, 3.0, 100.0),
    ]


def test_leading_bilabial_share_capped_on_short_note():
    # [b, e] over [0,5): d_b = min(3, 5×0.5=2.5) = 2.5(取り分上限でクランプ)。
    assert _shapes_spans(mapping.note_mouth_events(["b", "e"], 0.0, 5.0)) == [
        (MouthShape.BILABIAL, 0.0, 2.5),
        (MouthShape.E, 2.5, 5.0),
    ]


def test_multiple_vowels_equally_divided():
    # 1音符に母音2つ([a, i])は残り区間を等分。語頭両唇なしなら全区間を2分割。
    assert _shapes_spans(mapping.note_mouth_events(["a", "i"], 0.0, 100.0)) == [
        (MouthShape.A, 0.0, 50.0),
        (MouthShape.I, 50.0, 100.0),
    ]


def test_leading_bilabial_then_multiple_vowels_divided_in_remainder():
    # [m, a, i] over [0,100): d_b=3。残り [3,100)(長さ97)を母音2個で等分(各 48.5)。
    # 全区間を等分したり境界を 50 にする実装を落とす。
    assert _shapes_spans(mapping.note_mouth_events(["m", "a", "i"], 0.0, 100.0)) == [
        (MouthShape.BILABIAL, 0.0, 3.0),
        (MouthShape.A, 3.0, 51.5),
        (MouthShape.I, 51.5, 100.0),
    ]


def test_respects_nonzero_start_frame():
    # 区間始端が 0 でなくても [s, s+d_b)・[s+d_b, e) に配分する(s を無視する実装を落とす)。
    # [m, a, i] over [10,110): d_b=3 → BILABIAL[10,13)、残り [13,110) を2分割(各 48.5)。
    assert _shapes_spans(mapping.note_mouth_events(["m", "a", "i"], 10.0, 110.0)) == [
        (MouthShape.BILABIAL, 10.0, 13.0),
        (MouthShape.A, 13.0, 61.5),
        (MouthShape.I, 61.5, 110.0),
    ]


def test_non_leading_bilabial_makes_no_event():
    # 両唇閉鎖イベントは音符語頭の両唇音のみ。語中・語末の両唇音は自前イベントを作らない。
    # [k, m, a]: 語頭は k(その他子音)で d_b=0、語中の m は閉鎖イベントを作らず母音 a が全区間。
    assert _shapes_spans(mapping.note_mouth_events(["k", "m", "a"], 0.0, 30.0)) == [
        (MouthShape.A, 0.0, 30.0),
    ]
    # [a, m]: 語頭は母音、語末の m は閉鎖イベントを作らない。
    assert _shapes_spans(mapping.note_mouth_events(["a", "m"], 0.0, 30.0)) == [
        (MouthShape.A, 0.0, 30.0),
    ]


def test_no_vowel_falls_back_to_default_a():
    # 母音が得られない音符(子音のみ)は既定母音「あ」を1つ置く。
    assert _shapes_spans(mapping.note_mouth_events(["t"], 0.0, 30.0)) == [
        (MouthShape.A, 0.0, 30.0),
    ]


def test_no_vowel_keeps_leading_bilabial_then_default_a():
    # 語頭両唇音のみ([m])は閉鎖を保持し、残りへ既定母音「あ」。
    assert _shapes_spans(mapping.note_mouth_events(["m"], 0.0, 100.0)) == [
        (MouthShape.BILABIAL, 0.0, 3.0),
        (MouthShape.A, 3.0, 100.0),
    ]


def test_moraic_nasal_fills_note_with_n():
    # 撥音(後続母音を持たない単独の鼻音 N\)は音符全体を「ん」にする。
    assert _shapes_spans(mapping.note_mouth_events(["N\\"], 0.0, 30.0)) == [
        (MouthShape.N, 0.0, 30.0),
    ]


def test_moraic_nasal_mixed_with_consonant_is_not_standalone():
    # 撥音は「単独の鼻音」のときのみ N。子音を伴う(非単独)場合は撥音扱いせず、母音なし
    # フォールバック(既定母音あ)へ倒す。[t, N\] は t がその他子音で語頭両唇も無いため A 全区間。
    assert _shapes_spans(mapping.note_mouth_events(["t", "N\\"], 0.0, 30.0)) == [
        (MouthShape.A, 0.0, 30.0),
    ]


def test_other_consonant_before_vowel_has_no_own_event():
    # 両唇閉鎖以外の子音は自前イベントを作らず、母音が全区間を占める(協調調音へ委ねる)。
    assert _shapes_spans(mapping.note_mouth_events(["k", "o"], 0.0, 30.0)) == [
        (MouthShape.O, 0.0, 30.0),
    ]


def test_bilabial_fricative_is_not_bilabial_event():
    # p\(ふ=φ の両唇摩擦音)はその他子音。語頭でも閉鎖イベントを作らず、後続母音 う が
    # 全区間を占める。p 接頭辞などで p\ を誤って両唇閉鎖にする実装を落とす。
    assert _shapes_spans(mapping.note_mouth_events(["p\\", "M"], 0.0, 30.0)) == [
        (MouthShape.U, 0.0, 30.0),
    ]


def test_open_amount_placeholder_zero_for_all_event_kinds():
    # 開き量(ベロシティ写像)は後続ステップ。本段では母音・両唇閉鎖・撥音・フォールバックの
    # 全イベントで既定 0.0。
    for phonemes in (["a"], ["m", "a"], ["N\\"], ["t"]):
        events = mapping.note_mouth_events(phonemes, 0.0, 30.0)
        assert events
        assert all(e.open_amount == 0.0 for e in events)
