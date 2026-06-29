"""音符内の音素→口形イベント写像のテスト(vpr2vmd.md §3、音符内の時間配分)。

1つの採用音符(フレーム区間 [s, e))の音素列を、文脈なしで定まる口形イベントへ写像する。
時間配分: 語頭両唇閉鎖は音符先頭に取り分 d_b = min(公称長, (e-s)×取り分上限)、残り区間の母音 k 個を等分。
撥音「ん」(単独の鼻音)は ON で「ん」(N)、OFF(use_n_morph=False)で無音(閉口)。促音「っ」(Q)は無音(閉口)。
母音を持たず撥音/促音でもない音符(継続「-」・その他子音のみ等)は直前口形に依存するため None を返し、
組み立て段で直前口形を継続する(既定母音「あ」フォールバックは行わない)。
"""

from lipsync import ConsonantClass, MouthShape

from vpr2vmd import mapping


def _shapes_spans(eventlist):
    return [(e.shape, e.start, e.end) for e in eventlist]


def _classes(eventlist):
    return [(e.shape, e.consonant_class) for e in eventlist]


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


def test_no_vowel_returns_none_for_hold():
    # 母音を持たず撥音/促音でもない音符(その他子音のみ・継続・語頭両唇のみ・空)は None を返し、
    # 直前口形の継続を組み立て段へ委ねる(既定母音「あ」へ倒さない)。
    assert mapping.note_mouth_events(["t"], 0.0, 30.0) is None
    assert mapping.note_mouth_events(["-"], 0.0, 30.0) is None
    assert mapping.note_mouth_events(["m"], 0.0, 100.0) is None  # 語頭両唇のみ(母音なし)
    assert mapping.note_mouth_events([], 0.0, 30.0) is None


def test_moraic_nasal_fills_note_with_n_when_on():
    # 撥音(単独の鼻音 N\)は既定(ん ON)で音符全体を「ん」(N)にする。
    assert _shapes_spans(mapping.note_mouth_events(["N\\"], 0.0, 30.0)) == [
        (MouthShape.N, 0.0, 30.0),
    ]


def test_moraic_nasal_is_silence_when_off():
    # ん OFF(use_n_morph=False)では撥音を無音(閉口)にする。口を開けた母音「あ」へ倒さない。
    assert _shapes_spans(mapping.note_mouth_events(["N\\"], 0.0, 30.0, use_n_morph=False)) == [
        (MouthShape.SILENCE, 0.0, 30.0),
    ]


def test_moraic_nasal_mixed_with_consonant_is_not_standalone():
    # 撥音は「単独の鼻音」のときのみ。子音を伴う(非単独)は撥音扱いせず None(直前口形継続)。
    assert mapping.note_mouth_events(["t", "N\\"], 0.0, 30.0) is None


def test_geminate_stop_fills_note_with_silence():
    # 促音「っ」(Q)は無音(閉口)。子音と共起([w,Q] 等)しても口を開けず閉口にする。
    assert _shapes_spans(mapping.note_mouth_events(["Q"], 0.0, 30.0)) == [
        (MouthShape.SILENCE, 0.0, 30.0),
    ]
    assert _shapes_spans(mapping.note_mouth_events(["w", "Q"], 0.0, 30.0)) == [
        (MouthShape.SILENCE, 0.0, 30.0),
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


def test_onset_consonant_class_attached_to_leading_vowel():
    # 先頭母音に先頭子音の ConsonantClass を付ける。し(S=SPREAD)・ふぁ(p\=ROUNDED)・か(k=NEUTRAL)・
    # 母音単独(子音なし=NONE)。両唇音は別イベントで表すので語頭子音から除く。
    assert _classes(mapping.note_mouth_events(["S", "a"], 0.0, 30.0)) == [
        (MouthShape.A, ConsonantClass.SPREAD),
    ]
    assert _classes(mapping.note_mouth_events(["p\\", "a"], 0.0, 30.0)) == [
        (MouthShape.A, ConsonantClass.ROUNDED),
    ]
    assert _classes(mapping.note_mouth_events(["k", "o"], 0.0, 30.0)) == [
        (MouthShape.O, ConsonantClass.NEUTRAL),
    ]
    assert _classes(mapping.note_mouth_events(["a"], 0.0, 30.0)) == [
        (MouthShape.A, ConsonantClass.NONE),
    ]


def test_bilabial_onset_excluded_palatalized_keeps_spread():
    # 両唇音は ConsonantClass の対象外(別イベント)。ば は あ が NONE、両唇閉鎖イベントも NONE。
    assert _classes(mapping.note_mouth_events(["b", "a"], 0.0, 30.0)) == [
        (MouthShape.BILABIAL, ConsonantClass.NONE),
        (MouthShape.A, ConsonantClass.NONE),
    ]
    # みゃ(m=両唇は除外、j=拗音わたり=SPREAD): 両唇閉鎖イベント + あ が SPREAD。
    assert _classes(mapping.note_mouth_events(["m", "j", "a"], 0.0, 30.0)) == [
        (MouthShape.BILABIAL, ConsonantClass.NONE),
        (MouthShape.A, ConsonantClass.SPREAD),
    ]


def test_onset_class_only_on_first_mora_vowel():
    # 1音符に母音が複数あるとき、先頭母音にだけ子音種別を付け後続母音は NONE。
    assert _classes(mapping.note_mouth_events(["k", "a", "i"], 0.0, 30.0)) == [
        (MouthShape.A, ConsonantClass.NEUTRAL),
        (MouthShape.I, ConsonantClass.NONE),
    ]


def test_onset_class_priority_when_multiple_consonants():
    # 複数の語頭子音は優先順 ROUNDED > SPREAD > NEUTRAL で1つに決める。
    # きゃ(k=NEUTRAL + j=SPREAD)→ SPREAD、くゎ(k=NEUTRAL + w=ROUNDED)→ ROUNDED。
    assert mapping.note_mouth_events(["k", "j", "a"], 0.0, 30.0)[0].consonant_class is (
        ConsonantClass.SPREAD
    )
    assert mapping.note_mouth_events(["k", "w", "a"], 0.0, 30.0)[0].consonant_class is (
        ConsonantClass.ROUNDED
    )
    # ROUNDED は SPREAD より優先(両方が語頭にある場合の決定論的なタイブレーク)。
    assert mapping.note_mouth_events(["w", "j", "a"], 0.0, 30.0)[0].consonant_class is (
        ConsonantClass.ROUNDED
    )


def test_unknown_onset_consonant_is_neutral():
    # 未知の語頭子音は NEUTRAL(純母音扱い)。
    assert _classes(mapping.note_mouth_events(["zzz", "a"], 0.0, 30.0)) == [
        (MouthShape.A, ConsonantClass.NEUTRAL),
    ]


def test_open_amount_placeholder_zero_for_all_event_kinds():
    # 開き量は本段(mapping)では付与せず、組み立て段の build_mouth_events が刻印する。本段では
    # 母音・両唇閉鎖・撥音・促音(無音)の全イベントで既定 0.0。
    for phonemes in (["a"], ["m", "a"], ["N\\"], ["Q"]):
        events = mapping.note_mouth_events(phonemes, 0.0, 30.0)
        assert events
        assert all(e.open_amount == 0.0 for e in events)
