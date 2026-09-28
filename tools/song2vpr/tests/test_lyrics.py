import numpy as np
import pytest

from song2vpr import lyrics, notes
from vocal_analysis import RmsEnvelope, Segment


def _note(start, end, midi=69, syllable=0):
    return notes.Note(start_sec=start, end_sec=end, midi=midi, syllable=syllable)


def _vowel(start, end, phoneme="a"):
    return Segment(type="vowel", start_sec=start, end_sec=end, phoneme=phoneme, confidence=0.9)


def _consonant(start, end, phoneme="k"):
    return Segment(type="consonant", start_sec=start, end_sec=end, phoneme=phoneme, confidence=0.9)


def _gap(start, end):
    return Segment(type="gap", start_sec=start, end_sec=end, phoneme=None, confidence=None)


def _flat_rms(value=0.5, seconds=2.0):
    times = np.arange(0.0, seconds, 0.01)
    return RmsEnvelope(times_sec=times, values=np.full(len(times), value), dynamic_range_db=20.0)


def _rms_silent_between(start, end, seconds=2.0):
    times = np.arange(0.0, seconds, 0.01)
    values = np.where((times >= start) & (times < end), 0.0, 0.5)
    return RmsEnvelope(times_sec=times, values=values, dynamic_range_db=20.0)


def _one_syllable(*segments):
    return [list(segments)]


def _one_syllable_per_note(count, seconds=0.2, phoneme="a"):
    source = [_note(i * seconds, (i + 1) * seconds, syllable=i) for i in range(count)]
    segments = [[_vowel(i * seconds, (i + 1) * seconds, phoneme)] for i in range(count)]
    return source, segments


@pytest.mark.parametrize("ipa,xsampa", [
    pytest.param("a", "a", id="vowel_a"),
    pytest.param("i", "i", id="vowel_i"),
    pytest.param("ɯ", "M", id="vowel_u"),
    pytest.param("e̞", "e", id="vowel_e"),
    pytest.param("o̞", "o", id="vowel_o"),
    pytest.param("mʲ", "m'", id="palatalized_m"),
    pytest.param("ɸ", "p\\", id="bilabial_fricative"),
    pytest.param("ɕ", "S", id="alveolo_palatal_fricative"),
    pytest.param("dʑ", "dZ", id="voiced_alveolo_palatal_affricate"),
    pytest.param("tɕ", "tS", id="alveolo_palatal_affricate"),
    pytest.param("ɴ", "N\\", id="uvular_nasal"),
    pytest.param("ɡ", "g", id="voiced_velar_plosive"),
    pytest.param("ts", "ts", id="alveolar_affricate"),
    pytest.param("ɲ", "J", id="palatal_nasal"),
    pytest.param("ɾ", "4", id="alveolar_tap"),
    pytest.param("ç", "C", id="palatal_fricative_absent_from_recognizer_inventory"),
    pytest.param("v", "v", id="labiodental_fricative_absent_from_recognizer_inventory"),
])
def test_ipa_is_mapped_to_the_vpr_phoneme_notation(ipa, xsampa):
    kind = "vowel" if ipa in {"a", "i", "ɯ", "e̞", "o̞"} else "consonant"
    segment = Segment(type=kind, start_sec=0.0, end_sec=0.4, phoneme=ipa, confidence=0.9)
    result = lyrics.annotate([_note(0.0, 0.4)], _one_syllable(segment), _flat_rms())
    assert result.notes[0].phonemes == [xsampa]


def test_gap_and_unlabelled_segments_are_excluded_from_the_phonemes():
    result = lyrics.annotate([_note(0.0, 0.4)],
                             _one_syllable(_vowel(0.0, 0.2, "a"), _gap(0.2, 0.4)), _flat_rms())
    assert result.notes[0].phonemes == ["a"]


def test_symbols_without_a_vpr_notation_are_excluded_from_the_phonemes():
    result = lyrics.annotate([_note(0.0, 0.4)],
                             _one_syllable(_consonant(0.0, 0.1, "θ"), _vowel(0.1, 0.4, "a")),
                             _flat_rms())
    assert result.notes[0].phonemes == ["a"]


def test_phonemes_are_empty_when_every_segment_is_excluded():
    result = lyrics.annotate([_note(0.0, 0.4)], _one_syllable(_gap(0.0, 0.4)), _flat_rms())
    assert result.notes[0].phonemes == []


def test_the_mapping_covers_every_symbol_the_recognizer_can_emit():
    from vocal_analysis.phonemes import _G2P_TO_VOCAB_SYMBOL

    emitted = {symbol for symbol in _G2P_TO_VOCAB_SYMBOL.values() if symbol}
    assert emitted <= set(lyrics.IPA_TO_VPR_PHONEME)


@pytest.mark.parametrize("ipa,kana", [
    ("a", "あ"), ("i", "い"), ("ɯ", "う"), ("e̞", "え"), ("o̞", "お"),
])
def test_lyric_is_the_vowel_kana_of_the_nucleus(ipa, kana):
    result = lyrics.annotate([_note(0.0, 0.4)],
                             _one_syllable(Segment(type="vowel", start_sec=0.0, end_sec=0.4,
                                                   phoneme=ipa, confidence=0.9)), _flat_rms())
    assert result.notes[0].lyric == kana


def test_note_without_a_nucleus_gets_the_default_kana_and_is_counted_as_undetermined_vowel():
    result = lyrics.annotate([_note(0.0, 0.3)], _one_syllable(_gap(0.0, 0.3)), _flat_rms())
    assert result.notes[0].lyric == "あ"
    assert result.diagnostics.undetermined_vowel_notes == 1


def test_phonemes_come_from_the_syllable_even_when_the_note_overlaps_the_next_syllable():
    first = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.4, "a")]
    second = [_consonant(0.4, 0.5, "d"), _vowel(0.5, 0.8, "a")]
    result = lyrics.annotate([_note(0.0, 0.45, syllable=0), _note(0.45, 0.8, syllable=1)],
                             [first, second], _flat_rms())
    assert [note.phonemes for note in result.notes] == [["k", "a"], ["d", "a"]]


def test_later_notes_of_a_split_syllable_carry_the_continuation_mark_in_lyric_and_phonemes():
    syllable = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.6, "a")]
    result = lyrics.annotate([_note(0.0, 0.3, syllable=0), _note(0.3, 0.6, midi=71, syllable=0)],
                             [syllable], _flat_rms())
    assert [note.lyric for note in result.notes] == ["か", "-"]
    assert [note.phonemes for note in result.notes] == [["k", "a"], ["-"]]


@pytest.mark.parametrize(("head", "nucleus", "kana"), [
    pytest.param([], "a", "あ", id="no_head_consonant"),
    pytest.param(["k"], "a", "か", id="k_a"),
    pytest.param(["s"], "i", "すぃ", id="s_i_loanword_sound"),
    pytest.param(["t"], "ɯ", "とぅ", id="t_u_loanword_sound"),
    pytest.param(["ɸ"], "o̞", "ふぉ", id="bilabial_fricative_o_loanword_sound"),
    pytest.param(["ɕ"], "a", "しゃ", id="alveolo_palatal_fricative_a"),
    pytest.param(["kʲ"], "a", "きゃ", id="palatalized_consonant_gives_palatal_row"),
    pytest.param(["v"], "a", "ば", id="v_uses_the_b_row"),
    pytest.param(["k", "j"], "a", "きゃ", id="j_after_k_gives_palatal_row_of_k"),
    pytest.param(["ɾ", "j"], "o̞", "りょ", id="j_after_r_gives_palatal_row_of_r"),
    pytest.param(["ɡ", "j"], "a", "ぎゃ", id="j_after_g_gives_palatal_row_of_g"),
    pytest.param(["b", "j"], "ɯ", "びゅ", id="j_after_b_gives_palatal_row_of_b"),
    pytest.param(["p", "j"], "o̞", "ぴょ", id="j_after_p_gives_palatal_row_of_p"),
    pytest.param(["m", "j"], "a", "みゃ", id="j_after_m_gives_palatal_row_of_m"),
    pytest.param(["h", "j"], "a", "ひゃ", id="j_after_h_gives_palatal_row_of_h"),
    pytest.param(["s", "j"], "a", "や", id="j_after_consonant_without_palatal_row_gives_j_row"),
    pytest.param(["t", "k"], "a", "か", id="multiple_heads_use_the_one_next_to_the_nucleus"),
])
def test_lyric_comes_from_the_head_consonant_and_the_nucleus(head, nucleus, kana):
    segments = [_consonant(0.1 * i, 0.1 * (i + 1), phoneme) for i, phoneme in enumerate(head)]
    start = 0.1 * len(head)
    segments.append(Segment(type="vowel", start_sec=start, end_sec=start + 0.3,
                            phoneme=nucleus, confidence=0.9))
    result = lyrics.annotate([_note(0.0, start + 0.3, syllable=0)], [segments], _flat_rms())
    assert result.notes[0].lyric == kana


def test_head_consonants_that_do_not_decide_the_kana_stay_in_the_phonemes():
    syllable = [_consonant(0.0, 0.1, "t"), _consonant(0.1, 0.2, "k"), _vowel(0.2, 0.5, "a")]
    result = lyrics.annotate([_note(0.0, 0.5, syllable=0)], [syllable], _flat_rms())
    assert result.notes[0].lyric == "か"
    assert result.notes[0].phonemes == ["t", "k", "a"]


def test_a_moraic_nasal_nucleus_gets_the_nasal_kana_regardless_of_the_head_consonant():
    syllable = [_consonant(0.0, 0.1, "k"), _consonant(0.1, 0.4, "ɴ")]
    result = lyrics.annotate([_note(0.0, 0.4, syllable=0)], [syllable], _flat_rms())
    assert result.notes[0].lyric == "ん"


def test_the_kana_table_covers_every_consonant_the_recognizer_can_emit():
    consonants = set(lyrics.IPA_TO_VPR_PHONEME.values()) - set("aiMeo") - {"N\\"}
    assert consonants <= set(lyrics.KANA_BY_CONSONANT)


def test_every_cell_of_the_kana_table():
    assert lyrics.KANA_BY_CONSONANT == {
        "": ("あ", "い", "う", "え", "お"),
        "k": ("か", "き", "く", "け", "こ"),
        "k'": ("きゃ", "き", "きゅ", "きぇ", "きょ"),
        "g": ("が", "ぎ", "ぐ", "げ", "ご"),
        "g'": ("ぎゃ", "ぎ", "ぎゅ", "ぎぇ", "ぎょ"),
        "s": ("さ", "すぃ", "す", "せ", "そ"),
        "S": ("しゃ", "し", "しゅ", "しぇ", "しょ"),
        "z": ("ざ", "ずぃ", "ず", "ぜ", "ぞ"),
        "dZ": ("じゃ", "じ", "じゅ", "じぇ", "じょ"),
        "t": ("た", "てぃ", "とぅ", "て", "と"),
        "ts": ("つぁ", "つぃ", "つ", "つぇ", "つぉ"),
        "tS": ("ちゃ", "ち", "ちゅ", "ちぇ", "ちょ"),
        "d": ("だ", "でぃ", "どぅ", "で", "ど"),
        "n": ("な", "に", "ぬ", "ね", "の"),
        "J": ("にゃ", "に", "にゅ", "にぇ", "にょ"),
        "h": ("は", "ひ", "ふ", "へ", "ほ"),
        "C": ("ひゃ", "ひ", "ひゅ", "ひぇ", "ひょ"),
        "p\\": ("ふぁ", "ふぃ", "ふ", "ふぇ", "ふぉ"),
        "b": ("ば", "び", "ぶ", "べ", "ぼ"),
        "b'": ("びゃ", "び", "びゅ", "びぇ", "びょ"),
        "p": ("ぱ", "ぴ", "ぷ", "ぺ", "ぽ"),
        "p'": ("ぴゃ", "ぴ", "ぴゅ", "ぴぇ", "ぴょ"),
        "m": ("ま", "み", "む", "め", "も"),
        "m'": ("みゃ", "み", "みゅ", "みぇ", "みょ"),
        "j": ("や", "い", "ゆ", "いぇ", "よ"),
        "4": ("ら", "り", "る", "れ", "ろ"),
        "w": ("わ", "うぃ", "う", "うぇ", "うぉ"),
        "v": ("ば", "び", "ぶ", "べ", "ぼ"),
    }


def test_only_the_syllable_head_carrying_phonemes_is_protected():
    first = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.6, "a")]
    second = [_gap(0.6, 0.9)]
    result = lyrics.annotate(
        [_note(0.0, 0.3, syllable=0), _note(0.3, 0.6, midi=71, syllable=0),
         _note(0.6, 0.9, syllable=1)],
        [first, second], _flat_rms())
    assert [note.is_protected for note in result.notes] == [True, False, False]


def test_a_note_whose_volume_representative_is_zero_is_not_output_and_is_counted():
    result = lyrics.annotate([_note(0.0, 0.2, syllable=0), _note(0.2, 0.4, syllable=1)],
                             [[_vowel(0.0, 0.2, "a")], [_vowel(0.2, 0.4, "i")]],
                             _rms_silent_between(0.2, 0.4))
    assert [note.lyric for note in result.notes] == ["あ"]
    assert result.diagnostics.suppressed_notes == 1


def test_the_first_surviving_note_of_a_syllable_becomes_its_head():
    syllable = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.6, "a")]
    result = lyrics.annotate([_note(0.0, 0.3, syllable=0), _note(0.3, 0.6, midi=71, syllable=0)],
                             [syllable], _rms_silent_between(0.0, 0.3))
    assert [note.lyric for note in result.notes] == ["か"]
    assert [note.phonemes for note in result.notes] == [["k", "a"]]
    assert result.notes[0].is_protected is True


def test_a_syllable_with_no_surviving_note_takes_no_mora():
    source = [_note(0.0, 0.2, syllable=0), _note(0.2, 0.4, syllable=1),
              _note(0.4, 0.6, syllable=2)]
    segments = [[_vowel(0.0, 0.2, "a")], [_vowel(0.2, 0.4, "a")], [_vowel(0.4, 0.6, "a")]]
    result = lyrics.annotate(source, segments, _rms_silent_between(0.2, 0.4),
                             lyrics_text="さくら")
    assert [note.lyric for note in result.notes] == ["さ", "く"]
    assert result.diagnostics.discarded_morae == 1


def test_given_lyrics_are_assigned_one_mora_per_syllable_head():
    source, segments = _one_syllable_per_note(3)
    result = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="さくら")
    assert [note.lyric for note in result.notes] == ["さ", "く", "ら"]


def test_a_continuation_note_keeps_the_continuation_mark_when_lyrics_are_given():
    result = lyrics.annotate([_note(0.0, 0.2, syllable=0), _note(0.2, 0.4, syllable=0),
                              _note(0.4, 0.6, syllable=1)],
                             [[_vowel(0.0, 0.4, "a")], [_vowel(0.4, 0.6, "a")]],
                             _flat_rms(), lyrics_text="さく")
    assert [note.lyric for note in result.notes] == ["さ", "-", "く"]


def test_phonemes_still_come_from_the_segments_when_lyrics_are_given():
    result = lyrics.annotate([_note(0.0, 0.4)], _one_syllable(_vowel(0.0, 0.4, "i")), _flat_rms(),
                             lyrics_text="さ")
    assert result.notes[0].lyric == "さ"
    assert result.notes[0].phonemes == ["i"]


@pytest.mark.parametrize("text,expected", [
    pytest.param("きゃく", ["きゃ", "く"], id="small_kana_joins_the_preceding_mora"),
    pytest.param("がっき", ["が", "っき"], id="sokuon_joins_the_following_mora"),
    pytest.param("かー", ["かー"], id="long_mark_joins_the_preceding_mora"),
    pytest.param("はん", ["は", "ん"], id="moraic_nasal_forms_its_own_mora"),
    pytest.param("んー", ["んー"], id="long_mark_joins_a_preceding_moraic_nasal"),
    pytest.param("っか", ["っか"], id="leading_sokuon_joins_the_following_mora"),
    pytest.param("ーか", ["ーか"], id="long_mark_without_preceding_joins_the_following_mora"),
    pytest.param("かっ", ["かっ"], id="sokuon_without_following_joins_the_preceding_mora"),
    pytest.param("ー", ["ー"], id="long_mark_without_neighbours_stands_alone"),
    pytest.param("あ い", ["あ", "い"], id="space_is_skipped_and_does_not_join_its_neighbours"),
    pytest.param("かーん", ["かー", "ん"], id="long_mark_then_moraic_nasal"),
])
def test_mora_split_rules(text, expected):
    source, segments = _one_syllable_per_note(len(expected))
    result = lyrics.annotate(source, segments, _flat_rms(), lyrics_text=text)
    assert [note.lyric for note in result.notes] == expected


@pytest.mark.parametrize("reading,expected", [
    pytest.param("ゃ", ["ゃ"], id="small_kana_without_previous_stands_alone"),
    pytest.param("かっん", ["か", "っん"], id="sokuon_joins_following_moraic_nasal"),
    pytest.param("んゃ", ["んゃ"], id="small_kana_joins_preceding_moraic_nasal"),
    pytest.param("か ー", ["か", "ー"], id="long_mark_does_not_join_across_skipped_char"),
    pytest.param("かっ い", ["かっ", "い"], id="sokuon_does_not_join_across_skipped_char"),
    pytest.param("っ", ["っ"], id="sokuon_without_neighbours_stands_alone"),
])
def test_mora_split_rules_on_the_reading_as_given(monkeypatch, reading, expected):
    monkeypatch.setattr("vocal_analysis.reading.to_kana_reading", lambda text: reading)
    source, segments = _one_syllable_per_note(len(expected))
    result = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="歌詞")
    assert [note.lyric for note in result.notes] == expected


def test_more_notes_than_morae_fall_back_to_the_vowel_kana():
    source, segments = _one_syllable_per_note(2, phoneme="i")
    result = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="さ")
    assert [note.lyric for note in result.notes] == ["さ", "い"]
    assert result.diagnostics.notes_beyond_morae == 1


def test_more_morae_than_notes_are_discarded_and_counted():
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="さくら")
    assert [note.lyric for note in result.notes] == ["さ"]
    assert result.diagnostics.discarded_morae == 2


def test_layout_characters_are_skipped_and_not_counted_toward_kana_conversion():
    source, segments = _one_syllable_per_note(2)
    result = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="あ!い?")
    assert [note.lyric for note in result.notes] == ["あ", "い"]
    assert result.diagnostics.unconverted_chars == 0
    assert result.diagnostics.counted_chars == 2


def _stub_kana_reading(monkeypatch, reading):
    monkeypatch.setattr("vocal_analysis.reading.to_kana_reading", lambda text: reading)


def test_kanji_and_alphanumerics_left_in_the_reading_are_counted_as_unconverted(monkeypatch):
    _stub_kana_reading(monkeypatch, "あ漢A")
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="どんな表記でもよい")
    assert result.diagnostics.unconverted_chars == 2
    assert result.diagnostics.counted_chars == 3
    assert result.diagnostics.kana_reading_ineffective


def test_reading_without_any_countable_character_is_not_judged_ineffective(monkeypatch):
    _stub_kana_reading(monkeypatch, "!?")
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="どんな表記でもよい")
    assert result.diagnostics.counted_chars == 0
    assert not result.diagnostics.kana_reading_ineffective


def test_mostly_kana_reading_is_not_flagged(monkeypatch):
    _stub_kana_reading(monkeypatch, "あいうえおかき漢")
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="どんな表記でもよい")
    assert result.diagnostics.unconverted_chars == 1
    assert not result.diagnostics.kana_reading_ineffective


def test_katakana_is_folded_to_hiragana():
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="サ")
    assert result.notes[0].lyric == "さ"


@pytest.mark.parametrize("value", [0.1, 1.0])
def test_velocity_is_the_neutral_value_on_head_and_continuation_regardless_of_the_volume(value):
    result = lyrics.annotate(
        [_note(0.0, 0.3, syllable=0), _note(0.3, 0.6, midi=71, syllable=0)],
        _one_syllable(_vowel(0.0, 0.6, "a")), _flat_rms(value=value))
    assert [note.lyric for note in result.notes] == ["あ", "-"]
    assert [note.velocity for note in result.notes] == [64, 64]


@pytest.mark.parametrize(("value", "output"), [
    pytest.param(0.003, False, id="volume_mapped_to_zero_is_suppressed"),
    pytest.param(0.004, True, id="volume_mapped_above_zero_is_kept"),
])
def test_note_is_dropped_only_when_its_volume_rounds_to_zero_on_the_0_to_127_scale(value, output):
    result = lyrics.annotate([_note(0.0, 0.4)], _one_syllable(_vowel(0.0, 0.4, "a")),
                             _flat_rms(value=value))
    assert bool(result.notes) is output


@pytest.mark.parametrize(("loud_in_middle", "output"), [
    pytest.param(True, True, id="loud_only_in_the_middle_is_kept"),
    pytest.param(False, False, id="silent_only_in_the_middle_is_suppressed"),
])
def test_the_volume_comes_from_the_middle_of_the_note(loud_in_middle, output):
    times = np.arange(0.0, 1.0, 0.01)
    middle = (times >= 0.2) & (times < 0.8)
    values = np.where(middle if loud_in_middle else ~middle, 1.0, 0.0)
    envelope = RmsEnvelope(times_sec=times, values=values, dynamic_range_db=20.0)
    result = lyrics.annotate([_note(0.0, 1.0)], _one_syllable(_vowel(0.0, 1.0, "a")), envelope)
    assert bool(result.notes) is output


def test_the_span_and_the_pitch_are_carried_through():
    source = _note(0.1, 0.5, midi=72)
    result = lyrics.annotate([source], _one_syllable(_vowel(0.1, 0.5, "a")), _flat_rms())
    assert (result.notes[0].start_sec, result.notes[0].end_sec, result.notes[0].midi) == \
           (source.start_sec, source.end_sec, source.midi)


def test_same_input_gives_the_same_result():
    source, segments = _one_syllable_per_note(2)
    first = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="さく")
    second = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="さく")
    assert [(n.lyric, n.phonemes, n.velocity) for n in first.notes] == \
           [(n.lyric, n.phonemes, n.velocity) for n in second.notes]


def test_only_moraic_nasal_notes_are_counted():
    result = lyrics.annotate([_note(0.0, 0.2, syllable=0), _note(0.2, 0.4, syllable=1)],
                             [[_consonant(0.0, 0.2, "ɴ")], [_vowel(0.2, 0.4, "a")]], _flat_rms())
    assert [note.lyric for note in result.notes] == ["ん", "あ"]
    assert result.diagnostics.moraic_nasal_notes == 1


def test_moraic_nasal_from_the_given_lyrics_is_not_counted():
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="ん")
    assert result.notes[0].lyric == "ん"
    assert result.diagnostics.moraic_nasal_notes == 0


def test_undetermined_vowel_from_the_given_lyrics_is_not_counted():
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_gap(0.0, 0.2)), _flat_rms(),
                             lyrics_text="か")
    assert result.notes[0].lyric == "か"
    assert result.diagnostics.undetermined_vowel_notes == 0


def test_continuation_notes_are_not_counted_as_undetermined_vowels():
    result = lyrics.annotate([_note(0.0, 0.2), _note(0.2, 0.4, midi=71)],
                             _one_syllable(_gap(0.0, 0.4)), _flat_rms())
    assert [note.lyric for note in result.notes] == ["あ", "-"]
    assert result.diagnostics.undetermined_vowel_notes == 1


def test_only_notes_without_phonemes_are_counted():
    result = lyrics.annotate([_note(0.0, 0.2, syllable=0), _note(0.2, 0.4, syllable=1)],
                             [[_vowel(0.0, 0.2, "a")], [_gap(0.2, 0.4)]], _flat_rms())
    assert [note.phonemes for note in result.notes] == [["a"], []]
    assert result.diagnostics.no_phoneme_notes == 1
