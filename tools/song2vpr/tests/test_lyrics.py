"""song2vpr の表示歌詞・音素列・強弱の付与のテスト。

音符の区間と音高は前段が確定させているので、ここでは各音符へ何を載せるかだけを見る。
"""

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


# --- 音素列 ------------------------------------------------------------------


def test_phonemes_come_from_the_overlapping_segments_in_time_order():
    result = lyrics.annotate([_note(0.0, 0.4)],
                             [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.4, "a")], _flat_rms())
    assert result.notes[0].phonemes == ["k", "a"]


@pytest.mark.parametrize("ipa,xsampa", [
    ("a", "a"), ("i", "i"), ("ɯ", "M"), ("e̞", "e"), ("o̞", "o"),  # 母音5記号
    ("mʲ", "m'"), ("ɸ", "p\\"), ("ɕ", "S"), ("dʑ", "dZ"), ("tɕ", "tS"),
    ("ɴ", "N\\"), ("ɡ", "g"), ("ts", "ts"), ("ɲ", "J"), ("ɾ", "4"),
    ("ç", "C"), ("v", "v"),  # インベントリに現れない2記号(記号体系の定義から決まる)
])
def test_ipa_is_mapped_to_the_vpr_phoneme_notation(ipa, xsampa):
    kind = "vowel" if ipa in {"a", "i", "ɯ", "e̞", "o̞"} else "consonant"
    segment = Segment(type=kind, start_sec=0.0, end_sec=0.4, phoneme=ipa, confidence=0.9)
    result = lyrics.annotate([_note(0.0, 0.4)], [segment], _flat_rms())
    assert result.notes[0].phonemes == [xsampa]


def test_gap_and_unlabelled_segments_are_excluded_from_the_phonemes():
    result = lyrics.annotate([_note(0.0, 0.4)],
                             [_vowel(0.0, 0.2, "a"), _gap(0.2, 0.4)], _flat_rms())
    assert result.notes[0].phonemes == ["a"]


def test_phonemes_may_be_empty():
    """除外の結果として空になっても許容する。"""
    result = lyrics.annotate([_note(0.0, 0.4)], [_gap(0.0, 0.4)], _flat_rms())
    assert result.notes[0].phonemes == []


def test_the_mapping_covers_every_symbol_the_recognizer_can_emit():
    """写像表は認識器が音素ラベルに出しうる記号を全被覆する(取りこぼしを表の網羅で防ぐ)。"""
    from vocal_analysis.phonemes import _G2P_TO_VOCAB_SYMBOL

    emitted = {symbol for symbol in _G2P_TO_VOCAB_SYMBOL.values() if symbol}
    assert emitted <= set(lyrics.IPA_TO_VPR_PHONEME)


# --- 表示歌詞(歌詞テキストの指定なし)---------------------------------------


@pytest.mark.parametrize("ipa,kana", [
    ("a", "あ"), ("i", "い"), ("ɯ", "う"), ("e̞", "え"), ("o̞", "お"),
])
def test_lyric_is_the_vowel_kana_of_the_representative_vowel(ipa, kana):
    result = lyrics.annotate([_note(0.0, 0.4)],
                             [Segment(type="vowel", start_sec=0.0, end_sec=0.4,
                                      phoneme=ipa, confidence=0.9)], _flat_rms())
    assert result.notes[0].lyric == kana


def test_representative_vowel_is_the_longest_one_in_the_note():
    result = lyrics.annotate([_note(0.0, 0.4)],
                             [_vowel(0.0, 0.1, "a"), _vowel(0.1, 0.4, "i")], _flat_rms())
    assert result.notes[0].lyric == "い"


def test_note_of_a_moraic_nasal_gets_the_nasal_kana():
    """母音が無く鼻音が主体の音符は撥音にする。"""
    result = lyrics.annotate([_note(0.0, 0.3)], [_consonant(0.0, 0.3, "ɴ")], _flat_rms())
    assert result.notes[0].lyric == "ん"


def test_nasal_wins_only_when_it_outweighs_the_others():
    """鼻音の重なり時間がそれ以外を上回るときだけ撥音にする。"""
    segments = [_consonant(0.0, 0.1, "n"), _consonant(0.1, 0.3, "s")]
    result = lyrics.annotate([_note(0.0, 0.3)], segments, _flat_rms())
    assert result.notes[0].lyric != "ん"


def test_note_without_a_vowel_or_nasal_falls_back_and_is_counted():
    """母音も鼻音も得られない音符は既定の仮名を入れ、母音未確定として数える。"""
    result = lyrics.annotate([_note(0.0, 0.3)], [_gap(0.0, 0.3)], _flat_rms())
    assert result.notes[0].lyric == "あ"
    assert result.diagnostics.undetermined_vowel_notes == 1


# --- 表示歌詞(歌詞テキストの指定あり)---------------------------------------


def test_given_lyrics_are_assigned_one_mora_per_note():
    result = lyrics.annotate([_note(0.0, 0.2), _note(0.2, 0.4), _note(0.4, 0.6)],
                             [_vowel(0.0, 0.6, "a")], _flat_rms(), lyrics_text="さくら")
    assert [note.lyric for note in result.notes] == ["さ", "く", "ら"]


def test_phonemes_still_come_from_the_segments_when_lyrics_are_given():
    """歌詞を与えても音素列はセグメント由来のままにする。"""
    result = lyrics.annotate([_note(0.0, 0.4)], [_vowel(0.0, 0.4, "i")], _flat_rms(),
                             lyrics_text="さ")
    assert result.notes[0].lyric == "さ"
    assert result.notes[0].phonemes == ["i"]


@pytest.mark.parametrize("text,expected", [
    ("きゃく", ["きゃ", "く"]),  # 小書きのかなは直前のモーラへ
    ("がっき", ["が", "っき"]),  # 促音は後続のモーラへ(無声の詰まりで音符が増えない)
    ("かー", ["かー"]),  # 長音記号は直前のモーラへ(音を伸ばすだけで音符が増えない)
    ("はん", ["は", "ん"]),  # 撥音は新しいモーラを成す
    ("んー", ["んー"]),  # 撥音の直後の長音記号はそのモーラへ
    ("っか", ["っか"]),  # 先頭の促音は後続へ
    ("ーか", ["ーか"]),  # 直前が無い長音記号は後続へ
    ("かっ", ["かっ"]),  # 後続に結合先が無い促音は直前のモーラへ
    ("ー", ["ー"]),  # 前後どちらにも結合先が無ければ単独で1モーラ
    ("あ い", ["あ", "い"]),  # 空白は読み飛ばし、その前後を結合させない
    ("かーん", ["かー", "ん"]),
])
def test_mora_split_rules(text, expected):
    result = lyrics.annotate([_note(i * 0.2, i * 0.2 + 0.2) for i in range(len(expected))],
                             [_vowel(0.0, 2.0, "a")], _flat_rms(), lyrics_text=text)
    assert [note.lyric for note in result.notes] == expected


def test_more_notes_than_morae_fall_back_to_the_vowel_kana():
    result = lyrics.annotate([_note(0.0, 0.2), _note(0.2, 0.4)],
                             [_vowel(0.0, 0.4, "i")], _flat_rms(), lyrics_text="さ")
    assert [note.lyric for note in result.notes] == ["さ", "い"]
    assert result.diagnostics.notes_beyond_morae == 1


def test_more_morae_than_notes_are_discarded_and_counted():
    result = lyrics.annotate([_note(0.0, 0.2)], [_vowel(0.0, 0.2, "a")], _flat_rms(),
                             lyrics_text="さくら")
    assert [note.lyric for note in result.notes] == ["さ"]
    assert result.diagnostics.discarded_morae == 2


def test_layout_characters_are_skipped_without_being_counted():
    """空白や記号は体裁の文字なので読み飛ばし、変換の効きの判定には数えない。"""
    result = lyrics.annotate([_note(0.0, 0.2), _note(0.2, 0.4)], [_vowel(0.0, 0.4, "a")],
                             _flat_rms(), lyrics_text="あ!い?")
    assert [note.lyric for note in result.notes] == ["あ", "い"]
    assert result.diagnostics.unconverted_chars == 0
    assert result.diagnostics.counted_chars == 2  # かな2文字だけ


def _stub_reading(monkeypatch, reading):
    """かな読みへの変換を差し替える(変換が効かなかった読みを再現するため)。"""
    monkeypatch.setattr("vocal_analysis.reading.to_kana_reading", lambda text: reading)


def test_characters_that_should_have_been_kana_are_counted(monkeypatch):
    """読みに残った漢字・英数字は、かな読みが効いていないかの判定へ数える。"""
    _stub_reading(monkeypatch, "あ漢A")
    result = lyrics.annotate([_note(0.0, 0.2)], [_vowel(0.0, 0.2, "a")], _flat_rms(),
                             lyrics_text="どんな表記でもよい")
    assert result.diagnostics.unconverted_chars == 2
    assert result.diagnostics.counted_chars == 3
    assert result.diagnostics.kana_reading_ineffective


def test_reading_without_any_countable_character_is_not_judged(monkeypatch):
    """かなも未変換の文字も無ければ割合を求められないので、変換の効きを判定しない。"""
    _stub_reading(monkeypatch, "!?")
    result = lyrics.annotate([_note(0.0, 0.2)], [_vowel(0.0, 0.2, "a")], _flat_rms(),
                             lyrics_text="どんな表記でもよい")
    assert result.diagnostics.counted_chars == 0
    assert not result.diagnostics.kana_reading_ineffective


def test_mostly_kana_reading_is_not_flagged(monkeypatch):
    _stub_reading(monkeypatch, "あいうえおかき漢")
    result = lyrics.annotate([_note(0.0, 0.2)], [_vowel(0.0, 0.2, "a")], _flat_rms(),
                             lyrics_text="どんな表記でもよい")
    assert result.diagnostics.unconverted_chars == 1
    assert not result.diagnostics.kana_reading_ineffective


def test_katakana_is_folded_to_hiragana():
    result = lyrics.annotate([_note(0.0, 0.2)], [_vowel(0.0, 0.2, "a")], _flat_rms(),
                             lyrics_text="サ")
    assert result.notes[0].lyric == "さ"


# --- ベロシティ --------------------------------------------------------------


def test_velocity_comes_from_the_rms_of_the_note():
    result = lyrics.annotate([_note(0.0, 0.4)], [_vowel(0.0, 0.4, "a")], _flat_rms(value=0.5))
    assert result.notes[0].velocity == 64  # round(0.5 * 127)


@pytest.mark.parametrize("value,expected", [(0.0, 0), (1.0, 127)])
def test_velocity_spans_the_full_range(value, expected):
    result = lyrics.annotate([_note(0.0, 0.4)], [_vowel(0.0, 0.4, "a")], _flat_rms(value=value))
    assert result.notes[0].velocity == expected


@pytest.mark.parametrize("loud_in_middle,expected", [(True, 127), (False, 0)])
def test_velocity_uses_the_middle_of_the_note(loud_in_middle, expected):
    """代表値は音符区間の中央だけから取る(端の立ち上がり・減衰に引かれない)。"""
    times = np.arange(0.0, 1.0, 0.01)
    middle = (times >= 0.2) & (times < 0.8)
    values = np.where(middle if loud_in_middle else ~middle, 1.0, 0.0)
    envelope = RmsEnvelope(times_sec=times, values=values, dynamic_range_db=20.0)
    result = lyrics.annotate([_note(0.0, 1.0)], [_vowel(0.0, 1.0, "a")], envelope)
    assert result.notes[0].velocity == expected


# --- 区間と音高は変えない ----------------------------------------------------


def test_the_span_and_the_pitch_are_carried_through():
    source = _note(0.1, 0.5, midi=72)
    result = lyrics.annotate([source], [_vowel(0.1, 0.5, "a")], _flat_rms())
    assert (result.notes[0].start_sec, result.notes[0].end_sec, result.notes[0].midi) == \
           (source.start_sec, source.end_sec, source.midi)


def test_same_input_gives_the_same_result():
    source = [_note(0.0, 0.2), _note(0.2, 0.4)]
    segments = [_vowel(0.0, 0.4, "a")]
    first = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="さく")
    second = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="さく")
    assert [(n.lyric, n.phonemes, n.velocity) for n in first.notes] == \
           [(n.lyric, n.phonemes, n.velocity) for n in second.notes]


# --- 診断 --------------------------------------------------------------------


def test_moraic_nasal_notes_are_counted():
    """撥音「ん」を入れた音符だけを数える(母音の音符は数えない)。"""
    result = lyrics.annotate([_note(0.0, 0.2), _note(0.2, 0.4)],
                             [_consonant(0.0, 0.2, "ɴ"), _vowel(0.2, 0.4, "a")], _flat_rms())
    assert [note.lyric for note in result.notes] == ["ん", "あ"]
    assert result.diagnostics.moraic_nasal_notes == 1


def test_moraic_nasal_from_the_given_lyrics_is_not_counted():
    """数えるのは表示歌詞を音声から決めた音符だけ(モーラ由来の「ん」は数えない)。"""
    result = lyrics.annotate([_note(0.0, 0.2)], [_vowel(0.0, 0.2, "a")], _flat_rms(),
                             lyrics_text="ん")
    assert result.notes[0].lyric == "ん"
    assert result.diagnostics.moraic_nasal_notes == 0


def test_notes_without_phonemes_are_counted():
    """音素列が空になった音符だけを数える(音素を持つ音符は数えない)。"""
    result = lyrics.annotate([_note(0.0, 0.2), _note(0.2, 0.4)],
                             [_vowel(0.0, 0.2, "a"), _gap(0.2, 0.4)], _flat_rms())
    assert [note.phonemes for note in result.notes] == [["a"], []]
    assert result.diagnostics.no_phoneme_notes == 1
