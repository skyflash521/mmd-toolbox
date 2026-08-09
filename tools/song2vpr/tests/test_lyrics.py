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


def _rms_silent_between(start, end, seconds=2.0):
    """指定の区間だけ音量が 0 の包絡(その区間に収まる音符だけベロシティが 0 になる)。"""
    times = np.arange(0.0, seconds, 0.01)
    values = np.where((times >= start) & (times < end), 0.0, 0.5)
    return RmsEnvelope(times_sec=times, values=values, dynamic_range_db=20.0)


def _one_syllable(*segments):
    """音節1つぶんのセグメント帰属(付与の段が受け取る形)。"""
    return [list(segments)]


def _syllable_per_note(count, seconds=0.2, phoneme="a"):
    """音符ごとに1音節を持つ入力。音符列とセグメント帰属を返す。"""
    source = [_note(i * seconds, (i + 1) * seconds, syllable=i) for i in range(count)]
    segments = [[_vowel(i * seconds, (i + 1) * seconds, phoneme)] for i in range(count)]
    return source, segments


# --- 音素列 ------------------------------------------------------------------


@pytest.mark.parametrize("ipa,xsampa", [
    ("a", "a"), ("i", "i"), ("ɯ", "M"), ("e̞", "e"), ("o̞", "o"),  # 母音5記号
    ("mʲ", "m'"), ("ɸ", "p\\"), ("ɕ", "S"), ("dʑ", "dZ"), ("tɕ", "tS"),
    ("ɴ", "N\\"), ("ɡ", "g"), ("ts", "ts"), ("ɲ", "J"), ("ɾ", "4"),
    ("ç", "C"), ("v", "v"),  # インベントリに現れない2記号(記号体系の定義から決まる)
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


def test_phonemes_may_be_empty():
    """除外の結果として空になっても許容する。"""
    result = lyrics.annotate([_note(0.0, 0.4)], _one_syllable(_gap(0.0, 0.4)), _flat_rms())
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
def test_lyric_is_the_vowel_kana_of_the_nucleus(ipa, kana):
    result = lyrics.annotate([_note(0.0, 0.4)],
                             _one_syllable(Segment(type="vowel", start_sec=0.0, end_sec=0.4,
                                                   phoneme=ipa, confidence=0.9)), _flat_rms())
    assert result.notes[0].lyric == kana


def test_note_without_a_nucleus_falls_back_and_is_counted():
    """核が得られない音符は既定の仮名を入れ、母音未確定として数える。"""
    result = lyrics.annotate([_note(0.0, 0.3)], _one_syllable(_gap(0.0, 0.3)), _flat_rms())
    assert result.notes[0].lyric == "あ"
    assert result.diagnostics.undetermined_vowel_notes == 1


# --- 音節の帰属からの付与 ----------------------------------------------------
#
# 付与の段は、音符区間との時間の重なりでなく、分割の段が決めた音節の帰属で音素列と表示歌詞を決める。
# 下の各テストは音節ごとのセグメント列を直に渡す。


def test_phonemes_come_from_the_syllable_not_from_the_overlap():
    """隣の音節の子音を載せない。音符区間が隣の音節へはみ出していても帰属で決める。"""
    first = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.4, "a")]
    second = [_consonant(0.4, 0.5, "d"), _vowel(0.5, 0.8, "a")]
    result = lyrics.annotate([_note(0.0, 0.45, syllable=0), _note(0.45, 0.8, syllable=1)],
                             [first, second], _flat_rms())
    assert [note.phonemes for note in result.notes] == [["k", "a"], ["d", "a"]]


def test_a_continuation_note_carries_the_continuation_mark():
    """1つの音節が複数の音高に分かれたら、2つ目以降は表示歌詞も音素列も継続の表記にする。"""
    syllable = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.6, "a")]
    result = lyrics.annotate([_note(0.0, 0.3, syllable=0), _note(0.3, 0.6, midi=71, syllable=0)],
                             [syllable], _flat_rms())
    assert [note.lyric for note in result.notes] == ["か", "-"]
    assert [note.phonemes for note in result.notes] == [["k", "a"], ["-"]]


@pytest.mark.parametrize(("head", "nucleus", "kana"), [
    ([], "a", "あ"),  # 頭子音なし
    (["k"], "a", "か"),
    (["s"], "i", "すぃ"),  # 外来語音
    (["t"], "ɯ", "とぅ"),
    (["ɸ"], "o̞", "ふぉ"),
    (["ɕ"], "a", "しゃ"),
    (["kʲ"], "a", "きゃ"),  # 子音そのものが拗音
    (["v"], "a", "ば"),  # ヴ表記は受理を確認できていないので調音の近い行を使う
    (["k", "j"], "a", "きゃ"),  # 隣接する子音が j のときは前の子音の拗音行
    (["ɾ", "j"], "o̞", "りょ"),
    (["s", "j"], "a", "や"),  # 拗音行を持たない子音は j の行
    (["t", "k"], "a", "か"),  # 頭子音が複数なら核に隣接する子音で決める
])
def test_lyric_comes_from_the_head_consonant_and_the_nucleus(head, nucleus, kana):
    segments = [_consonant(0.1 * i, 0.1 * (i + 1), phoneme) for i, phoneme in enumerate(head)]
    start = 0.1 * len(head)
    segments.append(Segment(type="vowel", start_sec=start, end_sec=start + 0.3,
                            phoneme=nucleus, confidence=0.9))
    result = lyrics.annotate([_note(0.0, start + 0.3, syllable=0)], [segments], _flat_rms())
    assert result.notes[0].lyric == kana


def test_head_consonants_that_do_not_decide_the_kana_stay_in_the_phonemes():
    """かなを決めなかった頭子音も音素列には載せる(発音の情報を落とさない)。"""
    syllable = [_consonant(0.0, 0.1, "t"), _consonant(0.1, 0.2, "k"), _vowel(0.2, 0.5, "a")]
    result = lyrics.annotate([_note(0.0, 0.5, syllable=0)], [syllable], _flat_rms())
    assert result.notes[0].lyric == "か"
    assert result.notes[0].phonemes == ["t", "k", "a"]


def test_a_moraic_nasal_nucleus_gets_the_nasal_kana():
    """核が撥音の音符は、頭子音に依らず撥音のかなにする。"""
    syllable = [_consonant(0.0, 0.1, "k"), _consonant(0.1, 0.4, "ɴ")]
    result = lyrics.annotate([_note(0.0, 0.4, syllable=0)], [syllable], _flat_rms())
    assert result.notes[0].lyric == "ん"


def test_the_kana_table_covers_every_consonant_the_recognizer_can_emit():
    """かな表は認識器の音素語彙と5母音の全組を覆う(取りこぼしを表の網羅で防ぐ)。"""
    consonants = set(lyrics.IPA_TO_VPR_PHONEME.values()) - set("aiMeo") - {"N\\"}
    assert consonants <= set(lyrics.KANA_BY_CONSONANT)


def test_the_kana_table_holds_the_documented_cells():
    """かな表の全セルを固定する(1セルの誤りが表示歌詞をそのまま壊すため)。"""
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


def test_the_syllable_head_protects_its_phonemes():
    """音素列を載せた音節の先頭の音符は保護を真にする。継続と音素列が空の音符は偽のまま。"""
    first = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.6, "a")]
    second = [_gap(0.6, 0.9)]
    result = lyrics.annotate(
        [_note(0.0, 0.3, syllable=0), _note(0.3, 0.6, midi=71, syllable=0),
         _note(0.6, 0.9, syllable=1)],
        [first, second], _flat_rms())
    assert [note.is_protected for note in result.notes] == [True, False, False]


# --- ベロシティが 0 になる音符の抑制 ------------------------------------------


def test_a_note_with_zero_velocity_is_not_output():
    """ベロシティが 0 になる音符は出力せず、落とした件数を数える。"""
    result = lyrics.annotate([_note(0.0, 0.2, syllable=0), _note(0.2, 0.4, syllable=1)],
                             [[_vowel(0.0, 0.2, "a")], [_vowel(0.2, 0.4, "i")]],
                             _rms_silent_between(0.2, 0.4))
    assert [note.lyric for note in result.notes] == ["あ"]
    assert result.diagnostics.suppressed_notes == 1


def test_the_first_surviving_note_of_a_syllable_becomes_its_head():
    """先頭の音符が落ちた音節は、残った最初の音符が先頭になる(継続の表記のままにしない)。"""
    syllable = [_consonant(0.0, 0.1, "k"), _vowel(0.1, 0.6, "a")]
    result = lyrics.annotate([_note(0.0, 0.3, syllable=0), _note(0.3, 0.6, midi=71, syllable=0)],
                             [syllable], _rms_silent_between(0.0, 0.3))
    assert [note.lyric for note in result.notes] == ["か"]
    assert [note.phonemes for note in result.notes] == [["k", "a"]]
    assert result.notes[0].is_protected is True


def test_a_syllable_with_no_surviving_note_takes_no_mora():
    """音符が1つも残らない音節は表示に現れないので、モーラも消費しない。"""
    source = [_note(0.0, 0.2, syllable=0), _note(0.2, 0.4, syllable=1),
              _note(0.4, 0.6, syllable=2)]
    segments = [[_vowel(0.0, 0.2, "a")], [_vowel(0.2, 0.4, "a")], [_vowel(0.4, 0.6, "a")]]
    result = lyrics.annotate(source, segments, _rms_silent_between(0.2, 0.4),
                             lyrics_text="さくら")
    assert [note.lyric for note in result.notes] == ["さ", "く"]
    assert result.diagnostics.discarded_morae == 1


# --- 表示歌詞(歌詞テキストの指定あり)---------------------------------------


def test_given_lyrics_are_assigned_one_mora_per_syllable_head():
    source, segments = _syllable_per_note(3)
    result = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="さくら")
    assert [note.lyric for note in result.notes] == ["さ", "く", "ら"]


def test_a_continuation_note_keeps_the_continuation_mark_when_lyrics_are_given():
    """モーラを割り当てるのは音節の先頭の音符だけで、継続の音符は継続の表記のままにする。"""
    result = lyrics.annotate([_note(0.0, 0.2, syllable=0), _note(0.2, 0.4, syllable=0),
                              _note(0.4, 0.6, syllable=1)],
                             [[_vowel(0.0, 0.4, "a")], [_vowel(0.4, 0.6, "a")]],
                             _flat_rms(), lyrics_text="さく")
    assert [note.lyric for note in result.notes] == ["さ", "-", "く"]


def test_phonemes_still_come_from_the_segments_when_lyrics_are_given():
    """歌詞を与えても音素列はセグメント由来のままにする。"""
    result = lyrics.annotate([_note(0.0, 0.4)], _one_syllable(_vowel(0.0, 0.4, "i")), _flat_rms(),
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
    source, segments = _syllable_per_note(len(expected))
    result = lyrics.annotate(source, segments, _flat_rms(), lyrics_text=text)
    assert [note.lyric for note in result.notes] == expected


def test_more_notes_than_morae_fall_back_to_the_vowel_kana():
    source, segments = _syllable_per_note(2, phoneme="i")
    result = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="さ")
    assert [note.lyric for note in result.notes] == ["さ", "い"]
    assert result.diagnostics.notes_beyond_morae == 1


def test_more_morae_than_notes_are_discarded_and_counted():
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="さくら")
    assert [note.lyric for note in result.notes] == ["さ"]
    assert result.diagnostics.discarded_morae == 2


def test_layout_characters_are_skipped_without_being_counted():
    """空白や記号は体裁の文字なので読み飛ばし、変換の効きの判定には数えない。"""
    source, segments = _syllable_per_note(2)
    result = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="あ!い?")
    assert [note.lyric for note in result.notes] == ["あ", "い"]
    assert result.diagnostics.unconverted_chars == 0
    assert result.diagnostics.counted_chars == 2  # かな2文字だけ


def _stub_reading(monkeypatch, reading):
    """かな読みへの変換を差し替える(変換が効かなかった読みを再現するため)。"""
    monkeypatch.setattr("vocal_analysis.reading.to_kana_reading", lambda text: reading)


def test_characters_that_should_have_been_kana_are_counted(monkeypatch):
    """読みに残った漢字・英数字は、かな読みが効いていないかの判定へ数える。"""
    _stub_reading(monkeypatch, "あ漢A")
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="どんな表記でもよい")
    assert result.diagnostics.unconverted_chars == 2
    assert result.diagnostics.counted_chars == 3
    assert result.diagnostics.kana_reading_ineffective


def test_reading_without_any_countable_character_is_not_judged(monkeypatch):
    """かなも未変換の文字も無ければ割合を求められないので、変換の効きを判定しない。"""
    _stub_reading(monkeypatch, "!?")
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="どんな表記でもよい")
    assert result.diagnostics.counted_chars == 0
    assert not result.diagnostics.kana_reading_ineffective


def test_mostly_kana_reading_is_not_flagged(monkeypatch):
    _stub_reading(monkeypatch, "あいうえおかき漢")
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="どんな表記でもよい")
    assert result.diagnostics.unconverted_chars == 1
    assert not result.diagnostics.kana_reading_ineffective


def test_katakana_is_folded_to_hiragana():
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="サ")
    assert result.notes[0].lyric == "さ"


# --- ベロシティ --------------------------------------------------------------


@pytest.mark.xfail(reason="impl pending: ベロシティの中立値固定", strict=True)
@pytest.mark.parametrize("value", [0.1, 1.0])
def test_velocity_is_the_neutral_value_regardless_of_the_volume(value):
    """ベロシティは音量に依らず、全音符で中立値に固定する(ベロシティは音量の欄ではない)。

    音節の先頭と継続では音符を作る経路が別なので、同じ音節の2音符で両方を見る。音量を写した値が
    たまたま中立値と一致する強さ(0.5)は、区別が付かないので使わない。
    """
    result = lyrics.annotate(
        [_note(0.0, 0.3, syllable=0), _note(0.3, 0.6, midi=71, syllable=0)],
        _one_syllable(_vowel(0.0, 0.6, "a")), _flat_rms(value=value))
    assert [note.lyric for note in result.notes] == ["あ", "-"]
    assert [note.velocity for note in result.notes] == [64, 64]


@pytest.mark.parametrize(("value", "output"), [(0.003, False), (0.004, True)])
def test_the_volume_threshold_is_the_step_of_the_stored_scale(value, output):
    """抑制の境目は、音量の代表値を 0〜127 へ写したときの最小の刻み。

    正規化の下端は 0 へ写るので、そこにある音符だけが落ちる。刻みより上の音量は、どれだけ小さくても
    音符として残す。
    """
    result = lyrics.annotate([_note(0.0, 0.4)], _one_syllable(_vowel(0.0, 0.4, "a")),
                             _flat_rms(value=value))
    assert bool(result.notes) is output


def test_velocity_comes_from_the_rms_of_the_note():
    result = lyrics.annotate([_note(0.0, 0.4)], _one_syllable(_vowel(0.0, 0.4, "a")),
                             _flat_rms(value=0.5))
    assert result.notes[0].velocity == 64  # round(0.5 * 127)


@pytest.mark.parametrize("value,expected", [(0.004, 1), (1.0, 127)])
def test_velocity_spans_the_range_of_the_notes_that_are_output(value, expected):
    """出力する音符が取りうる下端と上端(0 になる音符は出力しないので下端は 1)。"""
    result = lyrics.annotate([_note(0.0, 0.4)], _one_syllable(_vowel(0.0, 0.4, "a")),
                             _flat_rms(value=value))
    assert result.notes[0].velocity == expected


@pytest.mark.parametrize("loud_in_middle,expected", [(True, 127), (False, 64)])
def test_velocity_uses_the_middle_of_the_note(loud_in_middle, expected):
    """代表値は音符区間の中央だけから取る(端の立ち上がり・減衰に引かれない)。"""
    times = np.arange(0.0, 1.0, 0.01)
    middle = (times >= 0.2) & (times < 0.8)
    values = np.where(middle if loud_in_middle else ~middle, 1.0, 0.5)
    envelope = RmsEnvelope(times_sec=times, values=values, dynamic_range_db=20.0)
    result = lyrics.annotate([_note(0.0, 1.0)], _one_syllable(_vowel(0.0, 1.0, "a")), envelope)
    assert result.notes[0].velocity == expected


# --- 区間と音高は変えない ----------------------------------------------------


def test_the_span_and_the_pitch_are_carried_through():
    source = _note(0.1, 0.5, midi=72)
    result = lyrics.annotate([source], _one_syllable(_vowel(0.1, 0.5, "a")), _flat_rms())
    assert (result.notes[0].start_sec, result.notes[0].end_sec, result.notes[0].midi) == \
           (source.start_sec, source.end_sec, source.midi)


def test_same_input_gives_the_same_result():
    source, segments = _syllable_per_note(2)
    first = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="さく")
    second = lyrics.annotate(source, segments, _flat_rms(), lyrics_text="さく")
    assert [(n.lyric, n.phonemes, n.velocity) for n in first.notes] == \
           [(n.lyric, n.phonemes, n.velocity) for n in second.notes]


# --- 診断 --------------------------------------------------------------------


def test_moraic_nasal_notes_are_counted():
    """撥音「ん」を入れた音符だけを数える(母音の音符は数えない)。"""
    result = lyrics.annotate([_note(0.0, 0.2, syllable=0), _note(0.2, 0.4, syllable=1)],
                             [[_consonant(0.0, 0.2, "ɴ")], [_vowel(0.2, 0.4, "a")]], _flat_rms())
    assert [note.lyric for note in result.notes] == ["ん", "あ"]
    assert result.diagnostics.moraic_nasal_notes == 1


def test_moraic_nasal_from_the_given_lyrics_is_not_counted():
    """数えるのは表示歌詞を音声から決めた音符だけ(モーラ由来の「ん」は数えない)。"""
    result = lyrics.annotate([_note(0.0, 0.2)], _one_syllable(_vowel(0.0, 0.2, "a")), _flat_rms(),
                             lyrics_text="ん")
    assert result.notes[0].lyric == "ん"
    assert result.diagnostics.moraic_nasal_notes == 0


def test_notes_without_phonemes_are_counted():
    """音素列が空になった音符だけを数える(音素を持つ音符は数えない)。"""
    result = lyrics.annotate([_note(0.0, 0.2, syllable=0), _note(0.2, 0.4, syllable=1)],
                             [[_vowel(0.0, 0.2, "a")], [_gap(0.2, 0.4)]], _flat_rms())
    assert [note.phonemes for note in result.notes] == [["a"], []]
    assert result.diagnostics.no_phoneme_notes == 1
