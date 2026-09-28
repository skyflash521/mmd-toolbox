import unicodedata
from dataclasses import dataclass, field

import numpy as np

from vocal_analysis.phonemes import espeak_ipa_to_vowel

from .notes import MORAIC_NASAL

# vpr の音素表現は VOCALOID 日本語の X-SAMPA。
IPA_TO_VPR_PHONEME = {
    "a": "a", "i": "i", "ɯ": "M", "e̞": "e", "o̞": "o",
    "m": "m", "mʲ": "m'", "b": "b", "bʲ": "b'", "p": "p", "pʲ": "p'", "ɸ": "p\\",
    "w": "w", "ɕ": "S", "dʑ": "dZ", "tɕ": "tS", "j": "j", "ɴ": "N\\",
    "k": "k", "kʲ": "k'", "ɡ": "g", "ɡʲ": "g'", "s": "s", "z": "z", "t": "t",
    "ts": "ts", "d": "d", "n": "n", "ɲ": "J", "h": "h", "ɾ": "4",
    "ç": "C", "v": "v",
}

KANA_BY_CONSONANT = {
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

_PALATALIZED_CONSONANT = {"k": "k'", "g": "g'", "b": "b'", "p": "p'", "m": "m'", "h": "C"}
_RA_ROW_CONSONANT = "4"
_RA_PALATALIZED_KANA_ROW = ("りゃ", "り", "りゅ", "りぇ", "りょ")

_KANA_COLUMN_VOWELS = "aiueo"

_PALATAL_APPROXIMANT = "j"

_UNDETERMINED_VOWEL_KANA = "あ"

# 実 vpr が音節の継続に用いる表記。
CONTINUATION = "-"

_LONG_MARK = "ー"
_SOKUON = "っ"
_MORAIC_NASAL_KANA = "ん"
_SMALL_KANA_EXCEPT_SOKUON = frozenset("ぁぃぅぇぉゃゅょゎ")
_KATAKANA_TO_HIRAGANA_OFFSET = 0x60

_KANA_BLOCK_BREAK = None

_UNCONVERTED_RATIO_THRESHOLD = 0.2

_VOLUME_CENTER_WINDOW_RATIO = 0.6
_VOLUME_MAX = 127

_NEUTRAL_VELOCITY = 64


@dataclass(frozen=True)
class SungNote:
    start_sec: float
    end_sec: float
    midi: int
    lyric: str
    phonemes: list[str]
    velocity: int
    is_protected: bool = False


@dataclass
class Diagnostics:
    undetermined_vowel_notes: int = 0
    moraic_nasal_notes: int = 0
    suppressed_notes: int = 0
    no_phoneme_notes: int = 0
    notes_beyond_morae: int = 0
    discarded_morae: int = 0
    unconverted_chars: int = 0
    counted_chars: int = 0

    @property
    def kana_reading_ineffective(self) -> bool:
        if self.counted_chars == 0:
            return False
        return self.unconverted_chars / self.counted_chars > _UNCONVERTED_RATIO_THRESHOLD


@dataclass
class AnnotationResult:
    notes: list[SungNote] = field(default_factory=list)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


def _vpr_phonemes_of(segments):
    return [IPA_TO_VPR_PHONEME[s.phoneme] for s in sorted(segments, key=lambda s: s.start_sec)
            if s.phoneme is not None and s.phoneme in IPA_TO_VPR_PHONEME]


def _nucleus_of(segments):
    for segment in segments:
        if segment.type == "vowel" or (segment.type == "consonant"
                                       and segment.phoneme == MORAIC_NASAL):
            return segment
    return None


def _kana_row(head_consonants):
    if not head_consonants:
        return KANA_BY_CONSONANT[""]
    adjacent_to_nucleus = head_consonants[-1]
    if adjacent_to_nucleus == _PALATAL_APPROXIMANT and len(head_consonants) > 1:
        previous = head_consonants[-2]
        if previous == _RA_ROW_CONSONANT:
            return _RA_PALATALIZED_KANA_ROW
        return KANA_BY_CONSONANT[_PALATALIZED_CONSONANT.get(previous, _PALATAL_APPROXIMANT)]
    return KANA_BY_CONSONANT[adjacent_to_nucleus]


def _lyric_of(segments, diagnostics):
    nucleus = _nucleus_of(segments)
    if nucleus is not None and nucleus.phoneme == MORAIC_NASAL:
        diagnostics.moraic_nasal_notes += 1
        return _MORAIC_NASAL_KANA

    vowel = espeak_ipa_to_vowel(nucleus.phoneme or "") if nucleus is not None else None
    if vowel is None:
        diagnostics.undetermined_vowel_notes += 1
        return _UNDETERMINED_VOWEL_KANA

    head_consonants = [IPA_TO_VPR_PHONEME[s.phoneme]
                       for s in sorted(segments, key=lambda s: s.start_sec)
                       if s is not nucleus and s.start_sec < nucleus.start_sec
                       and s.phoneme in IPA_TO_VPR_PHONEME]
    return _kana_row(head_consonants)[_KANA_COLUMN_VOWELS.index(vowel)]


def _should_have_been_kana(character):
    return unicodedata.category(character)[0] in {"L", "N"}


def _to_kana_tokens(reading, diagnostics) -> list[str | None]:
    normalized = unicodedata.normalize("NFKC", reading)
    tokens = []
    for character in normalized:
        if "ァ" <= character <= "ヶ":
            character = chr(ord(character) - _KATAKANA_TO_HIRAGANA_OFFSET)
        if character == _LONG_MARK or "ぁ" <= character <= "ゖ":
            tokens.append(character)
            diagnostics.counted_chars += 1
        else:
            tokens.append(_KANA_BLOCK_BREAK)
            if _should_have_been_kana(character):
                diagnostics.unconverted_chars += 1
                diagnostics.counted_chars += 1
    return tokens


def _morae_of_block(block):
    morae = []
    carry_to_next_mora = ""
    for character in block:
        if character == _SOKUON:
            carry_to_next_mora += character
        elif character == _LONG_MARK and morae and not carry_to_next_mora:
            morae[-1] += character
        elif character == _LONG_MARK:
            carry_to_next_mora += character
        elif character in _SMALL_KANA_EXCEPT_SOKUON and morae and not carry_to_next_mora:
            morae[-1] += character
        else:
            morae.append(carry_to_next_mora + character)
            carry_to_next_mora = ""
    if carry_to_next_mora:
        if morae:
            morae[-1] += carry_to_next_mora
        else:
            morae.append(carry_to_next_mora)
    return morae


def _split_morae(tokens):
    morae = []
    block = []
    for token in [*tokens, _KANA_BLOCK_BREAK]:
        if token is _KANA_BLOCK_BREAK:
            morae += _morae_of_block(block)
            block = []
        else:
            block.append(token)
    return morae


def _volume_of(rms, start_sec, end_sec):
    times = np.asarray(rms.times_sec)
    values = np.asarray(rms.values)
    margin = (end_sec - start_sec) * (1.0 - _VOLUME_CENTER_WINDOW_RATIO) / 2.0
    window = (times >= start_sec + margin) & (times < end_sec - margin)
    if not window.any():
        window = (times >= start_sec) & (times < end_sec)
    if not window.any():
        return 0
    return int(np.clip(round(float(values[window].mean()) * _VOLUME_MAX), 0, _VOLUME_MAX))


def annotate(source_notes, syllable_segments, rms, *, lyrics_text=None) -> AnnotationResult:
    diagnostics = Diagnostics()

    audible_notes = []
    for note in source_notes:
        if _volume_of(rms, note.start_sec, note.end_sec) == 0:
            diagnostics.suppressed_notes += 1
            continue
        audible_notes.append(note)
    syllable_count = len({note.syllable for note in audible_notes})

    morae = []
    if lyrics_text is not None:
        from vocal_analysis.reading import to_kana_reading

        morae = _split_morae(_to_kana_tokens(to_kana_reading(lyrics_text), diagnostics))
        if len(morae) > syllable_count:
            diagnostics.discarded_morae = len(morae) - syllable_count

    result = []
    started_syllables = set()
    syllable_head_position = 0
    for note in audible_notes:
        if note.syllable in started_syllables:
            result.append(SungNote(
                start_sec=note.start_sec, end_sec=note.end_sec, midi=note.midi,
                lyric=CONTINUATION, phonemes=[CONTINUATION], velocity=_NEUTRAL_VELOCITY))
            continue

        started_syllables.add(note.syllable)
        segments = syllable_segments[note.syllable]
        phonemes = _vpr_phonemes_of(segments)
        if not phonemes:
            diagnostics.no_phoneme_notes += 1
        if syllable_head_position < len(morae):
            lyric = morae[syllable_head_position]
        else:
            if lyrics_text is not None:
                diagnostics.notes_beyond_morae += 1
            lyric = _lyric_of(segments, diagnostics)
        syllable_head_position += 1
        result.append(SungNote(
            start_sec=note.start_sec, end_sec=note.end_sec, midi=note.midi, lyric=lyric,
            phonemes=phonemes, velocity=_NEUTRAL_VELOCITY, is_protected=bool(phonemes)))

    return AnnotationResult(notes=result, diagnostics=diagnostics)
