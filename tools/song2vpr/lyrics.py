"""song2vpr の表示歌詞・音素列・強弱の付与。

音符の区間と音高は前段が確定させているので、ここは各音符へ載せる値だけを決める。意味のある
歌詞の書き起こしはしない(音素セグメントから得られる範囲に留める)。
"""

import unicodedata
from dataclasses import dataclass, field

import numpy as np

from vocal_analysis.phonemes import espeak_ipa_to_vowel

from .notes import MORAIC_NASAL

# 認識器が音素ラベルに出しうる IPA を、vpr の音素表現(VOCALOID 日本語の X-SAMPA)へ写す表。
# 認識器の語彙は閉じているので全被覆で持ち、取りこぼしを表の網羅で防ぐ。
IPA_TO_VPR_PHONEME = {
    # 母音
    "a": "a", "i": "i", "ɯ": "M", "e̞": "e", "o̞": "o",
    # 子音
    "m": "m", "mʲ": "m'", "b": "b", "bʲ": "b'", "p": "p", "pʲ": "p'", "ɸ": "p\\",
    "w": "w", "ɕ": "S", "dʑ": "dZ", "tɕ": "tS", "j": "j", "ɴ": "N\\",
    "k": "k", "kʲ": "k'", "ɡ": "g", "ɡʲ": "g'", "s": "s", "z": "z", "t": "t",
    "ts": "ts", "d": "d", "n": "n", "ɲ": "J", "h": "h", "ɾ": "4",
    # 実 vpr のインベントリには現れないが、X-SAMPA の記号定義から一意に決まるもの。
    "ç": "C", "v": "v",
}

# 音節の頭子音(vpr の音素表現)と核の5母音から表示歌詞のかなを引く表。行は認識器の音素語彙を
# 写した記号の全体、列は a / i / u / e / o。空文字列の行は頭子音を持たない音節。
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
    # ヴ表記を受理する実 vpr を確認できていないので、調音の最も近い b の行を使う。
    "v": ("ば", "び", "ぶ", "べ", "ぼ"),
}

# 核に隣接する子音が j のとき、その前の子音から引く行。ら行の拗音だけは対応する子音の記号が
# 認識器の語彙に無いので、行そのものを持つ。
_PALATALIZED_ROW = {"k": "k'", "g": "g'", "b": "b'", "p": "p'", "m": "m'", "h": "C"}
_RA_PALATALIZED = ("りゃ", "り", "りゅ", "りぇ", "りょ")

# 列の並び。共有側が返す5母音の文字に対応する。
_VOWEL_ORDER = "aiueo"

# 半母音の音素記号。直前の子音と組んで拗音になる。
_PALATAL_APPROXIMANT = "j"

# 母音も撥音も得られない音符へ入れる仮名。
_FALLBACK_KANA = "あ"

# 1つの音節の2つ目以降の音符に入れる表記(実 vpr が用いる継続の表記)。
CONTINUATION = "-"

_LONG_MARK = "ー"
_SOKUON = "っ"
_MORAIC_NASAL_KANA = "ん"
# 促音を除く小書きのかな。直前のモーラに含める。
_SMALL_KANA = frozenset("ぁぃぅぇぉゃゅょゎ")

# かなへ変換されるべきだった文字が、判定に使う文字全体に占める割合がこれを超えると、かな読みへの
# 変換が効いていないおそれがあるとみなす。
_UNCONVERTED_RATIO_THRESHOLD = 0.2

# ベロシティの代表値を取る窓(音符区間の中央のこの割合)。端の立ち上がり・減衰を避ける。
_VELOCITY_WINDOW_RATIO = 0.6


@dataclass(frozen=True)
class SungNote:
    """表示歌詞・音素列・強弱まで載せた音符。"""

    start_sec: float
    end_sec: float
    midi: int
    lyric: str
    phonemes: list[str]
    velocity: int
    is_protected: bool = False  # 音素列の保護。書き出しの段がそのまま vpr の音符へ渡す


@dataclass
class Diagnostics:
    """付与の過程で数えた件数。利用者向けの診断と警告へ出す。

    unconverted_chars は読みに残った「かなへ変換されるべきだった文字」の数、counted_chars は判定に
    使った文字数(かな・長音記号と、かなへ変換されるべきだった文字の合計。体裁の文字は含めない)。
    """

    undetermined_vowel_notes: int = 0
    moraic_nasal_notes: int = 0
    no_phoneme_notes: int = 0
    notes_beyond_morae: int = 0
    discarded_morae: int = 0
    unconverted_chars: int = 0
    counted_chars: int = 0

    @property
    def kana_reading_ineffective(self) -> bool:
        """かな読みへの変換が効いていないおそれがあるか。

        判定に使う文字が1つも無ければ割合を求められないので、そのときは成立させない。
        """
        if self.counted_chars == 0:
            return False
        return self.unconverted_chars / self.counted_chars > _UNCONVERTED_RATIO_THRESHOLD


@dataclass
class AnnotationResult:
    notes: list[SungNote] = field(default_factory=list)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


def _phonemes_of(segments):
    """音節のセグメント由来の音素を時間順に並べ、vpr の音素表現へ写す。

    音素ラベルを持たないセグメントと gap は除外する(除外で空になっても許容する)。
    """
    return [IPA_TO_VPR_PHONEME[s.phoneme] for s in sorted(segments, key=lambda s: s.start_sec)
            if s.phoneme is not None and s.phoneme in IPA_TO_VPR_PHONEME]


def _nucleus_of(segments):
    """音節の核。母音と、母音を伴わない撥音がこれに当たる(分割の段と同じ判定)。"""
    for segment in segments:
        if segment.type == "vowel" or (segment.type == "consonant"
                                       and segment.phoneme == MORAIC_NASAL):
            return segment
    return None


def _kana_row(heads):
    """表示歌詞を引く行。核に隣接する子音で決め、それが半母音なら前の子音の拗音行を使う。"""
    if not heads:
        return KANA_BY_CONSONANT[""]
    adjacent = heads[-1]
    if adjacent == _PALATAL_APPROXIMANT and len(heads) > 1:
        previous = heads[-2]
        if previous == "4":
            return _RA_PALATALIZED
        return KANA_BY_CONSONANT[_PALATALIZED_ROW.get(previous, _PALATAL_APPROXIMANT)]
    return KANA_BY_CONSONANT[adjacent]


def _lyric_of(segments, diagnostics):
    """音節の頭子音と核の母音から表示歌詞を決める。核が撥音なら撥音の仮名。"""
    nucleus = _nucleus_of(segments)
    if nucleus is not None and nucleus.phoneme == MORAIC_NASAL:
        diagnostics.moraic_nasal_notes += 1
        return _MORAIC_NASAL_KANA

    vowel = espeak_ipa_to_vowel(nucleus.phoneme or "") if nucleus is not None else None
    if vowel is None:
        diagnostics.undetermined_vowel_notes += 1
        return _FALLBACK_KANA

    heads = [IPA_TO_VPR_PHONEME[s.phoneme] for s in sorted(segments, key=lambda s: s.start_sec)
             if s is not nucleus and s.start_sec < nucleus.start_sec
             and s.phoneme in IPA_TO_VPR_PHONEME]
    return _kana_row(heads)[_VOWEL_ORDER.index(vowel)]


def _should_have_been_kana(character):
    """かなへ変換されるべきだった文字か。

    漢字・英数字のような、読みとして残っては困る文字を指す。空白・改行や記号は体裁の文字なので
    含めない(判定に混ぜると、句読点の多い歌詞で割合が動いてしまう)。
    """
    return unicodedata.category(character)[0] in {"L", "N"}


def _to_kana_tokens(reading, diagnostics):
    """かな読みを、かなと長音記号の列へ正規化する。

    共有側が返す読みの表記に仮定を置かず、どんな文字列が来ても一意に定まる規則で畳む。かな・
    長音記号以外は読み飛ばすが、その位置に区切り(None)を残す(前後を結合させないため)。
    """
    normalized = unicodedata.normalize("NFKC", reading)
    tokens = []
    for character in normalized:
        # カタカナはひらがなへ写す(長音記号は写さずそのまま残す)。
        if "ァ" <= character <= "ヶ":
            character = chr(ord(character) - 0x60)
        if character == _LONG_MARK or "ぁ" <= character <= "ゖ":
            tokens.append(character)
            diagnostics.counted_chars += 1
        else:
            tokens.append(None)
            if _should_have_been_kana(character):
                diagnostics.unconverted_chars += 1
                diagnostics.counted_chars += 1
    return tokens


def _morae_of_block(block):
    """区切りで挟まれた1続きのかな列をモーラへ分ける。

    かな1文字を1モーラとし、小書きのかな(促音を除く)と長音記号は直前のモーラへ、促音は後続の
    モーラへ含める。結合先が無ければ反対側へ、どちらにも無ければ単独で1モーラにする。撥音は前の
    モーラに含めず新しいモーラを成す(母音を伴わない音節として独立した音符になるため)。
    """
    morae = []
    pending = ""  # 後続のモーラへ付けるもの(促音、または直前が無い長音記号)
    for character in block:
        if character == _SOKUON:
            pending += character
        elif character == _LONG_MARK and morae and not pending:
            morae[-1] += character
        elif character == _LONG_MARK:
            pending += character
        elif character in _SMALL_KANA and morae and not pending:
            morae[-1] += character
        else:
            morae.append(pending + character)
            pending = ""
    if pending:
        # 後続に結合先が無ければ直前へ、それも無ければ単独で1モーラ。
        if morae:
            morae[-1] += pending
        else:
            morae.append(pending)
    return morae


def _split_morae(tokens):
    """区切りをまたがずにモーラを作る。"""
    morae = []
    block = []
    for token in [*tokens, None]:
        if token is None:
            morae += _morae_of_block(block)
            block = []
        else:
            block.append(token)
    return morae


def _velocity_of(rms, start_sec, end_sec):
    """音符区間の中央から取った相対正規化RMSの代表値を 0〜127 へ写す。"""
    times = np.asarray(rms.times_sec)
    values = np.asarray(rms.values)
    margin = (end_sec - start_sec) * (1.0 - _VELOCITY_WINDOW_RATIO) / 2.0
    window = (times >= start_sec + margin) & (times < end_sec - margin)
    if not window.any():
        window = (times >= start_sec) & (times < end_sec)
    if not window.any():
        return 0
    return int(np.clip(round(float(values[window].mean()) * 127), 0, 127))


def annotate(source_notes, syllable_segments, rms, *, lyrics_text=None) -> AnnotationResult:
    """音符へ表示歌詞・音素列・強弱を載せる。

    表示歌詞と音素列は、音符が属する音節のセグメント(syllable_segments は音節番号で引く)から
    決める。1つの音節が複数の音符に分かれている場合、載せるのは先頭の音符だけで、2つ目以降は
    継続の表記にする。lyrics_text を与えると、かな読みへ変換してモーラへ分け、音節の先頭の音符へ
    先頭から 1音符=1モーラで割り当てる。音符数とモーラ数が食い違うぶんは診断へ数える(下書きなので
    厳密な整合は求めない)。
    """
    diagnostics = Diagnostics()
    heads = len({note.syllable for note in source_notes})

    morae = []
    if lyrics_text is not None:
        # 読みへの変換は追加依存を要するので、歌詞を与えられたときだけ取り込む。
        from vocal_analysis.reading import to_kana_reading

        morae = _split_morae(_to_kana_tokens(to_kana_reading(lyrics_text), diagnostics))
        if len(morae) > heads:
            diagnostics.discarded_morae = len(morae) - heads

    result = []
    started = set()
    position = 0  # 音節の先頭の音符の通し番号(モーラの割り当てに使う)
    for note in source_notes:
        velocity = _velocity_of(rms, note.start_sec, note.end_sec)
        if note.syllable in started:
            result.append(SungNote(
                start_sec=note.start_sec, end_sec=note.end_sec, midi=note.midi,
                lyric=CONTINUATION, phonemes=[CONTINUATION], velocity=velocity))
            continue

        started.add(note.syllable)
        segments = syllable_segments[note.syllable]
        phonemes = _phonemes_of(segments)
        if not phonemes:
            diagnostics.no_phoneme_notes += 1
        if position < len(morae):
            lyric = morae[position]
        else:
            if lyrics_text is not None:
                diagnostics.notes_beyond_morae += 1
            lyric = _lyric_of(segments, diagnostics)
        position += 1
        result.append(SungNote(
            start_sec=note.start_sec, end_sec=note.end_sec, midi=note.midi, lyric=lyric,
            phonemes=phonemes, velocity=velocity,
            # 音素列を載せた音符は、発音を音声由来の音素列の側で決めさせる。
            is_protected=bool(phonemes)))

    return AnnotationResult(notes=result, diagnostics=diagnostics)
