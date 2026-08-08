"""song2vpr の表示歌詞・音素列・強弱の付与。

音符の区間と音高は前段が確定させているので、ここは各音符へ載せる値だけを決める。意味のある
歌詞の書き起こしはしない(音素セグメントから得られる範囲に留める)。
"""

import unicodedata
from dataclasses import dataclass, field

import numpy as np

from vocal_analysis.phonemes import espeak_ipa_to_vowel

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

# 5母音から表示歌詞の仮名へ。音素記号から5母音への写像は共有側が持つので、ここはその先だけを持つ。
_VOWEL_KANA = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}

# 鼻音の音素記号。母音が無い音符でこれらが主体なら撥音として扱う。
_NASALS = frozenset({"m", "mʲ", "n", "ɲ", "ɴ"})

# 母音も鼻音も得られない音符へ入れる仮名。
_FALLBACK_KANA = "あ"

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


@dataclass
class Diagnostics:
    """付与の過程で数えた件数。利用者向けの診断と警告へ出す。

    unconverted_chars は読みに残った「かなへ変換されるべきだった文字」の数、counted_chars は判定に
    使った文字数(かな・長音記号と、かなへ変換されるべきだった文字の合計。体裁の文字は含めない)。
    """

    undetermined_vowel_notes: int = 0
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


def _overlap_sec(segment, start_sec, end_sec):
    return max(0.0, min(segment.end_sec, end_sec) - max(segment.start_sec, start_sec))


def _overlapping(segments, start_sec, end_sec):
    return [s for s in segments if _overlap_sec(s, start_sec, end_sec) > 0.0]


def _phonemes_of(segments, start_sec, end_sec):
    """音符区間に重なる音素セグメントを時間順に並べ、vpr の音素表現へ写す。

    音素ラベルを持たないセグメントと gap は除外する(除外で空になっても許容する)。
    """
    overlapping = sorted(_overlapping(segments, start_sec, end_sec), key=lambda s: s.start_sec)
    return [IPA_TO_VPR_PHONEME[s.phoneme] for s in overlapping
            if s.phoneme is not None and s.phoneme in IPA_TO_VPR_PHONEME]


def _lyric_from_segments(segments, start_sec, end_sec, diagnostics):
    """音符の代表母音から表示歌詞を決める。母音が無ければ鼻音の主体で撥音を判定する。"""
    overlapping = _overlapping(segments, start_sec, end_sec)

    vowels = [s for s in overlapping
              if s.type == "vowel" and espeak_ipa_to_vowel(s.phoneme or "") is not None]
    if vowels:
        longest = max(vowels, key=lambda s: _overlap_sec(s, start_sec, end_sec))
        return _VOWEL_KANA[espeak_ipa_to_vowel(longest.phoneme)]

    nasal = sum(_overlap_sec(s, start_sec, end_sec) for s in overlapping
                if s.type == "consonant" and s.phoneme in _NASALS)
    other = sum(_overlap_sec(s, start_sec, end_sec) for s in overlapping
                if s.type == "consonant" and s.phoneme not in _NASALS)
    if nasal > other:
        return _MORAIC_NASAL_KANA

    diagnostics.undetermined_vowel_notes += 1
    return _FALLBACK_KANA


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


def annotate(source_notes, segments, rms, *, lyrics_text=None) -> AnnotationResult:
    """音符へ表示歌詞・音素列・強弱を載せる。

    lyrics_text を与えると、かな読みへ変換してモーラへ分け、先頭から 1音符=1モーラで割り当てる。
    音符数とモーラ数が食い違うぶんは診断へ数える(下書きなので厳密な整合は求めない)。
    """
    diagnostics = Diagnostics()

    morae = []
    if lyrics_text is not None:
        # 読みへの変換は追加依存を要するので、歌詞を与えられたときだけ取り込む。
        from vocal_analysis.reading import to_kana_reading

        morae = _split_morae(_to_kana_tokens(to_kana_reading(lyrics_text), diagnostics))
        if len(morae) > len(source_notes):
            diagnostics.discarded_morae = len(morae) - len(source_notes)

    result = []
    for position, note in enumerate(source_notes):
        if position < len(morae):
            lyric = morae[position]
        else:
            if lyrics_text is not None:
                diagnostics.notes_beyond_morae += 1
            lyric = _lyric_from_segments(segments, note.start_sec, note.end_sec, diagnostics)
        result.append(SungNote(
            start_sec=note.start_sec, end_sec=note.end_sec, midi=note.midi, lyric=lyric,
            phonemes=_phonemes_of(segments, note.start_sec, note.end_sec),
            velocity=_velocity_of(rms, note.start_sec, note.end_sec)))

    return AnnotationResult(notes=result, diagnostics=diagnostics)
