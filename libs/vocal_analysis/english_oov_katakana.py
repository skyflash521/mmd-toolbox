"""英語未知語カタカナ化フォールバック。

pyopenjtalk-plusが1形態素ノードで完結する未知の英単語を正しく読めない場合(品詞がフィラーと
判定される場合)に、CMUdictで発音記号(ARPAbet)を引き、変換モデル(ENGLISH_OOV_KATAKANA_MODEL)
でカタカナへ補完変換する。CMUdictに無い語・変換結果がひらがな/カタカナ以外を含む場合は変換せず
元のテキストのまま残す(安全側フォールバック)。
"""

import unicodedata

from .config import ENGLISH_OOV_KATAKANA_MODEL

_FULLWIDTH_OFFSET = 0xFEE0
_FULLWIDTH_LO = 0xFF01
_FULLWIDTH_HI = 0xFF5E

_KANA_RANGES = (
    (0x3040, 0x309F),  # ひらがな
    (0x30A0, 0x30FF),  # カタカナ・長音符
)

_PROMPT_TEMPLATE = """英語とその単語単位の音素から、リエゾンを考慮したカタカナと繋がった音素列を生成してください。

英語: take it easy
単語音素: [T EY1 K] [IH1 T] [IY1 Z IY0]
カタカナ: テイキットイージー
繋がった音素: T EY1 K IH1 T IY1 Z IY0

英語: {word}
単語音素: [{phonemes}]
カタカナ: """


def _is_target_node(surface_normalized: str, pos: str) -> bool:
    """NFKC正規化後の表層形・品詞から対象ノードかどうかを判定する。

    ASCII英字のみ(1文字以上)で構成され、かつ品詞がフィラー(pyopenjtalk-plusが正しい読みを
    持たない未知語)であるノードだけを対象とする。品詞が名詞(既に正しく変換済み)・記号(複数
    ノードへ分裂した語の断片)のノードは対象にしない。
    """
    if not surface_normalized or not surface_normalized.isascii() or not surface_normalized.isalpha():
        return False
    return pos == "フィラー"


def _find_target_words(text: str) -> list[str]:
    """テキストを形態素解析し、対象語(NFKC正規化後の表層形)を出現順に列挙する(重複除去しない)。"""
    import pyopenjtalk

    targets = []
    for node in pyopenjtalk.run_frontend(text):
        surface = unicodedata.normalize("NFKC", node["string"])
        if _is_target_node(surface, node["pos"]):
            targets.append(surface)
    return targets


def _lookup_cmudict_phonemes(word: str) -> str | None:
    """CMUdictで英単語の発音記号(ARPAbet)を引く。複数発音があれば先頭を使う。未収録ならNone。"""
    from nltk.corpus import cmudict

    entries = cmudict.dict().get(word.lower())
    if not entries:
        return None
    return " ".join(entries[0])


def _is_valid_katakana(text: str) -> bool:
    """生成結果がひらがな・カタカナのみで構成されるかを判定する(出力妥当性検証)。"""
    if not text:
        return False
    return all(any(lo <= ord(ch) <= hi for lo, hi in _KANA_RANGES) for ch in text)


_katakana_model_cache = None


def _load_katakana_model():
    """変換モデルをロードする(プロセス内キャッシュ)。"""
    global _katakana_model_cache
    if _katakana_model_cache is not None:
        return _katakana_model_cache

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ENGLISH_OOV_KATAKANA_MODEL.model_id, revision=ENGLISH_OOV_KATAKANA_MODEL.model_revision
    )
    model = AutoModelForCausalLM.from_pretrained(
        ENGLISH_OOV_KATAKANA_MODEL.model_id,
        revision=ENGLISH_OOV_KATAKANA_MODEL.model_revision,
        dtype=torch.float32,
    )
    model.eval()
    _katakana_model_cache = (tokenizer, model)
    return _katakana_model_cache


def _generate_katakana(word: str, phonemes: str) -> str:
    """英単語・発音記号からカタカナを生成する(変換モデル呼び出し)。"""
    import torch

    tokenizer, model = _load_katakana_model()
    prompt = _PROMPT_TEMPLATE.format(word=word, phonemes=phonemes)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.split("\n")[0].strip()


def _to_halfwidth(text: str) -> str:
    """全角ラテン/数字/記号ブロック(Unicode fullwidth)を半角化する(1文字ずつの固定オフセット変換)。"""
    return "".join(
        chr(ord(ch) - _FULLWIDTH_OFFSET) if _FULLWIDTH_LO <= ord(ch) <= _FULLWIDTH_HI else ch
        for ch in text
    )


def _to_fullwidth(text: str) -> str:
    """半角ASCII英字を全角(Unicode fullwidth)へ変換する(_to_halfwidthの逆変換)。"""
    return "".join(chr(ord(ch) + _FULLWIDTH_OFFSET) for ch in text)


def _locate_and_replace(text: str, target_words: list[str], converted: dict[str, str]) -> str:
    """対象語を元テキスト中で特定し、変換に成功した語(convertedに値がある語)だけを置換した
    文字列を返す。

    各対象語は、半角表記(target_wordsはNFKC正規化済みのため常に半角)・全角表記(元テキスト自体が
    全角の場合に対応する)の両方でcursor以降を検索し、両方見つかった場合はより手前(cursorに近い)
    側を採る(元テキスト中に半角表記と全角表記の出現が混在する場合、常に半角を優先すると、cursorの
    直後にある全角の出現を飛び越して後方の半角の出現を誤って拾うため)。位置が特定できた対象語
    (converted に無い語も含む)は次の対象語の検索開始位置(cursor)を前進させる(位置特定できない
    語だけカーソルを進めないと、後続の対象語の検索がその語より前の位置から始まり誤った箇所を
    見つけうるため、変換の成否に関わらず位置特定できた時点でカーソルを進める)。converted に
    値が無い語(CMUdict未収録・生成結果が不正)は置換対象に加えず元のまま残す。まず全対象語の
    (開始位置, 終了位置, 変換後カタカナ)を先に求めてから(この走査中はtextを書き換えない)、
    1回の走査で最終的な文字列を組み立てる。
    """
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for word in target_words:
        candidates = [
            (index, needle)
            for index, needle in (
                (text.find(word, cursor), word),
                (text.find(_to_fullwidth(word), cursor), _to_fullwidth(word)),
            )
            if index != -1
        ]
        if not candidates:
            continue
        index, needle = min(candidates, key=lambda candidate: candidate[0])
        cursor = index + len(needle)
        katakana = converted.get(word)
        if katakana is not None:
            spans.append((index, index + len(needle), katakana))

    if not spans:
        return text

    result = []
    prev_end = 0
    for start, end, katakana in spans:
        result.append(text[prev_end:start])
        result.append(katakana)
        prev_end = end
    result.append(text[prev_end:])
    return "".join(result)


def convert_oov_words(text: str) -> str:
    """テキスト中の対象語(pyopenjtalkが読めない英単語)をカタカナへ変換する。対象語が無ければ
    元のテキストをそのまま返す。
    """
    target_words = _find_target_words(text)
    if not target_words:
        return text

    converted: dict[str, str] = {}
    for word in dict.fromkeys(target_words):
        phonemes = _lookup_cmudict_phonemes(word)
        if phonemes is None:
            continue
        katakana = _generate_katakana(word, phonemes)
        if _is_valid_katakana(katakana):
            converted[word] = katakana

    return _locate_and_replace(text, target_words, converted)
