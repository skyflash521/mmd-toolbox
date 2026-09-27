import unicodedata
from collections.abc import Callable

from .config import (
    DEFAULT_ENGLISH_KATAKANA_METHOD,
    ENGLISH_KATAKANA_MODEL,
    EnglishKatakanaMethod,
)
from .quiet import silence_third_party_output, suppress_native_stderr

_FULLWIDTH_OFFSET = 0xFEE0
_FULLWIDTH_LO = 0xFF01
_FULLWIDTH_HI = 0xFF5E

_HIRAGANA_RANGE = (0x3040, 0x309F)
_KATAKANA_RANGE = (0x30A0, 0x30FF)
_KANA_RANGES = (_HIRAGANA_RANGE, _KATAKANA_RANGE)

_PROMPT_TEMPLATE = """英語とその単語単位の音素から、リエゾンを考慮したカタカナと繋がった音素列を生成してください。

英語: take it easy
単語音素: [T EY1 K] [IH1 T] [IY1 Z IY0]
カタカナ: テイキットイージー
繋がった音素: T EY1 K IH1 T IY1 Z IY0

英語: {word}
単語音素: [{phonemes}]
カタカナ: """


def _is_target_node(surface_normalized: str, pos: str) -> bool:
    if not surface_normalized or not surface_normalized.isascii() or not surface_normalized.isalpha():
        return False
    # pyopenjtalk-plus は読みを持たない英単語の品詞をフィラーにする。
    return pos == "フィラー"


def _find_target_words(text: str) -> list[str]:
    import pyopenjtalk

    targets = []
    with suppress_native_stderr():
        nodes = pyopenjtalk.run_frontend(text)
    for node in nodes:
        surface = unicodedata.normalize("NFKC", node["string"])
        if _is_target_node(surface, node["pos"]):
            targets.append(surface)
    return targets


_cmudict_cache = None


def _get_cmudict_entries():
    global _cmudict_cache
    if _cmudict_cache is None:
        import nltk
        from nltk.corpus import cmudict

        # nltk の cmudict.dict() は呼ぶたびに辞書全体を作り直す。
        try:
            _cmudict_cache = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            _cmudict_cache = cmudict.dict()
    return _cmudict_cache


def _lookup_cmudict_phonemes(word: str) -> str | None:
    entries = _get_cmudict_entries().get(word.lower())
    if not entries:
        return None
    return " ".join(entries[0])


def _is_valid_katakana(text: str) -> bool:
    if not text:
        return False
    return all(any(lo <= ord(ch) <= hi for lo, hi in _KANA_RANGES) for ch in text)


def _select_katakana_model_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


_katakana_model_cache = None


def _hf_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, revision=revision, tqdm_class=tqdm_class)


def _prefetch_with_progress(repo_id: str, revision: str | None, on_progress: Callable[[str], None]) -> bool:
    """戻り値はバイト転送が実際に起きたか。"""
    from huggingface_hub.utils import tqdm as hf_tqdm

    state = {"shown": False}

    class _RelayTqdm(hf_tqdm):
        def __init__(self, *args, **kwargs):
            self._relay_unit = kwargs.get("unit")
            self._relay_n = kwargs.get("initial") or 0
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            result = super().update(n)
            # snapshot_download はファイル数を数えるバーと、バイト数を数える unit="B" のバーを作る。
            if self._relay_unit == "B" and self.total:
                self._relay_n += n or 0
                state["shown"] = True
                percent = min(100, int(self._relay_n * 100 / self.total))
                on_progress(f"ダウンロード中: {repo_id} {percent}%")
            return result

    _hf_snapshot_download(repo_id, revision=revision, tqdm_class=_RelayTqdm)
    return state["shown"]


def _load_katakana_model(on_progress: Callable[[str], None] | None = None):
    global _katakana_model_cache
    if _katakana_model_cache is not None:
        return _katakana_model_cache

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    silence_third_party_output()

    downloaded = False
    if on_progress is not None:
        downloaded = _prefetch_with_progress(
            ENGLISH_KATAKANA_MODEL.model_id, ENGLISH_KATAKANA_MODEL.model_revision, on_progress)

    device = _select_katakana_model_device()

    try:
        if on_progress is not None:
            on_progress(f"カタカナ生成モデル読み込み中: {ENGLISH_KATAKANA_MODEL.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            ENGLISH_KATAKANA_MODEL.model_id, revision=ENGLISH_KATAKANA_MODEL.model_revision
        )
        # CPU 実行では fp16 を使えない。
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            ENGLISH_KATAKANA_MODEL.model_id,
            revision=ENGLISH_KATAKANA_MODEL.model_revision,
            dtype=dtype,
        )
    finally:
        if downloaded:
            on_progress("")
    model.to(device)
    model.eval()
    _katakana_model_cache = (tokenizer, model)
    return _katakana_model_cache


def _generate_katakana_tinyllama(
    word: str, phonemes: str, on_progress: Callable[[str], None] | None = None
) -> str:
    import torch

    tokenizer, model = _load_katakana_model(on_progress=on_progress)
    try:
        prompt = _PROMPT_TEMPLATE.format(word=word, phonemes=phonemes)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated.split("\n")[0].strip()
    except Exception:
        return ""


def _generate_katakana_arpakana(word: str, phonemes: str) -> str:
    from arpakana import arpabet_to_kana

    try:
        return arpabet_to_kana(phonemes)
    except Exception:
        return ""


def _generate_katakana(
    word: str, phonemes: str, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    if method == "arpakana":
        return _generate_katakana_arpakana(word, phonemes)
    if method == "tinyllama-katakana-converter":
        return _generate_katakana_tinyllama(word, phonemes, on_progress=on_progress)
    # Literal 型の引数は実行時に値を検査されない。
    raise ValueError(f"未知の変換方式です: {method!r}")


def _to_halfwidth(text: str) -> str:
    return "".join(
        chr(ord(ch) - _FULLWIDTH_OFFSET) if _FULLWIDTH_LO <= ord(ch) <= _FULLWIDTH_HI else ch
        for ch in text
    )


def _to_fullwidth(text: str) -> str:
    return "".join(chr(ord(ch) + _FULLWIDTH_OFFSET) for ch in text)


def _locate_and_replace(text: str, target_words: list[str], converted: dict[str, str]) -> str:
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


def release_katakana_model() -> None:
    global _katakana_model_cache
    if _katakana_model_cache is None:
        return
    _katakana_model_cache = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def uncached_target_words(
    text: str, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD
) -> list[str]:
    """空リストなら、同じ text と method で convert_target_words を呼んでも変換モデルはロードされない。"""
    return [
        word
        for word in dict.fromkeys(_find_target_words(text))
        if (method, word) not in _conversion_cache
    ]


def convert_words(
    words: list[str], method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """method が "tinyllama-katakana-converter" なら、変換し終えた時点で変換モデルを解放する。"""
    try:
        for word in words:
            _convert_word(word, method, on_progress=on_progress)
    finally:
        if method == "tinyllama-katakana-converter":
            release_katakana_model()


def convert_target_words(
    text: str, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    target_words = _find_target_words(text)
    if not target_words:
        return text

    converted: dict[str, str] = {}
    for word in dict.fromkeys(target_words):
        result = _convert_word(word, method, on_progress=on_progress)
        if result is not None:
            converted[word] = result

    return _locate_and_replace(text, target_words, converted)


_conversion_cache: dict[tuple[EnglishKatakanaMethod, str], str | None] = {}


def _convert_word(
    word: str, method: EnglishKatakanaMethod, on_progress: Callable[[str], None] | None = None
) -> str | None:
    cache_key = (method, word)
    if cache_key in _conversion_cache:
        return _conversion_cache[cache_key]

    result = None
    phonemes = _lookup_cmudict_phonemes(word)
    if phonemes is not None:
        katakana = _generate_katakana(word, phonemes, method=method, on_progress=on_progress)
        if _is_valid_katakana(katakana):
            result = katakana
    _conversion_cache[cache_key] = result
    return result
