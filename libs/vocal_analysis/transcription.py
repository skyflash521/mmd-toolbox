import contextlib
from collections.abc import Callable, Iterable

import numpy as np

from . import english_katakana, model_store
from .config import (
    DEFAULT_ENGLISH_KATAKANA_METHOD,
    KANA_PROMPT,
    RECOGNIZER_CONFIG,
    ContentRecognizerModel,
    EnglishKatakanaMethod,
)
from .phonemes import _MIN_WORD_DURATION_SEC, RecognitionError
from .quiet import silenced_third_party_output, suppress_native_stderr

DENSITY_G2P_METHOD: EnglishKatakanaMethod = "arpakana"

# Whisper はプロンプトと生成を合わせたトークン数が最大コンテキスト長に収まる必要がある。
SEGMENT_MAX_NEW_TOKENS = 380
RETRY_MAX_NEW_TOKENS = 420


def load_pipeline(
    content_recognizer_model: ContentRecognizerModel,
    on_progress: Callable[[str], None] | None = None,
):
    """呼ぶたびに新しいパイプラインを作る。保持と解放は呼び出し側が行う。"""
    import transformers

    device = model_store.select_device()
    dtype = model_store.select_dtype(device)

    def build():
        return transformers.pipeline(
            "automatic-speech-recognition",
            model=content_recognizer_model.model_id,
            revision=content_recognizer_model.model_revision,
            device=device,
            dtype=dtype,
        )

    return model_store.load_with_progress(
        content_recognizer_model.model_id, content_recognizer_model.model_revision,
        on_progress=on_progress,
        loading_note=f"内容認識モデル読み込み中: {content_recognizer_model.model_id}",
        build=build,
    )


def sanitize_word_timestamps(
    words: list[tuple[str, float, float]], duration_sec: float
) -> list[tuple[str, float, float]]:
    sanitized: list[tuple[str, float, float]] = []
    prev_start = 0.0
    for text, start, end in words:
        start = min(max(start, 0.0), duration_sec)
        end = min(max(end, 0.0), duration_sec)
        start = max(start, prev_start)
        end = max(end, start + _MIN_WORD_DURATION_SEC)
        sanitized.append((text, start, end))
        prev_start = start
    return sanitized


def extract_word_timestamps(chunks: list[dict], duration_sec: float) -> list[tuple[str, float, float]]:
    words: list[tuple[str, float, float]] = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        start, end = chunk.get("timestamp", (None, None))
        if not text or start is None:
            continue
        end_sec = float(end) if end is not None else float(start) + _MIN_WORD_DURATION_SEC
        words.append((text, float(start), end_sec))
    return sanitize_word_timestamps(words, duration_sec)


def transcribe_text_only(pipeline, samples: np.ndarray) -> str:
    # transformers の Whisper は temperature を単一の値で受けると温度フォールバックを行わない。
    generate_kwargs = {
        "language": "japanese", "task": "transcribe",
        "num_beams": 1, "do_sample": False, "temperature": 0.0,
        "max_new_tokens": RETRY_MAX_NEW_TOKENS,
    }
    with silenced_third_party_output():
        result = pipeline(samples, generate_kwargs=generate_kwargs)
    return result["text"]


def transcribe_segment(
    pipeline, samples: np.ndarray
) -> tuple[str, list[tuple[str, float, float]] | None]:
    prompt_ids = pipeline.tokenizer.get_prompt_ids(KANA_PROMPT, return_tensors="pt").to(pipeline.device)
    with silenced_third_party_output():
        result = pipeline(
            samples,
            return_timestamps="word",
            generate_kwargs={
                "language": "japanese",
                "task": "transcribe",
                "num_beams": 1,
                "do_sample": False,
                "prompt_ids": prompt_ids,
                "max_new_tokens": SEGMENT_MAX_NEW_TOKENS,
            },
        )
    chunks = result.get("chunks")
    if not chunks:
        return result["text"], None
    duration_sec = len(samples) / RECOGNIZER_CONFIG.sample_rate
    return result["text"], extract_word_timestamps(chunks, duration_sec)


@contextlib.contextmanager
def _katakana_conversion_errors():
    try:
        yield
    except ImportError as e:
        raise RecognitionError(
            "arpakana、nltk、または transformers/torch が見つかりません。導入してください"
            "(vocal-analysis extra で導入されます)。"
        ) from e
    except LookupError as e:
        raise RecognitionError(
            "CMUdict(nltkのcmudictコーパス)が見つかりません。ネットワーク接続を確認してください。"
        ) from e
    except OSError as e:
        raise RecognitionError(
            "カタカナ生成モデルを取得できません。ネットワーク接続を確認してください。"
        ) from e


def prepare_dictionary(on_progress: Callable[[str], None] | None = None) -> None:
    """英単語の変換に使う発音辞書を使える状態にする。失敗は RecognitionError にする。"""
    with _katakana_conversion_errors():
        try:
            english_katakana.ensure_cmudict(on_progress)
        except OSError as e:
            raise RecognitionError("CMUdict を取得できません。ネットワーク接続を確認してください。") from e


def convert_english_words(
    texts: Iterable[str], method: EnglishKatakanaMethod,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """tinyllama 方式のとき、texts に含まれる未変換の英単語を変換モデルを1回ロードして変換し、変換後に
    モデルを解放する。他の方式では何もしない。"""
    if method != "tinyllama-katakana-converter":
        return
    with _katakana_conversion_errors():
        words = list(dict.fromkeys(
            word for text in texts for word in english_katakana.uncached_target_words(text, method)))
        if not words:
            return
        try:
            english_katakana.convert_words(words, method, on_progress=on_progress, keep_model=True)
        finally:
            english_katakana.release_katakana_model()


def convert_with_g2p(
    text: str, *, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None, kana: bool = False,
):
    """kana が真ならかな読みの文字列、偽なら音素記号のリストを返す。"""
    import pyopenjtalk

    convert_english_words([text], method, on_progress=on_progress)
    with _katakana_conversion_errors():
        text = english_katakana.convert_target_words(text, method=method, on_progress=on_progress)

    # pyopenjtalk のネイティブ拡張は Python のログ設定を経由せず標準エラーへ直接書く。
    with suppress_native_stderr():
        if kana:
            return pyopenjtalk.g2p(text, kana=True)
        return pyopenjtalk.g2p(text, kana=False, join=False)


def g2p(
    text: str, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> list[str]:
    return convert_with_g2p(text, method=method, on_progress=on_progress, kana=False)
