import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import sofa_align
from .config import (
    DEFAULT_CONTENT_RECOGNIZER_MODEL,
    DEFAULT_ENGLISH_KATAKANA_METHOD,
    DEFAULT_FORCED_ALIGNER,
    KANA_PROMPT,
    RECOGNIZER_CONFIG,
    ContentRecognizerModel,
    EnglishKatakanaMethod,
    ForcedAlignerId,
    SofaAlignerConfig,
)
from .english_katakana import convert_target_words, convert_words, uncached_target_words
from .phonemes import (
    _BLANK_G2P_SYMBOLS,
    _G2P_TO_VOCAB_SYMBOL,
    _MIN_WORD_DURATION_SEC,
    RecognitionError,
    _classify_symbol,
)
from .quiet import silence_third_party_output, suppress_native_stderr
from .types import Segment

# 採用した音素モデルの畳み込み層の総ストライドは 320 サンプルで、16kHz 入力の1フレームは 20ms になる。
FRAME_DURATION_SEC = 0.02

_SILENCE_FRAME_SEC = 0.1
_SILENCE_THRESHOLD_DB_BELOW_PEAK = 30.0
_SILENCE_MIN_RUN_SEC = 0.6
_SEGMENT_MIN_SEC = 1.5
_SEGMENT_MAX_SEC = 25.0
_TRIM_MARGIN_SEC = 0.1


def _frame_rms(mono: np.ndarray, sample_rate: int, frame_sec: float) -> np.ndarray:
    frame_len = max(1, round(frame_sec * sample_rate))
    num_frames = len(mono) // frame_len
    if num_frames == 0:
        return np.array([], dtype=np.float64)
    trimmed = mono[: num_frames * frame_len].astype(np.float64).reshape(num_frames, frame_len)
    return np.sqrt(np.mean(np.square(trimmed), axis=1))


def _silence_threshold(frame_rms: np.ndarray) -> float:
    if frame_rms.size == 0:
        return 0.0
    peak = float(np.percentile(frame_rms, 95))
    return peak * (10 ** (-_SILENCE_THRESHOLD_DB_BELOW_PEAK / 20.0))


def _detect_silence_split_points(mono: np.ndarray, sample_rate: int) -> list[float]:
    frame_rms = _frame_rms(mono, sample_rate, _SILENCE_FRAME_SEC)
    if frame_rms.size == 0:
        return []
    threshold = _silence_threshold(frame_rms)
    min_run_frames = max(1, round(_SILENCE_MIN_RUN_SEC / _SILENCE_FRAME_SEC))

    split_points: list[float] = []
    run_start: int | None = None
    for i, value in enumerate(frame_rms):
        is_silent = value <= threshold
        if is_silent and run_start is None:
            run_start = i
        elif not is_silent and run_start is not None:
            if i - run_start >= min_run_frames:
                split_points.append((run_start + i) / 2 * _SILENCE_FRAME_SEC)
            run_start = None
    if run_start is not None and frame_rms.size - run_start >= min_run_frames:
        split_points.append((run_start + frame_rms.size) / 2 * _SILENCE_FRAME_SEC)
    return split_points


def _build_segment_bounds(duration_sec: float, split_points: list[float]) -> list[tuple[float, float]]:
    boundaries = [0.0, *sorted(split_points), duration_sec]
    last_index = len(boundaries) - 1
    merged: list[tuple[float, float]] = []
    start = boundaries[0]
    for index in range(1, len(boundaries)):
        end = boundaries[index]
        if end - start >= _SEGMENT_MIN_SEC or index == last_index:
            merged.append((start, end))
            start = end

    result: list[tuple[float, float]] = []
    for seg_start, seg_end in merged:
        span = seg_end - seg_start
        if span <= _SEGMENT_MAX_SEC:
            result.append((seg_start, seg_end))
            continue
        piece_count = math.ceil(span / _SEGMENT_MAX_SEC)
        piece_len = span / piece_count
        result.extend(
            (seg_start + i * piece_len, seg_start + (i + 1) * piece_len) for i in range(piece_count)
        )
    return result


def _format_time_range(start_sec: float, end_sec: float) -> str:

    def _mmss(sec: float) -> str:
        total = int(sec)
        return f"{total // 60}:{total % 60:02d}"

    return f"{_mmss(start_sec)}-{_mmss(end_sec)}"


def _is_segment_silent(segment_samples: np.ndarray, threshold: float) -> bool:
    if segment_samples.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(segment_samples.astype(np.float64)))))
    return rms <= threshold


def _voiced_trim_bounds(segment_samples: np.ndarray, sample_rate: int, threshold: float) -> tuple[int, int]:
    frame_rms = _frame_rms(segment_samples, sample_rate, _SILENCE_FRAME_SEC)
    voiced = np.nonzero(frame_rms > threshold)[0]
    if voiced.size == 0:
        return 0, len(segment_samples)
    frame_len = max(1, round(_SILENCE_FRAME_SEC * sample_rate))
    margin = round(_TRIM_MARGIN_SEC * sample_rate)
    lo = max(0, int(voiced[0]) * frame_len - margin)
    hi = min(len(segment_samples), (int(voiced[-1]) + 1) * frame_len + margin)
    return lo, hi


_HALLUCINATION_PHONEME_RATE = 20.0
_ECHO_FRAGMENT_PREFIX_LEN = 5
_REPEAT_MIN_COUNT = 3
_REPEAT_MIN_TOTAL_CHARS = 8
_REPEAT_NORMALIZE_MORA_RATE = 3.5
_MORA_G2P_SYMBOLS = frozenset({"a", "i", "u", "e", "o", "I", "U", "N"})
# Whisper はプロンプトと生成を合わせたトークン数が最大コンテキスト長に収まる必要がある。
_SEGMENT_MAX_NEW_TOKENS = 380
_RETRY_MAX_NEW_TOKENS = 420


def _normalize_sentence(sentence: str) -> str:
    return re.sub(r"[\s、。,.]", "", sentence)


_PROMPT_SENTENCES = frozenset(
    normalized for normalized in (_normalize_sentence(part) for part in KANA_PROMPT.split("。")) if normalized
)


def _is_hallucinated_phoneme_density(phoneme_count: int, duration_sec: float) -> bool:
    if duration_sec <= 0:
        return False
    return phoneme_count / duration_sec > _HALLUCINATION_PHONEME_RATE


def _strip_prompt_echo(text: str) -> tuple[str, bool]:
    """戻り値は (除去後のテキスト, 1文以上除去したか)。"""
    parts = re.split(r"(?<=。)", text)
    kept = [part for part in parts if _normalize_sentence(part) not in _PROMPT_SENTENCES]
    removed = len(kept) < len(parts)
    if removed and kept:
        tail = _normalize_sentence(kept[-1])
        if tail and any(_common_prefix_len(tail, prompt) >= _ECHO_FRAGMENT_PREFIX_LEN
                        for prompt in _PROMPT_SENTENCES):
            kept.pop()
    return "".join(kept), removed


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        n += 1
    return n


def _find_suffix_repetition(text: str) -> tuple[str, int, str] | None:
    """戻り値は (反復の単位, 繰り返し数, 反復を除いた先頭部)。末尾に反復が無ければ None。"""
    best: tuple[str, int, str] | None = None
    n = len(text)
    for unit_len in range(1, n // _REPEAT_MIN_COUNT + 1):
        unit = text[n - unit_len:]
        count = 1
        while n - (count + 1) * unit_len >= 0 and text[n - (count + 1) * unit_len: n - count * unit_len] == unit:
            count += 1
        total = unit_len * count
        if count >= _REPEAT_MIN_COUNT and total >= _REPEAT_MIN_TOTAL_CHARS:
            if best is None or total > len(best[0]) * best[1]:
                best = (unit, count, text[: n - total])
    return best


def _text_mora_count(
    text: str, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    return sum(1 for symbol in _g2p(text, method=method, on_progress=on_progress) if symbol in _MORA_G2P_SYMBOLS)


def _text_phoneme_density(
    text: str, duration_sec: float, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> float:
    if not text.strip() or duration_sec <= 0:
        return 0.0
    return len(_g2p(text, method=method, on_progress=on_progress)) / duration_sec


def _resolve_transcription(
    chunk_samples: np.ndarray,
    trim_duration_sec: float,
    text: str,
    words: list[tuple[str, float, float]] | None,
    content_recognizer_model: ContentRecognizerModel,
    retry_enabled: bool,
    english_katakana_method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[str, list[tuple[str, float, float]] | None]:
    """戻り値は採用する (テキスト, 単語タイムスタンプ)。テキストを変えたときは単語タイムスタンプを None にする。"""
    text2, echo_removed = _strip_prompt_echo(text)
    if echo_removed:
        words = None

    def _needs_retry(candidate: str) -> bool:
        if not candidate.strip():
            return echo_removed
        density = _text_phoneme_density(
            candidate, trim_duration_sec, method=english_katakana_method, on_progress=on_progress)
        return density > _HALLUCINATION_PHONEME_RATE

    if retry_enabled and _needs_retry(text2):
        retry_text = _transcribe_text_only(
            chunk_samples, content_recognizer_model, on_progress=on_progress).strip()
        retry_text, _retry_echo_removed = _strip_prompt_echo(retry_text)
        text2, words = retry_text, None

    density = _text_phoneme_density(
        text2, trim_duration_sec, method=english_katakana_method, on_progress=on_progress)
    if text2.strip() and density > _HALLUCINATION_PHONEME_RATE:
        repetition = _find_suffix_repetition(text2)
        if repetition is not None:
            unit, count, head = repetition
            rescued = text2
            if head.strip():
                rescued = head
            else:
                unit_moras = max(
                    1, _text_mora_count(unit, method=english_katakana_method, on_progress=on_progress))
                normalized = max(1, round(trim_duration_sec * _REPEAT_NORMALIZE_MORA_RATE / unit_moras))
                if normalized < count:
                    rescued = unit * normalized
            if rescued != text2:
                text2 = rescued
                words = None

    return text2, words


def _assemble_phoneme_sequence(chunk_phonemes: list[list[str]]) -> list[str]:
    sequence = ["pau"]
    for i, phonemes in enumerate(chunk_phonemes):
        if i > 0:
            sequence.append("pau")
        sequence.extend(phonemes)
    sequence.append("pau")
    return sequence


_CRAMPED_WORD_THRESHOLD_SEC = 0.03
_CRAMPED_WORD_TARGET_SEC = 0.04
_CRAMPED_WORD_EPSILON_SEC = 1e-9


def _widen_cramped_words(
    words_phonemes: list[tuple[list[str], float, float]], duration_sec: float
) -> list[tuple[list[str], float, float]]:
    widened = []
    for phonemes, start, end in words_phonemes:
        threshold = len(phonemes) * _CRAMPED_WORD_THRESHOLD_SEC - _CRAMPED_WORD_EPSILON_SEC
        if phonemes and (end - start) < threshold:
            needed = len(phonemes) * _CRAMPED_WORD_TARGET_SEC
            end = min(end, duration_sec)
            start = max(0.0, end - needed)
            end = min(duration_sec, start + needed)
        widened.append((phonemes, start, end))
    return widened


def _assemble_with_word_windows(
    words_phonemes: list[tuple[list[str], float, float]], margin_sec: float, duration_sec: float
) -> tuple[list[str], list[tuple[float, float]]]:
    """words_phonemes の各要素は (音素記号列, 開始秒, 終了秒)。戻り値は (記号列, 各記号の (下端秒, 上端秒))。"""

    def ordered(lo: float, hi: float) -> tuple[float, float]:
        return (lo, hi) if lo <= hi else (hi, lo)

    first_start = words_phonemes[0][1]
    sequence = ["pau"]
    windows: list[tuple[float, float]] = [ordered(0.0, first_start + margin_sec)]

    for index, (phonemes, start, end) in enumerate(words_phonemes):
        word_window = ordered(start - margin_sec, end + margin_sec)
        sequence.extend(phonemes)
        windows.extend([word_window] * len(phonemes))
        if index + 1 < len(words_phonemes):
            next_start = words_phonemes[index + 1][1]
            sequence.append("pau")
            windows.append(ordered(end - margin_sec, next_start + margin_sec))

    last_end = words_phonemes[-1][2]
    sequence.append("pau")
    windows.append(ordered(last_end - margin_sec, duration_sec))

    return sequence, windows


def _g2p_symbols_to_token_ids(symbols: list[str], vocab: dict[str, int], blank_token_id: int) -> list[int]:
    token_ids = []
    for symbol in symbols:
        if symbol in _BLANK_G2P_SYMBOLS:
            token_ids.append(blank_token_id)
            continue
        vocab_symbol = _G2P_TO_VOCAB_SYMBOL.get(symbol)
        if vocab_symbol is None:
            raise RecognitionError(f"G2P記号 '{symbol}' の音素モデル語彙への写像が未定義です")
        token_id = vocab.get(vocab_symbol)
        if token_id is None:
            raise RecognitionError(f"音素モデルの語彙に記号 '{vocab_symbol}' が見つかりません")
        token_ids.append(token_id)
    return token_ids


_FORCED_ALIGN_BAND_SEC = 1.0
_MIN_STAY_FRAMES = 6
_VOICED_BLANK_PENALTY = 7.0
_WORD_WINDOW_MARGIN_SEC = 0.75
_EARLY_COMMIT_BONUS = 2.0
_MIN_WINDOW_WIDTH_SEC = 1e-9


def _apply_voiced_blank_penalty(
    log_probs: np.ndarray, chunk_samples: np.ndarray, threshold: float, blank_token_id: int
) -> np.ndarray:
    frame_rms = _frame_rms(chunk_samples, RECOGNIZER_CONFIG.sample_rate, FRAME_DURATION_SEC)
    n = min(len(frame_rms), log_probs.shape[0])
    penalized = log_probs.copy()
    voiced = frame_rms[:n] > threshold
    penalized[:n][voiced, blank_token_id] -= _VOICED_BLANK_PENALTY
    return penalized


def _expand_min_stay(
    token_ids: list[int], blank_token_id: int, num_frames: int
) -> tuple[list[int], list[int]]:
    """戻り値は (展開後のトークン ID 列, 各サブ状態が属する元トークンの index 列)。"""
    num_blank = sum(1 for tid in token_ids if tid == blank_token_id)
    num_phoneme = len(token_ids) - num_blank
    stay = _MIN_STAY_FRAMES
    if num_phoneme > 0:
        stay = min(stay, max(1, (num_frames - num_blank) // num_phoneme))
    sub_token_ids: list[int] = []
    sub_to_token: list[int] = []
    for index, tid in enumerate(token_ids):
        reps = 1 if tid == blank_token_id else stay
        sub_token_ids.extend([tid] * reps)
        sub_to_token.extend([index] * reps)
    return sub_token_ids, sub_to_token


def _expand_min_stay_local(
    token_ids: list[int],
    words_phonemes: list[tuple[list[str], float, float]],
    blank_token_id: int,
) -> tuple[list[int], list[int]]:
    """token_ids は words_phonemes から _assemble_with_word_windows が組んだ記号列を変換したもの。
    戻り値は (展開後のトークン ID 列, 各サブ状態が属する元トークンの index 列)。"""
    pau_stay = 1
    stays = [pau_stay]
    for i, (phonemes, start, end) in enumerate(words_phonemes):
        word_frames = (end - start) / FRAME_DURATION_SEC
        # round() は偶数丸めで、ちょうど .5 の商を切り捨てることがある。
        stay = min(_MIN_STAY_FRAMES, max(1, int(word_frames / len(phonemes) + 0.5))) if phonemes else 1
        stays.extend([stay] * len(phonemes))
        if i + 1 < len(words_phonemes):
            stays.append(pau_stay)
    stays.append(pau_stay)

    sub_token_ids: list[int] = []
    sub_to_token: list[int] = []
    for index, (tid, stay) in enumerate(zip(token_ids, stays, strict=True)):
        reps = 1 if tid == blank_token_id else stay
        sub_token_ids.extend([tid] * reps)
        sub_to_token.extend([index] * reps)
    return sub_token_ids, sub_to_token


def _viterbi_monotonic(
    log_probs: np.ndarray,
    token_ids: list[int],
    out_of_bounds: np.ndarray,
    state_bias: np.ndarray | None = None,
) -> list[int] | None:
    """out_of_bounds[t, l] が真のフレーム t には状態 l を置かない。戻り値は各フレームの状態 index で、
    最終フレームで末尾の状態に届く経路が無ければ None。"""
    num_frames, num_states = out_of_bounds.shape
    emission = log_probs[:, token_ids]
    if state_bias is not None:
        emission = emission + state_bias
    neg_inf = float("-inf")
    dp = np.full((num_frames, num_states), neg_inf, dtype=np.float64)
    backpointer = np.zeros((num_frames, num_states), dtype=np.int64)

    dp[0, 0] = emission[0, 0] if not out_of_bounds[0, 0] else neg_inf
    for t in range(1, num_frames):
        stay = dp[t - 1, :]
        advance = np.concatenate(([neg_inf], dp[t - 1, :-1]))
        take_advance = advance > stay
        candidate = np.where(take_advance, advance, stay) + emission[t, :]
        candidate[out_of_bounds[t]] = neg_inf
        dp[t, :] = candidate
        backpointer[t, :] = take_advance.astype(np.int64)

    if dp[num_frames - 1, num_states - 1] == neg_inf:
        return None

    path = [0] * num_frames
    state = num_states - 1
    path[num_frames - 1] = state
    for t in range(num_frames - 1, 0, -1):
        state -= int(backpointer[t, state])
        path[t - 1] = state
    return path


def _forced_align(log_probs: np.ndarray, token_ids: list[int]) -> list[int]:
    num_frames = log_probs.shape[0]
    num_states = len(token_ids)
    if num_states == 0 or num_frames < num_states:
        raise RecognitionError(
            f"強制アライメントが対応付け不能です(フレーム数{num_frames}、トークン数{num_states})"
        )

    band_frames = max(1, round(_FORCED_ALIGN_BAND_SEC / FRAME_DURATION_SEC))

    def expected_frame(state_index: int) -> float:
        if num_states == 1:
            return 0.0
        return (state_index / (num_states - 1)) * (num_frames - 1)

    out_of_band = np.array([
        [abs(t - expected_frame(state)) > band_frames for state in range(num_states)]
        for t in range(num_frames)
    ])

    path = _viterbi_monotonic(log_probs, token_ids, out_of_band)
    if path is None:
        raise RecognitionError("強制アライメントが末尾トークンへ到達できませんでした")
    return path


def _forced_align_windowed(
    log_probs: np.ndarray, token_ids: list[int], windows_sec: list[tuple[float, float]]
) -> list[int]:
    num_frames = log_probs.shape[0]
    num_states = len(token_ids)
    if num_states == 0 or num_frames < num_states:
        raise RecognitionError(
            f"強制アライメントが対応付け不能です(フレーム数{num_frames}、トークン数{num_states})"
        )

    frame_lo = np.arange(num_frames) * FRAME_DURATION_SEC
    frame_hi = frame_lo + FRAME_DURATION_SEC
    win_lo = np.array([w[0] for w in windows_sec])
    win_hi = np.array([w[1] for w in windows_sec])
    width = np.maximum(win_hi - win_lo, _MIN_WINDOW_WIDTH_SEC)

    out_of_window = ~((win_lo[None, :] < frame_hi[:, None]) & (win_hi[None, :] > frame_lo[:, None]))
    relative_position = np.clip((frame_lo[:, None] - win_lo[None, :]) / width[None, :], 0.0, 1.0)
    state_bias = _EARLY_COMMIT_BONUS * (1.0 - relative_position)

    path = _viterbi_monotonic(log_probs, token_ids, out_of_window, state_bias)
    if path is None:
        raise RecognitionError("単語窓制約下で強制アライメントが末尾トークンへ到達できませんでした")
    return path


def _path_to_segments(path: list[int], symbols: list[str], frame_duration_sec: float) -> list[Segment]:
    num_frames = len(path)
    num_states = len(symbols)
    state_start_frame: list[int | None] = [None] * num_states
    for t, state in enumerate(path):
        if state_start_frame[state] is None:
            state_start_frame[state] = t

    segments: list[Segment] = []
    for state in range(num_states):
        start_frame = state_start_frame[state]
        end_frame = state_start_frame[state + 1] if state + 1 < num_states else num_frames
        symbol = symbols[state]
        is_blank = symbol in _BLANK_G2P_SYMBOLS
        phoneme = None if is_blank else _G2P_TO_VOCAB_SYMBOL[symbol]
        seg_type = "gap" if is_blank else _classify_symbol(phoneme)
        new_segment = Segment(
            type=seg_type,
            start_sec=start_frame * frame_duration_sec,
            end_sec=end_frame * frame_duration_sec,
            phoneme=phoneme,
            confidence=None,
        )
        if segments and segments[-1].type == new_segment.type and segments[-1].phoneme == new_segment.phoneme:
            prev = segments[-1]
            segments[-1] = Segment(
                type=prev.type,
                start_sec=prev.start_sec,
                end_sec=new_segment.end_sec,
                phoneme=prev.phoneme,
                confidence=None,
            )
        else:
            segments.append(new_segment)
    return segments


def _merge_adjacent_segments(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for seg in segments:
        if merged and merged[-1].type == seg.type and merged[-1].phoneme == seg.phoneme:
            prev = merged[-1]
            merged[-1] = Segment(
                type=prev.type, start_sec=prev.start_sec, end_sec=seg.end_sec,
                phoneme=prev.phoneme, confidence=None,
            )
        else:
            merged.append(seg)
    return merged


def _ensure_phoneme_model(on_progress: Callable[[str], None] | None):
    try:
        processor, model = _load_model_and_processor(on_progress=on_progress)
    except ImportError as e:
        raise RecognitionError(
            "transformers または torch が見つかりません。導入してください"
            "(vocal-analysis extra で両方導入されます)。"
        ) from e
    except OSError as e:
        raise RecognitionError(
            f"認識モデル({RECOGNIZER_CONFIG.model_id}, revision={RECOGNIZER_CONFIG.model_revision})を"
            "取得できません。ネットワーク接続を確認してください。"
        ) from e
    vocab = processor.tokenizer.get_vocab()
    blank_token_id = processor.tokenizer.pad_token_id
    return processor, model, vocab, blank_token_id


def _align_wav2vec2_job(
    processor, model, vocab, blank_token_id, threshold: float, job: dict
) -> list[Segment]:
    chunk_samples = job["chunk_samples"]
    seq = job["seq"]
    windows_sec = job["windows_sec"]
    words_phonemes = job["words_phonemes"]
    trim_lo_sec = job["trim_lo_sec"]
    trim_hi_sec = job["trim_hi_sec"]

    token_ids = _g2p_symbols_to_token_ids(seq, vocab, blank_token_id)
    log_probs = _compute_log_probs(processor, model, chunk_samples)
    log_probs = _apply_voiced_blank_penalty(log_probs, chunk_samples, threshold, blank_token_id)

    if windows_sec is not None:
        sub_token_ids, sub_to_token = _expand_min_stay_local(token_ids, words_phonemes, blank_token_id)
        sub_windows = [windows_sec[i] for i in sub_to_token]
        try:
            sub_path = _forced_align_windowed(log_probs, sub_token_ids, sub_windows)
        except RecognitionError:
            sub_token_ids, sub_to_token = _expand_min_stay(token_ids, blank_token_id, log_probs.shape[0])
            sub_path = _forced_align(log_probs, sub_token_ids)
    else:
        sub_token_ids, sub_to_token = _expand_min_stay(token_ids, blank_token_id, log_probs.shape[0])
        sub_path = _forced_align(log_probs, sub_token_ids)

    path = [sub_to_token[s] for s in sub_path]
    local_segments = _path_to_segments(path, seq, FRAME_DURATION_SEC)
    if local_segments:
        last = local_segments[-1]
        local_segments[-1] = Segment(
            type=last.type, start_sec=last.start_sec, end_sec=trim_hi_sec - trim_lo_sec,
            phoneme=last.phoneme, confidence=None,
        )
    return [
        Segment(
            type=seg.type,
            start_sec=seg.start_sec + trim_lo_sec,
            end_sec=seg.end_sec + trim_lo_sec,
            phoneme=seg.phoneme,
            confidence=None,
        )
        for seg in local_segments
    ]


def _release_content_recognizer_pipeline() -> None:
    global _content_recognizer_pipeline_cache
    if _content_recognizer_pipeline_cache is None:
        return
    _content_recognizer_pipeline_cache = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class _SofaTarget(NamedTuple):
    samples: np.ndarray
    sample_rate: int
    phonemes: list[str]
    offset_sec: float
    duration_sec: float


def recognize(
    vocal_wav_path: Path,
    content_recognizer_model: ContentRecognizerModel = DEFAULT_CONTENT_RECOGNIZER_MODEL,
    *,
    retry: bool = True,
    forced_aligner: ForcedAlignerId = DEFAULT_FORCED_ALIGNER,
    sofa_aligner: SofaAlignerConfig | None = None,
    english_katakana_method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> list[Segment]:
    """forced_aligner が "sofa-forcedalign" のときは sofa_aligner が必須で、省略すると RecognitionError。"""
    if forced_aligner not in ("wav2vec2-ctc-forcedalign", "sofa-forcedalign"):
        raise RecognitionError(f"未知の forced_aligner です: {forced_aligner!r}")
    if forced_aligner == "sofa-forcedalign" and sofa_aligner is None:
        raise RecognitionError(
            "forced_aligner='sofa-forcedalign' を指定する場合は sofa_aligner(SofaAlignerConfig)が必須です"
        )

    samples, sample_rate = sf.read(vocal_wav_path, dtype="float32", always_2d=True)
    mono = _downmix_to_mono(samples)
    resampled = _resample_to_target(mono, sample_rate, RECOGNIZER_CONFIG.sample_rate)
    duration_sec = len(resampled) / RECOGNIZER_CONFIG.sample_rate

    frame_rms = _frame_rms(resampled, RECOGNIZER_CONFIG.sample_rate, _SILENCE_FRAME_SEC)
    threshold = _silence_threshold(frame_rms)
    split_points = _detect_silence_split_points(resampled, RECOGNIZER_CONFIG.sample_rate)
    segment_bounds = _build_segment_bounds(duration_sec, split_points)

    pending_sofa_targets: list[_SofaTarget] = []
    pending_align_jobs: list[dict] = []

    all_segments: list[Segment] = []
    segment_count = len(segment_bounds)
    for segment_index, (start_sec, end_sec) in enumerate(segment_bounds, start=1):
        segment_note = (
            f"歌詞書き起こし中: {_format_time_range(start_sec, end_sec)}({segment_index}/{segment_count})"
        )
        if on_progress is not None:
            on_progress(segment_note)
        start_index = round(start_sec * RECOGNIZER_CONFIG.sample_rate)
        end_index = round(end_sec * RECOGNIZER_CONFIG.sample_rate)
        chunk_samples = resampled[start_index:end_index]

        if _is_segment_silent(chunk_samples, threshold):
            all_segments.append(
                Segment(type="gap", start_sec=start_sec, end_sec=end_sec, phoneme=None, confidence=None)
            )
            continue

        trim_lo, trim_hi = _voiced_trim_bounds(chunk_samples, RECOGNIZER_CONFIG.sample_rate, threshold)
        trimmed_tail = trim_hi < len(chunk_samples)
        trim_lo_sec = start_sec + trim_lo / RECOGNIZER_CONFIG.sample_rate if trim_lo > 0 else start_sec
        trim_hi_sec = start_sec + trim_hi / RECOGNIZER_CONFIG.sample_rate if trimmed_tail else end_sec
        if trim_lo > 0:
            all_segments.append(
                Segment(type="gap", start_sec=start_sec, end_sec=trim_lo_sec, phoneme=None, confidence=None)
            )
        chunk_samples = chunk_samples[trim_lo:trim_hi]

        trim_duration_sec = trim_hi_sec - trim_lo_sec

        try:
            text, words = _transcribe_segment(
                _load_content_recognizer_for_segment(content_recognizer_model, on_progress, segment_note),
                chunk_samples,
            )
        except ImportError as e:
            raise RecognitionError(
                "transformers または torch が見つかりません。導入してください"
                "(vocal-analysis extra で両方導入されます)。"
            ) from e
        except OSError as e:
            raise RecognitionError(
                f"内容認識モデル({content_recognizer_model.model_id}, "
                f"revision={content_recognizer_model.model_revision})を"
                "取得できません。ネットワーク接続を確認してください。"
            ) from e

        try:
            text, words = _resolve_transcription(
                chunk_samples, trim_duration_sec, text, words,
                content_recognizer_model, retry,
                english_katakana_method=english_katakana_method,
                on_progress=on_progress,
            )
        except ImportError as e:
            raise RecognitionError(
                "pyopenjtalk-plus または transformers が見つかりません。導入してください"
                "(vocal-analysis extra で導入されます)。"
            ) from e

        if not text.strip():
            all_segments.append(
                Segment(type="gap", start_sec=trim_lo_sec, end_sec=trim_hi_sec, phoneme=None, confidence=None)
            )
            if trimmed_tail:
                all_segments.append(
                    Segment(type="gap", start_sec=trim_hi_sec, end_sec=end_sec, phoneme=None, confidence=None)
                )
            continue

        try:
            if words:
                words_phonemes = [
                    (_g2p(word_text, method=english_katakana_method, on_progress=on_progress),
                     w_start, w_end)
                    for word_text, w_start, w_end in words
                ]
                phonemes = [p for word_phonemes, _, _ in words_phonemes for p in word_phonemes]
            else:
                words_phonemes = None
                phonemes = _g2p(text, method=english_katakana_method, on_progress=on_progress)
        except ImportError as e:
            raise RecognitionError(
                "pyopenjtalk-plus が見つかりません。導入してください(vocal-analysis extra で導入されます)。"
            ) from e

        if _is_hallucinated_phoneme_density(len(phonemes), trim_duration_sec):
            all_segments.append(
                Segment(type="gap", start_sec=trim_lo_sec, end_sec=trim_hi_sec, phoneme=None, confidence=None)
            )
            if trimmed_tail:
                all_segments.append(
                    Segment(type="gap", start_sec=trim_hi_sec, end_sec=end_sec, phoneme=None, confidence=None)
                )
            continue

        if forced_aligner == "wav2vec2-ctc-forcedalign":
            if words_phonemes is not None:
                words_phonemes = _widen_cramped_words(words_phonemes, trim_duration_sec)
                seq, windows_sec = _assemble_with_word_windows(
                    words_phonemes, _WORD_WINDOW_MARGIN_SEC, trim_duration_sec
                )
            else:
                seq = _assemble_phoneme_sequence([phonemes])
                windows_sec = None

            pending_align_jobs.append({
                "chunk_samples": chunk_samples,
                "seq": seq,
                "windows_sec": windows_sec,
                "words_phonemes": words_phonemes,
                "trim_lo_sec": trim_lo_sec,
                "trim_hi_sec": trim_hi_sec,
                "insert_at": len(all_segments),
            })
        else:
            if words_phonemes is not None:
                valid_words = sofa_align._clamp_words_to_valid_list(words_phonemes, trim_duration_sec)
                for gap_start, gap_end in sofa_align._determine_word_gaps(valid_words, trim_duration_sec):
                    all_segments.append(
                        Segment(
                            type="gap",
                            start_sec=trim_lo_sec + gap_start,
                            end_sec=trim_lo_sec + gap_end,
                            phoneme=None,
                            confidence=None,
                        )
                    )
                for word_phonemes, w_start, w_end in valid_words:
                    w_start_index = round(w_start * RECOGNIZER_CONFIG.sample_rate)
                    w_end_index = round(w_end * RECOGNIZER_CONFIG.sample_rate)
                    pending_sofa_targets.append(
                        _SofaTarget(
                            chunk_samples[w_start_index:w_end_index],
                            RECOGNIZER_CONFIG.sample_rate,
                            word_phonemes,
                            trim_lo_sec + w_start,
                            w_end - w_start,
                        )
                    )
            else:
                pending_sofa_targets.append(
                    _SofaTarget(chunk_samples, RECOGNIZER_CONFIG.sample_rate, phonemes, trim_lo_sec, trim_duration_sec)
                )

        if trimmed_tail:
            all_segments.append(
                Segment(type="gap", start_sec=trim_hi_sec, end_sec=end_sec, phoneme=None, confidence=None)
            )

    if forced_aligner == "wav2vec2-ctc-forcedalign":
        _release_content_recognizer_pipeline()
        if pending_align_jobs:
            processor, model, vocab, blank_token_id = _ensure_phoneme_model(on_progress)
            job_count = len(pending_align_jobs)
            inserted = 0
            for job_index, job in enumerate(pending_align_jobs, start=1):
                if on_progress is not None:
                    on_progress(
                        f"音素アライメント中: {_format_time_range(job['trim_lo_sec'], job['trim_hi_sec'])}"
                        f"({job_index}/{job_count})"
                    )
                aligned = _align_wav2vec2_job(processor, model, vocab, blank_token_id, threshold, job)
                position = job["insert_at"] + inserted
                all_segments[position:position] = aligned
                inserted += len(aligned)
            del processor, model
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if forced_aligner == "sofa-forcedalign":
        _release_content_recognizer_pipeline()
        raw_by_basename = sofa_align._align_batch(
            [(samples, sr, ph) for samples, sr, ph, _, _ in pending_sofa_targets], sofa_aligner
        )
        for i, (_, _, _, offset_sec, rel_duration_sec) in enumerate(pending_sofa_targets):
            raw_segments = raw_by_basename[f"segment_{i:04d}"]
            validated = sofa_align._validate_and_normalize_segments(raw_segments, rel_duration_sec)
            for seg in sofa_align._segments_from_raw(validated):
                all_segments.append(
                    Segment(
                        type=seg.type,
                        start_sec=seg.start_sec + offset_sec,
                        end_sec=seg.end_sec + offset_sec,
                        phoneme=seg.phoneme,
                        confidence=None,
                    )
                )
        all_segments.sort(key=lambda seg: (seg.start_sec, seg.end_sec))

    return _merge_adjacent_segments(all_segments)


def _downmix_to_mono(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=1)


def _resample_to_target(mono: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return mono
    gcd = math.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd
    return resample_poly(mono, up, down).astype(np.float32)


def _prepare_english_katakana_conversion(
    text: str, method: EnglishKatakanaMethod, on_progress: Callable[[str], None] | None = None
) -> None:
    if method != "tinyllama-katakana-converter":
        return
    words = uncached_target_words(text, method)
    if not words:
        return
    _release_content_recognizer_pipeline()
    convert_words(words, method, on_progress=on_progress)


def convert_with_g2p(
    text: str, *, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None, kana: bool = False,
):
    """kana が真ならかな読みの文字列、偽なら音素記号のリストを返す。"""
    import pyopenjtalk

    try:
        _prepare_english_katakana_conversion(text, method, on_progress=on_progress)
        text = convert_target_words(text, method=method, on_progress=on_progress)
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

    # pyopenjtalk のネイティブ拡張は Python のログ設定を経由せず標準エラーへ直接書く。
    with suppress_native_stderr():
        if kana:
            return pyopenjtalk.g2p(text, kana=True)
        return pyopenjtalk.g2p(text, kana=False, join=False)


def _g2p(
    text: str, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> list[str]:
    return convert_with_g2p(text, method=method, on_progress=on_progress, kana=False)


def _sanitize_word_timestamps(
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


def _extract_word_timestamps(chunks: list[dict], duration_sec: float) -> list[tuple[str, float, float]]:
    words: list[tuple[str, float, float]] = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        start, end = chunk.get("timestamp", (None, None))
        if not text or start is None:
            continue
        end_sec = float(end) if end is not None else float(start) + _MIN_WORD_DURATION_SEC
        words.append((text, float(start), end_sec))
    return _sanitize_word_timestamps(words, duration_sec)


def _transcribe_text_only(
    samples: np.ndarray, content_recognizer_model: ContentRecognizerModel,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    pipeline = _load_content_recognizer_pipeline(content_recognizer_model, on_progress=on_progress)
    # transformers の Whisper は temperature を単一の値で受けると温度フォールバックを行わない。
    generate_kwargs = {
        "language": "japanese", "task": "transcribe",
        "num_beams": 1, "do_sample": False, "temperature": 0.0,
        "max_new_tokens": _RETRY_MAX_NEW_TOKENS,
    }
    result = pipeline(samples, generate_kwargs=generate_kwargs)
    return result["text"]


def _transcribe_segment(
    pipeline, samples: np.ndarray
) -> tuple[str, list[tuple[str, float, float]] | None]:
    prompt_ids = pipeline.tokenizer.get_prompt_ids(KANA_PROMPT, return_tensors="pt").to(pipeline.device)
    result = pipeline(
        samples,
        return_timestamps="word",
        generate_kwargs={
            "language": "japanese",
            "task": "transcribe",
            "num_beams": 1,
            "do_sample": False,
            "prompt_ids": prompt_ids,
            "max_new_tokens": _SEGMENT_MAX_NEW_TOKENS,
        },
    )
    chunks = result.get("chunks")
    if not chunks:
        return result["text"], None
    duration_sec = len(samples) / RECOGNIZER_CONFIG.sample_rate
    return result["text"], _extract_word_timestamps(chunks, duration_sec)


def _select_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


_content_recognizer_pipeline_cache: tuple[ContentRecognizerModel, object] | None = None


# transformers の PyTorch バックエンドは、これらの形式の重みファイルを読まない。
_PREFETCH_IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.tflite", "*.onnx", "*.ot", "*.mlmodel"]


def _hf_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id, revision=revision, tqdm_class=tqdm_class,
        ignore_patterns=_PREFETCH_IGNORE_PATTERNS,
    )


def _transformers_pipeline(*args, **kwargs):
    from transformers import pipeline as transformers_pipeline

    return transformers_pipeline(*args, **kwargs)


def _transformers_auto_processor_from_pretrained(*args, **kwargs):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(*args, **kwargs)


def _transformers_auto_model_for_ctc_from_pretrained(*args, **kwargs):
    from transformers import AutoModelForCTC

    return AutoModelForCTC.from_pretrained(*args, **kwargs)


def _prefetch_with_progress(repo_id: str, revision: str | None, on_progress: Callable[[str], None]) -> bool:
    """戻り値はバイト転送が実際に起きたか。"""
    from huggingface_hub.utils import tqdm as hf_tqdm

    state = {"shown": False}

    class _RelayTqdm(hf_tqdm):
        # 表示を無効化した tqdm は __init__ を途中で抜け、self.unit を持たず self.n も進めない。
        def __init__(self, *args, **kwargs):
            self._relay_unit = kwargs.get("unit")
            self._relay_n = kwargs.get("initial") or 0
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            result = super().update(n)
            # snapshot_download はファイル数を数えるバーと、実際に転送したバイト数を数える unit="B" のバーを作る。
            if self._relay_unit == "B" and self.total:
                self._relay_n += n or 0
                state["shown"] = True
                percent = min(100, int(self._relay_n * 100 / self.total))
                on_progress(f"ダウンロード中: {repo_id} {percent}%")
            return result

    _hf_snapshot_download(repo_id, revision=revision, tqdm_class=_RelayTqdm)
    return state["shown"]


def _load_content_recognizer_for_segment(content_recognizer_model, on_progress, segment_note):
    pipeline = _load_content_recognizer_pipeline(content_recognizer_model, on_progress=on_progress)
    if on_progress is not None:
        on_progress(segment_note)
    return pipeline


def _load_content_recognizer_pipeline(
    content_recognizer_model: ContentRecognizerModel,
    on_progress: Callable[[str], None] | None = None,
):
    global _content_recognizer_pipeline_cache
    if _content_recognizer_pipeline_cache is not None:
        cached_model, cached_pipeline = _content_recognizer_pipeline_cache
        if cached_model == content_recognizer_model:
            return cached_pipeline
        del cached_model, cached_pipeline
        _content_recognizer_pipeline_cache = None

    import torch

    silence_third_party_output()

    downloaded = False
    if on_progress is not None:
        downloaded = _prefetch_with_progress(
            content_recognizer_model.model_id, content_recognizer_model.model_revision, on_progress)

    device = _select_device()
    # CPU 実行では fp16 を使えない。
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        if on_progress is not None:
            on_progress(f"内容認識モデル読み込み中: {content_recognizer_model.model_id}")
        pipeline = _transformers_pipeline(
            "automatic-speech-recognition",
            model=content_recognizer_model.model_id,
            revision=content_recognizer_model.model_revision,
            device=device,
            dtype=dtype,
        )
    finally:
        if downloaded:
            on_progress("")
    _content_recognizer_pipeline_cache = (content_recognizer_model, pipeline)
    return pipeline


def _compute_log_probs(processor, model, samples: np.ndarray) -> np.ndarray:
    import torch

    inputs = processor(samples, sampling_rate=RECOGNIZER_CONFIG.sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(_select_device())).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[0].cpu().numpy()


def _load_model_and_processor(on_progress: Callable[[str], None] | None = None):
    import torch

    silence_third_party_output()

    downloaded = False
    if on_progress is not None:
        downloaded = _prefetch_with_progress(
            RECOGNIZER_CONFIG.model_id, RECOGNIZER_CONFIG.model_revision, on_progress)

    try:
        if on_progress is not None:
            on_progress(f"音素モデル読み込み中: {RECOGNIZER_CONFIG.model_id}")
        # wav2vec2-espeak のトークナイザは、既定では espeak の実行ファイル(phonemizer 経由)を要求する。
        processor = _transformers_auto_processor_from_pretrained(
            RECOGNIZER_CONFIG.model_id,
            revision=RECOGNIZER_CONFIG.model_revision,
            do_phonemize=False,
        )
        model = _transformers_auto_model_for_ctc_from_pretrained(
            RECOGNIZER_CONFIG.model_id,
            revision=RECOGNIZER_CONFIG.model_revision,
            dtype=getattr(torch, RECOGNIZER_CONFIG.dtype),
        )
    finally:
        if downloaded:
            on_progress("")
    model.to(_select_device())
    model.eval()
    return processor, model
