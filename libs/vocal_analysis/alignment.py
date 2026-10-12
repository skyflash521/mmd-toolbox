from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from . import model_store, sofa_align
from .config import RECOGNIZER_CONFIG, SofaAlignerConfig
from .phonemes import (
    _BLANK_G2P_SYMBOLS,
    _G2P_TO_VOCAB_SYMBOL,
    RecognitionError,
    _classify_symbol,
)
from .segmentation import frame_rms
from .types import Segment

# 採用した音素モデルの畳み込み層の総ストライドは 320 サンプルで、16kHz 入力の1フレームは 20ms になる。
FRAME_DURATION_SEC = 0.02

CRAMPED_WORD_THRESHOLD_SEC = 0.03
CRAMPED_WORD_TARGET_SEC = 0.04
CRAMPED_WORD_EPSILON_SEC = 1e-9

FORCED_ALIGN_BAND_SEC = 1.0
MIN_STAY_FRAMES = 6
VOICED_BLANK_PENALTY = 7.0
WORD_WINDOW_MARGIN_SEC = 0.75
EARLY_COMMIT_BONUS = 2.0
MIN_WINDOW_WIDTH_SEC = 1e-9


class AlignJob(NamedTuple):
    chunk_samples: np.ndarray
    seq: list[str]
    windows_sec: list[tuple[float, float]] | None
    words_phonemes: list[tuple[list[str], float, float]] | None
    trim_lo_sec: float
    trim_hi_sec: float


class SofaTarget(NamedTuple):
    samples: np.ndarray
    sample_rate: int
    phonemes: list[str]
    offset_sec: float
    duration_sec: float


def assemble_phoneme_sequence(chunk_phonemes: list[list[str]]) -> list[str]:
    sequence = ["pau"]
    for i, phonemes in enumerate(chunk_phonemes):
        if i > 0:
            sequence.append("pau")
        sequence.extend(phonemes)
    sequence.append("pau")
    return sequence


def widen_cramped_words(
    words_phonemes: list[tuple[list[str], float, float]], duration_sec: float
) -> list[tuple[list[str], float, float]]:
    widened = []
    for phonemes, start, end in words_phonemes:
        threshold = len(phonemes) * CRAMPED_WORD_THRESHOLD_SEC - CRAMPED_WORD_EPSILON_SEC
        if phonemes and (end - start) < threshold:
            needed = len(phonemes) * CRAMPED_WORD_TARGET_SEC
            end = min(end, duration_sec)
            start = max(0.0, end - needed)
            end = min(duration_sec, start + needed)
        widened.append((phonemes, start, end))
    return widened


def assemble_with_word_windows(
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


def g2p_symbols_to_token_ids(symbols: list[str], vocab: dict[str, int], blank_token_id: int) -> list[int]:
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


def apply_voiced_blank_penalty(
    log_probs: np.ndarray, chunk_samples: np.ndarray, threshold: float, blank_token_id: int
) -> np.ndarray:
    rms_per_frame = frame_rms(chunk_samples, RECOGNIZER_CONFIG.sample_rate, FRAME_DURATION_SEC)
    n = min(len(rms_per_frame), log_probs.shape[0])
    penalized = log_probs.copy()
    voiced = rms_per_frame[:n] > threshold
    penalized[:n][voiced, blank_token_id] -= VOICED_BLANK_PENALTY
    return penalized


def expand_min_stay(
    token_ids: list[int], blank_token_id: int, num_frames: int
) -> tuple[list[int], list[int]]:
    """戻り値は (展開後のトークン ID 列, 各サブ状態が属する元トークンの index 列)。"""
    num_blank = sum(1 for tid in token_ids if tid == blank_token_id)
    num_phoneme = len(token_ids) - num_blank
    stay = MIN_STAY_FRAMES
    if num_phoneme > 0:
        stay = min(stay, max(1, (num_frames - num_blank) // num_phoneme))
    sub_token_ids: list[int] = []
    sub_to_token: list[int] = []
    for index, tid in enumerate(token_ids):
        reps = 1 if tid == blank_token_id else stay
        sub_token_ids.extend([tid] * reps)
        sub_to_token.extend([index] * reps)
    return sub_token_ids, sub_to_token


def expand_min_stay_local(
    token_ids: list[int],
    words_phonemes: list[tuple[list[str], float, float]],
    blank_token_id: int,
) -> tuple[list[int], list[int]]:
    """token_ids は words_phonemes から assemble_with_word_windows が組んだ記号列を変換したもの。
    戻り値は (展開後のトークン ID 列, 各サブ状態が属する元トークンの index 列)。"""
    pau_stay = 1
    stays = [pau_stay]
    for i, (phonemes, start, end) in enumerate(words_phonemes):
        word_frames = (end - start) / FRAME_DURATION_SEC
        # round() は偶数丸めで、ちょうど .5 の商を切り捨てることがある。
        stay = min(MIN_STAY_FRAMES, max(1, int(word_frames / len(phonemes) + 0.5))) if phonemes else 1
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


def viterbi_monotonic(
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


def forced_align(log_probs: np.ndarray, token_ids: list[int]) -> list[int]:
    num_frames = log_probs.shape[0]
    num_states = len(token_ids)
    if num_states == 0 or num_frames < num_states:
        raise RecognitionError(
            f"強制アライメントが対応付け不能です(フレーム数{num_frames}、トークン数{num_states})"
        )

    band_frames = max(1, round(FORCED_ALIGN_BAND_SEC / FRAME_DURATION_SEC))

    def expected_frame(state_index: int) -> float:
        if num_states == 1:
            return 0.0
        return (state_index / (num_states - 1)) * (num_frames - 1)

    out_of_band = np.array([
        [abs(t - expected_frame(state)) > band_frames for state in range(num_states)]
        for t in range(num_frames)
    ])

    path = viterbi_monotonic(log_probs, token_ids, out_of_band)
    if path is None:
        raise RecognitionError("強制アライメントが末尾トークンへ到達できませんでした")
    return path


def forced_align_windowed(
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
    width = np.maximum(win_hi - win_lo, MIN_WINDOW_WIDTH_SEC)

    out_of_window = ~((win_lo[None, :] < frame_hi[:, None]) & (win_hi[None, :] > frame_lo[:, None]))
    relative_position = np.clip((frame_lo[:, None] - win_lo[None, :]) / width[None, :], 0.0, 1.0)
    state_bias = EARLY_COMMIT_BONUS * (1.0 - relative_position)

    path = viterbi_monotonic(log_probs, token_ids, out_of_window, state_bias)
    if path is None:
        raise RecognitionError("単語窓制約下で強制アライメントが末尾トークンへ到達できませんでした")
    return path


def path_to_segments(path: list[int], symbols: list[str], frame_duration_sec: float) -> list[Segment]:
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
        )
        if segments and segments[-1].type == new_segment.type and segments[-1].phoneme == new_segment.phoneme:
            prev = segments[-1]
            segments[-1] = Segment(
                type=prev.type,
                start_sec=prev.start_sec,
                end_sec=new_segment.end_sec,
                phoneme=prev.phoneme,
            )
        else:
            segments.append(new_segment)
    return segments


def merge_adjacent_segments(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for seg in segments:
        if merged and merged[-1].type == seg.type and merged[-1].phoneme == seg.phoneme:
            prev = merged[-1]
            merged[-1] = Segment(
                type=prev.type, start_sec=prev.start_sec, end_sec=seg.end_sec,
                phoneme=prev.phoneme,
            )
        else:
            merged.append(seg)
    return merged


def build_align_job(
    chunk_samples: np.ndarray,
    phonemes: list[str],
    words_phonemes: list[tuple[list[str], float, float]] | None,
    trim_lo_sec: float,
    trim_hi_sec: float,
) -> AlignJob:
    trim_duration_sec = trim_hi_sec - trim_lo_sec
    if words_phonemes is not None:
        words_phonemes = widen_cramped_words(words_phonemes, trim_duration_sec)
        seq, windows_sec = assemble_with_word_windows(
            words_phonemes, WORD_WINDOW_MARGIN_SEC, trim_duration_sec
        )
    else:
        seq = assemble_phoneme_sequence([phonemes])
        windows_sec = None
    return AlignJob(chunk_samples, seq, windows_sec, words_phonemes, trim_lo_sec, trim_hi_sec)


def load_phoneme_model(on_progress: Callable[[str], None] | None = None):
    """戻り値は (processor, model)。"""
    import torch
    import transformers

    def build():
        # wav2vec2-espeak のトークナイザは、既定では espeak の実行ファイル(phonemizer 経由)を要求する。
        processor = transformers.AutoProcessor.from_pretrained(
            RECOGNIZER_CONFIG.model_id,
            revision=RECOGNIZER_CONFIG.model_revision,
            do_phonemize=False,
        )
        model = transformers.AutoModelForCTC.from_pretrained(
            RECOGNIZER_CONFIG.model_id,
            revision=RECOGNIZER_CONFIG.model_revision,
            dtype=getattr(torch, RECOGNIZER_CONFIG.dtype),
        )
        return processor, model

    processor, model = model_store.load_with_progress(
        RECOGNIZER_CONFIG.model_id, RECOGNIZER_CONFIG.model_revision,
        on_progress=on_progress,
        loading_note=f"音素モデル読み込み中: {RECOGNIZER_CONFIG.model_id}",
        build=build,
    )
    model.to(model_store.select_device())
    model.eval()
    return processor, model


def compute_log_probs(processor, model, samples: np.ndarray) -> np.ndarray:
    import torch

    inputs = processor(samples, sampling_rate=RECOGNIZER_CONFIG.sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(model_store.select_device())).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[0].cpu().numpy()


def align_job(
    processor, model, vocab, blank_token_id, threshold: float, job: AlignJob
) -> list[Segment]:
    token_ids = g2p_symbols_to_token_ids(job.seq, vocab, blank_token_id)
    log_probs = compute_log_probs(processor, model, job.chunk_samples)
    log_probs = apply_voiced_blank_penalty(log_probs, job.chunk_samples, threshold, blank_token_id)

    if job.windows_sec is not None:
        sub_token_ids, sub_to_token = expand_min_stay_local(token_ids, job.words_phonemes, blank_token_id)
        sub_windows = [job.windows_sec[i] for i in sub_to_token]
        try:
            sub_path = forced_align_windowed(log_probs, sub_token_ids, sub_windows)
        except RecognitionError:
            sub_token_ids, sub_to_token = expand_min_stay(token_ids, blank_token_id, log_probs.shape[0])
            sub_path = forced_align(log_probs, sub_token_ids)
    else:
        sub_token_ids, sub_to_token = expand_min_stay(token_ids, blank_token_id, log_probs.shape[0])
        sub_path = forced_align(log_probs, sub_token_ids)

    path = [sub_to_token[s] for s in sub_path]
    local_segments = path_to_segments(path, job.seq, FRAME_DURATION_SEC)
    if local_segments:
        last = local_segments[-1]
        local_segments[-1] = Segment(
            type=last.type, start_sec=last.start_sec, end_sec=job.trim_hi_sec - job.trim_lo_sec,
            phoneme=last.phoneme,
        )
    return [
        Segment(
            type=seg.type,
            start_sec=seg.start_sec + job.trim_lo_sec,
            end_sec=seg.end_sec + job.trim_lo_sec,
            phoneme=seg.phoneme,
        )
        for seg in local_segments
    ]


def build_sofa_targets(
    chunk_samples: np.ndarray,
    phonemes: list[str],
    words_phonemes: list[tuple[list[str], float, float]] | None,
    trim_lo_sec: float,
    trim_duration_sec: float,
) -> tuple[list[Segment], list[SofaTarget]]:
    """戻り値は (単語間の gap, SOFA に渡す対象)。"""
    sample_rate = RECOGNIZER_CONFIG.sample_rate
    if words_phonemes is None:
        return [], [SofaTarget(chunk_samples, sample_rate, phonemes, trim_lo_sec, trim_duration_sec)]

    valid_words = sofa_align._clamp_words_to_valid_list(words_phonemes, trim_duration_sec)
    gaps = [
        Segment(type="gap", start_sec=trim_lo_sec + gap_start, end_sec=trim_lo_sec + gap_end, phoneme=None)
        for gap_start, gap_end in sofa_align._determine_word_gaps(valid_words, trim_duration_sec)
    ]
    targets = [
        SofaTarget(
            chunk_samples[round(w_start * sample_rate):round(w_end * sample_rate)],
            sample_rate,
            word_phonemes,
            trim_lo_sec + w_start,
            w_end - w_start,
        )
        for word_phonemes, w_start, w_end in valid_words
    ]
    return gaps, targets


def align_with_sofa(targets: list[SofaTarget], sofa_aligner: SofaAlignerConfig) -> list[Segment]:
    raw_by_basename = sofa_align._align_batch(
        [(samples, sr, ph) for samples, sr, ph, _, _ in targets], sofa_aligner
    )
    segments: list[Segment] = []
    for i, (_, _, _, offset_sec, rel_duration_sec) in enumerate(targets):
        raw_segments = raw_by_basename[f"segment_{i:04d}"]
        validated = sofa_align._validate_and_normalize_segments(raw_segments, rel_duration_sec)
        for seg in sofa_align._segments_from_raw(validated):
            segments.append(
                Segment(
                    type=seg.type,
                    start_sec=seg.start_sec + offset_sec,
                    end_sec=seg.end_sec + offset_sec,
                    phoneme=seg.phoneme,
                )
            )
    return segments
