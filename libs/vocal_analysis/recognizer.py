"""S2 音素/母音認識(vocal_analysis.md §5・§5.1・§5.2・§8.1・§8.3)。

内容認識(Whisper)・G2P(pyopenjtalk-plus)・音素モデルのCTC強制アライメント(§5.2)を組み合わせた
複合構成で、母音/子音/gap を区別したセグメント列を生成する。公開関数
recognize(vocal_wav_path) -> list[Segment] が唯一の公開面(Recognizer アダプタ契約。§8.1)。
"""

import math
import unicodedata
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import RECOGNIZER_CONFIG, WHISPER_CONFIG
from .types import Segment

FRAME_DURATION_SEC = 0.02  # §5.1: 採用モデルの畳み込み総ストライド320サンプル@16kHzで固定


class RecognitionError(Exception):
    """S2 の認識失敗(transformers 未導入、写像表に無い記号、強制アライメント失敗など)。"""


# §5.1: IPA母音チャートの基本母音28記号 + R音性母音2記号(ɚ・ɝ) + 拡張母音記号1(ᵻ)。
_VOWEL_BASE_CHARACTERS = frozenset("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɚɝᵻ")


def _classify_symbol(symbol: str) -> Literal["vowel", "consonant"]:
    """音素記号を母音/子音へ分類する(§5.1。言語非依存)。

    NFD 正規化後の先頭の基底文字(長音記号・鼻音化の結合チルダ等の修飾記号は正規化により基底文字の
    後ろに分離される)が母音記号基準集合に含まれれば母音、そうでなければ子音とする。
    """
    if not symbol:
        return "consonant"
    base = unicodedata.normalize("NFD", symbol)[0]
    return "vowel" if base in _VOWEL_BASE_CHARACTERS else "consonant"


# --- §5.2 複合構成(内容認識+G2P+強制アライメント)の区間化 ---

# §5.2 の写像表(確定): pyopenjtalk-plus の音素記号(無声化母音 I/U を含む)を音素モデルの語彙(espeak
# 表記)へ対応付ける。pau・cl はここに含めず、blank トークン(呼び出し側が渡す blank_token_id)へ変換する。
_G2P_TO_VOCAB_SYMBOL: dict[str, str] = {
    "a": "a", "i": "i", "u": "ɯ", "e": "e̞", "o": "o̞",
    "I": "i", "U": "ɯ",
    "k": "k", "ky": "kʲ", "g": "ɡ", "gy": "ɡʲ",
    "s": "s", "sh": "ɕ", "z": "z", "j": "dʑ",
    "t": "t", "ch": "tɕ", "ts": "ts", "d": "d",
    "n": "n", "ny": "ɲ", "h": "h", "hy": "ç", "f": "ɸ",
    "b": "b", "by": "bʲ", "p": "p", "py": "pʲ",
    "m": "m", "my": "mʲ", "y": "j", "r": "ɾ", "ry": "ɾ",
    "w": "w", "v": "v", "N": "ɴ",
}
_BLANK_G2P_SYMBOLS = frozenset({"pau", "cl"})


def _assemble_phoneme_sequence(chunk_phonemes: list[list[str]]) -> list[str]:
    """チャンクごとのG2P音素記号列を pau を挟んで結合する(§5.2手順3)。

    チャンク境界ごとに pau を1つ挟み、列の先頭と末尾にも pau を1つずつ補う。
    """
    sequence = ["pau"]
    for i, phonemes in enumerate(chunk_phonemes):
        if i > 0:
            sequence.append("pau")
        sequence.extend(phonemes)
    sequence.append("pau")
    return sequence


def _g2p_symbols_to_token_ids(symbols: list[str], vocab: dict[str, int], blank_token_id: int) -> list[int]:
    """G2P記号列を音素モデル語彙のトークンID列へ変換する(§5.2手順4)。

    pau・cl は blank_token_id へ変換する。写像表に無い記号、または写像先が実際のモデル語彙に
    無い場合は RecognitionError で停止する(黙って捨てない)。
    """
    token_ids = []
    for symbol in symbols:
        if symbol in _BLANK_G2P_SYMBOLS:
            token_ids.append(blank_token_id)
            continue
        vocab_symbol = _G2P_TO_VOCAB_SYMBOL.get(symbol)
        if vocab_symbol is None:
            raise RecognitionError(f"G2P記号 '{symbol}' の音素モデル語彙への写像が未定義です(§5.2写像表)")
        token_id = vocab.get(vocab_symbol)
        if token_id is None:
            raise RecognitionError(f"音素モデルの語彙に記号 '{vocab_symbol}' が見つかりません")
        token_ids.append(token_id)
    return token_ids


def _forced_align(log_probs: np.ndarray, token_ids: list[int]) -> list[int]:
    """既知のトークン列を対数確率行列へ単調に対応付ける(§5.2手順5。Viterbi)。

    各トークンを1状態とし、フレームごとに「同一状態に留まる」「次のトークンの状態へ進む」の
    2種の遷移のみ許す(読み飛ばし禁止)。各状態の対数確率には token_ids が指すその状態自身の
    語彙IDの列を使う(pau・cl由来の状態のblank列選択は、呼び出し側の _g2p_symbols_to_token_ids が
    そこへ blank_token_id を書き込み済みであることに由来し、本関数はトークン種別を区別しない)。
    フレーム0は状態0に固定し、最終フレームは状態 len(token_ids)-1 に到達している経路の中で
    最尤のものを採る。戻り値は各フレームが対応する状態(トークン列中のindex)。理論上到達不能
    (フレーム数がトークン数未満等)な場合は RecognitionError で停止する。
    """
    num_frames = log_probs.shape[0]
    num_states = len(token_ids)
    if num_states == 0 or num_frames < num_states:
        raise RecognitionError(
            f"強制アライメントが対応付け不能です(フレーム数{num_frames}、トークン数{num_states})"
        )

    emission = log_probs[:, token_ids]  # (num_frames, num_states)
    neg_inf = float("-inf")
    dp = np.full((num_frames, num_states), neg_inf, dtype=np.float64)
    backpointer = np.zeros((num_frames, num_states), dtype=np.int64)

    dp[0, 0] = emission[0, 0]
    for t in range(1, num_frames):
        stay = dp[t - 1, :]
        advance = np.concatenate(([neg_inf], dp[t - 1, :-1]))
        take_advance = advance > stay
        dp[t, :] = np.where(take_advance, advance, stay) + emission[t, :]
        backpointer[t, :] = take_advance.astype(np.int64)

    if dp[num_frames - 1, num_states - 1] == neg_inf:
        raise RecognitionError("強制アライメントが末尾トークンへ到達できませんでした")

    path = [0] * num_frames
    state = num_states - 1
    path[num_frames - 1] = state
    for t in range(num_frames - 1, 0, -1):
        state -= int(backpointer[t, state])
        path[t - 1] = state
    return path


def _path_to_segments(path: list[int], symbols: list[str], frame_duration_sec: float) -> list[Segment]:
    """強制アライメントの状態パスをSegment列へ変換する(§5.2手順6・7)。

    各トークンの区間は、自身の状態が経路上に最初に現れるフレームから次のトークンの状態が最初に
    現れるフレームまで(最後のトークンは音声終端まで)とする。隣接する区間が同一の出力(type・
    phoneme。pau・cl由来はいずれもgap/Noneで同一視される)を持つ場合は1区間へ結合する。
    """
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


def recognize(vocal_wav_path: Path) -> list[Segment]:
    """ボーカルWAVから母音/子音/gapのセグメント列を認識する(§8.1のRecognizerアダプタ契約。§5.2複合構成)。"""
    samples, sample_rate = sf.read(vocal_wav_path, dtype="float32", always_2d=True)
    mono = _downmix_to_mono(samples)
    resampled = _resample_to_target(mono, sample_rate, RECOGNIZER_CONFIG.sample_rate)

    try:
        chunk_texts = _transcribe_with_timestamps(resampled)
    except ImportError as e:
        raise RecognitionError(
            "transformers または torch が見つかりません。導入してください"
            "(vocal-analysis extra で両方導入されます)。"
        ) from e
    except OSError as e:
        raise RecognitionError(
            f"内容認識モデル({WHISPER_CONFIG.model_id}, revision={WHISPER_CONFIG.model_revision})を"
            "取得できません。ネットワーク接続を確認するか、モデルを事前にキャッシュしてください。"
        ) from e

    try:
        chunk_phonemes = [_g2p(text) for text in chunk_texts]
    except ImportError as e:
        raise RecognitionError(
            "pyopenjtalk-plus が見つかりません。導入してください(vocal-analysis extra で導入されます)。"
        ) from e
    phoneme_sequence = _assemble_phoneme_sequence(chunk_phonemes)

    try:
        processor, model = _load_model_and_processor()
    except ImportError as e:
        raise RecognitionError(
            "transformers または torch が見つかりません。導入してください"
            "(vocal-analysis extra で両方導入されます)。"
        ) from e
    except OSError as e:
        raise RecognitionError(
            f"認識モデル({RECOGNIZER_CONFIG.model_id}, revision={RECOGNIZER_CONFIG.model_revision})を"
            "取得できません。ネットワーク接続を確認するか、モデルを事前にキャッシュしてください。"
        ) from e

    vocab = processor.tokenizer.get_vocab()
    blank_token_id = processor.tokenizer.pad_token_id
    token_ids = _g2p_symbols_to_token_ids(phoneme_sequence, vocab, blank_token_id)

    log_probs = _compute_log_probs(processor, model, resampled)
    path = _forced_align(log_probs, token_ids)
    return _path_to_segments(path, phoneme_sequence, FRAME_DURATION_SEC)


def _downmix_to_mono(samples: np.ndarray) -> np.ndarray:
    """複数チャンネルの PCM を平均でモノラルへダウンミックスする(§5.1)。"""
    return samples.mean(axis=1)


def _resample_to_target(mono: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """モノラル PCM を目標サンプルレートへ再サンプリングする(§5.1。多相補間)。"""
    if sample_rate == target_sample_rate:
        return mono
    gcd = math.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd
    return resample_poly(mono, up, down).astype(np.float32)


def _g2p(text: str) -> list[str]:
    """テキストをG2Pで音素記号列へ変換する(§5.2手順2。pyopenjtalk-plus、ルールベース)。"""
    import pyopenjtalk

    return pyopenjtalk.g2p(text, kana=False, join=False)


def _transcribe_with_timestamps(samples: np.ndarray) -> list[str]:
    """Whisperでボーカル音声を書き起こし、チャンクごとのテキストを返す(§5.2手順1)。"""
    pipeline = _load_whisper_pipeline()
    result = pipeline(
        samples,
        return_timestamps=True,
        generate_kwargs={
            "language": "japanese",
            "task": "transcribe",
            "num_beams": 1,
            "do_sample": False,
        },
    )
    return [chunk["text"] for chunk in result["chunks"]]


def _load_whisper_pipeline():
    """内容認識器(Whisper)をS-1測定の固定条件(§8.3)でロードする。"""
    from transformers import pipeline as transformers_pipeline

    return transformers_pipeline(
        "automatic-speech-recognition",
        model=WHISPER_CONFIG.model_id,
        revision=WHISPER_CONFIG.model_revision,
        device=RECOGNIZER_CONFIG.device,
    )


def _compute_log_probs(processor, model, samples: np.ndarray) -> np.ndarray:
    """音素モデルで推論し、フレームごとの対数確率行列を返す(§5.2手順5の入力)。"""
    import torch

    inputs = processor(samples, sampling_rate=RECOGNIZER_CONFIG.sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(RECOGNIZER_CONFIG.device)).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[0].cpu().numpy()


def _load_model_and_processor():
    """音素モデル(強制アライメント用)をS-1測定の固定条件(§5.1・§8.3)でロードする。"""
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    torch.manual_seed(RECOGNIZER_CONFIG.random_seed)
    torch.set_num_threads(RECOGNIZER_CONFIG.num_threads)

    # §5.1: wav2vec2-espeak のトークナイザは既定で espeak ネイティブバイナリ(phonemizer)を
    # 要求する。音素IDのデコードのみが必要で音素へのエンコードは不要なため do_phonemize=False
    # でこの依存を回避する。
    processor = AutoProcessor.from_pretrained(
        RECOGNIZER_CONFIG.model_id,
        revision=RECOGNIZER_CONFIG.model_revision,
        do_phonemize=False,
    )
    model = AutoModelForCTC.from_pretrained(
        RECOGNIZER_CONFIG.model_id,
        revision=RECOGNIZER_CONFIG.model_revision,
        torch_dtype=getattr(torch, RECOGNIZER_CONFIG.dtype),
    )
    model.to(RECOGNIZER_CONFIG.device)
    model.eval()
    return processor, model
