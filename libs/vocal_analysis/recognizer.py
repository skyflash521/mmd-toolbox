"""S2 音素/母音認識(vocal_analysis.md §5・§5.1・§8.1・§8.3)。

CTC系認識のフレーム列を、母音/子音/gap を区別したセグメント列へ正規化する。公開関数
recognize(vocal_wav_path) -> list[Segment] が wav2vec2 の実行からセグメント列生成までを担う。
"""

import math
import unicodedata
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import RECOGNIZER_CONFIG
from .types import Segment

FRAME_DURATION_SEC = 0.02  # §5.1: 採用モデルの畳み込み総ストライド320サンプル@16kHzで固定
MIN_SEGMENT_DURATION_SEC = 0.06  # §5.1: 60ms吸収の閾値


class RecognitionError(Exception):
    """S2 の認識失敗(transformers 未導入など、原因が分かるエラー)。"""

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


def _merge_ctc_frames(frame_symbols: list[str | None], frame_duration: float) -> list[Segment]:
    """CTC出力のフレーム列をセグメント列へ正規化する(§5.1「区間化」)。

    連続する同一ラベル(同一音素記号、または連続する blank)を1区間に連結する。blank は None で表し
    gap になる。confidence は算出せず常に None(§2.1で任意のため)。全時間軸を隙間なく被覆する。
    """
    segments: list[Segment] = []
    start_index = 0
    for i in range(1, len(frame_symbols) + 1):
        if i < len(frame_symbols) and frame_symbols[i] == frame_symbols[start_index]:
            continue
        symbol = frame_symbols[start_index]
        seg_type = "gap" if symbol is None else _classify_symbol(symbol)
        segments.append(
            Segment(
                type=seg_type,
                start_sec=start_index * frame_duration,
                end_sec=i * frame_duration,
                phoneme=symbol,
                confidence=None,
            )
        )
        start_index = i
    return segments


def _absorb_short_segments(segments: list[Segment], min_duration_sec: float) -> list[Segment]:
    """最小区間長を満たさない区間を隣接へ吸収する(§5.1の60ms吸収規則)。

    吸収先はより長い隣接(母音/子音)区間、同長なら直前。先頭/末尾で片側しか無ければその側。
    gap は吸収先にしない。非gapの隣接が無い場合はその短区間自体を gap にする。

    二相で行う。フェーズA: 非gapの吸収先を持つ短区間を、無くなるまで繰り返し吸収する(各吸収で
    区間数が1つ減るため必ず停止する)。フェーズB: フェーズA終了時点でなお残る短区間(非gapの
    隣接を持たない、という理由で吸収できなかったもの)を一括で gap 化し隣接 gap と連結する
    (1回限りの終端処理。ここで生じた gap を再び60ms判定にかけて無限に処理し直すことはしない。
    §5.1の「その短区間自体を gap にする」は最終形としての指定であり、以後の再判定は要求しない)。
    """
    result = list(segments)
    while True:
        absorption = _find_absorption_target(result, min_duration_sec)
        if absorption is None:
            break
        short_index, target_index = absorption
        result = _absorb_into(result, short_index, target_index)

    result = [
        seg
        if seg.end_sec - seg.start_sec >= min_duration_sec
        else Segment(type="gap", start_sec=seg.start_sec, end_sec=seg.end_sec, phoneme=None, confidence=None)
        for seg in result
    ]
    return _merge_adjacent_same_type(result)


def _find_absorption_target(segments: list[Segment], min_duration_sec: float) -> tuple[int, int] | None:
    """吸収可能な最初の短区間について (短区間の index, 吸収先の index) を返す。

    非gapの隣接(より長い方。同長なら直前)が無ければ None(=フェーズBで処理する対象)として
    その短区間はスキップし、他の短区間を探し続ける。
    """
    for i, seg in enumerate(segments):
        if seg.end_sec - seg.start_sec >= min_duration_sec:
            continue

        left_index = i - 1 if i > 0 else None
        right_index = i + 1 if i + 1 < len(segments) else None
        # gap は吸収先にしないため、隣接が gap の側は候補から外す。
        if left_index is not None and segments[left_index].type == "gap":
            left_index = None
        if right_index is not None and segments[right_index].type == "gap":
            right_index = None

        if left_index is None and right_index is None:
            continue
        if left_index is None:
            return i, right_index
        if right_index is None:
            return i, left_index
        left_duration = segments[left_index].end_sec - segments[left_index].start_sec
        right_duration = segments[right_index].end_sec - segments[right_index].start_sec
        return (i, left_index) if left_duration >= right_duration else (i, right_index)
    return None


def _absorb_into(segments: list[Segment], short_index: int, target_index: int) -> list[Segment]:
    """short_index の区間を target_index の区間へ吸収し、区間長を合算した新しいリストを返す。"""
    short_seg = segments[short_index]
    target_seg = segments[target_index]
    merged_start = min(short_seg.start_sec, target_seg.start_sec)
    merged_end = max(short_seg.end_sec, target_seg.end_sec)
    merged = Segment(
        type=target_seg.type,
        start_sec=merged_start,
        end_sec=merged_end,
        phoneme=target_seg.phoneme,
        confidence=None,
    )
    keep_index = min(short_index, target_index)
    drop_index = max(short_index, target_index)
    result = list(segments)
    result[keep_index] = merged
    del result[drop_index]
    return result


def _merge_adjacent_same_type(segments: list[Segment]) -> list[Segment]:
    """隣接する同一 type(かつ gap は phoneme も None どうし)の区間を1つに連結する。"""
    result: list[Segment] = []
    for seg in segments:
        if result and result[-1].type == seg.type and result[-1].phoneme == seg.phoneme:
            prev = result[-1]
            result[-1] = Segment(
                type=prev.type,
                start_sec=prev.start_sec,
                end_sec=seg.end_sec,
                phoneme=prev.phoneme,
                confidence=None,
            )
        else:
            result.append(seg)
    return result


def recognize(vocal_wav_path: Path) -> list[Segment]:
    """ボーカルWAVから母音/子音/gapのセグメント列を認識する(§8.1のRecognizerアダプタ契約)。"""
    samples, sample_rate = sf.read(vocal_wav_path, dtype="float32", always_2d=True)
    mono = _downmix_to_mono(samples)
    resampled = _resample_to_target(mono, sample_rate, RECOGNIZER_CONFIG.sample_rate)

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

    token_ids = _run_model_inference(processor, model, resampled)
    decoder = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    frame_symbols = _ids_to_symbols(token_ids, processor.tokenizer.pad_token_id, decoder)

    segments = _merge_ctc_frames(frame_symbols, FRAME_DURATION_SEC)
    return _absorb_short_segments(segments, MIN_SEGMENT_DURATION_SEC)


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


def _ids_to_symbols(token_ids: list[int], pad_token_id: int, decoder: dict[int, str]) -> list[str | None]:
    """CTC出力のトークンID列を音素記号列へ変換する(blank=pad_token_idはNone)。"""
    return [None if tid == pad_token_id else decoder[tid] for tid in token_ids]


def _run_model_inference(processor, model, samples: np.ndarray) -> list[int]:
    """wav2vec2 モデルで推論し、フレームごとの argmax トークンID列を返す。"""
    import torch

    inputs = processor(samples, sampling_rate=RECOGNIZER_CONFIG.sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(RECOGNIZER_CONFIG.device)).logits
    return torch.argmax(logits, dim=-1)[0].tolist()


def _load_model_and_processor():
    """認識器(wav2vec2 espeak)をS-1測定の固定条件(§5.1・§8.3)でロードする。"""
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
