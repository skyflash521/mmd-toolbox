from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import alignment, hallucination, model_store, segmentation, transcription
from .config import (
    DEFAULT_CONTENT_RECOGNIZER_MODEL,
    DEFAULT_ENGLISH_KATAKANA_METHOD,
    DEFAULT_FORCED_ALIGNER,
    RECOGNIZER_CONFIG,
    ContentRecognizerModel,
    EnglishKatakanaMethod,
    ForcedAlignerId,
    SofaAlignerConfig,
)
from .phonemes import RecognitionError
from .transcription import convert_with_g2p
from .types import Segment

__all__ = [
    "RecognitionError", "RecognizerSession", "Transcription", "convert_with_g2p", "recognize",
]


@dataclass(frozen=True)
class _Utterance:
    """時刻は書き起こした音声の先頭を 0 とする。"""

    samples: np.ndarray
    text: str
    words: list[tuple[str, float, float]] | None
    trim_lo_sec: float
    trim_hi_sec: float


@dataclass(frozen=True)
class Transcription:
    """items は時刻順で、確定した gap か、アライメント待ちの発声区間。"""

    threshold: float
    items: tuple[Segment | _Utterance, ...]


@dataclass(frozen=True)
class _Plan:
    items: list[Segment | alignment.AlignJob]
    sofa_targets: list[alignment.SofaTarget]


def _gap(start_sec: float, end_sec: float) -> Segment:
    return Segment(type="gap", start_sec=start_sec, end_sec=end_sec, phoneme=None)


class RecognizerSession:
    """辞書の準備(prepare_dictionary)の後、全チャンクの書き起こし(transcribe)を終えてから全チャンクのアライメント(align)を行う。内容認識
    パイプライン・英単語の変換モデル・音素モデルのロードは、セッション全体で各最大1回。アライメントを
    始めたセッションでは書き起こせない。"""

    def __init__(
        self,
        content_recognizer_model: ContentRecognizerModel = DEFAULT_CONTENT_RECOGNIZER_MODEL,
        *,
        retry: bool = True,
        forced_aligner: ForcedAlignerId = DEFAULT_FORCED_ALIGNER,
        sofa_aligner: SofaAlignerConfig | None = None,
        english_katakana_method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    ):
        """forced_aligner が "sofa-forcedalign" のときは sofa_aligner が必須で、省略すると RecognitionError。"""
        if forced_aligner not in ("wav2vec2-ctc-forcedalign", "sofa-forcedalign"):
            raise RecognitionError(f"未知の forced_aligner です: {forced_aligner!r}")
        if forced_aligner == "sofa-forcedalign" and sofa_aligner is None:
            raise RecognitionError(
                "forced_aligner='sofa-forcedalign' を指定する場合は sofa_aligner(SofaAlignerConfig)が必須です"
            )
        self._content_recognizer_model = content_recognizer_model
        self._retry = retry
        self._forced_aligner = forced_aligner
        self._sofa_aligner = sofa_aligner
        self._english_katakana_method = english_katakana_method
        self._pipeline = None
        self._phoneme_model = None
        self._transcriptions: list[Transcription] = []
        self._plans: dict[int, _Plan] = {}
        self._aligning = False
        self._closed = False

    def __enter__(self) -> "RecognizerSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        holding_models = self._pipeline is not None or self._phoneme_model is not None
        self._pipeline = None
        self._phoneme_model = None
        self._transcriptions.clear()
        self._plans.clear()
        if holding_models:
            model_store.release_gpu_memory()

    def prepare_dictionary(self, on_progress: Callable[[str], None] | None = None) -> None:
        """英単語の変換に使う発音辞書を使える状態にする。無ければ取得し、on_progress に「辞書取得中」を渡す。
        transcribe・align は辞書を取得しないので、辞書が要る処理の前に呼ぶ。取得できないときは
        RecognitionError。"""
        if self._closed:
            raise RuntimeError("閉じたセッションは使えません")
        transcription.prepare_dictionary(on_progress)

    def transcribe(
        self, vocal_wav_path: Path, on_progress: Callable[[str], None] | None = None
    ) -> Transcription:
        if self._closed:
            raise RuntimeError("閉じたセッションは使えません")
        if self._aligning:
            raise RuntimeError("アライメントを始めたセッションでは書き起こせません")

        sample_rate = RECOGNIZER_CONFIG.sample_rate
        samples, file_sample_rate = sf.read(vocal_wav_path, dtype="float32", always_2d=True)
        mono = segmentation.downmix_to_mono(samples)
        resampled = segmentation.resample_to_target(mono, file_sample_rate, sample_rate)
        duration_sec = len(resampled) / sample_rate

        threshold = segmentation.silence_threshold(
            segmentation.frame_rms(resampled, sample_rate, segmentation.SILENCE_FRAME_SEC))
        split_points = segmentation.detect_silence_split_points(resampled, sample_rate)
        segment_bounds = segmentation.build_segment_bounds(duration_sec, split_points)

        items: list[Segment | _Utterance] = []
        segment_count = len(segment_bounds)
        for segment_index, (start_sec, end_sec) in enumerate(segment_bounds, start=1):
            segment_note = (
                f"歌詞書き起こし中: {segmentation.format_time_range(start_sec, end_sec)}"
                f"({segment_index}/{segment_count})"
            )
            if on_progress is not None:
                on_progress(segment_note)
            chunk_samples = resampled[round(start_sec * sample_rate):round(end_sec * sample_rate)]
            items.extend(
                self._transcribe_chunk(
                    chunk_samples, start_sec, end_sec, threshold, segment_note, on_progress))

        result = Transcription(threshold=threshold, items=tuple(items))
        self._transcriptions.append(result)
        return result

    def align(
        self, transcribed: Transcription, on_progress: Callable[[str], None] | None = None
    ) -> list[Segment]:
        """全チャンクの transcribe を終えてから呼ぶ。最初の呼び出しが、内容認識パイプラインの解放、英単語の
        変換、全チャンクの音素列の確定までを行う。"""
        if self._closed:
            raise RuntimeError("閉じたセッションは使えません")
        if not any(transcribed is known for known in self._transcriptions):
            raise ValueError("このセッションで書き起こした結果ではありません")
        if not self._aligning:
            self._begin_alignment(on_progress)
        plan = self._plans[id(transcribed)]

        if self._forced_aligner == "wav2vec2-ctc-forcedalign":
            segments = self._align_with_ctc(plan, transcribed.threshold, on_progress)
        else:
            segments = [
                *(item for item in plan.items if isinstance(item, Segment)),
                *alignment.align_with_sofa(plan.sofa_targets, self._sofa_aligner),
            ]
            segments.sort(key=lambda seg: (seg.start_sec, seg.end_sec))
        return alignment.merge_adjacent_segments(segments)

    def _transcribe_chunk(
        self, chunk_samples: np.ndarray, start_sec: float, end_sec: float, threshold: float,
        segment_note: str, on_progress: Callable[[str], None] | None,
    ) -> list[Segment | _Utterance]:
        sample_rate = RECOGNIZER_CONFIG.sample_rate
        if segmentation.is_segment_silent(chunk_samples, threshold):
            return [_gap(start_sec, end_sec)]

        trim_lo, trim_hi = segmentation.voiced_trim_bounds(chunk_samples, sample_rate, threshold)
        trimmed_tail = trim_hi < len(chunk_samples)
        trim_lo_sec = start_sec + trim_lo / sample_rate if trim_lo > 0 else start_sec
        trim_hi_sec = start_sec + trim_hi / sample_rate if trimmed_tail else end_sec
        items: list[Segment | _Utterance] = []
        if trim_lo > 0:
            items.append(_gap(start_sec, trim_lo_sec))
        chunk_samples = chunk_samples[trim_lo:trim_hi]

        text, words = self._recognize_text(
            chunk_samples, trim_hi_sec - trim_lo_sec, segment_note, on_progress)
        if text.strip():
            items.append(_Utterance(chunk_samples, text, words, trim_lo_sec, trim_hi_sec))
        else:
            items.append(_gap(trim_lo_sec, trim_hi_sec))
        if trimmed_tail:
            items.append(_gap(trim_hi_sec, end_sec))
        return items

    def _recognize_text(
        self, chunk_samples: np.ndarray, duration_sec: float, segment_note: str,
        on_progress: Callable[[str], None] | None,
    ) -> tuple[str, list[tuple[str, float, float]] | None]:
        model = self._content_recognizer_model
        try:
            if self._pipeline is None:
                self._pipeline = transcription.load_pipeline(model, on_progress=on_progress)
            pipeline = self._pipeline
            if on_progress is not None:
                on_progress(segment_note)
            text, words = transcription.transcribe_segment(pipeline, chunk_samples)
        except ImportError as e:
            raise RecognitionError(
                "transformers または torch が見つかりません。導入してください"
                "(vocal-analysis extra で両方導入されます)。"
            ) from e
        except OSError as e:
            raise RecognitionError(
                f"内容認識モデル({model.model_id}, revision={model.model_revision})を"
                "取得できません。ネットワーク接続を確認してください。"
            ) from e

        try:
            return hallucination.resolve_transcription(
                duration_sec, text, words,
                g2p=lambda candidate: transcription.g2p(
                    candidate, method=transcription.DENSITY_G2P_METHOD, on_progress=on_progress),
                retranscribe=(
                    (lambda: transcription.transcribe_text_only(pipeline, chunk_samples))
                    if self._retry else None),
            )
        except ImportError as e:
            raise RecognitionError(
                "pyopenjtalk-plus または transformers が見つかりません。導入してください"
                "(vocal-analysis extra で導入されます)。"
            ) from e

    def _begin_alignment(self, on_progress: Callable[[str], None] | None) -> None:
        self._aligning = True
        if self._pipeline is not None:
            self._pipeline = None
            model_store.release_gpu_memory()
        method = self._english_katakana_method
        transcription.convert_english_words(
            (text for t in self._transcriptions for item in t.items if isinstance(item, _Utterance)
             for text in self._g2p_texts(item)),
            method, on_progress)
        for t in self._transcriptions:
            self._plans[id(t)] = self._plan(t, on_progress)

    @staticmethod
    def _g2p_texts(utterance: _Utterance) -> list[str]:
        if utterance.words:
            return [word_text for word_text, _, _ in utterance.words]
        return [utterance.text]

    def _plan(self, transcribed: Transcription, on_progress: Callable[[str], None] | None) -> _Plan:
        method = self._english_katakana_method
        items: list[Segment | alignment.AlignJob] = []
        sofa_targets: list[alignment.SofaTarget] = []
        for item in transcribed.items:
            if isinstance(item, Segment):
                items.append(item)
                continue
            try:
                if item.words:
                    words_phonemes = [
                        (transcription.g2p(word_text, method=method, on_progress=on_progress), w_start, w_end)
                        for word_text, w_start, w_end in item.words
                    ]
                    phonemes = [p for word_phonemes, _, _ in words_phonemes for p in word_phonemes]
                else:
                    words_phonemes = None
                    phonemes = transcription.g2p(item.text, method=method, on_progress=on_progress)
            except ImportError as e:
                raise RecognitionError(
                    "pyopenjtalk-plus が見つかりません。導入してください(vocal-analysis extra で導入されます)。"
                ) from e

            duration_sec = item.trim_hi_sec - item.trim_lo_sec
            if hallucination.is_hallucinated_phoneme_density(len(phonemes), duration_sec):
                items.append(_gap(item.trim_lo_sec, item.trim_hi_sec))
            elif self._forced_aligner == "wav2vec2-ctc-forcedalign":
                items.append(alignment.build_align_job(
                    item.samples, phonemes, words_phonemes, item.trim_lo_sec, item.trim_hi_sec))
            else:
                gaps, targets = alignment.build_sofa_targets(
                    item.samples, phonemes, words_phonemes, item.trim_lo_sec, duration_sec)
                items.extend(gaps)
                sofa_targets.extend(targets)
        return _Plan(items=items, sofa_targets=sofa_targets)

    def _align_with_ctc(
        self, plan: _Plan, threshold: float, on_progress: Callable[[str], None] | None
    ) -> list[Segment]:
        jobs = [item for item in plan.items if isinstance(item, alignment.AlignJob)]
        if not jobs:
            return list(plan.items)

        processor, model, vocab, blank_token_id = self._ensure_phoneme_model(on_progress)
        segments: list[Segment] = []
        job_index = 0
        for item in plan.items:
            if not isinstance(item, alignment.AlignJob):
                segments.append(item)
                continue
            job_index += 1
            if on_progress is not None:
                on_progress(
                    "音素アライメント中: "
                    f"{segmentation.format_time_range(item.trim_lo_sec, item.trim_hi_sec)}"
                    f"({job_index}/{len(jobs)})"
                )
            segments.extend(
                alignment.align_job(processor, model, vocab, blank_token_id, threshold, item))
        return segments

    def _ensure_phoneme_model(self, on_progress: Callable[[str], None] | None):
        if self._phoneme_model is None:
            try:
                processor, model = alignment.load_phoneme_model(on_progress=on_progress)
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
            self._phoneme_model = (
                processor, model, processor.tokenizer.get_vocab(), processor.tokenizer.pad_token_id)
        return self._phoneme_model


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
    with RecognizerSession(
        content_recognizer_model, retry=retry, forced_aligner=forced_aligner,
        sofa_aligner=sofa_aligner, english_katakana_method=english_katakana_method,
    ) as session:
        session.prepare_dictionary(on_progress=on_progress)
        return session.align(
            session.transcribe(vocal_wav_path, on_progress=on_progress), on_progress=on_progress)
