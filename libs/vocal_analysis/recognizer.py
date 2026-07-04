"""S2 音素/母音認識(vocal_analysis.md §5・§5.1・§5.2・§8.1・§8.3)。

無音検出による区間分割・内容認識・G2P(pyopenjtalk-plus)・音素モデルのCTC強制アライメント(§5.2)を
組み合わせた複合構成で、母音/子音/gap を区別したセグメント列を生成する。公開関数
recognize(vocal_wav_path, adapter_id) -> list[Segment] が唯一の公開面(Recognizer アダプタ契約。§8.1)。
内容認識モデルは既定アダプタ `kana-whisper-ctc-forcedalign`(kana-whisper)と選択可能な代替アダプタ
`whisper-ctc-forcedalign`(whisper-medium+かな限定プロンプト)を切り替えられる(§5.2)。
"""

import math
import unicodedata
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import KANA_WHISPER_CONFIG, RECOGNIZER_CONFIG, WHISPER_CONFIG
from .types import Segment

DEFAULT_ADAPTER_ID = "kana-whisper-ctc-forcedalign"
ALTERNATE_ADAPTER_ID = "whisper-ctc-forcedalign"

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


# --- §5.2 手順1・2: 無音検出による区間分割 ---

_SILENCE_FRAME_SEC = 0.1  # §5.2手順1: 無音検出用フレーム幅
_SILENCE_THRESHOLD_DB_BELOW_PEAK = 30.0  # §5.2手順1: ピーク(95パーセンタイル)から下回るdB
_SILENCE_MIN_RUN_SEC = 0.6  # §5.2手順1: 分割点とみなす無音区間の最小長
_SEGMENT_MIN_SEC = 1.5  # §5.2手順1: 区間の最小長(未満は次の分割点まで結合)
_SEGMENT_MAX_SEC = 25.0  # §5.2手順1: 区間の最大長(超過は均等分割)


def _frame_rms(mono: np.ndarray, sample_rate: int, frame_sec: float) -> np.ndarray:
    """一定幅の連続フレームに区切ってRMSを求める(§5.2手順1)。末尾の不完全フレームは切り捨てる。"""
    frame_len = max(1, round(frame_sec * sample_rate))
    num_frames = len(mono) // frame_len
    if num_frames == 0:
        return np.array([], dtype=np.float64)
    trimmed = mono[: num_frames * frame_len].astype(np.float64).reshape(num_frames, frame_len)
    return np.sqrt(np.mean(np.square(trimmed), axis=1))


def _silence_threshold(frame_rms: np.ndarray) -> float:
    """フレームRMS列から無音しきい値(95パーセンタイルのピークから30dB下)を求める(§5.2手順1)。

    ピークが0(全フレームRMSが0)の場合はしきい値も0になり、RMSがしきい値"以下"かどうかで判定する
    呼び出し側(手順1・2)がRMS0のフレーム・区間を過不足なく無音と判定する。
    """
    if frame_rms.size == 0:
        return 0.0
    peak = float(np.percentile(frame_rms, 95))
    return peak * (10 ** (-_SILENCE_THRESHOLD_DB_BELOW_PEAK / 20.0))


def _detect_silence_split_points(mono: np.ndarray, sample_rate: int) -> list[float]:
    """無音区間(しきい値以下が0.6秒以上連続)の中点を分割点の時刻(秒)として返す(§5.2手順1)。"""
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
    """分割点から、最小長・最大長の制約を満たす区間の(開始, 終了)秒の列を作る(§5.2手順1)。

    分割点(0秒・音声終端を含む)で区切られた各区間を左から順に長さを累積し、1.5秒未満の区間は
    次の分割点まで結合を続ける(音声終端まで結合しても1.5秒に満たない場合はそのまま採用する)。
    結合後、25秒を超える区間は等分割して25秒以下の区間へ均等に分ける。
    """
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


def _is_segment_silent(segment_samples: np.ndarray, threshold: float) -> bool:
    """区間全体のRMSがしきい値以下かどうかを判定する(§5.2手順2)。"""
    if segment_samples.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(segment_samples.astype(np.float64)))))
    return rms <= threshold


def _merge_adjacent_segments(segments: list[Segment]) -> list[Segment]:
    """隣接する同一 type・phoneme の Segment を1つへ結合する(§5.2手順9。区間境界をまたぐ結合)。"""
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


# --- §5.2 手順4以降: 内容認識+G2P+強制アライメントの区間化 ---

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
    """チャンクごとのG2P音素記号列を pau を挟んで結合する(§5.2手順5)。

    チャンク境界ごとに pau を1つ挟み、列の先頭と末尾にも pau を1つずつ補う(区間ごとに独立して
    Whisper呼び出しを行う現行パイプラインでは chunk_phonemes は常に1要素で呼ばれ、実質的に
    その区間の音素記号列の先頭・末尾へ pau を補う処理になる)。
    """
    sequence = ["pau"]
    for i, phonemes in enumerate(chunk_phonemes):
        if i > 0:
            sequence.append("pau")
        sequence.extend(phonemes)
    sequence.append("pau")
    return sequence


def _g2p_symbols_to_token_ids(symbols: list[str], vocab: dict[str, int], blank_token_id: int) -> list[int]:
    """G2P記号列を音素モデル語彙のトークンID列へ変換する(§5.2手順6)。

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


_FORCED_ALIGN_BAND_SEC = 1.0  # §5.2手順7: blank支配下での押し込み崩壊を防ぐ位置バンド幅


def _forced_align(log_probs: np.ndarray, token_ids: list[int]) -> list[int]:
    """既知のトークン列を対数確率行列へ単調に対応付ける(§5.2手順7。バンド制限Viterbi)。

    各トークンを1状態とし、フレームごとに「同一状態に留まる」「次のトークンの状態へ進む」の
    2種の遷移のみ許す(読み飛ばし禁止)。各状態の対数確率には token_ids が指すその状態自身の
    語彙IDの列を使う(pau・cl由来の状態のblank列選択は、呼び出し側の _g2p_symbols_to_token_ids が
    そこへ blank_token_id を書き込み済みであることに由来し、本関数はトークン種別を区別しない)。

    **位置バンド制限**: 状態 l の期待フレーム位置を (l / (L-1)) * (T-1)(トークン列を区間内へ均等
    割り当てした場合の位置。L=1 なら0)とし、フレーム t が状態 l に遷移できるのは
    |t - 期待フレーム位置| <= バンド幅 を満たす場合に限る(バンド幅は1.0秒に相当するフレーム数で
    固定)。blank 支配下でも均等割り当てから大きく外れた押し込み崩壊を構造的に防ぐ。

    フレーム0は状態0に固定し、最終フレームは状態 len(token_ids)-1 に到達している経路の中で
    最尤のものを採る。戻り値は各フレームが対応する状態(トークン列中のindex)。理論上到達不能
    (フレーム数がトークン数未満、またはバンド制限により到達不能な場合)は RecognitionError で
    停止する。
    """
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

    out_of_band = np.array(
        [[abs(t - expected_frame(l)) > band_frames for l in range(num_states)] for t in range(num_frames)]
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
        candidate = np.where(take_advance, advance, stay) + emission[t, :]
        candidate[out_of_band[t]] = neg_inf
        dp[t, :] = candidate
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
    """強制アライメントの状態パスをSegment列へ変換する(§5.2手順8・9)。

    各トークンの区間は、自身の状態が経路上に最初に現れるフレームから次のトークンの状態が最初に
    現れるフレームまで(最後のトークンは区間終端まで)とする。隣接する区間が同一の出力(type・
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


def recognize(vocal_wav_path: Path, adapter_id: str = DEFAULT_ADAPTER_ID) -> list[Segment]:
    """ボーカルWAVから母音/子音/gapのセグメント列を認識する(§8.1のRecognizerアダプタ契約。§5.2複合構成)。

    adapter_id で内容認識モデルを選択する(既定 `kana-whisper-ctc-forcedalign`・選択可能な代替
    `whisper-ctc-forcedalign`。§5.2・§8.2)。後段のG2P・強制アライメントはどちらも共通。
    """
    if adapter_id == DEFAULT_ADAPTER_ID:
        transcribe_segment = _transcribe_segment
        content_model_id, content_model_revision = KANA_WHISPER_CONFIG.model_id, KANA_WHISPER_CONFIG.model_revision
    elif adapter_id == ALTERNATE_ADAPTER_ID:
        transcribe_segment = _transcribe_segment_whisper_medium
        content_model_id, content_model_revision = WHISPER_CONFIG.model_id, WHISPER_CONFIG.model_revision
    else:
        raise ValueError(
            f"未知の内容認識アダプタです: {adapter_id!r}"
            f"({DEFAULT_ADAPTER_ID!r} か {ALTERNATE_ADAPTER_ID!r} を指定してください)"
        )

    samples, sample_rate = sf.read(vocal_wav_path, dtype="float32", always_2d=True)
    mono = _downmix_to_mono(samples)
    resampled = _resample_to_target(mono, sample_rate, RECOGNIZER_CONFIG.sample_rate)
    duration_sec = len(resampled) / RECOGNIZER_CONFIG.sample_rate

    frame_rms = _frame_rms(resampled, RECOGNIZER_CONFIG.sample_rate, _SILENCE_FRAME_SEC)
    threshold = _silence_threshold(frame_rms)
    split_points = _detect_silence_split_points(resampled, RECOGNIZER_CONFIG.sample_rate)
    segment_bounds = _build_segment_bounds(duration_sec, split_points)

    # 音素モデルは無音でない区間が実際に現れるまでロードしない(§5.2手順2: 全区間が無音なら
    # モデルを一切必要としない。非無音区間でもWhisper・G2Pより先にロードして失敗境界を隠さない)。
    processor = model = vocab = blank_token_id = None

    all_segments: list[Segment] = []
    for start_sec, end_sec in segment_bounds:
        start_index = round(start_sec * RECOGNIZER_CONFIG.sample_rate)
        end_index = round(end_sec * RECOGNIZER_CONFIG.sample_rate)
        chunk_samples = resampled[start_index:end_index]

        if _is_segment_silent(chunk_samples, threshold):
            all_segments.append(
                Segment(type="gap", start_sec=start_sec, end_sec=end_sec, phoneme=None, confidence=None)
            )
            continue

        try:
            text = transcribe_segment(chunk_samples)
        except ImportError as e:
            raise RecognitionError(
                "transformers または torch が見つかりません。導入してください"
                "(vocal-analysis extra で両方導入されます)。"
            ) from e
        except OSError as e:
            raise RecognitionError(
                f"内容認識モデル({content_model_id}, revision={content_model_revision})を"
                "取得できません。ネットワーク接続を確認するか、モデルを事前にキャッシュしてください。"
            ) from e

        try:
            phonemes = _g2p(text)
        except ImportError as e:
            raise RecognitionError(
                "pyopenjtalk-plus が見つかりません。導入してください(vocal-analysis extra で導入されます)。"
            ) from e

        if processor is None:
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

        seq = _assemble_phoneme_sequence([phonemes])
        token_ids = _g2p_symbols_to_token_ids(seq, vocab, blank_token_id)
        log_probs = _compute_log_probs(processor, model, chunk_samples)
        path = _forced_align(log_probs, token_ids)
        local_segments = _path_to_segments(path, seq, FRAME_DURATION_SEC)
        # 最後の区切りは対数確率行列のフレーム数に由来する終端(local_segments[-1].end_sec)を
        # 使わず、区間自身の真の終端(end_sec - start_sec)へ強制的に揃える(フレーム数計算の
        # 丸め等で区間境界とわずかにずれ、隣接区間との欠落・重複を生むことを防ぐ)。
        if local_segments:
            last = local_segments[-1]
            local_segments[-1] = Segment(
                type=last.type, start_sec=last.start_sec, end_sec=end_sec - start_sec,
                phoneme=last.phoneme, confidence=None,
            )
        for seg in local_segments:
            all_segments.append(
                Segment(
                    type=seg.type,
                    start_sec=seg.start_sec + start_sec,
                    end_sec=seg.end_sec + start_sec,
                    phoneme=seg.phoneme,
                    confidence=None,
                )
            )

    return _merge_adjacent_segments(all_segments)


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
    """テキストをG2Pで音素記号列へ変換する(§5.2手順4。pyopenjtalk-plus、ルールベース)。"""
    import pyopenjtalk

    return pyopenjtalk.g2p(text, kana=False, join=False)


def _transcribe_segment(samples: np.ndarray) -> str:
    """区間のボーカル音声をkana-whisperで書き起こす(§5.2手順3。既定アダプタ
    kana-whisper-ctc-forcedalign。区間ごとに独立呼び出し)。かなを直接返す。"""
    pipeline = _load_kana_whisper_pipeline()
    result = pipeline(
        samples,
        generate_kwargs={
            "language": "japanese",
            "task": "transcribe",
            "num_beams": 1,
            "do_sample": False,
        },
    )
    return result["text"]


def _transcribe_segment_whisper_medium(samples: np.ndarray) -> str:
    """区間のボーカル音声をwhisper-medium+かな限定プロンプトで書き起こす(§5.2手順3。選択可能な
    代替アダプタ whisper-ctc-forcedalign。区間ごとに独立呼び出し)。かな化は区間により成功・失敗する。"""
    pipeline = _load_whisper_pipeline()
    prompt_ids = pipeline.tokenizer.get_prompt_ids(WHISPER_CONFIG.kana_prompt, return_tensors="pt").to(
        pipeline.device
    )
    result = pipeline(
        samples,
        generate_kwargs={
            "language": "japanese",
            "task": "transcribe",
            "num_beams": 1,
            "do_sample": False,
            "prompt_ids": prompt_ids,
        },
    )
    return result["text"]


def _load_kana_whisper_pipeline():
    """内容認識器(kana-whisper)をS-1測定の固定条件(§8.3)でロードする。"""
    from transformers import pipeline as transformers_pipeline

    return transformers_pipeline(
        "automatic-speech-recognition",
        model=KANA_WHISPER_CONFIG.model_id,
        revision=KANA_WHISPER_CONFIG.model_revision,
        device=RECOGNIZER_CONFIG.device,
    )


def _load_whisper_pipeline():
    """内容認識器(whisper-medium)をS-1測定の固定条件(§8.3)でロードする。"""
    from transformers import pipeline as transformers_pipeline

    return transformers_pipeline(
        "automatic-speech-recognition",
        model=WHISPER_CONFIG.model_id,
        revision=WHISPER_CONFIG.model_revision,
        device=RECOGNIZER_CONFIG.device,
    )


def _compute_log_probs(processor, model, samples: np.ndarray) -> np.ndarray:
    """音素モデルで推論し、フレームごとの対数確率行列を返す(§5.2手順7の入力)。"""
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
