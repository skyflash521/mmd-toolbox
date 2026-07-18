"""S2 音素/母音認識。

無音検出による区間分割・内容認識・G2P(pyopenjtalk-plus)は経路共通(手順1〜4)。強制アライメント段は
`forced_aligner`引数で選択できる: 既定のwav2vec2 CTC強制アライメントと、SOFA経路
(`sofa_align`モジュールへ委譲)。公開関数 recognize() が唯一の公開面(Recognizerアダプタ契約。
契約は`forced_aligner`の選択に関わらず不変)。内容認識モデルは `content_recognizer_model`
(`ContentRecognizerModel`)で指定する(既定値 `DEFAULT_CONTENT_RECOGNIZER_MODEL`・候補値
`KANA_WHISPER_MODEL`・任意指定も可)。`retry` はエコー幻覚・反復幻覚へのトリガ式リトライの
有効/無効を切り替える(既定True。主モデル自身をプロンプト無しで再認識する。別モデルは使わない)。
"""

import math
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import sofa_align
from .config import (
    DEFAULT_CONTENT_RECOGNIZER_MODEL,
    DEFAULT_ENGLISH_OOV_KATAKANA_METHOD,
    DEFAULT_FORCED_ALIGNER,
    KANA_PROMPT,
    RECOGNIZER_CONFIG,
    ContentRecognizerModel,
    EnglishOovKatakanaMethod,
    ForcedAlignerId,
    SofaAlignerConfig,
)
from .english_oov_katakana import convert_oov_words
from .phonemes import (
    _BLANK_G2P_SYMBOLS,
    _MIN_WORD_DURATION_SEC,
    RecognitionError,
    _classify_symbol,
    _G2P_TO_VOCAB_SYMBOL,
)
from .types import Segment

FRAME_DURATION_SEC = 0.02  # 採用モデルの畳み込み総ストライド320サンプル@16kHzで固定


# --- 手順1・2: 無音検出による区間分割 ---

_SILENCE_FRAME_SEC = 0.1  # 手順1: 無音検出用フレーム幅
_SILENCE_THRESHOLD_DB_BELOW_PEAK = 30.0  # 手順1: ピーク(95パーセンタイル)から下回るdB
_SILENCE_MIN_RUN_SEC = 0.6  # 手順1: 分割点とみなす無音区間の最小長
_SEGMENT_MIN_SEC = 1.5  # 手順1: 区間の最小長(未満は次の分割点まで結合)
_SEGMENT_MAX_SEC = 25.0  # 手順1: 区間の最大長(超過は均等分割)
_TRIM_MARGIN_SEC = 0.1  # 手順2: 有声スパンの外側に残す余白(しきい値未満の子音の助走を切らない)


def _frame_rms(mono: np.ndarray, sample_rate: int, frame_sec: float) -> np.ndarray:
    """一定幅の連続フレームに区切ってRMSを求める(手順1)。末尾の不完全フレームは切り捨てる。"""
    frame_len = max(1, round(frame_sec * sample_rate))
    num_frames = len(mono) // frame_len
    if num_frames == 0:
        return np.array([], dtype=np.float64)
    trimmed = mono[: num_frames * frame_len].astype(np.float64).reshape(num_frames, frame_len)
    return np.sqrt(np.mean(np.square(trimmed), axis=1))


def _silence_threshold(frame_rms: np.ndarray) -> float:
    """フレームRMS列から無音しきい値(95パーセンタイルのピークから30dB下)を求める(手順1)。

    ピークが0(全フレームRMSが0)の場合はしきい値も0になり、RMSがしきい値"以下"かどうかで判定する
    呼び出し側(手順1・2)がRMS0のフレーム・区間を過不足なく無音と判定する。
    """
    if frame_rms.size == 0:
        return 0.0
    peak = float(np.percentile(frame_rms, 95))
    return peak * (10 ** (-_SILENCE_THRESHOLD_DB_BELOW_PEAK / 20.0))


def _detect_silence_split_points(mono: np.ndarray, sample_rate: int) -> list[float]:
    """無音区間(しきい値以下が0.6秒以上連続)の中点を分割点の時刻(秒)として返す(手順1)。"""
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
    """分割点から、最小長・最大長の制約を満たす区間の(開始, 終了)秒の列を作る(手順1)。

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
    """区間全体のRMSがしきい値以下かどうかを判定する(手順2)。"""
    if segment_samples.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(segment_samples.astype(np.float64)))))
    return rms <= threshold


def _voiced_trim_bounds(segment_samples: np.ndarray, sample_rate: int, threshold: float) -> tuple[int, int]:
    """無音でない区間の有声スパン+余白のサンプル範囲を返す(手順2のトリム)。

    しきい値を上回る最初のフレームの開始から最後のフレームの終了までを有声スパンとし、
    その外側へ余白(_TRIM_MARGIN_SEC)を加えて区間内へクランプする。有声フレームが1つも
    無い場合(区間全体RMSはしきい値超だがフレーム単位では全て以下、の端ケース)は
    トリムせず区間全体を返す。
    """
    frame_rms = _frame_rms(segment_samples, sample_rate, _SILENCE_FRAME_SEC)
    voiced = np.nonzero(frame_rms > threshold)[0]
    if voiced.size == 0:
        return 0, len(segment_samples)
    frame_len = max(1, round(_SILENCE_FRAME_SEC * sample_rate))
    margin = round(_TRIM_MARGIN_SEC * sample_rate)
    lo = max(0, int(voiced[0]) * frame_len - margin)
    hi = min(len(segment_samples), (int(voiced[-1]) + 1) * frame_len + margin)
    return lo, hi


# --- 手順4以降: 内容認識+G2P+強制アライメントの区間化 ---

_HALLUCINATION_PHONEME_RATE = 20.0  # 手順4: 反復幻覚を疑う音素密度のしきい値(音素/秒)
_ECHO_FRAGMENT_PREFIX_LEN = 5  # エコー断片とみなすプロンプト文との共通接頭辞の最小文字数
_REPEAT_MIN_COUNT = 3  # 末尾反復として検出する最小繰り返し数
_REPEAT_MIN_TOTAL_CHARS = 8  # 末尾反復として検出する最小総長(文字)
_REPEAT_NORMALIZE_MORA_RATE = 3.5  # 全文反復の個数正規化レート(モーラ/秒)
_MORA_G2P_SYMBOLS = frozenset({"a", "i", "u", "e", "o", "I", "U", "N"})
# 反復ハルシネーション時の1呼び出しあたりの生成時間を有界化する上限(貪欲デコードは接頭辞不変のため、
# 上限未到達の正常な書き起こしの結果は変わらない)。プロンプト無しの再認識(リトライ)はプロンプト分の
# トークンを消費しない分、区間書き起こし(プロンプトあり)より上限を大きく取れる。いずれもWhisperの
# 最大コンテキスト長を超えないよう選定済み。
_SEGMENT_MAX_NEW_TOKENS = 380
_RETRY_MAX_NEW_TOKENS = 420


def _normalize_sentence(sentence: str) -> str:
    """文の照合用正規化(空白・句読点を除去)。エコー除去の照合に使う。"""
    return re.sub(r"[\s、。,.]", "", sentence)


_PROMPT_SENTENCES = frozenset(
    normalized for normalized in (_normalize_sentence(part) for part in KANA_PROMPT.split("。")) if normalized
)


def _is_hallucinated_phoneme_density(phoneme_count: int, duration_sec: float) -> bool:
    """音素密度(音素/秒)が反復幻覚を疑うしきい値を超えるかを判定する(手順4)。"""
    if duration_sec <= 0:
        return False
    return phoneme_count / duration_sec > _HALLUCINATION_PHONEME_RATE


def _strip_prompt_echo(text: str) -> tuple[str, bool]:
    """かな限定プロンプトの構成文と一致する文(エコー幻覚)を除去する。

    句点で文に分割し、正規化(空白・句読点除去)した各文がプロンプト構成文と一致するものを落とす。
    1文以上除去した場合、残りの末尾の文がプロンプト構成文と _ECHO_FRAGMENT_PREFIX_LEN 文字以上の
    共通接頭辞を持てば、区間末尾で切れたエコーの断片として同様に落とす。
    戻り値は (除去後テキスト, 除去が起きたか)。
    """
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
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _find_suffix_repetition(text: str) -> tuple[str, int, str] | None:
    """テキスト末尾の反復 単位×k を検出する(反復救済に使う)。

    末尾から一致する繰り返しを単位長1文字から順に調べ、繰り返し数が _REPEAT_MIN_COUNT 以上かつ
    総長が _REPEAT_MIN_TOTAL_CHARS 文字以上の候補のうち総切除長が最大のものを採る。
    戻り値は (単位, 繰り返し数, 反復を除いた先頭部) か None(反復なし)。
    """
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


def _text_mora_count(text: str, method: EnglishOovKatakanaMethod = DEFAULT_ENGLISH_OOV_KATAKANA_METHOD) -> int:
    """テキストのモーラ数(G2P結果の母音・撥音の数)。反復救済の個数正規化に使う。"""
    return sum(1 for symbol in _g2p(text, method=method) if symbol in _MORA_G2P_SYMBOLS)


def _text_phoneme_density(
    text: str, duration_sec: float, method: EnglishOovKatakanaMethod = DEFAULT_ENGLISH_OOV_KATAKANA_METHOD
) -> float:
    """テキスト全体をG2Pした音素密度(音素/秒)。エコー・反復のトリガ判定に使う。"""
    if not text.strip() or duration_sec <= 0:
        return 0.0
    return len(_g2p(text, method=method)) / duration_sec


def _resolve_transcription(
    chunk_samples: np.ndarray,
    trim_duration_sec: float,
    text: str,
    words: list[tuple[str, float, float]] | None,
    content_recognizer_model: ContentRecognizerModel,
    retry_enabled: bool,
    english_oov_katakana_method: EnglishOovKatakanaMethod = DEFAULT_ENGLISH_OOV_KATAKANA_METHOD,
) -> tuple[str, list[tuple[str, float, float]] | None]:
    """書き起こしの後処理: エコー除去→トリガ式区間リトライ→反復救済。

    採用する (テキスト, 単語タイムスタンプ) を返す。テキストが空になった場合は呼び出し側が
    区間をgap確定する。エコー除去・トリガ式リトライ・反復救済でテキストが変化した場合、残った文と
    単語の対応が保証できないため単語タイムスタンプを破棄する(単語窓が無い区間として位置バンド
    制限で整列するフォールバックに帰着させる)。
    """
    text2, echo_removed = _strip_prompt_echo(text)
    if echo_removed:
        words = None

    def _needs_retry(candidate: str) -> bool:
        # トリガ式リトライの条件はエコー幻覚(エコー除去で空になった場合)と反復幻覚(音素密度
        # 超過)のみ。ASCII英字の残存はリトライ条件から除外済み。
        if not candidate.strip():
            # 元から空の書き起こしは既存のgap確定に委ねる。エコー除去で空になった場合のみ再認識する。
            return echo_removed
        density = _text_phoneme_density(candidate, trim_duration_sec, method=english_oov_katakana_method)
        return density > _HALLUCINATION_PHONEME_RATE

    if retry_enabled and _needs_retry(text2):
        # トリガ式リトライは主モデル自身で、プロンプト無し・タイムスタンプ無しに再認識する。
        # エコーはプロンプト起因の幻覚のため、プロンプトを外すだけで同じモデルでも回復できる。
        # 別モデルはここには使わない(別モデルを主モデルと同時にGPUへ常駐させるとVRAMを圧迫し
        # 推論が不安定になり、処理時間を有界にできない)。
        # 単語タイムスタンプは要求せず破棄する(位置バンド制限で整列するフォールバックへ帰着)。
        retry_text = _transcribe_text_only(chunk_samples, content_recognizer_model).strip()
        retry_text, _retry_echo_removed = _strip_prompt_echo(retry_text)
        text2, words = retry_text, None

    density = _text_phoneme_density(text2, trim_duration_sec, method=english_oov_katakana_method)
    if text2.strip() and density > _HALLUCINATION_PHONEME_RATE:
        repetition = _find_suffix_repetition(text2)
        if repetition is not None:
            unit, count, head = repetition
            rescued = text2
            if head.strip():
                rescued = head
            else:
                unit_moras = max(1, _text_mora_count(unit, method=english_oov_katakana_method))
                normalized = max(1, round(trim_duration_sec * _REPEAT_NORMALIZE_MORA_RATE / unit_moras))
                if normalized < count:
                    rescued = unit * normalized
            if rescued != text2:
                text2 = rescued
                words = None

    return text2, words


def _assemble_phoneme_sequence(chunk_phonemes: list[list[str]]) -> list[str]:
    """チャンクごとのG2P音素記号列を pau を挟んで結合する(手順5。単語タイムスタンプ非取得時)。

    チャンク境界ごとに pau を1つ挟み、列の先頭と末尾にも pau を1つずつ補う(区間ごとに独立して
    内容認識呼び出しを行う現行パイプラインでは chunk_phonemes は常に1要素で呼ばれ、実質的に
    その区間の音素記号列の先頭・末尾へ pau を補う処理になる)。単語窓は対応付けない(手順7の
    フォールバック=位置バンド制限で整列する)。
    """
    sequence = ["pau"]
    for i, phonemes in enumerate(chunk_phonemes):
        if i > 0:
            sequence.append("pau")
        sequence.extend(phonemes)
    sequence.append("pau")
    return sequence


def _assemble_with_word_windows(
    words_phonemes: list[tuple[list[str], float, float]], margin_sec: float, duration_sec: float
) -> tuple[list[str], list[tuple[float, float]]]:
    """単語ごとの音素記号列と時間窓から、pauで連結したトークン記号列と各記号の時間窓を組み立てる
    (手順5。単語タイムスタンプ取得時)。

    words_phonemes は単語の時系列順の (音素記号列, 開始時刻, 終了時刻) の列(開始・終了はトリムした
    入力範囲内の相対秒。手順3で単調化・クランプ済み)。単語内の音素記号(句読点由来の pau を
    含む)は自身の単語の [開始-余白, 終了+余白] を窓とする。単語境界に挿入する pau は隣接する
    2単語の [前の終了-余白, 次の開始+余白]、先頭の pau は [0, 最初の開始+余白]、末尾の pau は
    [最後の終了-余白, duration_sec] を窓とする。窓の下端が上端を上回る場合は入れ替える(空窓に
    しない)。
    """

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
    """G2P記号列を音素モデル語彙のトークンID列へ変換する(手順6)。

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
            raise RecognitionError(f"G2P記号 '{symbol}' の音素モデル語彙への写像が未定義です")
        token_id = vocab.get(vocab_symbol)
        if token_id is None:
            raise RecognitionError(f"音素モデルの語彙に記号 '{vocab_symbol}' が見つかりません")
        token_ids.append(token_id)
    return token_ids


_FORCED_ALIGN_BAND_SEC = 1.0  # 手順7: blank支配下での押し込み崩壊を防ぐ位置バンド幅(フォールバック)
_MIN_STAY_FRAMES = 6  # 手順7: 非blank(音素)状態の最小滞在フレーム数(120ms)
_VOICED_BLANK_PENALTY = 7.0  # 手順7: 有声フレームのblank列から引く対数確率ペナルティ
_WORD_WINDOW_MARGIN_SEC = 0.75  # 手順7: 単語窓制約の余白(0.2/0.5/0.75秒の実測比較で採用)
_EARLY_COMMIT_BONUS = 2.0  # 手順7: 単語窓制約の早期遷移ボーナスの加算項上限(実測比較で採用)


def _apply_voiced_blank_penalty(
    log_probs: np.ndarray, chunk_samples: np.ndarray, threshold: float, blank_token_id: int
) -> np.ndarray:
    """有声フレームのblank列へ固定ペナルティを適用する(手順7)。

    blank優勢の歌唱では、声が出ているフレームでもblank(pau)に留まる経路が最尤になりやすく、
    フレーズ先頭のモーラが実際の発声より数百ms遅れて置かれるため、有声フレーム(20msフレームRMSが
    手順1のしきい値超)ではblankを不利にする。無音フレーム(トリム余白・息継ぎ)は変更しない。
    """
    frame_rms = _frame_rms(chunk_samples, RECOGNIZER_CONFIG.sample_rate, FRAME_DURATION_SEC)
    n = min(len(frame_rms), log_probs.shape[0])
    penalized = log_probs.copy()
    voiced = frame_rms[:n] > threshold
    penalized[:n][voiced, blank_token_id] -= _VOICED_BLANK_PENALTY
    return penalized


def _expand_min_stay(
    token_ids: list[int], blank_token_id: int, num_frames: int
) -> tuple[list[int], list[int]]:
    """非blankトークンを最小滞在フレーム数ぶんの連鎖サブ状態へ展開する(手順7)。

    blank優勢の歌唱では音素状態を1フレームで通過する経路が最尤になりやすく、モーラが数十msへ
    潰れるため、非blankトークンをサブ状態の連鎖(各1フレーム以上滞在)で表して合計滞在を強制する。
    入力範囲のフレーム数がサブ状態総数に足りない場合は、成立する最大の滞在数へ引き下げる
    (1未満にはしない)。

    戻り値は (展開後トークンID列, 各サブ状態が対応する元トークンindexの列)。
    """
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
    """最小滞在フレーム数を単語ごとに局所適応させて展開する(手順7の局所適応。単語
    タイムスタンプが取得できた場合のみ使う)。

    _expand_min_stay(チャンク全体で1回だけ最小滞在を計算する版)と異なり、単語ごとに
    「その単語の実時間(フレーム数)を音素記号列の要素数(pau・cl由来のblank記号を含む)で割った値を
    四捨五入したもの」を最小滞在とする(1〜_MIN_STAY_FRAMESにクランプ)。歌唱ではフレーズ内で
    テンポが不均一なことがあり、チャンク全体の平均では十分な余裕があっても、テンポの速い一部の
    単語群だけが局所的に逼迫し、単調遷移と単語窓を同時に満たす経路が理論上到達不能になることが
    実測で確認された。単語ごとの実時間に応じて最小滞在を短縮することで、この逼迫を解消する。

    token_ids は手順5・6で組み立てたトークン列(先頭pau・各単語の音素・単語間/末尾pauを含む)の
    語彙IDへの変換結果で、words_phonemes と同じ単語の並びから組み立てられていることを前提とする。

    戻り値は (展開後トークンID列, 各サブ状態が対応する元トークンindexの列)。
    """
    stays = [1]  # 先頭pau(blank)
    for i, (phonemes, start, end) in enumerate(words_phonemes):
        word_frames = (end - start) / FRAME_DURATION_SEC
        # 四捨五入は int(x + 0.5)(真の四捨五入)を使う。Python組み込みのround()は偶数丸めで
        # ちょうど.5の商(例: 単語の実時間が最小長0.05秒=2.5フレームにクランプされた場合)を
        # 切り捨てうるため使わない。浮動小数点誤差でfloor()が1フレーム低く切り捨てる境界値
        # (例: (0.06-0.02)/0.02 が2.0でなく1.9999999999999996になる)もこの式で正しく丸まる。
        stay = min(_MIN_STAY_FRAMES, max(1, int(word_frames / len(phonemes) + 0.5))) if phonemes else 1
        stays.extend([stay] * len(phonemes))
        if i + 1 < len(words_phonemes):
            stays.append(1)  # 単語間pau(blank)
    stays.append(1)  # 末尾pau(blank)

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
    """トークン列を対数確率行列へ単調に対応付ける共通Viterbi(手順7)。

    各トークンを1状態とし、フレームごとに「同一状態に留まる」「次のトークンの状態へ進む」の
    2種の遷移のみ許す(読み飛ばし禁止)。各状態の対数確率には token_ids が指すその状態自身の
    語彙IDの列を使う(pau・cl由来の状態のblank列選択は、呼び出し側の _g2p_symbols_to_token_ids が
    そこへ blank_token_id を書き込み済みであることに由来し、本関数はトークン種別を区別しない)。
    状態 l がフレーム t に遷移できるかは呼び出し側が渡す out_of_bounds[t, l](到達不能なら True)
    で制約する(単語窓制約・位置バンド制限のいずれもこの共通形へ帰着する)。state_bias を渡すと
    (num_frames, num_states)の加算項として対数確率へ足し込む(単語窓制約の早期遷移ボーナス。
    下記 _forced_align_windowed)。省略時(None)は何も加算しない(_forced_align のフォールバックは
    渡さない)。

    フレーム0は状態0に固定し、最終フレームは状態 len(token_ids)-1 に到達している経路の中で
    最尤のものを採る。戻り値は各フレームが対応する状態(トークン列中のindex)。最終フレームで
    末尾状態へ到達する経路が無い場合は None を返す(呼び出し側が RecognitionError にするか
    フォールバックへ切り替えるかを判断する)。
    """
    num_frames, num_states = out_of_bounds.shape
    emission = log_probs[:, token_ids]  # (num_frames, num_states)
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
    """既知のトークン列を対数確率行列へ単調に対応付ける(手順7の**フォールバック**。位置バンド制限)。

    単語タイムスタンプが取得できない場合、または単語窓制約(下記 _forced_align_windowed)で
    最終フレームへ到達できない場合に用いる。

    **位置バンド制限**: 状態 l の期待フレーム位置を (l / (L-1)) * (T-1)(トークン列を区間内へ均等
    割り当てした場合の位置。L=1 なら0)とし、フレーム t が状態 l に遷移できるのは
    |t - 期待フレーム位置| <= バンド幅 を満たす場合に限る(バンド幅は1.0秒に相当するフレーム数で
    固定)。blank 支配下でも均等割り当てから大きく外れた押し込み崩壊を構造的に防ぐ。

    理論上到達不能(フレーム数がトークン数未満、またはバンド制限により到達不能な場合)は
    RecognitionError で停止する。
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

    path = _viterbi_monotonic(log_probs, token_ids, out_of_band)
    if path is None:
        raise RecognitionError("強制アライメントが末尾トークンへ到達できませんでした")
    return path


def _forced_align_windowed(
    log_probs: np.ndarray, token_ids: list[int], windows_sec: list[tuple[float, float]]
) -> list[int]:
    """単語タイムスタンプの窓で状態ごとの到達可能フレームを制約した強制アライメント
    (手順7の**単語窓制約**。主経路)。

    状態 l がフレーム t に遷移できるのは、windows_sec[l] = (lo, hi) がフレーム区間
    [t*frame_dur, (t+1)*frame_dur) と交差する場合(lo < (t+1)*frame_dur かつ hi > t*frame_dur)に
    限る。windows_sec は token_ids と同じ長さ(サブ状態展開後、各サブ状態は由来するトークンの窓を
    共有する)。開始・終端条件・遷移規則は _forced_align と同じ(共通の _viterbi_monotonic を使う)。

    **早期遷移ボーナス**: 各状態の対数確率に、その状態の窓内での相対位置(早いほど大きい)に応じた
    加算項 `_EARLY_COMMIT_BONUS * (1 - 窓内相対位置)` を加える(窓の下端で最大 `_EARLY_COMMIT_BONUS`、
    上端で0)。単語窓制約は「窓内のどこでもよい」というハード制約のみで窓内の位置選好を持たないため、
    blank優勢の歌唱では有声フレームのblank抑制だけでは覆いきれず、窓の遅い側(上端寄り)へ配置が
    偏る実測済みの不具合への対処(先行する状態に留まるほどその状態自身の窓内相対位置が進み加算項が
    減衰する一方、後続状態はその状態自身の窓の下端に近いフレームで遷移するほど加算項が大きいため、
    早く遷移するほど総和が相対的に有利になる)。窓外は到達不能のまま変えない。

    最終フレームで末尾状態へ到達する経路が無い場合(単語タイムスタンプの誤り等)は
    RecognitionError で停止する(呼び出し側が _forced_align へフォールバックする)。
    """
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
    width = np.maximum(win_hi - win_lo, 1e-9)  # 幅0の窓(hi==lo)での0除算を避ける下駄

    out_of_window = ~((win_lo[None, :] < frame_hi[:, None]) & (win_hi[None, :] > frame_lo[:, None]))
    relative_position = np.clip((frame_lo[:, None] - win_lo[None, :]) / width[None, :], 0.0, 1.0)
    state_bias = _EARLY_COMMIT_BONUS * (1.0 - relative_position)

    path = _viterbi_monotonic(log_probs, token_ids, out_of_window, state_bias)
    if path is None:
        raise RecognitionError("単語窓制約下で強制アライメントが末尾トークンへ到達できませんでした")
    return path


def _path_to_segments(path: list[int], symbols: list[str], frame_duration_sec: float) -> list[Segment]:
    """強制アライメントの状態パスをSegment列へ変換する(手順8・9)。

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


def _merge_adjacent_segments(segments: list[Segment]) -> list[Segment]:
    """隣接する同一 type・phoneme の Segment を1つへ結合する(手順9。区間境界をまたぐ結合)。"""
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
    """音素モデル(強制アライメント用)をロードし (processor, model, vocab, blank_token_id) を返す。

    ロード失敗の例外は recognize() の既存契約どおり RecognitionError へ写像する。
    """
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
            "取得できません。ネットワーク接続を確認するか、モデルを事前にキャッシュしてください。"
        ) from e
    vocab = processor.tokenizer.get_vocab()
    blank_token_id = processor.tokenizer.pad_token_id
    return processor, model, vocab, blank_token_id


def _align_wav2vec2_job(
    processor, model, vocab, blank_token_id, threshold: float, job: dict
) -> list[Segment]:
    """1区間ぶんの wav2vec2 CTC 強制アライメント(手順6〜9)を実行し、絶対時刻の Segment 列を返す。

    job は区間ループ(手順1〜5)が積んだ辞書(chunk_samples・seq・windows_sec・words_phonemes・
    trim_lo_sec・trim_hi_sec)。
    """
    chunk_samples = job["chunk_samples"]
    seq = job["seq"]
    windows_sec = job["windows_sec"]
    words_phonemes = job["words_phonemes"]
    trim_lo_sec = job["trim_lo_sec"]
    trim_hi_sec = job["trim_hi_sec"]

    token_ids = _g2p_symbols_to_token_ids(seq, vocab, blank_token_id)
    log_probs = _compute_log_probs(processor, model, chunk_samples)
    # 手順7の有声フレームのblank抑制と最小滞在制約: 有声フレームでblankを不利にし、
    # 非blankトークンをサブ状態へ展開してアライメントし、経路を元トークンindexへ戻してから
    # Segment化する。
    log_probs = _apply_voiced_blank_penalty(log_probs, chunk_samples, threshold, blank_token_id)

    # 手順7: 単語窓制約を主経路とし、窓が無い(単語タイムスタンプ非取得)、または窓制約下で
    # 末尾トークンへ到達できない場合は位置バンド制限へフォールバックする。単語窓がある場合、
    # 最小滞在は単語ごとの局所適応(_expand_min_stay_local)を使う。局所適応でも窓制約が
    # 到達不能ならチャンク全体の最小滞在(_expand_min_stay)へ計算し直し、位置バンド制限を使う。
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
    # 最後の区切りは対数確率行列のフレーム数に由来する終端(local_segments[-1].end_sec)を
    # 使わず、トリム後区間の真の終端(trim_hi_sec - trim_lo_sec)へ強制的に揃える(フレーム数
    # 計算の丸め等で区間境界とわずかにずれ、隣接区間との欠落・重複を生むことを防ぐ)。
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
    """内容認識パイプラインのプロセス内キャッシュを解放し、GPUのキャッシュ済みメモリを返す。

    書き起こしフェーズ完了後、音素モデル(またはSOFAサブプロセス)の実行前に呼ぶことで、
    内容認識モデルと音素モデルの同時GPU常駐によるVRAMピークを避ける。
    """
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


def recognize(
    vocal_wav_path: Path,
    content_recognizer_model: ContentRecognizerModel = DEFAULT_CONTENT_RECOGNIZER_MODEL,
    *,
    retry: bool = True,
    forced_aligner: ForcedAlignerId = DEFAULT_FORCED_ALIGNER,
    sofa_aligner: SofaAlignerConfig | None = None,
    english_oov_katakana_method: EnglishOovKatakanaMethod = DEFAULT_ENGLISH_OOV_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> list[Segment]:
    """ボーカルWAVから母音/子音/gapのセグメント列を認識する(Recognizerアダプタ契約)。

    content_recognizer_model で内容認識モデルを指定する(既定値・候補値・任意指定)。
    retry はエコー幻覚・反復幻覚へのトリガ式リトライの有効/無効を切り替える
    (既定True。主モデル自身をプロンプト無しで再認識する。別モデルは使わない)。内容認識・
    G2Pはどのモデル・どの強制アライメント経路でも共通。forced_aligner で強制アライメント段を選択する
    (既定`wav2vec2-ctc-forcedalign`)。`forced_aligner="sofa-forcedalign"`を選ぶ場合は
    sofa_aligner(`SofaAlignerConfig`)が必須で、省略(`None`)すると`RecognitionError`にする
    (黙ってwav2vec2へフォールバックしない)。english_oov_katakana_method で英語未知語カタカナ化
    フォールバックの変換方式を選択する(既定`arpakana`)。on_progress はモデル(内容認識モデル・
    音素モデル)の初回取得がネットワークダウンロードを要した区間だけ、進捗文言を都度渡して呼ぶ
    (キャッシュ済みなら一切呼ばない。モデルロード関数へそのまま転送するだけで、判定・文言の
    組み立ては行わない)。
    """
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

    # 実行は二相に分ける: まず全区間の書き起こし(内容認識・G2P・音素密度チェック)を済ませ、
    # 内容認識パイプラインを解放してから、強制アライメント(音素モデルまたはSOFA)をまとめて行う。
    # 内容認識モデルと音素モデル(いずれもGPU実行時は数GB)を同時にGPUへ常駐させると、VRAMの
    # 逼迫(ページング)で処理時間が大きく悪化し、他プロセスの描画も阻害するため。
    # 内容認識パイプライン(Whisper系)は無音でない最初の区間で遅延ロードし、以降の区間では
    # 使い回す(区間ごとの再ロードによるモデル転送コスト(特にGPU使用時)の浪費を避ける)。
    # 音素モデルは強制アライメント対象が1件以上あるときだけアライメントフェーズでロードする
    # (全区間が無音・gap確定ならモデルを一切必要としない。ロード自体の失敗は握りつぶさず
    # 例外にして失敗境界を隠さない)。
    content_pipeline = None
    # SOFA経路専用: 対象区間・対象単語をここへ積み、recognize()呼び出し全体で1回だけ
    # SOFAへまとめて渡す(ループの外、全区間処理後)。各要素は
    # (音声サンプル, サンプルレート, G2P音素記号列, 絶対オフセット秒, 相対長さ秒)。
    pending_sofa_targets: list[tuple[np.ndarray, int, list[str], float, float]] = []
    # wav2vec2 CTC経路の二相化用: 書き起こしフェーズで積んだ強制アライメント対象。
    # 各要素は _align_wav2vec2_job が受け取る辞書+挿入位置(insert_at)。
    pending_align_jobs: list[dict] = []

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

        # 手順2: 有声スパンへトリムしてから内容認識・アライメントへ渡す。区間の境界は無音区間の
        # 中点のため端に長い無音を含みうる。トリムで除いた先頭・末尾はgapとして直接確定する。
        # トリムが無い側の境界は丸め誤差を持ち込まないよう区間自身の境界秒をそのまま使う。
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
            if content_pipeline is None:
                content_pipeline = _load_content_recognizer_pipeline(
                    content_recognizer_model, on_progress=on_progress)
            text, words = _transcribe_segment(content_pipeline, chunk_samples)
        except ImportError as e:
            raise RecognitionError(
                "transformers または torch が見つかりません。導入してください"
                "(vocal-analysis extra で両方導入されます)。"
            ) from e
        except OSError as e:
            raise RecognitionError(
                f"内容認識モデル({content_recognizer_model.model_id}, "
                f"revision={content_recognizer_model.model_revision})を"
                "取得できません。ネットワーク接続を確認するか、モデルを事前にキャッシュしてください。"
            ) from e

        try:
            # エコー除去→トリガ式リトライ(主モデル)→反復救済。テキスト変更時は単語タイムスタンプを破棄。
            text, words = _resolve_transcription(
                chunk_samples, trim_duration_sec, text, words,
                content_recognizer_model, retry,
                english_oov_katakana_method=english_oov_katakana_method,
            )
        except ImportError as e:
            raise RecognitionError(
                "pyopenjtalk-plus または transformers が見つかりません。導入してください"
                "(vocal-analysis extra で導入されます)。"
            ) from e

        # 書き起こしが空文字列(単語0個。エコー除去で空になりリトライでも回復しなかった場合を
        # 含む)の区間はG2P・アライメントを試みず、区間全体をgapとして直接確定する(無音確定・
        # 音素密度超過確定と同様の扱い)。
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
            # 手順4: 単語タイムスタンプが取得できた場合は単語ごとに個別にG2Pし(手順5の窓割り当てに
            # 使う)、取得できなかった場合は区間の書き起こし全体を1回で変換する(単語単位に分割しない)。
            if words:
                words_phonemes = [
                    (_g2p(word_text, method=english_oov_katakana_method), w_start, w_end)
                    for word_text, w_start, w_end in words
                ]
                phonemes = [p for word_phonemes, _, _ in words_phonemes for p in word_phonemes]
            else:
                words_phonemes = None
                phonemes = _g2p(text, method=english_oov_katakana_method)
        except ImportError as e:
            raise RecognitionError(
                "pyopenjtalk-plus が見つかりません。導入してください(vocal-analysis extra で導入されます)。"
            ) from e

        # エコー除去・リトライ・反復救済(_resolve_transcription)を経てなお音素密度がしきい値を
        # 超える区間は、書き起こし・G2P・強制アライメントの結果を用いず、区間全体をgapとして
        # 確定する(利用先のgap解決へ委ねる)。
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
            # 手順5: 単語タイムスタンプがあれば単語窓を対応付けて組み立て、無ければ
            # (単語分割していない一括の)音素記号列の前後にpauを補うだけにする。
            if words_phonemes is not None:
                seq, windows_sec = _assemble_with_word_windows(
                    words_phonemes, _WORD_WINDOW_MARGIN_SEC, trim_duration_sec
                )
            else:
                seq = _assemble_phoneme_sequence([phonemes])
                windows_sec = None

            # 二相化: 書き起こしフェーズでは音素モデルの推論を行わず対象を積むだけにし、
            # 全区間の書き起こし完了後に内容認識モデルを解放してからまとめて実行する
            # (両モデルの同時GPU常駐によるVRAMピークを避ける)。
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
            # forced_aligner == "sofa-forcedalign"(関数入口でこの2値のいずれかであることを
            # 検証済み): 単語単位分割してSOFA対象を積み、gapはここで直接確定する。実際のSOFA呼び出しは
            # recognize()呼び出し全体で1回にまとめる(全区間処理後にまとめて行う。バッチ単位の確定)。
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
                        (
                            chunk_samples[w_start_index:w_end_index],
                            RECOGNIZER_CONFIG.sample_rate,
                            word_phonemes,
                            trim_lo_sec + w_start,
                            w_end - w_start,
                        )
                    )
            else:
                pending_sofa_targets.append(
                    (chunk_samples, RECOGNIZER_CONFIG.sample_rate, phonemes, trim_lo_sec, trim_duration_sec)
                )

        if trimmed_tail:
            all_segments.append(
                Segment(type="gap", start_sec=trim_hi_sec, end_sec=end_sec, phoneme=None, confidence=None)
            )

    if forced_aligner == "wav2vec2-ctc-forcedalign":
        # 二相化のアライメントフェーズ: まず内容認識パイプラインを解放してから音素モデルを
        # ロードし、積んだ対象を順に実行して確定済み gap 区間の間の記録位置へ挿入する。
        # 挿入位置はフェーズ実行前の all_segments 上の位置(insert_at)なので、先行する挿入で
        # 増えた要素数を累積オフセットとして加算する。解放は対象0件(全区間が無音・gap確定)でも
        # 行う(内容認識だけ行われた場合もこの時点で以降の工程に内容認識モデルは不要なため)。
        content_pipeline = None
        _release_content_recognizer_pipeline()
        if pending_align_jobs:
            processor, model, vocab, blank_token_id = _ensure_phoneme_model(on_progress)
            inserted = 0
            for job in pending_align_jobs:
                aligned = _align_wav2vec2_job(processor, model, vocab, blank_token_id, threshold, job)
                position = job["insert_at"] + inserted
                all_segments[position:position] = aligned
                inserted += len(aligned)
            # 音素モデルの参照を落としてGPUのキャッシュ済みメモリを返す(以降のS3・モーフ生成、
            # および長尺分割時の次チャンクの書き起こしフェーズにVRAMを明け渡す)。
            del processor, model
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if forced_aligner == "sofa-forcedalign":
        # バッチ単位の確定: 積んだ対象をrecognize()呼び出し1回につき1回だけSOFAへまとめて渡す
        # (対象が0件ならsofa_align._align_batchがサブプロセスの起動自体を省略する)。
        # SOFAサブプロセスもGPUを使うため、実行前に内容認識パイプラインを解放して
        # VRAMを明け渡す(二相化と同じ理由)。
        content_pipeline = None
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
    """複数チャンネルの PCM を平均でモノラルへダウンミックスする。"""
    return samples.mean(axis=1)


def _resample_to_target(mono: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """モノラル PCM を目標サンプルレートへ再サンプリングする(多相補間)。"""
    if sample_rate == target_sample_rate:
        return mono
    gcd = math.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd
    return resample_poly(mono, up, down).astype(np.float32)


def _g2p(text: str, method: EnglishOovKatakanaMethod = DEFAULT_ENGLISH_OOV_KATAKANA_METHOD) -> list[str]:
    """テキストをG2Pで音素記号列へ変換する(手順4。pyopenjtalk-plus、ルールベース)。

    pyopenjtalk-plusへ渡す前に、英語未知語カタカナ化フォールバック(convert_oov_words)を適用する。
    methodで変換方式を選択する(既定`arpakana`)。
    """
    import pyopenjtalk

    try:
        text = convert_oov_words(text, method=method)
    except ImportError as e:
        raise RecognitionError(
            "arpakana、nltk、または transformers/torch が見つかりません。導入してください"
            "(vocal-analysis extra で導入されます)。"
        ) from e
    except LookupError as e:
        raise RecognitionError(
            "CMUdict(nltkのcmudictコーパス)が見つかりません。`nltk.download('cmudict')`で"
            "取得するか、事前にキャッシュしてください。"
        ) from e
    except OSError as e:
        raise RecognitionError(
            "英語未知語カタカナ化フォールバックの変換モデルを取得できません。ネットワーク接続を"
            "確認するか、モデルを事前にキャッシュしてください。"
        ) from e

    return pyopenjtalk.g2p(text, kana=False, join=False)


def _sanitize_word_timestamps(
    words: list[tuple[str, float, float]], duration_sec: float
) -> list[tuple[str, float, float]]:
    """単語タイムスタンプを単調化する(手順3)。

    まず開始・終了をトリムした入力範囲 [0, duration_sec] へクランプし、続けて開始時刻を直前の
    単語の開始時刻以上へ、終了時刻を自身の開始時刻+最小長以上へ、単語の時系列順にクランプする。
    """
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
    """内容認識パイプラインの chunks から単語タイムスタンプを抽出し単調化する(手順3)。

    テキストが空、または開始時刻が無い要素は読み飛ばす。終了時刻が無い要素(生成が単語の途中で
    打ち切られた場合)は開始時刻+最小長で補う。
    """
    words: list[tuple[str, float, float]] = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        start, end = chunk.get("timestamp", (None, None))
        if not text or start is None:
            continue
        end_sec = float(end) if end is not None else float(start) + _MIN_WORD_DURATION_SEC
        words.append((text, float(start), end_sec))
    return _sanitize_word_timestamps(words, duration_sec)


def _transcribe_text_only(samples: np.ndarray, content_recognizer_model: ContentRecognizerModel) -> str:
    """音声をタイムスタンプ無しで書き起こす(トリガ式リトライの再認識用。主モデル自身を呼ぶ)。

    リトライは区間全体のテキストを丸ごと採用し単語タイムスタンプを使わないため、単語単位の
    タイムスタンプ抽出(呼び出しコストが増す)は不要。プロンプトは渡さない(プロンプト起因の
    エコー幻覚を再発させないため)。温度フォールバックも無効化して1呼び出しの所要時間を
    有界化する(温度0固定の貪欲1パスに限定する)。上限で打ち切られた出力は後段の音素密度
    チェックへ通常どおり渡り、密度超過ならgap確定に落ちる(_SEGMENT_MAX_NEW_TOKENS・
    _RETRY_MAX_NEW_TOKENS)。パイプラインは _load_content_recognizer_pipeline の
    プロセス内キャッシュを共有する(主モデルと同じデバイス自動選択・ロード済み再利用が効く)。
    """
    pipeline = _load_content_recognizer_pipeline(content_recognizer_model)
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
    """区間のボーカル音声を書き起こし、単語タイムスタンプも得る(手順3。区間ごとに独立呼び出し)。

    pipeline は呼び出し側(recognize())が無音でない最初の区間で一度だけロードし、以降の区間へ
    使い回す(区間ごとの再ロードによるモデル転送コストの浪費を避ける)。
    どの content_recognizer_model でも同じ手順(かな限定プロンプトでprompt_ids取得→単語
    タイムスタンプ付き貪欲デコード)を適用する(モデルによる分岐は持たない)。単語タイムスタンプを
    返さないモデルは None を返す(手順7のフォールバックに帰着)。生成トークン数の上限
    (_SEGMENT_MAX_NEW_TOKENS)も指定し、反復ハルシネーション時の生成時間を有界化する
    (貪欲デコードは接頭辞不変のため、上限に達しない正常な書き起こしの結果は変わらない)。
    """
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
    """音素モデル(強制アライメント用)・内容認識モデル(Whisper系)の両方の実行デバイスを、
    この関数で共通に環境から自動選択する。GPU(CUDA)が利用可能ならGPUを使う。
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


_content_recognizer_pipeline_cache: tuple[ContentRecognizerModel, object] | None = None


def _hf_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
    """huggingface_hub.snapshot_download への薄いラッパー(モンキーパッチの受け口)。"""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, revision=revision, tqdm_class=tqdm_class)


def _transformers_pipeline(*args, **kwargs):
    """transformers.pipeline への薄いラッパー(モンキーパッチの受け口)。"""
    from transformers import pipeline as transformers_pipeline

    return transformers_pipeline(*args, **kwargs)


def _transformers_auto_processor_from_pretrained(*args, **kwargs):
    """transformers.AutoProcessor.from_pretrained への薄いラッパー(モンキーパッチの受け口)。"""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(*args, **kwargs)


def _transformers_auto_model_for_ctc_from_pretrained(*args, **kwargs):
    """transformers.AutoModelForCTC.from_pretrained への薄いラッパー(モンキーパッチの受け口)。"""
    from transformers import AutoModelForCTC

    return AutoModelForCTC.from_pretrained(*args, **kwargs)


def _prefetch_with_progress(repo_id: str, revision: str | None, on_progress: Callable[[str], None]) -> bool:
    """repo_id のファイル群を事前フェッチし、実際にバイト転送が発生した区間だけ
    on_progress(f"ダウンロード中: {repo_id} {percent}%") を呼ぶ。事前フェッチ後に呼び出し元が
    通常どおり from_pretrained 等でロードすると、キャッシュ済みのため高速に完了する。戻り値は
    実際にダウンロードが発生したか(呼び出し元がロード完了後の空文字列クリア通知を出すべきか)。

    huggingface_hub の snapshot_download は tqdm_class 差し込み口(公式拡張点)で2種類のバーを
    生成する: 対象ファイル数バー(unit指定なし。キャッシュ済みでも全ファイルぶん update される)と、
    実転送バイト集約バー(unit="B"。total=0 で生成後、実際に転送したバイト数だけ加算される)。
    ファイル数バーには反応せず、バイト集約バーの update だけに反応することで、キャッシュ済み
    (実転送無し)のときは on_progress を一度も呼ばない。
    """
    from huggingface_hub.utils import tqdm as hf_tqdm

    state = {"shown": False}

    class _RelayTqdm(hf_tqdm):
        # 表示が無効化されたバー(非TTY等)では tqdm の __init__ が早期 return し self.unit が
        # 未設定・self.n も update() で進まないため、tqdm 内部状態に依存せず、生成時引数の unit と
        # 自前の累積カウンタで中継する。self.total は無効化時も設定され、snapshot_download が
        # バー生成後に加算更新するため、都度読む。
        def __init__(self, *args, **kwargs):
            self._relay_unit = kwargs.get("unit")
            self._relay_n = kwargs.get("initial") or 0
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            result = super().update(n)
            if self._relay_unit == "B" and self.total:
                self._relay_n += n or 0
                state["shown"] = True
                percent = min(100, int(self._relay_n * 100 / self.total))
                on_progress(f"ダウンロード中: {repo_id} {percent}%")
            return result

    _hf_snapshot_download(repo_id, revision=revision, tqdm_class=_RelayTqdm)
    return state["shown"]


def _load_content_recognizer_pipeline(
    content_recognizer_model: ContentRecognizerModel,
    on_progress: Callable[[str], None] | None = None,
):
    """内容認識器を content_recognizer_model が指すモデル・revisionでロードする。

    実行デバイスは環境から自動選択する(_select_device)。1呼び出し中は
    区間ごと・トリガ式リトライごとに同じ content_recognizer_model を使い回すため、直前に
    ロードした1件だけを保持する単一枠キャッシュで足りる(再ロードを避ける)。別の
    content_recognizer_model が指定されると、直前のキャッシュは破棄して差し替える(複数の
    モデルを同時にプロセス内保持しない)。on_progress はモデルの初回取得が実際にネットワーク
    ダウンロードを要した区間だけ、進捗文言を渡して呼ぶ(vocal_analysis.md §5)。
    """
    global _content_recognizer_pipeline_cache
    if _content_recognizer_pipeline_cache is not None:
        cached_model, cached_pipeline = _content_recognizer_pipeline_cache
        if cached_model == content_recognizer_model:
            return cached_pipeline
        # 新モデルのロードを始める前に、旧パイプラインへの参照(グローバル・ローカルとも)を
        # 解放する。ローカル変数 cached_pipeline を残したままだと、グローバルキャッシュを
        # None にしても旧パイプラインが参照されたまま生き続け、新モデルのロード中に新旧
        # モデルが同時にGPU上へ残る窓ができてしまう。
        del cached_model, cached_pipeline
        _content_recognizer_pipeline_cache = None

    import torch

    downloaded = False
    if on_progress is not None:
        downloaded = _prefetch_with_progress(
            content_recognizer_model.model_id, content_recognizer_model.model_revision, on_progress)

    device = _select_device()
    # GPU実行時はfp16でロードし、既定のfp32に対して重み・アクティベーションのメモリ使用量を
    # 半減させる(限られたVRAMでの他モデル(音素モデル・分離器)との競合・アロケータの逼迫による
    # 速度低下を避ける)。CPU実行時はfp16未対応のためfp32のまま。
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        pipeline = _transformers_pipeline(
            "automatic-speech-recognition",
            model=content_recognizer_model.model_id,
            revision=content_recognizer_model.model_revision,
            device=device,
            dtype=dtype,
        )
    finally:
        # ロードが完了した時点で通知を終える(ダウンロードが実際に発生した場合のみ。
        # vocal_analysis.md §5)。例外時もライブ表示側の後始末に合わせクリアする。
        if downloaded:
            on_progress("")
    _content_recognizer_pipeline_cache = (content_recognizer_model, pipeline)
    return pipeline


def _compute_log_probs(processor, model, samples: np.ndarray) -> np.ndarray:
    """音素モデルで推論し、フレームごとの対数確率行列を返す(手順7の入力)。

    入力テンソルは _select_device() の戻り値のデバイスへ置く(ロード時
    (_load_model_and_processor)にモデルを配置したデバイスと、同一プロセス内で自動選択結果は
    変わらないため常に一致する)。
    """
    import torch

    inputs = processor(samples, sampling_rate=RECOGNIZER_CONFIG.sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(_select_device())).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[0].cpu().numpy()


def _load_model_and_processor(on_progress: Callable[[str], None] | None = None):
    """音素モデル(強制アライメント用)をロードする。

    モデル id・revision・dtype は S-1 測定の固定条件どおりに適用し、実行デバイスは
    _select_device() の自動選択で決める(実行デバイスは固定条件に含まれない)。on_progress は
    モデルの初回取得が実際にネットワークダウンロードを要した区間だけ、進捗文言を渡して呼ぶ
    (vocal_analysis.md §5)。
    """
    import torch

    downloaded = False
    if on_progress is not None:
        downloaded = _prefetch_with_progress(
            RECOGNIZER_CONFIG.model_id, RECOGNIZER_CONFIG.model_revision, on_progress)

    try:
        # wav2vec2-espeak のトークナイザは既定で espeak ネイティブバイナリ(phonemizer)を
        # 要求する。音素IDのデコードのみが必要で音素へのエンコードは不要なため do_phonemize=False
        # でこの依存を回避する。
        processor = _transformers_auto_processor_from_pretrained(
            RECOGNIZER_CONFIG.model_id,
            revision=RECOGNIZER_CONFIG.model_revision,
            do_phonemize=False,
        )
        model = _transformers_auto_model_for_ctc_from_pretrained(
            RECOGNIZER_CONFIG.model_id,
            revision=RECOGNIZER_CONFIG.model_revision,
            torch_dtype=getattr(torch, RECOGNIZER_CONFIG.dtype),
        )
    finally:
        # ロードが完了した時点で通知を終える(ダウンロードが実際に発生した場合のみ。
        # vocal_analysis.md §5)。例外時もライブ表示側の後始末に合わせクリアする。
        if downloaded:
            on_progress("")
    model.to(_select_device())
    model.eval()
    return processor, model
