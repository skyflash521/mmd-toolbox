"""S2 音素/母音認識(vocal_analysis.md §5・§5.1・§5.2・§8.1・§8.3)。

無音検出による区間分割・内容認識・G2P(pyopenjtalk-plus)・音素モデルのCTC強制アライメント(§5.2)を
組み合わせた複合構成で、母音/子音/gap を区別したセグメント列を生成する。公開関数
recognize(vocal_wav_path, content_recognizer_model) -> list[Segment] が唯一の公開面(Recognizer
アダプタ契約。§8.1)。内容認識モデルは `content_recognizer_model`(`ContentRecognizerModel`。§5.2)で
指定する(既定値 `DEFAULT_CONTENT_RECOGNIZER_MODEL`・候補値 `KANA_WHISPER_MODEL`・任意指定も可)。
"""

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import DEFAULT_CONTENT_RECOGNIZER_MODEL, KANA_PROMPT, RECOGNIZER_CONFIG, ContentRecognizerModel
from .phonemes import _BLANK_G2P_SYMBOLS, RecognitionError, _classify_symbol, _G2P_TO_VOCAB_SYMBOL
from .types import Segment

FRAME_DURATION_SEC = 0.02  # §5.1: 採用モデルの畳み込み総ストライド320サンプル@16kHzで固定


# --- §5.2 手順1・2: 無音検出による区間分割 ---

_SILENCE_FRAME_SEC = 0.1  # §5.2手順1: 無音検出用フレーム幅
_SILENCE_THRESHOLD_DB_BELOW_PEAK = 30.0  # §5.2手順1: ピーク(95パーセンタイル)から下回るdB
_SILENCE_MIN_RUN_SEC = 0.6  # §5.2手順1: 分割点とみなす無音区間の最小長
_SEGMENT_MIN_SEC = 1.5  # §5.2手順1: 区間の最小長(未満は次の分割点まで結合)
_SEGMENT_MAX_SEC = 25.0  # §5.2手順1: 区間の最大長(超過は均等分割)
_TRIM_MARGIN_SEC = 0.1  # §5.2手順2: 有声スパンの外側に残す余白(しきい値未満の子音の助走を切らない)


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


def _voiced_trim_bounds(segment_samples: np.ndarray, sample_rate: int, threshold: float) -> tuple[int, int]:
    """無音でない区間の有声スパン+余白のサンプル範囲を返す(§5.2手順2のトリム)。

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


# --- §5.2 手順4以降: 内容認識+G2P+強制アライメントの区間化 ---

_HALLUCINATION_PHONEME_RATE = 20.0  # §5.2手順4: 反復幻覚を疑う音素密度のしきい値(音素/秒)


def _is_hallucinated_phoneme_density(phoneme_count: int, duration_sec: float) -> bool:
    """音素密度(音素/秒)が反復幻覚を疑うしきい値を超えるかを判定する(§5.2手順4)。"""
    if duration_sec <= 0:
        return False
    return phoneme_count / duration_sec > _HALLUCINATION_PHONEME_RATE


def _assemble_phoneme_sequence(chunk_phonemes: list[list[str]]) -> list[str]:
    """チャンクごとのG2P音素記号列を pau を挟んで結合する(§5.2手順5。単語タイムスタンプ非取得時)。

    チャンク境界ごとに pau を1つ挟み、列の先頭と末尾にも pau を1つずつ補う(区間ごとに独立して
    内容認識呼び出しを行う現行パイプラインでは chunk_phonemes は常に1要素で呼ばれ、実質的に
    その区間の音素記号列の先頭・末尾へ pau を補う処理になる)。単語窓は対応付けない(§5.2手順7の
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
    (§5.2手順5。単語タイムスタンプ取得時)。

    words_phonemes は単語の時系列順の (音素記号列, 開始時刻, 終了時刻) の列(開始・終了はトリムした
    入力範囲内の相対秒。§5.2手順3で単調化・クランプ済み)。単語内の音素記号(句読点由来の pau を
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


_FORCED_ALIGN_BAND_SEC = 1.0  # §5.2手順7: blank支配下での押し込み崩壊を防ぐ位置バンド幅(フォールバック)
_MIN_STAY_FRAMES = 6  # §5.2手順7: 非blank(音素)状態の最小滞在フレーム数(120ms)
_VOICED_BLANK_PENALTY = 7.0  # §5.2手順7: 有声フレームのblank列から引く対数確率ペナルティ
_WORD_WINDOW_MARGIN_SEC = 0.75  # §5.2手順7: 単語窓制約の余白(0.2/0.5/0.75秒の実測比較で採用)
_EARLY_COMMIT_BONUS = 2.0  # §5.2手順7: 単語窓制約の早期遷移ボーナスの加算項上限(実測比較で採用)


def _apply_voiced_blank_penalty(
    log_probs: np.ndarray, chunk_samples: np.ndarray, threshold: float, blank_token_id: int
) -> np.ndarray:
    """有声フレームのblank列へ固定ペナルティを適用する(§5.2手順7)。

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
    """非blankトークンを最小滞在フレーム数ぶんの連鎖サブ状態へ展開する(§5.2手順7)。

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
    """最小滞在フレーム数を単語ごとに局所適応させて展開する(§5.2手順7の局所適応。単語
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
    """トークン列を対数確率行列へ単調に対応付ける共通Viterbi(§5.2手順7)。

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
    """既知のトークン列を対数確率行列へ単調に対応付ける(§5.2手順7の**フォールバック**。位置バンド制限)。

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
    (§5.2手順7の**単語窓制約**。主経路)。

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


def recognize(
    vocal_wav_path: Path,
    content_recognizer_model: ContentRecognizerModel = DEFAULT_CONTENT_RECOGNIZER_MODEL,
) -> list[Segment]:
    """ボーカルWAVから母音/子音/gapのセグメント列を認識する(§8.1のRecognizerアダプタ契約。§5.2複合構成)。

    content_recognizer_model で内容認識モデルを指定する(既定値・候補値・任意指定。§5.2)。
    後段のG2P・強制アライメントはどのモデルでも共通。
    """
    samples, sample_rate = sf.read(vocal_wav_path, dtype="float32", always_2d=True)
    mono = _downmix_to_mono(samples)
    resampled = _resample_to_target(mono, sample_rate, RECOGNIZER_CONFIG.sample_rate)
    duration_sec = len(resampled) / RECOGNIZER_CONFIG.sample_rate

    frame_rms = _frame_rms(resampled, RECOGNIZER_CONFIG.sample_rate, _SILENCE_FRAME_SEC)
    threshold = _silence_threshold(frame_rms)
    split_points = _detect_silence_split_points(resampled, RECOGNIZER_CONFIG.sample_rate)
    segment_bounds = _build_segment_bounds(duration_sec, split_points)

    # 音素モデルは実際に強制アライメントへ進む区間が現れるまでロードしない(§5.2手順2: 全区間が
    # 無音ならモデルを一切必要としない。非無音区間でも内容認識・G2P・手順4の音素密度チェックより
    # 先にロードするのではなく、それらを経てなお進む区間だけで遅延ロードする。幻覚検出でgap確定
    # した区間はロードせずスキップする。ロード自体の失敗は握りつぶさず例外にして失敗境界を隠さない)。
    # 内容認識パイプライン(Whisper系)も同様に無音でない最初の区間で遅延ロードし、以降の区間では
    # 使い回す(区間ごとの再ロードによるモデル転送コスト(特にGPU使用時)の浪費を避ける)。
    processor = model = vocab = blank_token_id = None
    content_pipeline = None

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

        # §5.2手順2: 有声スパンへトリムしてから内容認識・アライメントへ渡す。区間の境界は無音区間の
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

        try:
            if content_pipeline is None:
                content_pipeline = _load_content_recognizer_pipeline(content_recognizer_model)
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

        # §5.2手順3: 書き起こしが空文字列(単語0個)の区間はG2P・アライメントを試みず、区間全体を
        # gapとして直接確定する(手順2の無音確定・手順4の音素密度超過確定と同様の扱い)。
        if not text.strip():
            all_segments.append(
                Segment(type="gap", start_sec=trim_lo_sec, end_sec=trim_hi_sec, phoneme=None, confidence=None)
            )
            if trimmed_tail:
                all_segments.append(
                    Segment(type="gap", start_sec=trim_hi_sec, end_sec=end_sec, phoneme=None, confidence=None)
                )
            continue

        trim_duration_sec = trim_hi_sec - trim_lo_sec

        try:
            # §5.2手順4: 単語タイムスタンプが取得できた場合は単語ごとに個別にG2Pし(手順5の窓割り当てに
            # 使う)、取得できなかった場合は区間の書き起こし全体を1回で変換する(単語単位に分割しない)。
            if words:
                words_phonemes = [(_g2p(word_text), w_start, w_end) for word_text, w_start, w_end in words]
                phonemes = [p for word_phonemes, _, _ in words_phonemes for p in word_phonemes]
            else:
                words_phonemes = None
                phonemes = _g2p(text)
        except ImportError as e:
            raise RecognitionError(
                "pyopenjtalk-plus が見つかりません。導入してください(vocal-analysis extra で導入されます)。"
            ) from e

        # §5.2手順4の音素密度による幻覚検出: 内容認識の反復幻覚(同一文・同一フレーズの繰り返し)は
        # 区間の実際の発声より著しく多い音素列を生む。書き起こし・G2P・強制アライメントの結果を
        # 用いず、区間全体をgapとして確定する(利用先のgap解決へ委ねる)。
        if _is_hallucinated_phoneme_density(len(phonemes), trim_duration_sec):
            all_segments.append(
                Segment(type="gap", start_sec=trim_lo_sec, end_sec=trim_hi_sec, phoneme=None, confidence=None)
            )
            if trimmed_tail:
                all_segments.append(
                    Segment(type="gap", start_sec=trim_hi_sec, end_sec=end_sec, phoneme=None, confidence=None)
                )
            continue

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

        # §5.2手順5: 単語タイムスタンプがあれば単語窓を対応付けて組み立て、無ければ
        # (単語分割していない一括の)音素記号列の前後にpauを補うだけにする。
        if words_phonemes is not None:
            seq, windows_sec = _assemble_with_word_windows(words_phonemes, _WORD_WINDOW_MARGIN_SEC, trim_duration_sec)
        else:
            seq = _assemble_phoneme_sequence([phonemes])
            windows_sec = None

        token_ids = _g2p_symbols_to_token_ids(seq, vocab, blank_token_id)
        log_probs = _compute_log_probs(processor, model, chunk_samples)
        # §5.2手順7の有声フレームのblank抑制と最小滞在制約: 有声フレームでblankを不利にし、
        # 非blankトークンをサブ状態へ展開してアライメントし、経路を元トークンindexへ戻してから
        # Segment化する。
        log_probs = _apply_voiced_blank_penalty(log_probs, chunk_samples, threshold, blank_token_id)

        # §5.2手順7: 単語窓制約を主経路とし、窓が無い(単語タイムスタンプ非取得)、または窓制約下で
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
        for seg in local_segments:
            all_segments.append(
                Segment(
                    type=seg.type,
                    start_sec=seg.start_sec + trim_lo_sec,
                    end_sec=seg.end_sec + trim_lo_sec,
                    phoneme=seg.phoneme,
                    confidence=None,
                )
            )
        if trimmed_tail:
            all_segments.append(
                Segment(type="gap", start_sec=trim_hi_sec, end_sec=end_sec, phoneme=None, confidence=None)
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


_MIN_WORD_DURATION_SEC = 0.05  # §5.2手順3: 単語タイムスタンプ単調化の最小長


def _sanitize_word_timestamps(
    words: list[tuple[str, float, float]], duration_sec: float
) -> list[tuple[str, float, float]]:
    """単語タイムスタンプを単調化する(§5.2手順3)。

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
    """内容認識パイプラインの chunks から単語タイムスタンプを抽出し単調化する(§5.2手順3)。

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


def _transcribe_segment(
    pipeline, samples: np.ndarray
) -> tuple[str, list[tuple[str, float, float]] | None]:
    """区間のボーカル音声を書き起こし、単語タイムスタンプも得る(§5.2手順3。区間ごとに独立呼び出し)。

    pipeline は呼び出し側(recognize())が無音でない最初の区間で一度だけロードし、以降の区間へ
    使い回す(区間ごとの再ロードによるモデル転送コストの浪費を避ける)。
    どの content_recognizer_model でも同じ手順(かな限定プロンプトでprompt_ids取得→単語
    タイムスタンプ付き貪欲デコード)を適用する(モデルによる分岐は持たない)。単語タイムスタンプを
    返さないモデルは None を返す(§5.2手順7のフォールバックに帰着)。
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
        },
    )
    chunks = result.get("chunks")
    if not chunks:
        return result["text"], None
    duration_sec = len(samples) / RECOGNIZER_CONFIG.sample_rate
    return result["text"], _extract_word_timestamps(chunks, duration_sec)


def _select_content_recognizer_device() -> str:
    """内容認識モデル(Whisper系)の実行デバイスを環境から自動選択する(§5.1・§5.2)。

    GPU(CUDA)が利用可能ならGPUを使う。強制アライメント用の音素モデル(RECOGNIZER_CONFIG.device)は
    決定論のためCPU固定のままで、この自動選択の対象外(§5.1「実行条件の固定」)。内容認識の貪欲デコード
    (ビーム幅1・サンプリング無し)はサンプリング由来の乱数的非決定性を排除するが、実行デバイス・
    スレッド数の違いによる浮動小数点演算の丸め誤差までは排除しない。環境が異なれば僅差のトークン
    選択が割れ、書き起こし結果がわずかに変わりうる(§5.2「決定論」)。
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_content_recognizer_pipeline(content_recognizer_model: ContentRecognizerModel):
    """内容認識器を content_recognizer_model が指すモデル・revisionでロードする(§5.2・§8.3)。

    実行デバイスは環境から自動選択する(_select_content_recognizer_device。§5.1)。
    """
    from transformers import pipeline as transformers_pipeline

    return transformers_pipeline(
        "automatic-speech-recognition",
        model=content_recognizer_model.model_id,
        revision=content_recognizer_model.model_revision,
        device=_select_content_recognizer_device(),
    )


def _compute_log_probs(processor, model, samples: np.ndarray) -> np.ndarray:
    """音素モデルで推論し、フレームごとの対数確率行列を返す(§5.2手順7の入力)。

    スレッド数の固定(RECOGNIZER_CONFIG.num_threads。§5.1)はこの音素モデル推論だけへ局所的に
    適用し、呼び出し前後の設定へ復元する(内容認識(Whisper系。手順3)のCPU実行はこの制約を
    受けず、環境のデフォルトスレッド数で並列に動く)。
    """
    import torch

    inputs = processor(samples, sampling_rate=RECOGNIZER_CONFIG.sample_rate, return_tensors="pt")
    prev_num_threads = torch.get_num_threads()
    torch.set_num_threads(RECOGNIZER_CONFIG.num_threads)
    try:
        with torch.no_grad():
            logits = model(inputs.input_values.to(RECOGNIZER_CONFIG.device)).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[0].cpu().numpy()
    finally:
        torch.set_num_threads(prev_num_threads)


def _load_model_and_processor():
    """音素モデル(強制アライメント用)をS-1測定の固定条件(§5.1・§8.3)でロードする。

    スレッド数の固定(RECOGNIZER_CONFIG.num_threads)はロード時ではなく推論時(_compute_log_probs)
    に局所適用する(内容認識(Whisper系)のCPU実行を道連れにしないため)。
    """
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    torch.manual_seed(RECOGNIZER_CONFIG.random_seed)

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
