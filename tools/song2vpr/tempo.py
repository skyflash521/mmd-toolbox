"""song2vpr のテンポ・拍子の推定と、秒から tick への変換。

拍を刻むのは主に伴奏側なので、推定は分離前の入力に対して行う(分離後のボーカルからは拍が取りにくい)。
`--tempo`・`--time-signature` を指定された項目は推定せずその値を使う。
"""

from dataclasses import dataclass

import numpy as np

# vpr 形式が固定している分解能(tick / 四分音符)。
RESOLUTION = 480

# 周期を取り出せなかったときに仮置きするテンポと拍子。
_DEFAULT_BPM = 120.0
_DEFAULT_TIME_SIGNATURE = (4, 4)

# オンセット強度包絡の窓長とホップ(秒)。
_WINDOW_SEC = 0.046
_HOP_SEC = 0.012

# 自己相関のラグを走査する BPM の範囲と、拍と感じる帯域を選ぶ重みの中心。
_BPM_RANGE = (30.0, 300.0)
_BPM_WEIGHT_CENTER = 120.0

# 正規化自己相関のピークがこれ未満なら、周期が得られなかったとみなす。
_PERIODICITY_FLOOR = 0.1

# 未指定時に探索する拍子の分子。
_NUMERATOR_CANDIDATES = (2, 3, 4)


def quantize_bpm(bpm: float) -> float:
    """テンポを vpr 形式が格納できる粒度(BPM の 1/100)へ丸める。

    出力ファイルと診断・秒から tick への変換で同じ値を使うため、採用の時点で丸めておく。
    """
    return round(bpm * 100) / 100.0


@dataclass(frozen=True)
class TempoEstimate:
    """採用したテンポ・拍子と、最初の小節線の位置。"""

    bpm: float
    numerator: int
    denominator: int
    beat_offset_sec: float
    first_bar_sec: float
    tempo_defaulted: bool
    time_signature_defaulted: bool
    resolution: int = RESOLUTION

    def to_tick(self, seconds: float) -> int:
        """入力音声の時刻を tick へ写す。入力の 0 秒が tick の 0。"""
        return round(seconds * self.bpm * self.resolution / 60.0)


def _onset_envelope(pcm):
    """オンセット強度の包絡と、そのホップ幅(秒)を返す。"""
    samples = np.asarray(pcm.samples, dtype=np.float64)
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    sample_rate = pcm.sample_rate

    window_length = int(round(_WINDOW_SEC * sample_rate))
    hop = int(round(_HOP_SEC * sample_rate))
    if window_length < 2 or hop < 1 or len(mono) < window_length:
        return np.zeros(0), _HOP_SEC

    fft_length = 1 << (window_length - 1).bit_length()
    window = np.hanning(window_length)
    starts = range(0, len(mono) - window_length + 1, hop)
    # 各フレームの振幅スペクトルを圧縮し、隣接フレーム間の増加分だけを足す。
    spectra = np.array([np.abs(np.fft.rfft(mono[s:s + window_length] * window, n=fft_length))
                        for s in starts])
    if len(spectra) < 2:
        return np.zeros(0), _HOP_SEC
    compressed = np.log1p(spectra)
    envelope = np.maximum(np.diff(compressed, axis=0), 0.0).sum(axis=1)

    deviation = envelope.std()
    if deviation == 0.0:
        return np.zeros(0), _HOP_SEC
    return (envelope - envelope.mean()) / deviation, hop / sample_rate


def _estimate_bpm(envelope, hop_sec, denominator):
    """包絡の自己相関から四分音符あたりの BPM を求める。得られなければ None。"""
    if len(envelope) < 2:
        return None
    zero_lag = float(np.dot(envelope, envelope))
    if zero_lag == 0.0:
        return None

    def bpm_of(lag):
        return 60.0 / (lag * hop_sec) * 4.0 / denominator

    best = None
    for lag in range(1, len(envelope)):
        bpm = bpm_of(lag)
        if not _BPM_RANGE[0] <= bpm <= _BPM_RANGE[1]:
            continue
        correlation = float(np.dot(envelope[:-lag], envelope[lag:])) / zero_lag
        # 同じ周期性の 1/2 倍・2 倍のピークから、人が拍と感じる帯域を選ぶ。
        weight = np.exp(-(np.log2(bpm / _BPM_WEIGHT_CENTER) ** 2) / 2.0)
        score = correlation * weight
        if best is None or score > best[0] + 1e-12:
            best = (score, bpm, lag)
        elif abs(score - best[0]) <= 1e-12:
            # 同値なら 120 に近い方、それも同値なら小さいラグ。
            if abs(bpm - _BPM_WEIGHT_CENTER) < abs(best[1] - _BPM_WEIGHT_CENTER):
                best = (score, bpm, lag)

    if best is None or best[0] < _PERIODICITY_FLOOR:
        return None
    return best[1]


def _beat_values(envelope, hop_sec, offset_sec, period_sec):
    """拍の位置に最も近い包絡値(負は 0 へ倒す)を並べる。"""
    if period_sec <= 0.0 or len(envelope) == 0:
        return np.zeros(0)
    duration = len(envelope) * hop_sec
    count = int(max(0.0, duration - offset_sec) / period_sec) + 1
    indices = np.clip(np.rint((offset_sec + np.arange(count) * period_sec) / hop_sec).astype(int),
                      0, len(envelope) - 1)
    return np.maximum(envelope[indices], 0.0)


def _estimate_phase(envelope, hop_sec, period_sec):
    """拍間隔に対する位相(最初の拍の時刻)を求める。"""
    if len(envelope) == 0 or period_sec <= 0.0:
        return 0.0
    best = None
    for step in range(max(1, int(round(period_sec / hop_sec)))):
        offset = step * hop_sec
        total = float(_beat_values(envelope, hop_sec, offset, period_sec).sum())
        if best is None or total > best[0] + 1e-12:
            best = (total, offset)
    return 0.0 if best is None else best[1]


def _estimate_numerator(envelope, hop_sec, offset_sec, period_sec, candidates):
    """小節内で強い拍が現れる周期から、拍子の分子と小節内オフセットを求める。

    判定材料が得られなければ既定へ倒し、そのことを3つ目の戻り値で示す(分子が指定されている
    実行では、その値を保つので倒したことにならない)。
    """
    values = _beat_values(envelope, hop_sec, offset_sec, period_sec)
    overall = float(values.mean()) if len(values) else 0.0
    if overall == 0.0:
        if len(candidates) == 1:
            return candidates[0], 0, False
        return _DEFAULT_TIME_SIGNATURE[0], 0, True

    best = None
    for numerator in candidates:
        for phase in range(numerator):
            selected = values[phase::numerator]
            if not len(selected):
                continue
            ratio = float(selected.mean()) / overall
            # 同値なら分子が大きい方、さらに同値なら小節内オフセットが小さい方。
            key = (ratio, numerator, -phase)
            if best is None or key > best[0]:
                best = (key, numerator, phase)
    if best is None:
        return candidates[0], 0, len(candidates) > 1
    return best[1], best[2], False


def estimate(pcm, *, tempo_bpm=None, time_signature=None) -> TempoEstimate:
    """分離前の入力からテンポ・拍子・最初の小節線を求める。

    tempo_bpm・time_signature を与えられた項目は推定せずその値を使う。周期を取り出せない入力では
    テンポを既定へ倒し、そのことを tempo_defaulted で示す(拍子の指定があればそれは保つ)。
    """
    numerator, denominator = time_signature if time_signature is not None else (None, 4)

    envelope, hop_sec = _onset_envelope(pcm)
    estimated_bpm = None if tempo_bpm is not None else _estimate_bpm(envelope, hop_sec, denominator)

    if tempo_bpm is None and estimated_bpm is None:
        # 周期が得られないので既定へ倒す。位相も求められないため小節線は先頭に置く。
        return TempoEstimate(
            bpm=_DEFAULT_BPM,
            numerator=numerator if numerator is not None else _DEFAULT_TIME_SIGNATURE[0],
            denominator=denominator, beat_offset_sec=0.0, first_bar_sec=0.0,
            tempo_defaulted=True, time_signature_defaulted=numerator is None)

    bpm = quantize_bpm(tempo_bpm if tempo_bpm is not None else estimated_bpm)
    # 自己相関が拾うのは拍(拍子の分母が表す音価)の周期。四分音符あたりの BPM から戻す。
    period_sec = 60.0 / bpm * 4.0 / denominator
    # 位相は小節線の位置だけを決めるので、テンポが指定値でも推定する。
    offset_sec = _estimate_phase(envelope, hop_sec, period_sec)

    candidates = (numerator,) if numerator is not None else _NUMERATOR_CANDIDATES
    adopted_numerator, bar_phase, numerator_defaulted = _estimate_numerator(
        envelope, hop_sec, offset_sec, period_sec, candidates)

    return TempoEstimate(
        bpm=bpm, numerator=adopted_numerator, denominator=denominator,
        beat_offset_sec=offset_sec, first_bar_sec=offset_sec + bar_phase * period_sec,
        tempo_defaulted=False, time_signature_defaulted=numerator_defaulted)
