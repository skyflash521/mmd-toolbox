"""song2vpr のテンポ推定と拍子の決定、秒から tick への変換。

拍を刻むのは主に伴奏側なので、テンポの推定は分離前の入力に対して行う(分離後のボーカルからは拍が
取りにくい)。`--tempo` を指定されたときは推定せずその値を使う。拍子は音声から決めず、
`--time-signature` の指定があればその値、無ければ既定を使う。
"""

from dataclasses import dataclass

import numpy as np

# vpr 形式が固定している分解能(tick / 四分音符)。
RESOLUTION = 480

# 周期を取り出せなかったときに仮置きするテンポ。
_DEFAULT_BPM = 120.0
# 拍子の既定。指定が無ければ常にこれを使う。
_DEFAULT_TIME_SIGNATURE = (4, 4)

# オンセット強度包絡の窓長とホップ(秒)。
_WINDOW_SEC = 0.046
_HOP_SEC = 0.012

# 自己相関のラグを走査する BPM の範囲と、拍と感じる帯域を選ぶ重みの中心・幅(オクターブ)。
# 幅は、候補どうしのタイブレークとして働く程度に取る。狭いと中心から離れた速い拍を半分の周期で採り、
# 広いと相関が半分の周期に傾いた入力を中心へ引き戻せなくなる。
_BPM_RANGE = (30.0, 300.0)
_BPM_WEIGHT_CENTER = 120.0
_BPM_WEIGHT_SIGMA = 1.5

# 重みを掛けた得点がこれ未満なら、周期が得られなかったとみなす。判定に重みが入るので、中心から
# 離れたテンポほど高い周期性が要る。幅を広げるとその要求が下がり、下がり方は中心から遠いほど大きい。
_PERIODICITY_FLOOR = 0.1

# 半分読み(重みが倍の周期=半分のテンポを選ばせた状態)を倍へ昇格する判定の定数。
# しきい値は、既知テンポの合成信号(素の拍と、8分・16分の細分を重ねた信号を 60〜240 BPM で
# 生成)を推定に通し、半分読みになった信号と正しく読めた信号を分ける値として計測で決めた。
# 昇格先の上限。半分読みとして観測された昇格先の上端(240)より上、正しく読めている通常の
# 帯の曲(135 前後)を倍にしてしまわない位置に置く。
_PROMOTION_MAX_BPM = 250.0
# 表拍と裏拍の同格性(線形流束の正規化差の中央値)の上限。半分読み側の最大 0.061 と
# 正読側の最小 0.11 を分ける中間。
_PROMOTION_CONTRAST_MAX = 0.085
# 倍のテンポ側の周期(採用周期の半分)の自己相関比の下限。半分読み側の最小 0.93 の下。
_PROMOTION_CORR_MIN = 0.9
# 判定に足る表拍・裏拍の対の最小数と、有効な対とみなす流束の下限(分布の上端に対する比)。
_PROMOTION_MIN_PAIRS = 8
_PROMOTION_PAIR_FLOOR = 0.05
# 自己相関比を測るときの包絡の補間倍率。半ラグはホップの整数に載らないことがあり、丸めの
# 罰で真の周期の相関が過小評価されるのを補間で防ぐ。
_PROMOTION_UPSAMPLE = 4


def quantize_bpm(bpm: float) -> float:
    """テンポを vpr 形式が格納できる粒度(BPM の 1/100)へ丸める。

    出力ファイルと診断・秒から tick への変換で同じ値を使うため、採用の時点で丸めておく。
    """
    return round(bpm * 100) / 100.0


@dataclass(frozen=True)
class TempoEstimate:
    """採用したテンポと拍子。"""

    bpm: float
    numerator: int
    denominator: int
    beat_offset_sec: float
    tempo_source: str  # "option" 指定値 / "estimated" 推定値 / "default" 仮置き
    time_signature_source: str  # "option" 指定値 / "default" 既定値
    resolution: int = RESOLUTION

    # 仮置きへ倒したかは出どころから導く(同じ事実を2つ持つと、警告と診断が食い違いうるため)。
    @property
    def tempo_defaulted(self) -> bool:
        return self.tempo_source == "default"

    def to_tick(self, seconds: float) -> int:
        """入力音声の時刻を tick へ写す。入力の 0 秒が tick の 0。"""
        return round(seconds * self.bpm * self.resolution / 60.0)


def _onset_envelopes(pcm):
    """オンセット強度の包絡2種(標準化した対数流束・生の線形流束)とホップ幅(秒)を返す。

    対数側は周期の推定と位相に使う(圧縮で音量差に頑健)。線形側は昇格判定の強さ比較に
    使う(対数圧縮は表拍と裏拍の振幅差を潰し、同格かどうかを見分けられなくするため)。
    """
    samples = np.asarray(pcm.samples, dtype=np.float64)
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    sample_rate = pcm.sample_rate

    window_length = int(round(_WINDOW_SEC * sample_rate))
    hop = int(round(_HOP_SEC * sample_rate))
    if window_length < 2 or hop < 1 or len(mono) < window_length:
        return np.zeros(0), np.zeros(0), _HOP_SEC

    fft_length = 1 << (window_length - 1).bit_length()
    window = np.hanning(window_length)
    starts = range(0, len(mono) - window_length + 1, hop)
    # 各フレームの振幅スペクトルを圧縮し、隣接フレーム間の増加分だけを足す。
    spectra = np.array([np.abs(np.fft.rfft(mono[s:s + window_length] * window, n=fft_length))
                        for s in starts])
    if len(spectra) < 2:
        return np.zeros(0), np.zeros(0), _HOP_SEC
    flux = np.maximum(np.diff(spectra, axis=0), 0.0).sum(axis=1)
    compressed = np.log1p(spectra)
    envelope = np.maximum(np.diff(compressed, axis=0), 0.0).sum(axis=1)

    deviation = envelope.std()
    if deviation == 0.0:
        return np.zeros(0), np.zeros(0), _HOP_SEC
    return (envelope - envelope.mean()) / deviation, flux, hop / sample_rate


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
        weight = np.exp(-(np.log2(bpm / _BPM_WEIGHT_CENTER) ** 2) / (2.0 * _BPM_WEIGHT_SIGMA ** 2))
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


def _cell_peak(flux, hop_sec, center_sec, half_width_sec):
    """中心時刻の周りのセルにある線形流束のピーク値。

    セルは半開区間で切り、隣り合うセルが境界のビンを共有しないようにする(境界にある
    オンセットを表拍と裏拍の両方のピークに数えると、正規化差が不当に 0 へ寄るため)。
    """
    lo = max(0, int(round((center_sec - half_width_sec) / hop_sec)))
    hi = min(len(flux), int(round((center_sec + half_width_sec) / hop_sec)))
    if hi <= lo:
        return 0.0
    return float(flux[lo:hi].max())


def _offbeat_contrast(envelope, flux, hop_sec, period_sec):
    """表拍と裏拍(拍の中間)の強さの差の代表値。同格なら 0 に近づく。

    位相は対数包絡で拍の周期に合わせ、拍とその中間のそれぞれの周り(周期の 1/4 幅)の
    セルから線形流束のピークを拾い、対ごとの正規化差の中央値の絶対値を返す。
    弱すぎる対は数えず、有効な対が足りなければ None(判定不能)。
    """
    if len(flux) == 0:
        return None
    phase = _estimate_phase(envelope, hop_sec, period_sec)
    floor = _PROMOTION_PAIR_FLOOR * float(np.percentile(flux, 99))
    duration = len(flux) * hop_sec
    diffs = []
    index = 0
    while True:
        beat_at = phase + index * period_sec
        mid_at = beat_at + period_sec / 2.0
        if mid_at >= duration:
            break
        on_beat = _cell_peak(flux, hop_sec, beat_at, period_sec / 4.0)
        off_beat = _cell_peak(flux, hop_sec, mid_at, period_sec / 4.0)
        if on_beat + off_beat > floor:
            diffs.append((on_beat - off_beat) / (on_beat + off_beat))
        index += 1
    if len(diffs) < _PROMOTION_MIN_PAIRS:
        return None
    return abs(float(np.median(diffs)))


def _half_period_correlation_ratio(envelope, hop_sec, period_sec):
    """倍のテンポ側の周期(採用周期の半分)の自己相関を、採用周期の自己相関で割った比。

    半ラグはホップの整数に載らないことがあるため、包絡を補間し、目標に最も近い
    補間グリッド上の1ラグを評価する。
    """
    if len(envelope) < 2:
        return 0.0
    positions = np.arange(0, len(envelope) - 1 + 1e-9, 1.0 / _PROMOTION_UPSAMPLE)
    upsampled = np.interp(positions, np.arange(len(envelope)), envelope)
    zero_lag = float(np.dot(upsampled, upsampled))
    if zero_lag == 0.0:
        return 0.0
    lag = period_sec / hop_sec * _PROMOTION_UPSAMPLE

    def correlation(target):
        # 目標に最も近い補間グリッド上の1ラグだけを評価する(近傍の最大を採ると、目標から
        # 離れた別のピークとの比になり、正確な半周期では弱い相関でも条件を通ってしまう)。
        candidate = int(round(target))
        if candidate < 1 or candidate >= len(upsampled):
            return 0.0
        return float(np.dot(upsampled[:-candidate], upsampled[candidate:])) / zero_lag

    full = correlation(lag)
    if full <= 0.0:
        return 0.0
    return correlation(lag / 2.0) / full


def _promote_folded_bpm(bpm, envelope, flux, hop_sec, denominator):
    """半分読みと判定できたとき、推定テンポを倍へ昇格して返す。そうでなければそのまま返す。

    120 中心の重みは、速い曲で倍の周期(=半分のテンポ)を選ばせることがある。採用した拍の
    中間に同格の拍が並ぶ証拠——線形流束で表拍と裏拍の強さが釣り合い、倍のテンポ側の周期
    (採用周期の半分)にも同等の周期性がある——が揃うときだけ倍へ昇格する。証拠が足りない
    ときは昇格しない(正しく読めた曲を倍へ壊すより、半分読みを残す方が安全なため)。
    """
    doubled = bpm * 2.0
    if doubled > _PROMOTION_MAX_BPM:
        return bpm
    period_sec = 60.0 / bpm * 4.0 / denominator
    contrast = _offbeat_contrast(envelope, flux, hop_sec, period_sec)
    if contrast is None or contrast >= _PROMOTION_CONTRAST_MAX:
        return bpm
    if _half_period_correlation_ratio(envelope, hop_sec, period_sec) < _PROMOTION_CORR_MIN:
        return bpm
    return doubled


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


def estimate(pcm, *, tempo_bpm=None, time_signature=None) -> TempoEstimate:
    """分離前の入力からテンポを求め、拍子を決める。

    tempo_bpm を与えられた実行ではテンポを推定せずその値を使う。周期を取り出せない入力では
    テンポを既定へ倒し、そのことを tempo_defaulted で示す。拍子は音声から決めず、time_signature の
    指定があればその値、無ければ既定を使う。
    """
    numerator, denominator = time_signature if time_signature is not None else (None, 4)

    envelope, flux, hop_sec = _onset_envelopes(pcm)
    estimated_bpm = None if tempo_bpm is not None else _estimate_bpm(envelope, hop_sec, denominator)
    if estimated_bpm is not None:
        # 指定テンポには掛けない(推定の誤りを直す判定のため)。丸めの前に倍へ直す。
        estimated_bpm = _promote_folded_bpm(estimated_bpm, envelope, flux, hop_sec, denominator)

    if tempo_bpm is None and estimated_bpm is None:
        # 周期が得られないので既定へ倒す。位相も求められないため 0 を置く。
        return TempoEstimate(
            bpm=_DEFAULT_BPM,
            numerator=numerator if numerator is not None else _DEFAULT_TIME_SIGNATURE[0],
            denominator=denominator, beat_offset_sec=0.0,
            tempo_source="default",
            time_signature_source="option" if numerator is not None else "default")

    bpm = quantize_bpm(tempo_bpm if tempo_bpm is not None else estimated_bpm)
    # 自己相関が拾うのは拍(拍子の分母が表す音価)の周期。四分音符あたりの BPM から戻す。
    period_sec = 60.0 / bpm * 4.0 / denominator
    # 位相はテンポと別に決まるので、テンポが指定値でも音声から求める。
    offset_sec = _estimate_phase(envelope, hop_sec, period_sec)

    return TempoEstimate(
        bpm=bpm,
        numerator=numerator if numerator is not None else _DEFAULT_TIME_SIGNATURE[0],
        denominator=denominator,
        beat_offset_sec=offset_sec,
        tempo_source="option" if tempo_bpm is not None else "estimated",
        time_signature_source="option" if numerator is not None else "default")
