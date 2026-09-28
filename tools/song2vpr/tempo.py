from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np

RESOLUTION = 480

_VPR_BPM_STORAGE_SCALE = 100

_FALLBACK_BPM = 120.0
_DEFAULT_TIME_SIGNATURE = (4, 4)

_ONSET_WINDOW_SEC = 0.046
_ONSET_HOP_SEC = 0.012

_LAG_SCAN_BPM_RANGE = (30.0, 300.0)
_BPM_WEIGHT_CENTER = 120.0
_BPM_WEIGHT_SIGMA_OCTAVES = 1.5
_MIN_WEIGHTED_PERIODICITY_SCORE = 0.1
_SCORE_TIE_TOLERANCE = 1e-12

_PROMOTED_BPM_MAX = 250.0
_PROMOTION_OFFBEAT_CONTRAST_MAX = 0.085
_PROMOTION_HALF_PERIOD_CORR_RATIO_MIN = 0.9
_PROMOTION_MIN_BEAT_PAIRS = 8
_PROMOTION_PAIR_FLUX_FLOOR_RATIO = 0.05
_PROMOTION_FLUX_REFERENCE_PERCENTILE = 99
_PROMOTION_ENVELOPE_UPSAMPLE = 4

TempoSource = Literal["option", "estimated", "default"]
TimeSignatureSource = Literal["option", "default"]


def quantize_bpm(bpm: float) -> float:
    return round(bpm * _VPR_BPM_STORAGE_SCALE) / float(_VPR_BPM_STORAGE_SCALE)


@dataclass(frozen=True)
class TempoEstimate:
    bpm: float
    numerator: int
    denominator: int
    tempo_source: TempoSource
    time_signature_source: TimeSignatureSource
    resolution: int = RESOLUTION

    @property
    def tempo_defaulted(self) -> bool:
        return self.tempo_source == "default"

    def to_tick(self, seconds: float) -> int:
        return round(seconds * self.bpm * self.resolution / 60.0)


class _OnsetEnvelopes(NamedTuple):
    standardized_log_flux: np.ndarray
    linear_flux: np.ndarray
    hop_sec: float


def _onset_envelopes(pcm) -> _OnsetEnvelopes:
    samples = np.asarray(pcm.samples, dtype=np.float64)
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    sample_rate = pcm.sample_rate

    window_length = int(round(_ONSET_WINDOW_SEC * sample_rate))
    hop = int(round(_ONSET_HOP_SEC * sample_rate))
    if window_length < 2 or hop < 1 or len(mono) < window_length:
        return _OnsetEnvelopes(np.zeros(0), np.zeros(0), _ONSET_HOP_SEC)

    fft_length = 1 << (window_length - 1).bit_length()
    window = np.hanning(window_length)
    starts = range(0, len(mono) - window_length + 1, hop)
    spectra = np.array([np.abs(np.fft.rfft(mono[s:s + window_length] * window, n=fft_length))
                        for s in starts])
    if len(spectra) < 2:
        return _OnsetEnvelopes(np.zeros(0), np.zeros(0), _ONSET_HOP_SEC)
    linear_flux = np.maximum(np.diff(spectra, axis=0), 0.0).sum(axis=1)
    compressed = np.log1p(spectra)
    log_flux = np.maximum(np.diff(compressed, axis=0), 0.0).sum(axis=1)

    deviation = log_flux.std()
    if deviation == 0.0:
        return _OnsetEnvelopes(np.zeros(0), np.zeros(0), _ONSET_HOP_SEC)
    return _OnsetEnvelopes((log_flux - log_flux.mean()) / deviation, linear_flux,
                           hop / sample_rate)


def _estimate_quarter_note_bpm(envelope, hop_sec, denominator) -> float | None:
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
        if not _LAG_SCAN_BPM_RANGE[0] <= bpm <= _LAG_SCAN_BPM_RANGE[1]:
            continue
        correlation = float(np.dot(envelope[:-lag], envelope[lag:])) / zero_lag
        perceived_tempo_weight = np.exp(-(np.log2(bpm / _BPM_WEIGHT_CENTER) ** 2)
                                        / (2.0 * _BPM_WEIGHT_SIGMA_OCTAVES ** 2))
        score = correlation * perceived_tempo_weight
        if best is None or score > best[0] + _SCORE_TIE_TOLERANCE:
            best = (score, bpm, lag)
        elif abs(score - best[0]) <= _SCORE_TIE_TOLERANCE:
            if abs(bpm - _BPM_WEIGHT_CENTER) < abs(best[1] - _BPM_WEIGHT_CENTER):
                best = (score, bpm, lag)

    if best is None or best[0] < _MIN_WEIGHTED_PERIODICITY_SCORE:
        return None
    return best[1]


def _half_open_cell_peak(flux, hop_sec, center_sec, half_width_sec):
    lo = max(0, int(round((center_sec - half_width_sec) / hop_sec)))
    hi = min(len(flux), int(round((center_sec + half_width_sec) / hop_sec)))
    if hi <= lo:
        return 0.0
    return float(flux[lo:hi].max())


def _median_onbeat_offbeat_contrast(envelope, flux, hop_sec, period_sec) -> float | None:
    if len(flux) == 0:
        return None
    first_beat_sec = _estimate_first_beat_sec(envelope, hop_sec, period_sec)
    pair_floor = _PROMOTION_PAIR_FLUX_FLOOR_RATIO * float(
        np.percentile(flux, _PROMOTION_FLUX_REFERENCE_PERCENTILE))
    duration = len(flux) * hop_sec
    normalized_diffs = []
    index = 0
    while True:
        beat_at = first_beat_sec + index * period_sec
        offbeat_at = beat_at + period_sec / 2.0
        if offbeat_at >= duration:
            break
        on_beat = _half_open_cell_peak(flux, hop_sec, beat_at, period_sec / 4.0)
        off_beat = _half_open_cell_peak(flux, hop_sec, offbeat_at, period_sec / 4.0)
        if on_beat + off_beat > pair_floor:
            normalized_diffs.append((on_beat - off_beat) / (on_beat + off_beat))
        index += 1
    if len(normalized_diffs) < _PROMOTION_MIN_BEAT_PAIRS:
        return None
    return abs(float(np.median(normalized_diffs)))


def _half_period_correlation_ratio(envelope, hop_sec, period_sec):
    if len(envelope) < 2:
        return 0.0
    positions = np.arange(0, len(envelope) - 1 + 1e-9, 1.0 / _PROMOTION_ENVELOPE_UPSAMPLE)
    upsampled = np.interp(positions, np.arange(len(envelope)), envelope)
    zero_lag = float(np.dot(upsampled, upsampled))
    if zero_lag == 0.0:
        return 0.0
    period_lag = period_sec / hop_sec * _PROMOTION_ENVELOPE_UPSAMPLE

    def correlation_at_nearest_lag(target_lag):
        lag = int(round(target_lag))
        if lag < 1 or lag >= len(upsampled):
            return 0.0
        return float(np.dot(upsampled[:-lag], upsampled[lag:])) / zero_lag

    full = correlation_at_nearest_lag(period_lag)
    if full <= 0.0:
        return 0.0
    return correlation_at_nearest_lag(period_lag / 2.0) / full


def _double_if_half_tempo_reading(bpm, envelope, flux, hop_sec, denominator):
    doubled = bpm * 2.0
    if doubled > _PROMOTED_BPM_MAX:
        return bpm
    period_sec = 60.0 / bpm * 4.0 / denominator
    contrast = _median_onbeat_offbeat_contrast(envelope, flux, hop_sec, period_sec)
    if contrast is None or contrast >= _PROMOTION_OFFBEAT_CONTRAST_MAX:
        return bpm
    if (_half_period_correlation_ratio(envelope, hop_sec, period_sec)
            < _PROMOTION_HALF_PERIOD_CORR_RATIO_MIN):
        return bpm
    return doubled


def _nonnegative_envelope_at_beats(envelope, hop_sec, offset_sec, period_sec):
    if period_sec <= 0.0 or len(envelope) == 0:
        return np.zeros(0)
    duration = len(envelope) * hop_sec
    count = int(max(0.0, duration - offset_sec) / period_sec) + 1
    indices = np.clip(np.rint((offset_sec + np.arange(count) * period_sec) / hop_sec).astype(int),
                      0, len(envelope) - 1)
    return np.maximum(envelope[indices], 0.0)


def _estimate_first_beat_sec(envelope, hop_sec, period_sec):
    if len(envelope) == 0 or period_sec <= 0.0:
        return 0.0
    best = None
    for step in range(max(1, int(round(period_sec / hop_sec)))):
        offset = step * hop_sec
        total = float(_nonnegative_envelope_at_beats(envelope, hop_sec, offset, period_sec).sum())
        if best is None or total > best[0] + _SCORE_TIE_TOLERANCE:
            best = (total, offset)
    return 0.0 if best is None else best[1]


def estimate(unseparated_pcm, *, tempo_bpm=None, time_signature=None) -> TempoEstimate:
    numerator, denominator = time_signature if time_signature is not None else (None, 4)

    envelope, flux, hop_sec = _onset_envelopes(unseparated_pcm)
    estimated_bpm = (None if tempo_bpm is not None
                     else _estimate_quarter_note_bpm(envelope, hop_sec, denominator))
    if estimated_bpm is not None:
        estimated_bpm = _double_if_half_tempo_reading(estimated_bpm, envelope, flux, hop_sec,
                                                      denominator)

    if tempo_bpm is None and estimated_bpm is None:
        return TempoEstimate(
            bpm=_FALLBACK_BPM,
            numerator=numerator if numerator is not None else _DEFAULT_TIME_SIGNATURE[0],
            denominator=denominator,
            tempo_source="default",
            time_signature_source="option" if numerator is not None else "default")

    bpm = quantize_bpm(tempo_bpm if tempo_bpm is not None else estimated_bpm)
    return TempoEstimate(
        bpm=bpm,
        numerator=numerator if numerator is not None else _DEFAULT_TIME_SIGNATURE[0],
        denominator=denominator,
        tempo_source="option" if tempo_bpm is not None else "estimated",
        time_signature_source="option" if numerator is not None else "default")
