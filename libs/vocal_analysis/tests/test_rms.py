"""S3 強弱RMSのテスト(vocal_analysis.md §6・§6.1)。

純関数の核(フレーム化・チャンネル方向の扱い・パーセンタイル相対正規化・dynamic_range_db 算出)は
合成配列で決定論的に検証し、公開関数 compute_rms は AudioPcm からの一連の流れとゲイン不変性
(§6.1 の要である「ミックス全体の一様ゲイン差でRMSが不変」)を検証する。
"""

import math

import numpy as np
import pytest

# vocal_analysis.rms が未実装の間は import が失敗するため xfail 印で緑を保つ。
# strict=True: 未実装印を外し忘れたまま通ると XPASS が失敗になり検出できる。
pytestmark = pytest.mark.xfail(reason="impl pending: vocal_analysis.rms", strict=True)


def test_frame_rms_center_times_and_count():
    from vocal_analysis.rms import _frame_rms

    # sample_rate=1000 で計算しやすくする。frame=25サンプル・hop=10サンプル・信号長100サンプル。
    # start は 0,10,...,70(70+25=95<=100)の8個(80+25=105>100 で打ち切り。§6.1)。
    sample_rate = 1000
    samples = np.full((100, 1), 0.5, dtype=np.float32)

    times_sec, raw_rms = _frame_rms(samples, sample_rate)

    assert len(times_sec) == 8
    assert len(raw_rms) == 8
    expected_starts = np.arange(0, 71, 10)
    expected_centers_sec = (expected_starts + 25 / 2) / sample_rate
    assert np.allclose(times_sec, expected_centers_sec)


def test_frame_rms_rounds_frame_and_hop_samples():
    from vocal_analysis.rms import _frame_rms

    # §6.1: フレーム長25ms・ホップ10msは対象サンプルレートで round() してサンプル数へ換算する。
    # sample_rate=1370 は 0.025*1370=34.25→round 34(ceil なら35)、0.010*1370=13.7→round 14
    # (floor なら13)と、frame側でceilと、hop側でfloorと食い違うため round 固有の値を区別できる。
    sample_rate = 1370
    length = 200
    samples = np.full((length, 1), 0.5, dtype=np.float32)

    times_sec, raw_rms = _frame_rms(samples, sample_rate)

    frame_samples = round(0.025 * sample_rate)
    hop_samples = round(0.010 * sample_rate)
    expected_starts = np.arange(0, length - frame_samples + 1, hop_samples)
    expected_centers_sec = (expected_starts + frame_samples / 2) / sample_rate

    assert len(times_sec) == len(expected_starts)
    assert len(raw_rms) == len(expected_starts)
    assert np.allclose(times_sec, expected_centers_sec)


def test_frame_rms_constant_amplitude_matches_amplitude():
    from vocal_analysis.rms import _frame_rms

    # 振幅一定(0.5)の信号は、どのフレームも RMS = 0.5 になる。
    sample_rate = 1000
    samples = np.full((100, 1), 0.5, dtype=np.float32)

    _, raw_rms = _frame_rms(samples, sample_rate)

    assert np.allclose(raw_rms, 0.5, atol=1e-6)


def test_frame_rms_pools_across_channels():
    from vocal_analysis.rms import _frame_rms

    # §6.1: 複数チャンネルはフレーム内の全チャンネル・全サンプルをまとめてRMSを取る
    # (チャンネルごとの独立計算・平均はしない)。左ch一定0.6・右ch一定0.8 →
    # pooled RMS = sqrt((0.6^2+0.8^2)/2) = sqrt(0.5)。
    sample_rate = 1000
    left = np.full(100, 0.6, dtype=np.float32)
    right = np.full(100, 0.8, dtype=np.float32)
    samples = np.stack([left, right], axis=1)

    _, raw_rms = _frame_rms(samples, sample_rate)

    assert np.allclose(raw_rms, math.sqrt(0.5), atol=1e-6)


def test_frame_rms_signal_shorter_than_frame_returns_empty():
    from vocal_analysis.rms import _frame_rms

    # §6.1: 信号長がフレーム長未満でフレームが1つも取れない場合は空配列(エラーにしない)。
    sample_rate = 1000
    samples = np.full((10, 1), 0.5, dtype=np.float32)  # frame_samples=25 > 10

    times_sec, raw_rms = _frame_rms(samples, sample_rate)

    assert len(times_sec) == 0
    assert len(raw_rms) == 0


def test_normalize_rms_reference_formula():
    from vocal_analysis.rms import _normalize_rms

    # 0.0〜1.0 の等間隔11点。線形補間で p10=0.1・p90=0.9(§6.1)。
    raw_rms = np.linspace(0.0, 1.0, 11)

    values = _normalize_rms(raw_rms)

    p10 = np.percentile(raw_rms, 10, method="linear")
    p90 = np.percentile(raw_rms, 90, method="linear")
    expected = np.clip((raw_rms - p10) / (p90 - p10), 0.0, 1.0)
    assert np.allclose(values, expected)


def test_normalize_rms_p90_equals_p10_returns_zero():
    from vocal_analysis.rms import _normalize_rms

    # §6.1: p90 == p10(無音・定常)は0除算を避け全フレーム 0.0。
    raw_rms = np.full(20, 0.3)

    values = _normalize_rms(raw_rms)

    assert np.array_equal(values, np.zeros(20))


def test_normalize_rms_is_gain_invariant():
    from vocal_analysis.rms import _normalize_rms

    # §6.1: 差の比なので入力ゲインに不変。
    raw_rms = np.array([0.0, 0.05, 0.2, 0.5, 0.8, 1.0])
    gained = raw_rms * 3.0

    assert np.allclose(_normalize_rms(raw_rms), _normalize_rms(gained), atol=1e-9)


def test_dynamic_range_db_reference_formula():
    from vocal_analysis.rms import _dynamic_range_db

    raw_rms = np.linspace(0.0, 1.0, 11)

    result = _dynamic_range_db(raw_rms)

    p5 = np.percentile(raw_rms, 5, method="linear")
    p95 = np.percentile(raw_rms, 95, method="linear")
    expected = 20 * math.log10(p95 / p5)
    assert result == pytest.approx(expected)


def test_dynamic_range_db_p95_zero_returns_zero():
    from vocal_analysis.rms import _dynamic_range_db

    # §6.1: 線形補間で得た p95 の値自体が0(比の基準が無い)場合は dynamic_range_db = 0.0。
    raw_rms = np.zeros(20)

    assert _dynamic_range_db(raw_rms) == 0.0


def test_dynamic_range_db_p5_zero_uses_relative_floor():
    from vocal_analysis.rms import _dynamic_range_db

    # p5==0 かつ p95>0(§6.1)。前半10個が0、後半10個が1.0 → p5=0.0・p95=1.0(線形補間)。
    # p5 を p95*1e-6 に置き換えるため dynamic_range_db = 20*log10(1.0 / 1e-6) = 120.0。
    raw_rms = np.concatenate([np.zeros(10), np.ones(10)])

    assert _dynamic_range_db(raw_rms) == pytest.approx(120.0)


def test_dynamic_range_db_is_gain_invariant():
    from vocal_analysis.rms import _dynamic_range_db

    # p5==0 の相対フロア分岐を含め、正の一様ゲインで不変(§6.1)。
    raw_rms = np.concatenate([np.zeros(10), np.ones(10)])
    gained = raw_rms * 4.0

    assert _dynamic_range_db(raw_rms) == pytest.approx(_dynamic_range_db(gained))


def test_compute_rms_returns_envelope_with_expected_shape():
    from vocal_analysis.rms import compute_rms
    from vocal_analysis.types import AudioPcm, RmsEnvelope

    sample_rate = 1000
    rng = np.random.default_rng(0)
    samples = (rng.random((500, 1), dtype=np.float32) - 0.5) * 2  # [-1, 1) のノイズ
    pcm = AudioPcm(samples=samples, sample_rate=sample_rate)

    envelope = compute_rms(pcm)

    assert isinstance(envelope, RmsEnvelope)
    assert len(envelope.times_sec) == len(envelope.values)
    assert len(envelope.times_sec) > 0
    assert np.all(envelope.values >= 0.0) and np.all(envelope.values <= 1.0)


def test_compute_rms_is_gain_invariant():
    from vocal_analysis.rms import compute_rms
    from vocal_analysis.types import AudioPcm

    # §6.1: ミックス全体の一様ゲイン差でRMSが不変(パーセンタイル正規化・dB算出とも比なので不変)。
    sample_rate = 1000
    rng = np.random.default_rng(1)
    samples = (rng.random((500, 1), dtype=np.float32) - 0.5) * 0.4
    pcm = AudioPcm(samples=samples, sample_rate=sample_rate)
    gained_pcm = AudioPcm(samples=(samples * 2.5).astype(np.float32), sample_rate=sample_rate)

    envelope = compute_rms(pcm)
    gained_envelope = compute_rms(gained_pcm)

    assert np.allclose(envelope.times_sec, gained_envelope.times_sec)
    assert np.allclose(envelope.values, gained_envelope.values, atol=1e-5)
    assert envelope.dynamic_range_db == pytest.approx(gained_envelope.dynamic_range_db, abs=1e-3)


def test_compute_rms_signal_shorter_than_frame_returns_empty_envelope():
    from vocal_analysis.rms import compute_rms
    from vocal_analysis.types import AudioPcm

    pcm = AudioPcm(samples=np.full((10, 1), 0.5, dtype=np.float32), sample_rate=1000)

    envelope = compute_rms(pcm)

    assert len(envelope.times_sec) == 0
    assert len(envelope.values) == 0
    assert envelope.dynamic_range_db == 0.0
