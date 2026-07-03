"""公開データ型のテスト(vocal_analysis.md §2.1)。

各ステージをつなぐ正規化中間形式(AudioPcm/Segment/RmsEnvelope/AnalysisResult)の
公開型が仕様どおりのフィールドを持つことを確認する。S1 出力(ボーカル WAV)は専用
dataclass を設けず pathlib.Path で表す(§2.1)。
"""

import dataclasses
import typing
from pathlib import Path

import numpy as np
import pytest

# vocal_analysis の公開型が未実装の間は import が失敗するため xfail 印で緑を保つ。
# strict=True: 未実装印を外し忘れたまま通ると XPASS が失敗になり検出できる。
pytestmark = pytest.mark.xfail(reason="impl pending: vocal_analysis public API", strict=True)


def test_public_types_are_dataclasses():
    from vocal_analysis import AnalysisResult, AudioPcm, RmsEnvelope, Segment

    # §2.1: S1 出力(Path)以外の公開中間形式は Python の dataclass で表す。
    assert dataclasses.is_dataclass(AudioPcm)
    assert dataclasses.is_dataclass(Segment)
    assert dataclasses.is_dataclass(RmsEnvelope)
    assert dataclasses.is_dataclass(AnalysisResult)


def test_public_type_field_contracts():
    from vocal_analysis import AnalysisResult, AudioPcm, RmsEnvelope, Segment

    # §2.1 が規定する公開フィールドの集合を固定し、実装が余計なフィールドを足す/
    # 必要なフィールドを落とす逸脱を検出する(名前の完全一致で確認する)。
    def field_names(cls):
        return {f.name for f in dataclasses.fields(cls)}

    assert field_names(AudioPcm) == {"samples", "sample_rate"}
    assert field_names(Segment) == {"type", "start_sec", "end_sec", "phoneme", "confidence"}
    assert field_names(RmsEnvelope) == {"times_sec", "values", "dynamic_range_db"}
    assert field_names(AnalysisResult) == {"vocal_wav", "segments", "rms"}


def test_public_type_annotations_pin_key_contracts():
    from vocal_analysis import AnalysisResult, AudioPcm, RmsEnvelope, Segment

    # §2.1 の型注釈のうち契約価値の高いものを固定する。フィールド名一致だけでは実装が
    # 注釈を緩めても通るため、注釈も検証する。表現差(Optional[X] と X|None 等)に頑健な
    # get_origin/get_args で意味を確認し、注釈オブジェクトの厳密同値には依存しない。
    audio = typing.get_type_hints(AudioPcm)
    assert audio["samples"] is np.ndarray
    assert audio["sample_rate"] is int

    seg = typing.get_type_hints(Segment)
    # type は vowel/consonant/gap の3値 Literal(§2.1)。
    assert typing.get_origin(seg["type"]) is typing.Literal
    assert typing.get_args(seg["type"]) == ("vowel", "consonant", "gap")
    # 時刻は秒 float(§2.1)。
    assert seg["start_sec"] is float
    assert seg["end_sec"] is float
    # phoneme(IPA)・confidence は Optional(gap は phoneme=None、confidence は任意)。
    assert set(typing.get_args(seg["phoneme"])) == {str, type(None)}
    assert set(typing.get_args(seg["confidence"])) == {float, type(None)}

    rms = typing.get_type_hints(RmsEnvelope)
    assert rms["times_sec"] is np.ndarray
    assert rms["values"] is np.ndarray
    assert rms["dynamic_range_db"] is float

    res = typing.get_type_hints(AnalysisResult)
    assert res["vocal_wav"] is Path
    # segments は list[Segment]。
    assert typing.get_origin(res["segments"]) is list
    assert typing.get_args(res["segments"]) == (Segment,)
    assert res["rms"] is RmsEnvelope


def test_audio_pcm_holds_samples_and_sample_rate():
    from vocal_analysis import AudioPcm

    samples = np.zeros((100, 2), dtype=np.float32)
    pcm = AudioPcm(samples=samples, sample_rate=44100)

    assert pcm.samples.dtype == np.float32
    # 形状は (フレーム数, チャンネル数)。チャンネル数は shape[1] で得る(§2.1)。
    assert pcm.samples.shape == (100, 2)
    assert pcm.samples.shape[1] == 2
    assert pcm.sample_rate == 44100


def test_segment_vowel_carries_ipa_phoneme():
    from vocal_analysis import Segment

    seg = Segment(
        type="vowel",
        start_sec=0.5,
        end_sec=0.8,
        phoneme="a",
        confidence=0.9,
    )

    assert seg.type == "vowel"
    assert seg.start_sec == 0.5
    assert seg.end_sec == 0.8
    assert seg.phoneme == "a"
    assert seg.confidence == 0.9


def test_segment_gap_has_no_phoneme():
    from vocal_analysis import Segment

    # gap は音素ラベルを持たない(§2.1)。confidence は任意。
    seg = Segment(type="gap", start_sec=0.0, end_sec=0.5, phoneme=None, confidence=None)

    assert seg.type == "gap"
    assert seg.phoneme is None
    assert seg.confidence is None


def test_segment_consonant_carries_ipa_phoneme():
    from vocal_analysis import Segment

    # §2.1: type は vowel/consonant/gap の3種。子音も音素ラベル(IPA)を持つ。
    seg = Segment(type="consonant", start_sec=0.3, end_sec=0.4, phoneme="k", confidence=0.7)

    assert seg.type == "consonant"
    assert seg.phoneme == "k"


def test_rms_envelope_holds_arrays_and_dynamic_range():
    from vocal_analysis import RmsEnvelope

    times = np.array([0.005, 0.015, 0.025], dtype=np.float64)
    values = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    rms = RmsEnvelope(times_sec=times, values=values, dynamic_range_db=18.0)

    assert np.allclose(rms.times_sec, times)
    assert np.allclose(rms.values, values)
    # dynamic_range_db は正規化済み values からは復元できないため別フィールドで持つ(§2.1)。
    assert rms.dynamic_range_db == 18.0


def test_analysis_result_bundles_shared_outputs():
    from vocal_analysis import AnalysisResult, RmsEnvelope, Segment

    segs = [Segment(type="vowel", start_sec=0.0, end_sec=0.3, phoneme="a", confidence=None)]
    rms = RmsEnvelope(
        times_sec=np.array([0.005]),
        values=np.array([1.0]),
        dynamic_range_db=12.0,
    )
    result = AnalysisResult(vocal_wav=Path("vocal.wav"), segments=segs, rms=rms)

    assert result.vocal_wav == Path("vocal.wav")
    assert result.segments == segs
    assert result.rms is rms
