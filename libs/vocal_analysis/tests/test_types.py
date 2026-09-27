import dataclasses
import typing
from pathlib import Path

import numpy as np


def test_public_types_are_dataclasses():
    from vocal_analysis import AnalysisResult, AudioPcm, RmsEnvelope, Segment

    assert dataclasses.is_dataclass(AudioPcm)
    assert dataclasses.is_dataclass(Segment)
    assert dataclasses.is_dataclass(RmsEnvelope)
    assert dataclasses.is_dataclass(AnalysisResult)


def test_public_types_have_exactly_the_contracted_fields():
    from vocal_analysis import AnalysisResult, AudioPcm, RmsEnvelope, Segment

    def field_names(cls):
        return {f.name for f in dataclasses.fields(cls)}

    assert field_names(AudioPcm) == {"samples", "sample_rate"}
    assert field_names(Segment) == {"type", "start_sec", "end_sec", "phoneme", "confidence"}
    assert field_names(RmsEnvelope) == {"times_sec", "values", "dynamic_range_db"}
    assert field_names(AnalysisResult) == {"vocal_wav", "segments", "rms"}


def test_public_type_annotations_pin_key_contracts():
    from vocal_analysis import AnalysisResult, AudioPcm, RmsEnvelope, Segment

    audio = typing.get_type_hints(AudioPcm)
    assert audio["samples"] is np.ndarray
    assert audio["sample_rate"] is int

    seg = typing.get_type_hints(Segment)
    assert typing.get_origin(seg["type"]) is typing.Literal
    assert typing.get_args(seg["type"]) == ("vowel", "consonant", "gap")
    assert seg["start_sec"] is float
    assert seg["end_sec"] is float
    assert set(typing.get_args(seg["phoneme"])) == {str, type(None)}
    assert set(typing.get_args(seg["confidence"])) == {float, type(None)}

    rms = typing.get_type_hints(RmsEnvelope)
    assert rms["times_sec"] is np.ndarray
    assert rms["values"] is np.ndarray
    assert rms["dynamic_range_db"] is float

    res = typing.get_type_hints(AnalysisResult)
    assert res["vocal_wav"] is Path
    assert typing.get_origin(res["segments"]) is list
    assert typing.get_args(res["segments"]) == (Segment,)
    assert res["rms"] is RmsEnvelope


def test_audio_pcm_holds_samples_and_sample_rate():
    from vocal_analysis import AudioPcm

    samples = np.zeros((100, 2), dtype=np.float32)
    pcm = AudioPcm(samples=samples, sample_rate=44100)

    assert pcm.samples.dtype == np.float32
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

    seg = Segment(type="gap", start_sec=0.0, end_sec=0.5, phoneme=None, confidence=None)

    assert seg.type == "gap"
    assert seg.phoneme is None
    assert seg.confidence is None


def test_segment_consonant_carries_ipa_phoneme():
    from vocal_analysis import Segment

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
