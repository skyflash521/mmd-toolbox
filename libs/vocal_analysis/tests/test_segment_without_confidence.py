import dataclasses
import inspect

import pytest

from vocal_analysis import Segment


def test_segment_has_no_confidence_field():
    assert "confidence" not in {f.name for f in dataclasses.fields(Segment)}


def test_segment_fields_are_exactly_type_start_end_phoneme():
    assert [f.name for f in dataclasses.fields(Segment)] == ["type", "start_sec", "end_sec", "phoneme"]


def test_segment_constructor_has_no_confidence_parameter():
    assert "confidence" not in inspect.signature(Segment).parameters


def test_segment_constructor_rejects_confidence_keyword():
    with pytest.raises(TypeError):
        Segment(type="vowel", start_sec=0.0, end_sec=1.0, phoneme="a", confidence=0.9)


def test_segment_constructor_rejects_fifth_positional_argument():
    with pytest.raises(TypeError):
        Segment("vowel", 0.0, 1.0, "a", 0.9)


def test_segment_instance_has_no_confidence_attribute():
    segment = Segment(type="vowel", start_sec=0.0, end_sec=1.0, phoneme="a")

    assert not hasattr(segment, "confidence")
    assert "confidence" not in dataclasses.asdict(segment)
