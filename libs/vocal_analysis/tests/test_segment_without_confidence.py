import dataclasses

from vocal_analysis import Segment


def test_segment_fields_are_exactly_type_start_end_phoneme():
    assert [f.name for f in dataclasses.fields(Segment)] == ["type", "start_sec", "end_sec", "phoneme"]
