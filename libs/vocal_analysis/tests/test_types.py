import dataclasses


def test_public_types_have_exactly_the_contracted_fields():
    from vocal_analysis import AnalysisResult, AudioPcm, RmsEnvelope, Segment

    def field_names(cls):
        return {f.name for f in dataclasses.fields(cls)}

    assert field_names(AudioPcm) == {"samples", "sample_rate"}
    assert field_names(Segment) == {"type", "start_sec", "end_sec", "phoneme"}
    assert field_names(RmsEnvelope) == {"times_sec", "values", "dynamic_range_db"}
    assert field_names(AnalysisResult) == {"vocal_wav", "segments", "rms"}
