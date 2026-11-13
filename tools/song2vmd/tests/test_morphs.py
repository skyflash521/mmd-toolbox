from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape
from song2vmd import morphs


def ev(shape, start, end, open_amount=0.0, consonant_class=ConsonantClass.NONE):
    return MouthEvent(shape=shape, start=start, end=end, open_amount=open_amount,
                       consonant_class=consonant_class)


def test_build_vmd_document_contains_generated_morph_keys():
    events = [
        ev(MouthShape.SILENCE, 0, 5),
        ev(MouthShape.A, 5, 20, open_amount=0.6),
        ev(MouthShape.SILENCE, 20, 25),
    ]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="")
    names = {key.name for key in document.morph}
    assert "あ" in names


def test_model_name_is_cp932_encoded_and_padded_to_20_bytes():
    events = [ev(MouthShape.A, 0, 10, open_amount=0.5)]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="モデル")
    assert document.model_name_raw == "モデル".encode("cp932").ljust(20, b"\x00")


def test_empty_model_name_produces_all_zero_padding():
    events = [ev(MouthShape.A, 0, 10, open_amount=0.5)]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="")
    assert document.model_name_raw == b"\x00" * 20


def test_frame0_neutral_key_exists_for_every_used_morph_even_when_first_key_is_later():
    events = [
        ev(MouthShape.SILENCE, 0, 10),
        ev(MouthShape.A, 10, 30, open_amount=0.6),
    ]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="")
    names_used = {key.name for key in document.morph}
    for name in names_used:
        frames = [key.frame for key in document.morph if key.name == name]
        assert 0 in frames


def test_morph_keys_are_sorted_and_deduplicated_per_name():
    events = [
        ev(MouthShape.A, 0, 15, open_amount=0.5),
        ev(MouthShape.I, 15, 30, open_amount=0.5),
        ev(MouthShape.A, 30, 45, open_amount=0.7),
    ]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="")
    by_name = {}
    for key in document.morph:
        by_name.setdefault(key.name, []).append(key.frame)
    for frames in by_name.values():
        assert frames == sorted(frames)
        assert len(frames) == len(set(frames))


def test_only_morph_section_is_populated():
    events = [ev(MouthShape.A, 0, 10, open_amount=0.5)]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="")
    assert document.bone == []
    assert document.camera == []
    assert document.light == []
    assert document.self_shadow == []
    assert document.ik_property == []


def test_empty_events_produce_empty_morph_document():
    document = morphs.build_vmd_document([], GenerationParams(), model_name="")
    assert document.morph == []
