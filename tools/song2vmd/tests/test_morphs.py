import pytest

from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape
from song2vmd import morphs
from vmd import MorphKey, io


def ev(shape, start, end, open_amount=0.0, consonant_class=ConsonantClass.NONE):
    return MouthEvent(shape=shape, start=start, end=end, open_amount=open_amount,
                       consonant_class=consonant_class)


def test_build_vmd_document_passes_events_to_lipsync_and_stores_returned_keys(monkeypatch):
    events = [ev(MouthShape.A, 0, 10, open_amount=0.5)]
    params = GenerationParams()
    captured = {}
    fake_keys = [MorphKey("Fake".encode("cp932").ljust(15, b"\x00"), 3, 0.42)]

    def fake_generate_morph_keys(passed_events, passed_params):
        captured["events"] = passed_events
        captured["params"] = passed_params
        return fake_keys

    monkeypatch.setattr(morphs, "generate_morph_keys", fake_generate_morph_keys)
    document = morphs.build_vmd_document(events, params, model_name="")
    assert captured["events"] == events
    assert captured["params"] is params
    assert any(key.name == "Fake" and key.frame == 3 and key.weight == pytest.approx(0.42)
               for key in document.morph)


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


def test_round_trip_write_and_read_keeps_keys_within_float32_precision(tmp_path):
    events = [
        ev(MouthShape.SILENCE, 0, 5),
        ev(MouthShape.A, 5, 20, open_amount=0.6),
        ev(MouthShape.N, 20, 26, open_amount=0.3),
        ev(MouthShape.SILENCE, 26, 30),
    ]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="テスト")
    out = tmp_path / "out.vmd"
    io.write_file(document, str(out))
    read_back, warnings = io.read(str(out))
    assert warnings == []
    assert read_back.model_name == "テスト"
    read_morphs = sorted((key.name, key.frame) for key in read_back.morph)
    expected_names_frames = sorted((key.name, key.frame) for key in document.morph)
    assert read_morphs == expected_names_frames
    expected_weight = {(key.name, key.frame): key.weight for key in document.morph}
    for key in read_back.morph:
        assert key.weight == pytest.approx(expected_weight[(key.name, key.frame)], abs=1e-6)
