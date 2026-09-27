import pytest

import lipsync
from vmd import MorphKey, VmdDocument, read, write

_STANDARD_MORPHS = {"あ", "い", "う", "え", "お", "ん"}
_MORPH_NAME_BYTES = 15


def test_mouth_shape_members():
    names = {m.name for m in lipsync.MouthShape}
    assert names == {"A", "I", "U", "E", "O", "N", "BILABIAL", "SILENCE", "LEGATO_GAP"}
    assert lipsync.MouthShape.N.value == "n"
    assert lipsync.MouthShape.LEGATO_GAP.value == "legato_gap"


def test_consonant_class_members():
    names = {c.name for c in lipsync.ConsonantClass}
    assert names == {"NONE", "NEUTRAL", "ROUNDED", "SPREAD"}


def test_aperture_class_members():
    names = {c.name for c in lipsync.ApertureClass}
    assert names == {"NONE", "FIRM_CLOSURE", "NARROW_CHANNEL", "SLIGHT_CLOSURE"}


def test_mouth_event_defaults():
    ev = lipsync.MouthEvent(shape=lipsync.MouthShape.A, start=0.0, end=10.0)
    assert ev.shape is lipsync.MouthShape.A
    assert ev.start == 0.0
    assert ev.end == 10.0
    assert ev.open_amount == 0.0
    assert ev.consonant_class is lipsync.ConsonantClass.NONE
    assert ev.aperture_class is lipsync.ApertureClass.NONE


def test_mouth_event_positional_field_order():
    ev = lipsync.MouthEvent(
        lipsync.MouthShape.A,
        0.0,
        10.0,
        0.5,
        lipsync.ConsonantClass.ROUNDED,
        lipsync.ApertureClass.FIRM_CLOSURE,
    )
    assert ev.open_amount == 0.5
    assert ev.consonant_class is lipsync.ConsonantClass.ROUNDED
    assert ev.aperture_class is lipsync.ApertureClass.FIRM_CLOSURE


def test_generation_params_defaults():
    p = lipsync.GenerationParams()
    assert p.open_cap == 0.8
    assert p.vowel_scale == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert p.attack_frames == 2
    assert p.release_frames == 2
    assert p.min_hold_frames == 3
    assert p.triangle_min_frames == 2.0
    assert p.coartic_overlap_max == 2
    assert p.anticipation_frames == 1
    assert p.legato_valley_shallow == 0.4
    assert p.legato_valley_deep == 0.2
    assert p.legato_valley_slope == 0.025
    assert p.exaggeration == 1.0
    assert p.vibrato_threshold == 18
    assert p.vibrato_amp == 0.05
    assert p.vibrato_period == 15
    assert p.mora_valley_frames == 12.0
    assert p.mora_valley_min_gap_frames == 4.0


def test_output_keys_are_standard_mouth_morphs_with_fixed_width_names_in_frame_order():
    events = [
        lipsync.MouthEvent(lipsync.MouthShape.A, 0.0, 10.0, 0.5),
        lipsync.MouthEvent(lipsync.MouthShape.I, 10.0, 20.0, 0.5),
        lipsync.MouthEvent(lipsync.MouthShape.N, 20.0, 30.0, 0.5),
    ]
    keys = lipsync.generate_morph_keys(events, lipsync.GenerationParams())
    assert isinstance(keys, list)
    assert keys
    for k in keys:
        assert isinstance(k, MorphKey)
        assert len(k.name_raw) == _MORPH_NAME_BYTES
        assert k.name in _STANDARD_MORPHS
        assert isinstance(k.frame, int)
    frames = [k.frame for k in keys]
    assert frames == sorted(frames)


def test_generated_keys_roundtrip_through_vmd():
    events = [
        lipsync.MouthEvent(lipsync.MouthShape.A, 0.0, 10.0, 0.5),
        lipsync.MouthEvent(lipsync.MouthShape.I, 10.0, 20.0, 0.5),
    ]
    keys = lipsync.generate_morph_keys(events, lipsync.GenerationParams())
    assert keys
    restored, _warnings = read(write(VmdDocument(morph=keys)))
    assert len(restored.morph) == len(keys)
    for src, dst in zip(keys, restored.morph, strict=True):
        assert dst.name == src.name
        assert dst.frame == src.frame
        assert dst.weight == pytest.approx(src.weight)
