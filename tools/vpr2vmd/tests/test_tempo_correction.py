import pytest

from lipsync import GenerationParams
from vpr2vmd.tempo_correction import apply_tempo_correction


def test_no_correction_at_reference_tempo():
    p = GenerationParams(min_hold_frames=3, attack_frames=2, release_frames=2)
    out = apply_tempo_correction(p, 120.0, ref_bpm=120.0, s_min=0.5)
    assert (out.min_hold_frames, out.attack_frames, out.release_frames) == (3, 2, 2)


def test_slower_than_reference_not_stretched():
    p = GenerationParams(min_hold_frames=4, attack_frames=3, release_frames=3)
    out = apply_tempo_correction(p, 60.0, ref_bpm=120.0, s_min=0.5)
    assert (out.min_hold_frames, out.attack_frames, out.release_frames) == (4, 3, 3)


def test_fast_tempo_shrinks_min_hold():
    p = GenerationParams(min_hold_frames=3, attack_frames=2, release_frames=2)
    out = apply_tempo_correction(p, 190.0, ref_bpm=120.0, s_min=0.5)
    assert out.min_hold_frames == 2


def test_attack_release_floor_two():
    p = GenerationParams(min_hold_frames=3, attack_frames=2, release_frames=2)
    out = apply_tempo_correction(p, 190.0, ref_bpm=120.0)
    assert out.attack_frames == 2
    assert out.release_frames == 2


def test_preset_below_floor_not_raised():
    p = GenerationParams(min_hold_frames=2, attack_frames=1, release_frames=1)
    at_ref = apply_tempo_correction(p, 120.0, ref_bpm=120.0)
    assert (at_ref.attack_frames, at_ref.release_frames) == (1, 1)
    at_fast = apply_tempo_correction(p, 190.0, ref_bpm=120.0)
    assert (at_fast.attack_frames, at_fast.release_frames) == (1, 1)


def test_min_hold_floor_one():
    p = GenerationParams(min_hold_frames=1, attack_frames=2, release_frames=2)
    out = apply_tempo_correction(p, 600.0, ref_bpm=120.0, s_min=0.2)
    assert out.min_hold_frames == 1


def test_s_min_clamps_extreme_tempo():
    p = GenerationParams(min_hold_frames=4, attack_frames=2, release_frames=2)
    out = apply_tempo_correction(p, 600.0, ref_bpm=120.0, s_min=0.5)
    assert out.min_hold_frames == 2


def test_other_params_untouched():
    p = GenerationParams(
        min_hold_frames=3, attack_frames=2, release_frames=2,
        anticipation_frames=11, coartic_overlap_max=6, vibrato_amp=0.13,
        legato_valley_shallow=0.45, legato_valley_deep=0.30,
    )
    out = apply_tempo_correction(p, 190.0)
    assert out.anticipation_frames == 11
    assert out.coartic_overlap_max == 6
    assert out.vibrato_amp == pytest.approx(0.13)
    assert out.legato_valley_shallow == pytest.approx(0.45)
    assert out.legato_valley_deep == pytest.approx(0.30)
