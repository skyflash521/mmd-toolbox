"""テンポ補正のテスト(vpr2vmd.md §3)。

代表BPMが基準テンポより速いほど保持・アタック・リリースを縮め、高速テンポでの短母音消失を防ぐ。
補正対象は min_hold/attack/release のみで、他のパラメータは変えない。
"""

import pytest
from lipsync import GenerationParams

from vpr2vmd.tempo_correction import apply_tempo_correction


def test_no_correction_at_reference_tempo():
    # 代表BPM=ref_bpm → s=1.0。縮めない。
    p = GenerationParams(min_hold_frames=3, attack_frames=2, release_frames=2)
    out = apply_tempo_correction(p, 120.0, ref_bpm=120.0, s_min=0.5)
    assert (out.min_hold_frames, out.attack_frames, out.release_frames) == (3, 2, 2)


def test_slower_than_reference_not_stretched():
    # 基準より遅い(比>1)は 1.0 にクランプ。縮めも伸ばしもしない。
    p = GenerationParams(min_hold_frames=4, attack_frames=3, release_frames=3)
    out = apply_tempo_correction(p, 60.0, ref_bpm=120.0, s_min=0.5)
    assert (out.min_hold_frames, out.attack_frames, out.release_frames) == (4, 3, 3)


def test_fast_tempo_shrinks_min_hold():
    # 190bpm, ref120 → s=clamp(120/190,0.5,1)=0.6316。min_hold=round(3×0.6316)=2。
    p = GenerationParams(min_hold_frames=3, attack_frames=2, release_frames=2)
    out = apply_tempo_correction(p, 190.0, ref_bpm=120.0, s_min=0.5)
    assert out.min_hold_frames == 2


def test_attack_release_floor_two():
    # attack/release は下限2。round(2×0.6316)=1 でも 2 にクランプ(onset 急変回避)。
    p = GenerationParams(min_hold_frames=3, attack_frames=2, release_frames=2)
    out = apply_tempo_correction(p, 190.0, ref_bpm=120.0)
    assert out.attack_frames == 2
    assert out.release_frames == 2


def test_preset_below_floor_not_raised():
    # プリセットが元から下限未満に取った attack/release(powerful・rap の 1)は、補正で増やさない。
    # 基準テンポでも高速テンポでも 1 のまま(下限2は補正による過小化を防ぐためで、選択を上書きしない)。
    p = GenerationParams(min_hold_frames=2, attack_frames=1, release_frames=1)
    at_ref = apply_tempo_correction(p, 120.0, ref_bpm=120.0)
    assert (at_ref.attack_frames, at_ref.release_frames) == (1, 1)
    at_fast = apply_tempo_correction(p, 190.0, ref_bpm=120.0)
    assert (at_fast.attack_frames, at_fast.release_frames) == (1, 1)


def test_min_hold_floor_one():
    # 極端な高速でも min_hold は 1 を下回らない。
    p = GenerationParams(min_hold_frames=1, attack_frames=2, release_frames=2)
    out = apply_tempo_correction(p, 600.0, ref_bpm=120.0, s_min=0.2)
    assert out.min_hold_frames == 1


def test_s_min_clamps_extreme_tempo():
    # 600bpm で比=0.2 だが s_min=0.5 で下げ止まる。min_hold=round(4×0.5)=2。
    p = GenerationParams(min_hold_frames=4, attack_frames=2, release_frames=2)
    out = apply_tempo_correction(p, 600.0, ref_bpm=120.0, s_min=0.5)
    assert out.min_hold_frames == 2


def test_other_params_untouched():
    # 補正は min_hold/attack/release のみ。他は不変。
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
