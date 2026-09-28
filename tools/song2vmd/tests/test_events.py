import numpy as np
import pytest

from lipsync import ApertureClass, ConsonantClass, MouthShape
from song2vmd import events
from vocal_analysis import AudioPcm, RmsEnvelope, Segment
from vocal_analysis import rms as _va_rms

FRAME_RATE = 30.0


def seg(type_, start, end, phoneme=None, confidence=None):
    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=phoneme, confidence=confidence)


def rms_env(times_sec, values, dynamic_range_db=20.0):
    return RmsEnvelope(
        times_sec=np.array(times_sec, dtype=float),
        values=np.array(values, dtype=float),
        dynamic_range_db=dynamic_range_db,
    )


_FRAME_CENTER_OFFSET_SEC = 0.0125


def flat_rms(duration_sec, value, dynamic_range_db=20.0, hop_sec=0.010):
    n = max(2, round(duration_sec / hop_sec) + 1)
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop_sec for i in range(n)]
    values = [value] * n
    return rms_env(times, values, dynamic_range_db)


_DEFAULT_KW = dict(
    open_lo=0.30, open_hi=0.75, open_max=0.90, intensity_curve=0.6,
    silence_on=0.06, use_n_morph=True,
)
_IDENTITY_MAPPING_KW = dict(_DEFAULT_KW, open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)


def confirm(segments, rms, **overrides):
    kw = dict(_DEFAULT_KW)
    kw.update(overrides)
    mouth_events, diagnostics, _group_sizes = events.confirm_mouth_events(segments, rms, **kw)
    return mouth_events, diagnostics


_ANCHOR_LO_RMS = 0.10
_ANCHOR_HI_RMS = 1.00


def _normalized_between_anchors(raw_rms):
    return (raw_rms - _ANCHOR_LO_RMS) / (_ANCHOR_HI_RMS - _ANCHOR_LO_RMS)


def confirm_between_anchor_morae(target_rms_value, *, kind="vowel", vowel_phoneme="a",
                                 confidence=0.9, mora_dur=0.3, **kw):
    """対象モーラを下アンカー・上アンカーの母音モーラで挟んで確定し、(対象モーラのイベント, 診断) を返す。"""
    lo_end = mora_dur
    target_end = 2 * mora_dur
    hi_end = 3 * mora_dur
    if kind == "vowel":
        target_seg = seg("vowel", lo_end, target_end, phoneme=vowel_phoneme, confidence=confidence)
    else:
        target_seg = seg("consonant", lo_end, target_end, phoneme="ɴ")
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        target_seg,
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [
        _ANCHOR_LO_RMS if t < lo_end else (target_rms_value if t < target_end else _ANCHOR_HI_RMS)
        for t in times
    ]
    rms = rms_env(times, values)
    mouth_events, diag = confirm(segments, rms, **kw)
    return mouth_events[1], diag


def _clamped_open_amount(normalized):
    return min(max(normalized ** 0.6, 0.30), 0.75)


def test_vowel_ipa_maps_to_mouth_shape():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.3, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A]


@pytest.mark.parametrize("phoneme,letter", [
    ("a", MouthShape.A), ("i", MouthShape.I), ("ɯ", MouthShape.U),
    ("e̞", MouthShape.E), ("o̞", MouthShape.O),
])
def test_all_five_vowels_map(phoneme, letter):
    segments = [seg("vowel", 0.0, 0.3, phoneme=phoneme, confidence=0.9)]
    rms = flat_rms(0.3, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == letter


def test_unmapped_vowel_symbol_at_start_becomes_silence_as_a_leading_gap():
    segments = [seg("vowel", 0.0, 0.3, phoneme="ʔ", confidence=0.9)]
    rms = flat_rms(0.3, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.SILENCE


@pytest.mark.parametrize("phoneme", ["m", "mʲ", "b", "bʲ", "p", "pʲ"])
def test_bilabial_consonants_become_independent_closure_event(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.BILABIAL
    assert mouth_events[1].shape == MouthShape.A


def test_fu_consonant_is_not_a_closure_and_is_absorbed_into_the_vowel():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="ɸ"),
        seg("vowel", 0.05, 0.35, phoneme="ɯ", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.U
    assert mouth_events[0].start == pytest.approx(0.0)


def test_moraic_nasal_becomes_n_shape():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("consonant", 0.2, 0.35, phoneme="ɴ"),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.N]


def test_no_n_morph_flag_forces_silence_instead_of_n():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("consonant", 0.2, 0.35, phoneme="ɴ"),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms, use_n_morph=False)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.SILENCE]


def test_moraic_nasal_is_silence_when_use_n_morph_omitted():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("consonant", 0.2, 0.35, phoneme="ɴ"),
    ]
    rms = flat_rms(0.35, 0.8)
    kw = {k: v for k, v in _DEFAULT_KW.items() if k != "use_n_morph"}
    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **kw)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.SILENCE]


def test_head_nasal_consonant_before_vowel_is_not_n_mora():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="n"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A]


def test_plain_consonant_creates_no_independent_event():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="k"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert len(mouth_events) == 1
    assert mouth_events[0].shape == MouthShape.A


def test_consonant_class_none_when_no_preceding_consonant():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.3, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].consonant_class == ConsonantClass.NONE


@pytest.mark.parametrize("phoneme", ["ɸ", "w"])
def test_consonant_class_rounded(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].consonant_class == ConsonantClass.ROUNDED


@pytest.mark.parametrize("phoneme", ["ɕ", "tɕ", "dʑ", "ɲ", "ç", "kʲ", "bʲ"])
def test_consonant_class_spread(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].consonant_class == ConsonantClass.SPREAD


def test_consonant_class_neutral_for_unlisted_consonant():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="k"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].consonant_class == ConsonantClass.NEUTRAL


def test_aperture_class_none_when_no_preceding_consonant():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.3, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].aperture_class == ApertureClass.NONE


@pytest.mark.parametrize("phoneme", ["t", "d", "n", "ts", "ɲ"])
def test_aperture_class_firm_closure(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.FIRM_CLOSURE


@pytest.mark.parametrize("phoneme", ["s", "z", "ɕ", "tɕ", "dʑ", "ç", "j"])
def test_aperture_class_narrow_channel(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.NARROW_CHANNEL


@pytest.mark.parametrize("phoneme", ["k", "ɡ", "ɾ", "kʲ", "ɡʲ"])
def test_aperture_class_slight_closure(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.SLIGHT_CLOSURE


@pytest.mark.parametrize("phoneme", ["ɸ", "w", "h", "v"])
def test_aperture_class_none_for_listed_none_phonemes(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="ɯ" if phoneme in ("ɸ", "w") else "a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.NONE


def test_aperture_class_none_for_unlisted_consonant():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="ʔ"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.NONE


def test_aperture_class_is_none_for_palatalized_consonant_absent_from_table():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="sʲ"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.NONE


def test_aperture_class_uses_strongest_among_consonant_run_when_strong_comes_last():
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="k"),
        seg("consonant", 0.03, 0.06, phoneme="t"),
        seg("vowel", 0.06, 0.36, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.36, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.FIRM_CLOSURE


def test_aperture_class_strongest_wins_when_strong_comes_first():
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="t"),
        seg("consonant", 0.03, 0.06, phoneme="k"),
        seg("vowel", 0.06, 0.36, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.36, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.FIRM_CLOSURE


def test_aperture_class_resets_after_bilabial_closure():
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="t"),
        seg("consonant", 0.03, 0.08, phoneme="m"),
        seg("vowel", 0.08, 0.38, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.38, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    assert a_event.aperture_class == ApertureClass.NONE


def test_aperture_class_resets_at_moraic_nasal_between_consonants():
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="t"),
        seg("consonant", 0.03, 0.13, phoneme="ɴ"),
        seg("consonant", 0.13, 0.16, phoneme="k"),
        seg("vowel", 0.16, 0.46, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.46, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert i_event.aperture_class == ApertureClass.SLIGHT_CLOSURE


def test_aperture_class_resets_after_gap():
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="t"),
        seg("gap", 0.03, 0.13),
        seg("vowel", 0.13, 0.43, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.43, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    assert a_event.aperture_class == ApertureClass.NONE


def test_nya_consonant_is_spread_and_firm_closure_on_independent_axes():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="ɲ"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].consonant_class == ConsonantClass.SPREAD
    assert mouth_events[-1].aperture_class == ApertureClass.FIRM_CLOSURE


def test_consonant_class_takes_last_consonant_while_aperture_takes_strongest():
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="k"),
        seg("consonant", 0.03, 0.06, phoneme="w"),
        seg("vowel", 0.06, 0.36, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.36, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].consonant_class == ConsonantClass.ROUNDED
    assert mouth_events[-1].aperture_class == ApertureClass.SLIGHT_CLOSURE


def test_aperture_class_resets_after_vowel_boundary():
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="t"),
        seg("vowel", 0.03, 0.33, phoneme="a", confidence=0.9),
        seg("consonant", 0.33, 0.36, phoneme="k"),
        seg("vowel", 0.36, 0.66, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.66, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert i_event.aperture_class == ApertureClass.SLIGHT_CLOSURE


def test_leading_and_trailing_gap_are_silence_even_when_loud():
    segments = [
        seg("gap", 0.0, 0.1),
        seg("vowel", 0.1, 0.4, phoneme="a", confidence=0.9),
        seg("gap", 0.4, 0.5),
    ]
    rms = flat_rms(0.5, 0.9)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.SILENCE
    assert mouth_events[-1].shape == MouthShape.SILENCE


def test_gap_silent_from_its_start_becomes_silence_without_extending_preceding_vowel():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.8)
    for i in range(len(rms.times_sec)):
        if 0.2 <= rms.times_sec[i] < 0.4:
            rms.values[i] = 0.02
    mouth_events, diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.A, MouthShape.SILENCE, MouthShape.I]
    assert mouth_events[0].end == pytest.approx(0.2 * FRAME_RATE)
    assert diag.merged_morae == 0


def test_voiced_gap_continues_preceding_vowel_and_merges_with_following_same_vowel():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.5)
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A]
    assert mouth_events[0].start == pytest.approx(0.0 * FRAME_RATE)
    assert mouth_events[0].end == pytest.approx(0.6 * FRAME_RATE)


def test_voiced_gap_after_moraic_nasal_extends_the_n_event():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("consonant", 0.2, 0.3, phoneme="ɴ"),
        seg("gap", 0.3, 0.5),
        seg("vowel", 0.5, 0.7, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.7, 0.5)
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.A, MouthShape.N, MouthShape.I]
    n_event = mouth_events[1]
    assert n_event.end == pytest.approx(0.5 * FRAME_RATE)


def test_voiced_gap_after_bilabial_becomes_silence():
    segments = [
        seg("consonant", 0.0, 0.1, phoneme="m"),
        seg("gap", 0.1, 0.3),
        seg("vowel", 0.3, 0.5, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.5, 0.5)
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.BILABIAL, MouthShape.SILENCE, MouthShape.A]


def test_voiced_gap_after_silence_stays_silence():
    segments = [
        seg("gap", 0.0, 0.1),
        seg("gap", 0.1, 0.3),
        seg("vowel", 0.3, 0.5, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.5, 0.5)
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.SILENCE, MouthShape.A]


def test_gap_with_voiced_head_and_silent_tail_splits_at_voice_end():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="e̞", confidence=0.9),
        seg("gap", 0.2, 1.4),
        seg("vowel", 1.4, 1.6, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(1.6, 0.8)
    for i in range(len(rms.times_sec)):
        if 0.6 <= rms.times_sec[i] < 1.4:
            rms.values[i] = 0.02
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.E, MouthShape.SILENCE, MouthShape.A]
    assert mouth_events[0].end == pytest.approx(0.6 * FRAME_RATE, abs=0.5)
    assert mouth_events[1].end == pytest.approx(1.4 * FRAME_RATE, abs=0.5)


def test_gap_with_dip_shorter_than_run_length_does_not_close_when_voice_resumes():
    segments = [
        seg("vowel", 0.0, 0.15, phoneme="a", confidence=0.9),
        seg("gap", 0.15, 0.75),
        seg("vowel", 0.75, 0.9, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.9, 0.8)
    for i in range(len(rms.times_sec)):
        if 0.375 <= rms.times_sec[i] < 0.45:
            rms.values[i] = 0.02
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A]


def test_gap_stays_closed_after_scan_close_even_if_voice_returns():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 1.4),
        seg("vowel", 1.4, 1.6, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(1.6, 0.8)
    for i in range(len(rms.times_sec)):
        if 0.4 <= rms.times_sec[i] < 0.8:
            rms.values[i] = 0.02
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.A, MouthShape.SILENCE, MouthShape.I]
    assert mouth_events[1].start == pytest.approx(0.4 * FRAME_RATE, abs=0.5)
    assert mouth_events[1].end == pytest.approx(1.4 * FRAME_RATE, abs=0.5)


def test_gap_short_quiet_tail_reaching_gap_end_closes_without_minimum_run_length():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 1.0),
        seg("vowel", 1.0, 1.2, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(1.2, 0.8)
    for i in range(len(rms.times_sec)):
        if 0.9 <= rms.times_sec[i] < 1.0:
            rms.values[i] = 0.02
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.A, MouthShape.SILENCE, MouthShape.I]
    assert mouth_events[1].start == pytest.approx(0.9 * FRAME_RATE, abs=0.5)


def test_low_dynamics_suppresses_gap_silence_except_leading_trailing():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="a", confidence=0.9),
        seg("gap", 0.6, 0.7),
    ]
    rms = flat_rms(0.7, 0.5, dynamic_range_db=5.0)
    idx_lo = int(0.2 / 0.010)
    idx_hi = int(0.4 / 0.010)
    for i in range(idx_lo, idx_hi + 1):
        rms.values[i] = 0.01
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.A, MouthShape.SILENCE]


@pytest.mark.parametrize("dynamic_range_db, expected_shape, expected_low_dynamics", [
    pytest.param(12.0, MouthShape.SILENCE, False, id="12dBちょうどは抑制しない"),
    pytest.param(11.9, MouthShape.A, True, id="12dB未満は抑制する"),
])
def test_low_dynamics_suppression_starts_below_twelve_decibels(
        dynamic_range_db, expected_shape, expected_low_dynamics):
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.95)]
    rms = flat_rms(0.3, 0.01, dynamic_range_db=dynamic_range_db)
    mouth_events, diag = confirm(segments, rms)
    assert mouth_events[0].shape == expected_shape
    assert diag.low_dynamics is expected_low_dynamics


def test_gap_rms_exactly_at_silence_threshold_closes_instead_of_continuing_vowel():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.6),
        seg("vowel", 0.6, 0.8, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.8, 0.8)
    for i in range(len(rms.times_sec)):
        if 0.2 <= rms.times_sec[i] < 0.6:
            rms.values[i] = _DEFAULT_KW["silence_on"]
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.SILENCE, MouthShape.A]


def test_quiet_vowel_segment_is_silenced_even_with_high_confidence():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.95)]
    rms = flat_rms(0.3, 0.01)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.SILENCE


def test_low_dynamics_suppresses_vowel_silence_override():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.95)]
    rms = flat_rms(0.3, 0.01, dynamic_range_db=5.0)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.A


def _loud_consonant_with_narrow_quiet_vowel_nucleus():
    segments = [
        seg("consonant", 0.0, 0.9, phoneme="n"),
        seg("vowel", 0.9, 0.92, phoneme="i", confidence=0.9),
    ]
    hop = 0.010
    n = 93
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [0.9 if t <= 0.85 else 0.02 for t in times]
    return segments, rms_env(times, values)


def test_vowel_with_quiet_core_but_loud_mora_span_is_not_silenced():
    segments, rms = _loud_consonant_with_narrow_quiet_vowel_nucleus()
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.I


def _loud_consonant_with_narrow_quiet_vowel_nucleus_between_anchor_morae():
    offset = 0.3
    hi_end = offset + 0.92 + offset
    segments = [
        seg("vowel", 0.0, offset, phoneme="ɯ", confidence=0.9),
        seg("consonant", offset, offset + 0.9, phoneme="n"),
        seg("vowel", offset + 0.9, offset + 0.92, phoneme="i", confidence=0.9),
        seg("vowel", offset + 0.92, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = []
    for t in times:
        if t < offset:
            values.append(_ANCHOR_LO_RMS)
        elif t <= offset + 0.85:
            values.append(0.9)
        elif t < offset + 0.92:
            values.append(0.02)
        else:
            values.append(_ANCHOR_HI_RMS)
    return segments, rms_env(times, values)


def test_vowel_with_quiet_core_but_loud_mora_span_opens_from_the_louder_window():
    segments, rms = _loud_consonant_with_narrow_quiet_vowel_nucleus_between_anchor_morae()
    mouth_events, _diag = confirm(
        segments, rms, intensity_curve=1.0, open_lo=0.0, open_hi=1.0, open_max=1.0)
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert i_event.open_amount == pytest.approx(_normalized_between_anchors(0.9), abs=1e-6)


def test_vowel_with_absorbed_consonant_still_silenced_when_whole_mora_is_quiet():
    segments = [
        seg("consonant", 0.0, 0.2, phoneme="n"),
        seg("vowel", 0.2, 0.3, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.3, 0.01)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.SILENCE


def test_adjacent_same_vowel_segments_merge_into_one_event():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("vowel", 0.2, 0.4, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.4, 0.8)
    mouth_events, diag = confirm(segments, rms)
    assert len(mouth_events) == 1
    assert mouth_events[0].shape == MouthShape.A
    assert mouth_events[0].start == pytest.approx(0.0)
    assert mouth_events[0].end == pytest.approx(0.4 * FRAME_RATE)
    assert diag.merged_morae == 1


def test_three_adjacent_same_vowel_segments_merge_with_merged_count_two():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("vowel", 0.2, 0.4, phoneme="a", confidence=0.9),
        seg("vowel", 0.4, 0.6, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.8)
    mouth_events, diag = confirm(segments, rms)
    assert len(mouth_events) == 1
    assert diag.merged_morae == 2


def test_gap_continuation_fragment_does_not_count_as_merged_mora():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.5)
    mouth_events, diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.I]
    assert diag.merged_morae == 0


def test_gap_continuation_between_same_vowels_counts_only_the_real_mora():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.5)
    mouth_events, diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A]
    assert diag.merged_morae == 1


def test_adjacent_moraic_nasal_segments_merge_with_merged_count_one():
    segments = [
        seg("consonant", 0.0, 0.2, phoneme="ɴ"),
        seg("consonant", 0.2, 0.4, phoneme="ɴ"),
    ]
    rms = flat_rms(0.4, 0.5)
    mouth_events, diag = confirm(segments, rms, use_n_morph=True)
    assert [e.shape for e in mouth_events] == [MouthShape.N]
    assert diag.merged_morae == 1


def test_gap_continuation_fragment_of_moraic_nasal_does_not_count():
    segments = [
        seg("consonant", 0.0, 0.2, phoneme="ɴ"),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.5)
    mouth_events, diag = confirm(segments, rms, use_n_morph=True)
    assert [e.shape for e in mouth_events] == [MouthShape.N, MouthShape.I]
    assert diag.merged_morae == 0


def test_adjacent_different_vowels_do_not_merge():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("vowel", 0.2, 0.4, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.4, 0.8)
    mouth_events, diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.I]
    assert diag.merged_morae == 0


@pytest.mark.parametrize("phoneme", ["t", "k", "s"])
def test_repeated_consonant_between_same_vowels_does_not_merge(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.25, phoneme="a", confidence=0.9),
        seg("consonant", 0.25, 0.30, phoneme=phoneme),
        seg("vowel", 0.30, 0.50, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.50, 0.8)
    mouth_events, diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.A]
    assert diag.merged_morae == 0
    assert mouth_events[0].end == pytest.approx(0.25 * FRAME_RATE)
    assert mouth_events[1].start == pytest.approx(0.25 * FRAME_RATE)


def test_different_intervening_consonants_between_same_vowels_do_not_merge():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="t"),
        seg("vowel", 0.05, 0.25, phoneme="a", confidence=0.9),
        seg("consonant", 0.25, 0.30, phoneme="k"),
        seg("vowel", 0.30, 0.50, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.50, 0.8)
    mouth_events, diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.A]
    assert diag.merged_morae == 0


def test_adjacent_silence_segments_merging_does_not_count_as_merged_morae():
    segments = [
        seg("gap", 0.0, 0.2),
        seg("gap", 0.2, 0.4),
    ]
    rms = flat_rms(0.4, 0.01)
    mouth_events, diag = confirm(segments, rms)
    assert len(mouth_events) == 1
    assert mouth_events[0].shape == MouthShape.SILENCE
    assert diag.merged_morae == 0


def _rms_rising_at(rise_sec, n=60, low=0.05, high=0.8, hop=0.010):
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    rise_index = round((rise_sec - _FRAME_CENTER_OFFSET_SEC) / hop)
    values = [low if i < rise_index else high for i in range(n)]
    return rms_env(times, values), times[rise_index]


def test_vowel_onset_shifts_to_nearby_rms_rise():
    rms, rise_time = _rms_rising_at(0.28)
    segments = [
        seg("gap", 0.0, 0.30),
        seg("vowel", 0.30, 0.59, phoneme="a", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, rms)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    assert a_event.start < 0.30 * FRAME_RATE
    assert a_event.start == pytest.approx(rise_time * FRAME_RATE, abs=0.03 * FRAME_RATE)


def test_vowel_onset_ignores_rise_outside_the_60ms_window():
    rms, _rise_time = _rms_rising_at(0.20)
    segments = [
        seg("gap", 0.0, 0.30),
        seg("vowel", 0.30, 0.59, phoneme="a", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, rms)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    assert a_event.start == pytest.approx(0.30 * FRAME_RATE)


def test_vowel_onset_keeps_token_boundary_when_no_rise_nearby():
    rms = flat_rms(0.6, 0.5)
    segments = [
        seg("gap", 0.0, 0.30),
        seg("vowel", 0.30, 0.59, phoneme="a", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, rms)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    assert a_event.start == pytest.approx(0.30 * FRAME_RATE)


def step_rms(duration_sec, rise_sec, low=0.2, high=0.9, hop_sec=0.010):
    n = max(2, round(duration_sec / hop_sec) + 1)
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop_sec for i in range(n)]
    values = [high if t >= rise_sec else low for t in times]
    return rms_env(times, values)


_ONE_FRAME_SEC = 1.0 / FRAME_RATE


def test_onset_keeps_one_frame_for_the_preceding_unit():
    segments = [
        seg("vowel", 0.0, 0.25, phoneme="a", confidence=0.9),
        seg("consonant", 0.25, 0.28, phoneme="m"),
        seg("vowel", 0.28, 0.60, phoneme="i", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, step_rms(0.60, 0.24))
    bilabial = next(e for e in mouth_events if e.shape == MouthShape.BILABIAL)
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert bilabial.end == pytest.approx((0.25 + _ONE_FRAME_SEC) * FRAME_RATE)
    assert i_event.start == bilabial.end
    assert bilabial.end - bilabial.start == pytest.approx(_ONE_FRAME_SEC * FRAME_RATE)


def test_onset_keeps_one_frame_for_the_corrected_vowel():
    segments = [
        seg("gap", 0.0, 0.30),
        seg("vowel", 0.30, 0.32, phoneme="a", confidence=0.9),
        seg("consonant", 0.32, 0.45, phoneme="m"),
        seg("vowel", 0.45, 0.60, phoneme="i", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, step_rms(0.60, 0.34))
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    assert a_event.start == pytest.approx((0.32 - _ONE_FRAME_SEC) * FRAME_RATE)
    assert a_event.end - a_event.start == pytest.approx(_ONE_FRAME_SEC * FRAME_RATE)


def test_onset_lower_bound_follows_the_corrected_start_of_the_preceding_vowel():
    segments = [
        seg("vowel", 0.20, 0.30, phoneme="a", confidence=0.9),
        seg("vowel", 0.30, 0.60, phoneme="i", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, step_rms(0.60, 0.24))
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert i_event.start == a_event.end
    assert a_event.end - a_event.start == pytest.approx(_ONE_FRAME_SEC * FRAME_RATE)


def test_onset_is_not_applied_when_the_allowed_range_is_empty():
    segments = [
        seg("vowel", 0.0, 0.25, phoneme="a", confidence=0.9),
        seg("consonant", 0.25, 0.28, phoneme="m"),
        seg("vowel", 0.28, 0.29, phoneme="i", confidence=0.9),
        seg("consonant", 0.29, 0.33, phoneme="m"),
        seg("vowel", 0.33, 0.60, phoneme="e̞", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, step_rms(0.60, 0.24))
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert i_event.start == pytest.approx(0.28 * FRAME_RATE)


def test_low_confidence_and_low_rms_vowel_is_weakened():
    target_event, _diag = confirm_between_anchor_morae(0.25, confidence=0.2)
    expected_base = _clamped_open_amount(_normalized_between_anchors(0.25))
    assert target_event.open_amount == pytest.approx(expected_base * 0.5)


def test_vowel_not_weakened_when_confidence_is_high_even_with_low_rms():
    target_event, _diag = confirm_between_anchor_morae(0.25, confidence=0.9)
    expected = _clamped_open_amount(_normalized_between_anchors(0.25))
    assert target_event.open_amount == pytest.approx(expected)


def test_vowel_not_weakened_when_rms_is_high_even_with_low_confidence():
    target_event, _diag = confirm_between_anchor_morae(0.5, confidence=0.2)
    expected = _clamped_open_amount(_normalized_between_anchors(0.5))
    assert target_event.open_amount == pytest.approx(expected)


def test_without_confidence_the_weak_rms_threshold_is_lower():
    weak_event, _ = confirm_between_anchor_morae(0.15, confidence=None)
    expected_weak_base = _clamped_open_amount(_normalized_between_anchors(0.15))
    assert weak_event.open_amount == pytest.approx(expected_weak_base * 0.5)

    strong_event, _ = confirm_between_anchor_morae(0.25, confidence=None)
    expected_strong = _clamped_open_amount(_normalized_between_anchors(0.25))
    assert strong_event.open_amount == pytest.approx(expected_strong)


def test_open_amount_uses_intensity_curve_and_clamps_to_style_range():
    target_event, _diag = confirm_between_anchor_morae(0.5, open_lo=0.30, open_hi=0.75, intensity_curve=0.6)
    expected = _clamped_open_amount(_normalized_between_anchors(0.5))
    assert target_event.open_amount == pytest.approx(expected)


def test_open_amount_saturates_at_open_max_below_open_hi():
    target_event, _diag = confirm_between_anchor_morae(
        0.99, open_lo=0.30, open_hi=0.95, open_max=0.5, intensity_curve=1.0)
    assert target_event.open_amount == pytest.approx(0.5)


def test_bilabial_and_silence_have_zero_open_amount():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="m"),
        seg("vowel", 0.05, 0.2, phoneme="a", confidence=0.01),
    ]
    rms = flat_rms(0.2, 0.01)
    mouth_events, _diag = confirm(segments, rms)
    assert all(e.open_amount == 0.0 for e in mouth_events)


def test_open_amount_uses_middle_60_percent_of_vowel_segment():
    offset = 0.3
    target_dur = 0.9
    hi_end = offset + target_dur + offset
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = []
    target_values = []
    for t in times:
        if t < offset:
            values.append(_ANCHOR_LO_RMS)
        elif t < offset + target_dur:
            v = 0.8 if 0.2 * target_dur <= (t - offset) <= 0.8 * target_dur else 0.05
            values.append(v)
            target_values.append(v)
        else:
            values.append(_ANCHOR_HI_RMS)
    rms = rms_env(times, values)
    segments = [
        seg("vowel", 0.0, offset, phoneme="ɯ", confidence=0.9),
        seg("vowel", offset, offset + target_dur, phoneme="a", confidence=0.9),
        seg("vowel", offset + target_dur, hi_end, phoneme="o̞", confidence=0.9),
    ]
    mouth_events, _diag = confirm(
        segments, rms, intensity_curve=1.0, open_lo=0.0, open_hi=1.0, open_max=1.0)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    naive_full_average = sum(target_values) / len(target_values)
    assert a_event.open_amount > _normalized_between_anchors(naive_full_average) + 0.1
    assert a_event.open_amount == pytest.approx(_normalized_between_anchors(0.8), abs=1e-6)


def test_n_mora_open_amount_is_derived_from_rms():
    target_event, _diag = confirm_between_anchor_morae(
        0.7, kind="n", open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)
    assert target_event.shape == MouthShape.N
    assert target_event.open_amount == pytest.approx(_normalized_between_anchors(0.7), abs=1e-6)


def _renormalized(values):
    p_lo, p_hi = events._percentile_bounds(values)
    return [events._normalize_with_bounds(v, p_lo, p_hi) for v in values]


def test_renormalize_open_rms_stretches_skewed_distribution():
    values = [0.70, 0.75, 0.80, 0.81, 0.85, 0.89, 0.90, 0.95, 0.98, 1.00]
    result = _renormalized(values)
    arr = np.array(values)
    p10 = np.percentile(arr, 10, method="linear")
    p90 = np.percentile(arr, 90, method="linear")
    expected = np.clip((arr - p10) / (p90 - p10), 0.0, 1.0)
    assert result == pytest.approx(list(expected))
    assert min(result) < 0.15
    assert max(result) > 0.85


def test_renormalize_open_rms_single_value_is_degenerate_midpoint():
    assert _renormalized([0.42]) == [0.5]


def test_renormalize_open_rms_all_identical_values_are_degenerate_midpoint():
    assert _renormalized([0.6, 0.6, 0.6, 0.6]) == [0.5, 0.5, 0.5, 0.5]


def test_renormalize_open_rms_range_within_isclose_tolerance_is_not_degenerate():
    values = [0.500000, 0.500000, 0.5000005, 0.500001]
    result = _renormalized(values)
    arr = np.array(values)
    p10 = np.percentile(arr, 10, method="linear")
    p90 = np.percentile(arr, 90, method="linear")
    assert p90 != p10
    assert np.isclose(p90, p10)
    expected = np.clip((arr - p10) / (p90 - p10), 0.0, 1.0)
    assert result == pytest.approx(list(expected))
    assert result != pytest.approx([0.5] * len(values))


def test_renormalize_open_rms_empty_list_returns_empty():
    assert _renormalized([]) == []


def test_renormalize_open_rms_is_invariant_to_uniform_gain():
    values = [0.10, 0.35, 0.62, 0.77, 0.91]
    scaled = [v * 0.4 for v in values]
    assert _renormalized(values) == pytest.approx(
        _renormalized(scaled))


def test_confirm_mouth_events_output_is_invariant_to_uniform_audio_gain():
    sample_rate = 8000
    seg_dur_sec = 0.4
    n_per_seg = int(seg_dur_sec * sample_rate)
    silenced_floor, anchor_lo, anchor_hi = 0.05, 0.20, 0.95
    target_levels = [0.40, 0.50, 0.60]
    levels = [silenced_floor, anchor_lo, *target_levels, anchor_hi]
    n = n_per_seg * len(levels)
    t = np.arange(n) / sample_rate
    envelope = np.concatenate([np.full(n_per_seg, lv) for lv in levels])
    tone = (np.sin(2 * np.pi * 220 * t) * envelope).astype(np.float32).reshape(-1, 1)

    bounds = [i * seg_dur_sec for i in range(len(levels) + 1)]
    phonemes = ["o̞", "ɯ", "a", "i", "u", "e̞"]
    segments = [
        seg("vowel", bounds[i], bounds[i + 1], phoneme=phoneme, confidence=0.9)
        for i, phoneme in enumerate(phonemes)
    ]

    def confirm_at_gain(gain):
        pcm = AudioPcm(samples=tone * gain, sample_rate=sample_rate)
        rms_envelope = _va_rms.compute_rms(pcm)
        return confirm(segments, rms_envelope)

    events_full_gain, diag_full_gain = confirm_at_gain(1.0)
    events_low_gain, diag_low_gain = confirm_at_gain(0.3)

    assert len(events_full_gain) == len(events_low_gain)
    for e_full, e_low in zip(events_full_gain, events_low_gain, strict=True):
        assert e_full.shape == e_low.shape
        assert e_full.consonant_class == e_low.consonant_class
        assert e_full.aperture_class == e_low.aperture_class
        assert e_full.start == pytest.approx(e_low.start, abs=1e-3)
        assert e_full.end == pytest.approx(e_low.end, abs=1e-3)
        assert e_full.open_amount == pytest.approx(e_low.open_amount, abs=1e-5)
    assert diag_full_gain == diag_low_gain
    assert events_full_gain[0].shape == MouthShape.SILENCE
    target_amounts = [e.open_amount for e in events_full_gain[2:5]]
    assert all(0.30 < a < 0.75 for a in target_amounts)
    assert len(set(target_amounts)) == len(target_amounts)


def test_song_without_morae_yields_single_closed_event():
    segments = [seg("gap", 0.0, 0.5)]
    rms = flat_rms(0.5, 0.01)
    mouth_events, _diag = confirm(segments, rms)
    assert len(mouth_events) == 1
    assert mouth_events[0].shape == MouthShape.SILENCE
    assert mouth_events[0].open_amount == 0.0


def test_degenerate_uniform_morae_all_get_midpoint_open_amount():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("vowel", 0.2, 0.4, phoneme="i", confidence=0.9),
        seg("vowel", 0.4, 0.6, phoneme="ɯ", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.73)
    mouth_events, _diag = confirm(
        segments, rms, open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)
    assert [e.open_amount for e in mouth_events] == pytest.approx([0.5, 0.5, 0.5])


def test_open_amount_uses_whole_song_percentiles_not_a_chunk_subset():
    mora_dur = 0.9
    segments = [
        seg("vowel", 0.0 * mora_dur, 1.0 * mora_dur, phoneme="ɯ", confidence=0.9),
        seg("vowel", 1.0 * mora_dur, 2.0 * mora_dur, phoneme="e̞", confidence=0.9),
        seg("vowel", 2.0 * mora_dur, 3.0 * mora_dur, phoneme="a", confidence=0.9),
        seg("vowel", 3.0 * mora_dur, 4.0 * mora_dur, phoneme="i", confidence=0.9),
        seg("vowel", 4.0 * mora_dur, 5.0 * mora_dur, phoneme="o̞", confidence=0.9),
        seg("vowel", 5.0 * mora_dur, 6.0 * mora_dur, phoneme="ɯ", confidence=0.9),
    ]
    raw_values = [0.10, 0.30, 0.50, 0.60, 0.80, 0.99]
    hop = 0.010
    total = 6.0 * mora_dur
    n = round(total / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [raw_values[min(int(t // mora_dur), 5)] for t in times]
    rms = rms_env(times, values)

    mouth_events, _diag = confirm(
        segments, rms, open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)

    whole_song_arr = np.array(raw_values)
    p10 = np.percentile(whole_song_arr, 10, method="linear")
    p90 = np.percentile(whole_song_arr, 90, method="linear")
    expected_boundary_next = np.clip((0.60 - p10) / (p90 - p10), 0.0, 1.0)

    chunk2_arr = np.array([0.60, 0.80, 0.99])
    p10_chunk = np.percentile(chunk2_arr, 10, method="linear")
    p90_chunk = np.percentile(chunk2_arr, 90, method="linear")
    per_chunk_boundary_next = np.clip((0.60 - p10_chunk) / (p90_chunk - p10_chunk), 0.0, 1.0)

    boundary_next_event = mouth_events[3]
    assert boundary_next_event.open_amount == pytest.approx(float(expected_boundary_next), abs=1e-6)
    assert boundary_next_event.open_amount != pytest.approx(float(per_chunk_boundary_next), abs=1e-3)


def test_silence_classification_unaffected_by_other_loud_morae_context():
    def last_shape(with_other_mora):
        base = 0.3 if with_other_mora else 0.0
        segments = []
        if with_other_mora:
            segments.append(seg("vowel", 0.0, base, phoneme="i", confidence=0.9))
        segments.append(seg("vowel", base, base + 0.3, phoneme="a", confidence=0.95))
        hop = 0.010
        n = round((base + 0.3) / hop) + 1
        times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
        values = [0.9 if t < base else 0.01 for t in times]
        rms = rms_env(times, values)
        mouth_events, _diag = confirm(segments, rms)
        return mouth_events[-1].shape

    assert last_shape(with_other_mora=False) == MouthShape.SILENCE
    assert last_shape(with_other_mora=True) == MouthShape.SILENCE


def test_weak_vowel_classification_unaffected_by_other_loud_morae_context():
    def a_event(confidence, with_other_mora):
        base = 0.3 if with_other_mora else 0.0
        segments = []
        if with_other_mora:
            segments.append(seg("vowel", 0.0, base, phoneme="i", confidence=0.9))
        segments.append(seg("vowel", base, base + 0.3, phoneme="a", confidence=confidence))
        hop = 0.010
        n = round((base + 0.3) / hop) + 1
        times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
        values = [0.9 if t < base else 0.25 for t in times]
        rms = rms_env(times, values)
        mouth_events, _diag = confirm(segments, rms)
        return next(e for e in mouth_events if e.shape == MouthShape.A)

    for with_other_mora in (False, True):
        weak_event = a_event(confidence=0.2, with_other_mora=with_other_mora)
        strong_event = a_event(confidence=0.9, with_other_mora=with_other_mora)
        assert weak_event.open_amount == pytest.approx(strong_event.open_amount * 0.5)


def test_low_dynamics_suppression_unaffected_by_other_morae_context():
    def shapes(with_other_mora):
        base = 0.2 if with_other_mora else 0.0
        segments = []
        if with_other_mora:
            segments.append(seg("vowel", 0.0, base, phoneme="i", confidence=0.9))
        segments.append(seg("vowel", base, base + 0.2, phoneme="a", confidence=0.9))
        segments.append(seg("gap", base + 0.2, base + 0.4))
        segments.append(seg("vowel", base + 0.4, base + 0.6, phoneme="a", confidence=0.9))
        total = base + 0.6
        rms = flat_rms(total, 0.5, dynamic_range_db=5.0)
        idx_lo = int((base + 0.2) / 0.010)
        idx_hi = int((base + 0.4) / 0.010)
        for i in range(idx_lo, idx_hi + 1):
            rms.values[i] = 0.01
        mouth_events, _diag = confirm(segments, rms)
        return [e.shape for e in mouth_events]

    without_other_mora = shapes(with_other_mora=False)
    with_other_mora = shapes(with_other_mora=True)
    assert without_other_mora == [MouthShape.A]
    assert with_other_mora[-len(without_other_mora):] == without_other_mora


def test_events_are_time_ordered_contiguous_and_cover_full_range():
    segments = [
        seg("gap", 0.0, 0.05),
        seg("consonant", 0.05, 0.10, phoneme="k"),
        seg("vowel", 0.10, 0.30, phoneme="a", confidence=0.9),
        seg("consonant", 0.30, 0.35, phoneme="m"),
        seg("vowel", 0.35, 0.55, phoneme="i", confidence=0.9),
        seg("gap", 0.55, 0.65),
    ]
    rms = flat_rms(0.65, 0.6)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].start == pytest.approx(0.0)
    assert mouth_events[-1].end == pytest.approx(0.65 * FRAME_RATE)
    for event in mouth_events:
        assert event.start <= event.end
    for prev, nxt in zip(mouth_events, mouth_events[1:], strict=False):
        assert prev.end == pytest.approx(nxt.start)
        assert prev.start <= nxt.start


def test_empty_segment_list_returns_empty_events():
    rms = flat_rms(0.1, 0.5)
    mouth_events, diag = confirm([], rms)
    assert mouth_events == []
    assert diag.merged_morae == 0


def test_confirmation_is_deterministic():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="k"),
        seg("vowel", 0.05, 0.30, phoneme="a", confidence=0.9),
        seg("gap", 0.30, 0.40),
        seg("vowel", 0.40, 0.60, phoneme="i", confidence=0.4),
    ]
    rms = flat_rms(0.60, 0.5)
    first, _ = confirm(segments, rms)
    second, _ = confirm(segments, rms)
    assert first == second


@pytest.mark.parametrize("duration_sec,expected_count", [
    pytest.param(0.3, 2, id="1は最低の2へ引き上げ"),
    pytest.param(0.6, 2, id="2ちょうど"),
    pytest.param(0.9, 3, id="3ちょうど"),
    pytest.param(1.0, 3, id="3.33は3へ丸め"),
    pytest.param(1.05, 4, id="3.5は4へ切り上げ"),
    pytest.param(1.5, 5, id="5ちょうど"),
])
def test_subwindow_count_rounds_half_up_and_enforces_minimum_two(duration_sec, expected_count):
    assert events._subwindow_count(duration_sec) == expected_count


def test_split_into_subwindows_are_contiguous_equal_width_and_cover_range():
    start, end = 10.0, 11.5
    bounds = events._split_into_subwindows(start, end)
    assert len(bounds) == 5
    assert bounds[0][0] == pytest.approx(start)
    assert bounds[-1][1] == pytest.approx(end)
    width = (end - start) / 5
    for sub_start, sub_end in bounds:
        assert sub_end - sub_start == pytest.approx(width)
    for (_, e1), (s2, _) in zip(bounds, bounds[1:], strict=False):
        assert e1 == pytest.approx(s2)


def test_mora_below_one_second_remains_single_event():
    segments = [seg("vowel", 0.0, 0.99, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.99, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert len(mouth_events) == 1


def test_long_mora_splits_into_contiguous_events_tracking_rms_changes():
    anchor_dur = 0.2
    mora_dur = 1.5
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("vowel", lo_end, target_end, phoneme="e̞", confidence=0.9),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    step_values = [0.20, 0.35, 0.50, 0.65, 0.80]
    step_dur = mora_dur / 5
    values = []
    for t in times:
        if t < lo_end:
            values.append(_ANCHOR_LO_RMS)
        elif t < target_end:
            step_idx = min(4, int((t - lo_end) / step_dur))
            values.append(step_values[step_idx])
        else:
            values.append(_ANCHOR_HI_RMS)
    rms = rms_env(times, values)
    mouth_events, _diag, group_sizes = events.confirm_mouth_events(segments, rms, **_IDENTITY_MAPPING_KW)
    e_events = [e for e in mouth_events if e.shape == MouthShape.E]
    assert len(e_events) == 5
    assert group_sizes == [1, 5, 1]
    assert e_events[0].start == pytest.approx(lo_end * FRAME_RATE)
    assert e_events[-1].end == pytest.approx(target_end * FRAME_RATE)
    for prev, nxt in zip(e_events, e_events[1:], strict=False):
        assert prev.end == pytest.approx(nxt.start)
    amounts = [e.open_amount for e in e_events]
    assert amounts == sorted(amounts)
    assert len(set(amounts)) == 5


def test_mora_of_exactly_one_second_is_split():
    anchor_dur = 0.2
    mora_dur = 1.0
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("vowel", lo_end, target_end, phoneme="a", confidence=0.9),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    rms = flat_rms(hi_end, 0.6)
    mouth_events, _diag, group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 3
    assert group_sizes == [1, 3, 1]


def test_split_subwindow_rms_uses_middle_60_percent_not_whole_subwindow_average():
    anchor_dur = 0.2
    mora_dur = 1.2
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("vowel", lo_end, target_end, phoneme="a", confidence=0.9),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    sub_dur = mora_dur / 4
    center_value, edge_value = 0.9, 0.0
    inner_subwindow_index = 2
    target_start = inner_subwindow_index * sub_dur
    target_end_sub = (inner_subwindow_index + 1) * sub_dur
    target_center_lo = target_start + 0.2 * sub_dur
    target_center_hi = target_start + 0.8 * sub_dur

    values = []
    for t in times:
        if t < lo_end:
            values.append(_ANCHOR_LO_RMS)
        elif t < target_end:
            offset = t - lo_end
            if target_start <= offset < target_end_sub:
                is_center = target_center_lo <= offset < target_center_hi
                values.append(center_value if is_center else edge_value)
            else:
                values.append(center_value)
        else:
            values.append(_ANCHOR_HI_RMS)
    rms = rms_env(times, values)
    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_IDENTITY_MAPPING_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 4
    for e in a_events:
        assert e.open_amount == pytest.approx(a_events[0].open_amount, abs=1e-6)


def _long_mora_between_anchor_morae(segments_between, *, lo_end, target_end, hi_end, target_value=0.6):
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        *segments_between,
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [
        _ANCHOR_LO_RMS if t < lo_end else (target_value if t < target_end else _ANCHOR_HI_RMS)
        for t in times
    ]
    return segments, rms_env(times, values)


def test_split_head_only_keeps_consonant_and_aperture_class_others_are_none():
    lo_end = 0.2
    cons_end = lo_end + 0.05
    target_end = lo_end + 1.2
    segments, rms = _long_mora_between_anchor_morae(
        [seg("consonant", lo_end, cons_end, phoneme="t"),
         seg("vowel", cons_end, target_end, phoneme="a", confidence=0.9)],
        lo_end=lo_end, target_end=target_end, hi_end=target_end + 0.2)
    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 4
    assert a_events[0].consonant_class == ConsonantClass.NEUTRAL
    assert a_events[0].aperture_class == ApertureClass.FIRM_CLOSURE
    for e in a_events[1:]:
        assert e.consonant_class == ConsonantClass.NONE
        assert e.aperture_class == ApertureClass.NONE


def test_split_preserves_none_head_classes_when_original_has_no_preceding_consonant():
    lo_end = 0.2
    target_end = lo_end + 1.2
    segments, rms = _long_mora_between_anchor_morae(
        [seg("vowel", lo_end, target_end, phoneme="a", confidence=0.9)],
        lo_end=lo_end, target_end=target_end, hi_end=target_end + 0.2)
    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 4
    for e in a_events:
        assert e.consonant_class == ConsonantClass.NONE
        assert e.aperture_class == ApertureClass.NONE


def test_split_weak_vowel_scaling_applies_uniformly_to_all_subwindows():
    anchor_dur = 0.2
    mora_dur = 1.2
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur

    def a_events(confidence):
        segments = [
            seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
            seg("vowel", lo_end, target_end, phoneme="a", confidence=confidence),
            seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
        ]
        hop = 0.010
        n = round(hi_end / hop) + 1
        times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
        half = mora_dur / 2
        values = []
        for t in times:
            if t < lo_end:
                values.append(_ANCHOR_LO_RMS)
            elif t < target_end:
                offset = t - lo_end
                values.append(0.25 if offset < half else 0.28)
            else:
                values.append(_ANCHOR_HI_RMS)
        rms = rms_env(times, values)
        mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
        return [e for e in mouth_events if e.shape == MouthShape.A]

    weak_events = a_events(confidence=0.2)
    strong_events = a_events(confidence=0.9)
    assert len(weak_events) == len(strong_events) == 4
    for w, s in zip(weak_events, strong_events, strict=True):
        assert w.open_amount == pytest.approx(s.open_amount * 0.5)


def test_split_does_not_affect_population_percentiles_for_other_morae():
    def mid_percentile_open_amounts(long_mora_dur):
        segments = [
            seg("vowel", 0.0, 0.2, phoneme="ɯ", confidence=0.9),
            seg("vowel", 0.2, 0.4, phoneme="i", confidence=0.9),
            seg("vowel", 0.4, 0.4 + long_mora_dur, phoneme="e̞", confidence=0.9),
            seg("vowel", 0.4 + long_mora_dur, 0.6 + long_mora_dur, phoneme="a", confidence=0.9),
            seg("vowel", 0.6 + long_mora_dur, 0.8 + long_mora_dur, phoneme="o̞", confidence=0.9),
        ]
        hop = 0.010
        total = 0.8 + long_mora_dur
        n = round(total / hop) + 1
        times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
        values = []
        for t in times:
            if t < 0.2:
                values.append(0.10)
            elif t < 0.4:
                values.append(0.40)
            elif t < 0.4 + long_mora_dur:
                values.append(0.60)
            elif t < 0.6 + long_mora_dur:
                values.append(0.50)
            else:
                values.append(0.99)
        rms = rms_env(times, values)
        mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
        target1 = next(e for e in mouth_events if e.shape == MouthShape.I)
        target2 = next(e for e in mouth_events if e.shape == MouthShape.A)
        return target1.open_amount, target2.open_amount

    four_subwindows = mid_percentile_open_amounts(1.2)
    six_subwindows = mid_percentile_open_amounts(1.8)
    assert four_subwindows == pytest.approx(six_subwindows)
    assert 0.30 < four_subwindows[0] < 0.75
    assert 0.30 < four_subwindows[1] < 0.75


def test_song_without_morae_has_empty_group_sizes():
    segments = [seg("gap", 0.0, 0.5)]
    rms = flat_rms(0.5, 0.01)
    mouth_events, _diag, group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    assert len(mouth_events) == 1
    assert group_sizes == []


def test_split_subwindows_get_same_open_amount_when_population_is_degenerate():
    mora_dur = 1.2
    other_value = 0.5
    delta = 0.3
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="ɯ", confidence=0.9),
        seg("vowel", 0.2, 0.2 + mora_dur, phoneme="a", confidence=0.9),
    ]
    hop = 0.010
    total = 0.2 + mora_dur
    n = round(total / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    half = mora_dur / 2
    values = []
    for t in times:
        if t < 0.2:
            values.append(other_value)
        else:
            offset = t - 0.2
            values.append(other_value - delta if offset < half else other_value + delta)
    rms = rms_env(times, values)
    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 4
    amounts = [e.open_amount for e in a_events]
    assert amounts == pytest.approx([amounts[0]] * 4)


def test_n_mora_splits_when_long():
    anchor_dur = 0.2
    mora_dur = 1.2
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("consonant", lo_end, target_end, phoneme="ɴ"),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    step_values = [0.2, 0.4, 0.6, 0.8]
    step_dur = mora_dur / 4
    values = []
    for t in times:
        if t < lo_end:
            values.append(_ANCHOR_LO_RMS)
        elif t < target_end:
            idx = min(3, int((t - lo_end) / step_dur))
            values.append(step_values[idx])
        else:
            values.append(_ANCHOR_HI_RMS)
    rms = rms_env(times, values)
    mouth_events, _diag, group_sizes = events.confirm_mouth_events(segments, rms, **_IDENTITY_MAPPING_KW)
    n_events = [e for e in mouth_events if e.shape == MouthShape.N]
    assert len(n_events) == 4
    assert group_sizes == [1, 4, 1]
    amounts = [e.open_amount for e in n_events]
    assert amounts == sorted(amounts)
    assert len(set(amounts)) == 4


@pytest.mark.parametrize("normalized", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("open_max,cap_side", [(0.60, "open_max"), (0.90, "open_hi")])
def test_map_open_amount_continuous_follows_formula_without_clamping(normalized, open_max, cap_side):
    open_lo, open_hi, intensity_curve = 0.30, 0.75, 0.6
    cap = min(open_hi, open_max)
    floor = min(open_lo, cap)
    expected = floor + (cap - floor) * (normalized ** intensity_curve)
    actual = events._map_open_amount_continuous(
        normalized, open_lo=open_lo, open_hi=open_hi, open_max=open_max, intensity_curve=intensity_curve,
    )
    assert actual == pytest.approx(expected)
    assert cap == pytest.approx(open_max if cap_side == "open_max" else open_hi)


def test_map_open_amount_continuous_degenerates_to_constant_when_open_max_below_open_lo():
    open_lo, open_hi, open_max, intensity_curve = 0.30, 0.75, 0.10, 0.6
    amounts = [
        events._map_open_amount_continuous(
            v, open_lo=open_lo, open_hi=open_hi, open_max=open_max, intensity_curve=intensity_curve,
        )
        for v in (0.0, 0.3, 0.7, 1.0)
    ]
    assert amounts == pytest.approx([open_max] * 4)


def test_map_open_amount_continuous_degenerates_to_constant_when_open_max_equals_open_lo():
    open_lo, open_hi, open_max, intensity_curve = 0.30, 0.75, 0.30, 0.6
    amounts = [
        events._map_open_amount_continuous(
            v, open_lo=open_lo, open_hi=open_hi, open_max=open_max, intensity_curve=intensity_curve,
        )
        for v in (0.0, 0.3, 0.7, 1.0)
    ]
    assert amounts == pytest.approx([open_lo] * 4)


def test_split_subwindow_open_amounts_are_distinct_in_clamp_saturation_range():
    anchor_dur = 0.20
    three_subwindow_mora_dur = 1.02
    lo_end = anchor_dur
    target_end = lo_end + three_subwindow_mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("vowel", lo_end, target_end, phoneme="a", confidence=0.9),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    sub_dur = three_subwindow_mora_dur / 3
    sub_values = [0.73, 0.856, 0.955]
    values = []
    for t in times:
        if t < lo_end:
            values.append(_ANCHOR_LO_RMS)
        elif t < target_end:
            offset = t - lo_end
            idx = min(2, int(offset / sub_dur))
            values.append(sub_values[idx])
        else:
            values.append(_ANCHOR_HI_RMS)
    rms = rms_env(times, values)

    mora_representative = events._mora_rms(rms, lo_end, target_end)
    p_lo, p_hi = events._percentile_bounds([_ANCHOR_LO_RMS, mora_representative, _ANCHOR_HI_RMS])
    normalized_values = [events._normalize_with_bounds(v, p_lo, p_hi) for v in sub_values]

    cap = min(_DEFAULT_KW["open_hi"], _DEFAULT_KW["open_max"])
    saturation_threshold = cap ** (1.0 / _DEFAULT_KW["intensity_curve"])
    assert all(v >= saturation_threshold for v in normalized_values)

    expected_amounts = [
        events._map_open_amount_continuous(
            v, open_lo=_DEFAULT_KW["open_lo"], open_hi=_DEFAULT_KW["open_hi"],
            open_max=_DEFAULT_KW["open_max"], intensity_curve=_DEFAULT_KW["intensity_curve"],
        )
        for v in normalized_values
    ]

    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 3
    actual_amounts = [e.open_amount for e in a_events]
    assert actual_amounts == pytest.approx(expected_amounts)
    assert actual_amounts == sorted(actual_amounts)
    assert len(set(actual_amounts)) == 3
