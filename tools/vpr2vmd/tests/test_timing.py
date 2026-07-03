"""tick→秒→フレーム変換のテスト(vpr2vmd.md §3・§6)。

vpr の tick(整数、resolution=tick/四分音符)とテンポマップから 30fps の float フレームへ
変換する。テンポマップの区分積分で秒を求め、30 を掛けてフレーム化する(量子化は lipsync 側)。
"""

import pytest
from vpr import Note, TempoEvent

from vpr2vmd import timing


def _note(start, dur):
    return Note(
        start_tick=start, duration_tick=dur, pitch=60, lyric="x",
        velocity=64, phonemes=["a"],
    )


def test_tick_zero_is_frame_zero():
    assert timing.tick_to_frame(0, [TempoEvent(0, 120.0)], 480) == 0.0


def test_single_tempo_one_beat():
    # 120bpm・resolution 480 → 1拍(480tick)=0.5秒=15フレーム。
    assert timing.tick_to_frame(480, [TempoEvent(0, 120.0)], 480) == pytest.approx(15.0)


def test_single_tempo_fractional_frame_not_quantized():
    # 240tick=半拍=0.25秒=7.5フレーム(整数化しない)。
    assert timing.tick_to_frame(240, [TempoEvent(0, 120.0)], 480) == pytest.approx(7.5)


def test_tempo_change_integrates_piecewise():
    # [0,480) は 120bpm(0.5秒)、以降 240bpm。960tick = 0.5 + (480tick @240bpm=0.25) = 0.75秒
    # = 22.5フレーム。
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    assert timing.tick_to_frame(960, tempos, 480) == pytest.approx(22.5)


def test_region_boundary_cumulative_is_preceding_integral():
    # テンポ境界 tick=480 ちょうどの累積秒は、先行区間 [0,480)@120bpm の積分=0.5秒=15フレーム。
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    assert timing.tick_to_frame(480, tempos, 480) == pytest.approx(15.0)


def test_just_after_boundary_uses_new_region_rate():
    # 区間は半開 [tick_i, tick_{i+1}) なので tick=480 以降は新テンポ 240bpm。
    # tick=481: 0.5秒 + 1tick@240bpm(=60/(240*480)) = 0.500520833秒 = 15.015625フレーム。
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    assert timing.tick_to_frame(481, tempos, 480) == pytest.approx(15.015625)


def test_first_tempo_not_at_zero_applies_retroactively():
    # 最初の TempoEvent.tick が 0 でなくても、その bpm を tick 0 まで遡って適用する。
    tempos = [TempoEvent(240, 120.0)]
    assert timing.tick_to_frame(480, tempos, 480) == pytest.approx(15.0)


def test_seconds_helper_matches_frame_over_30():
    tempos = [TempoEvent(0, 120.0)]
    assert timing.tick_to_seconds(480, tempos, 480) == pytest.approx(0.5)
    assert timing.tick_to_frame(480, tempos, 480) == pytest.approx(
        timing.tick_to_seconds(480, tempos, 480) * 30.0
    )


def test_seconds_integrates_tempo_change_directly():
    # tick_to_seconds 自体がテンポ変化の区分積分を正しく行うことを直接検証する
    # (tick_to_frame が独自に正しくても tick_to_seconds の誤りを見逃さないため)。
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    assert timing.tick_to_seconds(960, tempos, 480) == pytest.approx(0.75)


def test_seconds_three_regions_accumulate():
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 60.0), TempoEvent(960, 240.0)]
    assert timing.tick_to_seconds(1440, tempos, 480) == pytest.approx(1.75)


def test_seconds_first_tempo_not_at_zero_applies_retroactively():
    assert timing.tick_to_seconds(480, [TempoEvent(240, 120.0)], 480) == pytest.approx(0.5)


def test_three_regions_accumulate():
    # 120bpm[0,480) + 60bpm[480,960) + 240bpm[960,..)。
    # tick1440: 0.5 + (480tick@60bpm=1.0) + (480tick@240bpm=0.25) = 1.75秒 = 52.5フレーム。
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 60.0), TempoEvent(960, 240.0)]
    assert timing.tick_to_frame(1440, tempos, 480) == pytest.approx(52.5)


# --- 代表BPM(note_effective_bpm / representative_bpm) ---


def test_note_effective_bpm_single_tempo_equals_bpm():
    # 単一テンポでは有効BPMはそのテンポに一致(発音長に依らない)。
    assert timing.note_effective_bpm(_note(0, 480), [TempoEvent(0, 150.0)], 480) == pytest.approx(150.0)
    assert timing.note_effective_bpm(_note(960, 120), [TempoEvent(0, 150.0)], 480) == pytest.approx(150.0)


def test_note_effective_bpm_spans_tempo_change_is_time_weighted():
    # 音符[0,960): [0,480)@120bpm(1拍0.5秒)+[480,960)@240bpm(1拍0.25秒)=2拍0.75秒。
    # 有効BPM = 60×2/0.75 = 160(実発音秒へ畳んだ一定BPM)。
    tempos = [TempoEvent(0, 120.0), TempoEvent(480, 240.0)]
    assert timing.note_effective_bpm(_note(0, 960), tempos, 480) == pytest.approx(160.0)


def test_note_effective_bpm_zero_duration_is_none():
    assert timing.note_effective_bpm(_note(100, 0), [TempoEvent(0, 120.0)], 480) is None


def test_representative_bpm_uniform_returns_that_bpm():
    tempos = [TempoEvent(0, 190.0)]
    notes = [_note(0, 240), _note(240, 240), _note(480, 240)]
    assert timing.representative_bpm(notes, tempos, 480) == pytest.approx(190.0)


def test_representative_bpm_long_fast_note_dominates():
    # 100bpm 1拍(0.6秒,重み18)と 200bpm 4拍(1.2秒,重み36)。総54・半分27。
    # BPM昇順 100(累積18<27)→200(累積54≥27) で代表=200(秒で長く鳴る側が支配)。
    tempos = [TempoEvent(0, 100.0), TempoEvent(480, 200.0)]
    slow = _note(0, 480)
    fast = _note(480, 480 * 4)
    assert timing.representative_bpm([slow, fast], tempos, 480) == pytest.approx(200.0)


def test_representative_bpm_slow_note_dominates_when_longer_in_seconds():
    # 100bpm 1拍(0.6秒,18)と 200bpm 1拍(0.3秒,9)。総27・半分13.5。
    # 100(累積18≥13.5) で代表=100。同じ拍数でも遅い側は秒で長く重みが大きい。
    tempos = [TempoEvent(0, 100.0), TempoEvent(480, 200.0)]
    assert timing.representative_bpm([_note(0, 480), _note(480, 480)], tempos, 480) == pytest.approx(100.0)


def test_representative_bpm_empty_returns_default():
    assert timing.representative_bpm([], [TempoEvent(0, 120.0)], 480) == pytest.approx(120.0)


def test_representative_bpm_all_zero_duration_returns_default():
    notes = [_note(0, 0), _note(100, 0)]
    assert timing.representative_bpm(notes, [TempoEvent(0, 120.0)], 480, default_bpm=120.0) == pytest.approx(120.0)
