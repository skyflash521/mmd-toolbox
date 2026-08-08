"""声量(ダイナミクス)コントローラ→開き量の写像のテスト。

声量コントローラの選別(採用するのは dynamics のみ。無ければ None)、モーラ区間の
階段平均、コントローラ値域での 0〜1 正規化と開き量写像を検証する。
"""

import pytest

from vpr import ControllerCurve, ControllerEvent, Note, Part
from vpr2vmd import loudness


def _part(controllers):
    return Part(name="p", start_tick=0, controllers=controllers)


def _curve(name, points):
    return ControllerCurve(name=name, events=[ControllerEvent(t, v) for t, v in points])


def _note(start, dur):
    return Note(
        start_tick=start, duration_tick=dur, pitch=60, lyric="x", velocity=64, phonemes=["a"]
    )


def test_choose_dynamics_ignoring_non_loudness_controllers():
    # 採用するのは dynamics だけ。s5Expression・音色等が同居しても dynamics を選ぶ。
    parts = [_part([_curve("s5Expression", [(0, 30)]), _curve("dynamics", [(0, 64)])])]
    assert loudness.choose_loudness_controller(parts) == "dynamics"


def test_choose_none_when_only_s5expression():
    # s5Expression は中立値・被覆の扱いが未確定なので声量として採用しない(None=フォールバック)。
    parts = [_part([_curve("s5Expression", [(0, 30)])])]
    assert loudness.choose_loudness_controller(parts) is None


def test_choose_none_when_no_loudness_controller():
    # 声量以外(音色等)しか無いときは None(velocity フォールバックへ)。
    parts = [_part([_curve("brightness", [(0, 64)]), _curve("character", [(0, 5)])])]
    assert loudness.choose_loudness_controller(parts) is None


def test_choose_ignores_empty_loudness_curve():
    parts = [_part([_curve("dynamics", [])])]
    assert loudness.choose_loudness_controller(parts) is None


def test_merged_events_stable_for_same_tick():
    # 同一 tick に複数の制御点があっても tick のみでソートし入力順(収集順)を保つ。値の大小で並べ替えない
    # (そうしないと曲線の値の引き当てが同 tick 衝突で「値の大きい方」を採ってしまう)。
    parts = [_part([_curve("dynamics", [(100, 80), (100, 50)])])]
    assert loudness._merged_events(parts, "dynamics") == [(100, 80), (100, 50)]


def test_step_average_constant():
    curve = loudness._StepCurve([(0, 64), (1000, 64)], default=64)
    assert curve.average(0, 480) == pytest.approx(64.0)


def test_step_average_step_midway():
    # [0,240) は値0、[240,480) は値120 → 平均 (0*240 + 120*240)/480 = 60。
    curve = loudness._StepCurve([(0, 0), (240, 120)], default=0)
    assert curve.average(0, 480) == pytest.approx(60.0)


def test_step_average_before_first_event_uses_default():
    # 区間が曲線始端より前なら default 値を保持する。
    curve = loudness._StepCurve([(1000, 100)], default=50)
    assert curve.average(0, 480) == pytest.approx(50.0)


def test_step_average_spanning_curve_start_mixes_default_and_curve():
    # 曲線始端をまたぐ区間は、始端より前を default・以降を曲線の値で平均する。
    # [0,480) は default 20、[480,960) は 100 → (20*480 + 100*480)/960 = 60。
    curve = loudness._StepCurve([(480, 100)], default=20)
    assert curve.average(0, 960) == pytest.approx(60.0)


def test_step_average_zero_length_interval_is_start_value():
    curve = loudness._StepCurve([(0, 30), (480, 90)], default=30)
    assert curve.average(480, 480) == pytest.approx(90.0)


def test_step_curve_value_at_takes_last_point_on_duplicate_tick():
    # 同一 tick に複数の点があるときは後に現れた点の値を採る(_merged_events の並び順が決める)。
    curve = loudness._StepCurve([(100, 80), (100, 50)], default=80)
    assert curve.value_at(100) == 50
    assert curve.value_at(99) == 80  # 始端より前は default


def test_open_amounts_loud_higher_than_quiet():
    # dynamics: note0 大音量(120)・note1 小音量(10)。大きいほど開き量が大きい。
    parts = [_part([_curve("dynamics", [(0, 120), (480, 10)])])]
    notes = [_note(0, 480), _note(480, 480)]
    out = loudness.open_amounts_from_loudness(parts, notes, lo=0.3, hi=0.75, open_max=0.9, gamma=0.6)
    assert out is not None
    assert out[0] > out[1]
    assert out[0] == pytest.approx(min(0.3 + 0.45 * (120 / 127) ** 0.6, 0.75, 0.9))
    assert out[1] == pytest.approx(min(0.3 + 0.45 * (10 / 127) ** 0.6, 0.75, 0.9))


def test_open_amounts_none_when_no_loudness_controller():
    parts = [_part([_curve("brightness", [(0, 64)])])]
    out = loudness.open_amounts_from_loudness(
        parts, [_note(0, 480)], lo=0.3, hi=0.75, open_max=0.9, gamma=0.6
    )
    assert out is None


def test_open_amounts_normalizes_over_controller_range():
    # dynamics の値域は 0..127。値127 → n=1.0 → 開き量は hi(open_max 内)。
    parts = [_part([_curve("dynamics", [(0, 127)])])]
    out = loudness.open_amounts_from_loudness(
        parts, [_note(0, 480)], lo=0.3, hi=0.75, open_max=0.9, gamma=0.6
    )
    assert out[0] == pytest.approx(0.75)


def test_open_amounts_clamped_by_open_max():
    # open_max が hi より小さければ上限になる。
    parts = [_part([_curve("dynamics", [(0, 127)])])]
    out = loudness.open_amounts_from_loudness(
        parts, [_note(0, 480)], lo=0.3, hi=0.9, open_max=0.5, gamma=0.6
    )
    assert out[0] == pytest.approx(0.5)
