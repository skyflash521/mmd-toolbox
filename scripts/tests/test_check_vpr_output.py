"""出力 vpr の機械検査スクリプトのテスト。

検査ロジックを合成した VprProject で検証する(実素材のファイルには依存しない)。
基準値は引数で渡した正解 vpr から算出するため、基準の固定値はここにも実装にも現れない。
"""

import check_vpr_output as check
import pytest

from vpr import Note, Part, TempoEvent, TimeSignature, Track, VprProject

RESOLUTION = 480

# 120 BPM・分解能480では 1 tick = 1/960 秒(960 tick = 1 秒)。


def _note(start: int, dur: int, lyric: str = "あ") -> Note:
    return Note(start_tick=start, duration_tick=dur, pitch=60, lyric=lyric, velocity=64)


def _project(notes, *, tempos=None, signatures=None) -> VprProject:
    return VprProject(
        resolution=RESOLUTION,
        tempos=tempos or [TempoEvent(tick=0, bpm=120.0)],
        time_signatures=signatures
        or [TimeSignature(tick=0, numerator=4, denominator=4)],
        tracks=[Track(name="song", parts=[Part(name="part", start_tick=0, notes=notes)])],
    )


def _reference(lyric_durs, cont_durs=(), *, gap=960) -> VprProject:
    """表示歌詞を持つ音符と継続の音符を、互いに離して並べた正解相当のプロジェクトを作る。"""
    notes = []
    pos = 0
    for dur in lyric_durs:
        notes.append(_note(pos, dur))
        pos += dur + gap
    for dur in cont_durs:
        notes.append(_note(pos, dur, lyric="-"))
        pos += dur + gap
    return _project(notes)


# --- 秒換算(テンポイベント列に沿った区分換算) ---


def test_seconds_at_follows_piecewise_tempo():
    # 120 BPM の区間は 960 tick = 1 秒、60 BPM へ変わった後は 480 tick = 1 秒。
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=960, bpm=60.0)]
    assert check.seconds_at(480, tempos, RESOLUTION) == pytest.approx(0.5)
    assert check.seconds_at(960, tempos, RESOLUTION) == pytest.approx(1.0)
    # 変化点より後を先頭テンポだけで換算すると 1.5 秒になり、区分換算の 2.0 秒とずれる。
    assert check.seconds_at(1440, tempos, RESOLUTION) == pytest.approx(2.0)


def test_seconds_at_single_tempo():
    tempos = [TempoEvent(tick=0, bpm=120.0)]
    assert check.seconds_at(0, tempos, RESOLUTION) == pytest.approx(0.0)
    assert check.seconds_at(2400, tempos, RESOLUTION) == pytest.approx(2.5)


# --- 小節番号(編集器の表示に合わせた1始まり) ---


def test_measure_number_is_one_based():
    signatures = [TimeSignature(tick=0, numerator=4, denominator=4)]
    # 4/4・分解能480では1小節 = 1920 tick。
    assert check.measure_number(0, signatures, RESOLUTION) == 1
    assert check.measure_number(1919, signatures, RESOLUTION) == 1
    assert check.measure_number(1920, signatures, RESOLUTION) == 2


def test_measure_number_follows_time_signature_change():
    # 2小節の 4/4(3840 tick)の後に 3/4(1小節 = 1440 tick)へ変わる。
    signatures = [
        TimeSignature(tick=0, numerator=4, denominator=4),
        TimeSignature(tick=3840, numerator=3, denominator=4),
    ]
    assert check.measure_number(3839, signatures, RESOLUTION) == 2
    assert check.measure_number(3840, signatures, RESOLUTION) == 3
    assert check.measure_number(3840 + 1440, signatures, RESOLUTION) == 4


# --- 正解から算出する基準 ---


def test_thresholds_take_min_duration_and_max_ratio_across_references():
    # 正解1: 表示歌詞 100ms 最短・継続なし。正解2: 表示歌詞 200ms 最短・継続1/3。
    ref1 = _reference([96, 192])
    ref2 = _reference([192, 384], cont_durs=[192])
    thresholds = check.compute_thresholds([ref1, ref2])
    assert thresholds.min_lyric_note_sec == pytest.approx(0.1)
    assert thresholds.max_continuation_ratio == pytest.approx(1.5 * (1 / 3))


def test_thresholds_mora_uses_min_across_references():
    # 各正解の「切れ目なく続く並びの1モーラあたりの最短」のうち小さい方を基準にする。
    ref1 = _reference([240])  # 単独の並び: 250ms/モーラ
    ref2 = _reference([96])  # 単独の並び: 100ms/モーラ
    thresholds = check.compute_thresholds([ref1, ref2])
    assert thresholds.min_mora_sec == pytest.approx(0.1)


def test_thresholds_change_with_different_references():
    # 別の正解を渡せば別の基準になる(基準が定数で返らないことの固定)。
    thresholds = check.compute_thresholds([_reference([240, 480])])
    assert thresholds.min_lyric_note_sec == pytest.approx(0.25)
    assert thresholds.min_mora_sec == pytest.approx(0.25)
    assert thresholds.max_continuation_ratio == pytest.approx(0.0)


def test_thresholds_mora_comes_from_runs_not_note_minimum():
    # 正解の中の切れ目なく続く並び(表示歌詞・継続・表示歌詞)は、全長 300ms を表示歌詞
    # 2つで割った 150ms/モーラとして効く。音符単体の最短 100ms をモーラ基準に流用したら誤り。
    run = [_note(0, 96), _note(96, 96, lyric="-"), _note(192, 96)]
    isolated = [_note(5000, 240)]
    ref = _project(run + isolated)
    thresholds = check.compute_thresholds([ref])
    assert thresholds.min_mora_sec == pytest.approx(0.15)
    assert thresholds.min_lyric_note_sec == pytest.approx(0.1)


def test_thresholds_convert_ticks_after_tempo_change_piecewise():
    # テンポ変化点より後の音符の長さは変化後のテンポで換算する。60 BPM 区間の 96 tick は
    # 200ms で、先頭の 120 BPM で換算した 100ms を基準にしたら誤り。
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=960, bpm=60.0)]
    ref = _project([_note(960, 96)], tempos=tempos)
    thresholds = check.compute_thresholds([ref])
    assert thresholds.min_lyric_note_sec == pytest.approx(0.2)


# --- 仕様の不変条件の検査 ---


def test_overlapping_notes_are_reported_with_measure():
    project = _project([_note(0, 960), _note(480, 480)])
    thresholds = check.compute_thresholds([_reference([96])])
    violations = [v for v in check.check_project(project, thresholds) if v.kind == "overlap"]
    assert len(violations) == 1
    assert violations[0].measure == 1


def test_adjacent_notes_do_not_overlap():
    project = _project([_note(0, 480), _note(480, 480)])
    thresholds = check.compute_thresholds([_reference([96])])
    assert [v for v in check.check_project(project, thresholds) if v.kind == "overlap"] == []


def test_zero_duration_note_is_reported():
    project = _project([_note(1920, 0)])
    thresholds = check.compute_thresholds([_reference([96])])
    violations = [
        v for v in check.check_project(project, thresholds) if v.kind == "zero_duration"
    ]
    assert len(violations) == 1
    assert violations[0].measure == 2


def test_zero_duration_note_inside_another_is_not_an_overlap():
    # 長さ0の音符の発音区間は空なので、他の音符の区間の中にあっても重なりにはしない
    # (長さ0そのものは別の違反として報告する)。
    thresholds = check.compute_thresholds([_reference([96])])
    project = _project([_note(0, 960), _note(480, 0)])
    violations = check.check_project(project, thresholds)
    assert [v for v in violations if v.kind == "overlap"] == []
    assert len([v for v in violations if v.kind == "zero_duration"]) == 1


# --- 正解の分布と突き合わせる検査 ---


def test_short_lyric_note_is_reported_against_reference_minimum():
    thresholds = check.compute_thresholds([_reference([96, 192])])  # 基準 100ms
    project = _project([_note(0, 48), _note(1920, 96)])  # 50ms と 100ms
    violations = [
        v for v in check.check_project(project, thresholds) if v.kind == "short_lyric_note"
    ]
    assert len(violations) == 1
    assert violations[0].measure == 1


def test_thresholds_convert_note_crossing_tempo_change_piecewise():
    # テンポ変化点をまたぐ音符は前後の区間ごとに換算して足す。120 BPM 側 480 tick(500ms)と
    # 60 BPM 側 480 tick(1000ms)で計 1500ms。開始位置のテンポだけで全長を換算すると
    # 1000ms になり誤り。並び(この音符単独)の1モーラあたりにも同じ換算が効く。
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=960, bpm=60.0)]
    ref = _project([_note(480, 960)], tempos=tempos)
    thresholds = check.compute_thresholds([ref])
    assert thresholds.min_lyric_note_sec == pytest.approx(1.5)
    assert thresholds.min_mora_sec == pytest.approx(1.5)


def test_check_mora_run_crossing_tempo_change_uses_piecewise_seconds():
    # 変化点をまたぐ並び: 120 BPM 側 96 tick(100ms)と 60 BPM 側 48 tick(100ms)の
    # 全長 200ms を表示歌詞2つで割ると 100ms/モーラで基準に達する。先頭テンポだけで
    # 換算すると 75ms/モーラになり誤検出する。
    thresholds = check.compute_thresholds([_reference([96])])  # 基準 100ms/モーラ
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=960, bpm=60.0)]
    run = [_note(864, 96), _note(960, 48)]
    project = _project(run, tempos=tempos)
    assert [
        v for v in check.check_project(project, thresholds) if v.kind == "short_mora_run"
    ] == []


def test_gap_splits_runs_in_checked_project():
    # 間に空きのある音符は別の並びとして扱う。短い並びだけが違反になり、離れた長い音符と
    # 合算して見逃してはならない。
    thresholds = check.compute_thresholds([_reference([96])])  # 基準 100ms/モーラ
    notes = [_note(0, 48), _note(2000, 480)]  # 50ms の単独の並びと 500ms の単独の並び
    violations = [
        v for v in check.check_project(_project(notes), thresholds) if v.kind == "short_mora_run"
    ]
    assert len(violations) == 1
    assert violations[0].measure == 1


def test_check_converts_ticks_after_tempo_change_piecewise():
    # 60 BPM 区間にある 48 tick の音符は 100ms で基準に達する。先頭の 120 BPM で
    # 換算すると 50ms になり誤検出する。
    thresholds = check.compute_thresholds([_reference([96])])  # 基準 100ms
    tempos = [TempoEvent(tick=0, bpm=120.0), TempoEvent(tick=960, bpm=60.0)]
    project = _project([_note(960, 48)], tempos=tempos)
    assert [
        v for v in check.check_project(project, thresholds) if v.kind == "short_lyric_note"
    ] == []


def test_lyric_note_equal_to_reference_minimum_passes():
    # 基準と等しい長さは「下回らない」ので合格。
    thresholds = check.compute_thresholds([_reference([96])])
    project = _project([_note(0, 96)])
    assert [
        v for v in check.check_project(project, thresholds) if v.kind == "short_lyric_note"
    ] == []


def test_short_continuation_note_is_not_a_lyric_violation():
    # 継続の音符(表示歌詞が「-」)は表示歌詞を持つ音符の最短検査の対象にしない。
    thresholds = check.compute_thresholds([_reference([96])])
    project = _project([_note(0, 480), _note(480, 48, lyric="-")])
    assert [
        v for v in check.check_project(project, thresholds) if v.kind == "short_lyric_note"
    ] == []


def test_short_mora_run_is_reported():
    # 前の終端と次の開始が一致して続く3音符の並び。全長 150ms を表示歌詞2つで割ると
    # 75ms/モーラで、正解の 100ms/モーラを下回る。
    thresholds = check.compute_thresholds([_reference([96])])
    run = [_note(0, 48), _note(48, 48, lyric="-"), _note(96, 48)]
    violations = [
        v for v in check.check_project(_project(run), thresholds) if v.kind == "short_mora_run"
    ]
    assert len(violations) == 1
    assert violations[0].measure == 1


def test_mora_run_with_enough_length_passes():
    # 全長 400ms・表示歌詞2つ = 200ms/モーラは基準 100ms を満たす。
    thresholds = check.compute_thresholds([_reference([96])])
    run = [_note(0, 192), _note(192, 96, lyric="-"), _note(288, 96)]
    assert [
        v for v in check.check_project(_project(run), thresholds) if v.kind == "short_mora_run"
    ] == []


def test_mora_run_equal_to_reference_minimum_passes():
    # 1モーラあたりが基準とちょうど等しい並びは「下回らない」ので合格。
    thresholds = check.compute_thresholds([_reference([96])])  # 基準 100ms/モーラ
    run = [_note(0, 96), _note(96, 96)]  # 全長 200ms・表示歌詞2つ = 100ms/モーラ
    assert [
        v for v in check.check_project(_project(run), thresholds) if v.kind == "short_mora_run"
    ] == []


def test_violation_measure_follows_time_signature_denominator():
    # 4/4 の1小節(1920 tick)の後に 3/8(1小節 = 720 tick)へ変わる。tick 3360 は
    # 3/8 の3小節目 = 通算4小節目。分母を無視して 3/4 で数えると3小節目になり誤り。
    signatures = [
        TimeSignature(tick=0, numerator=4, denominator=4),
        TimeSignature(tick=1920, numerator=3, denominator=8),
    ]
    thresholds = check.compute_thresholds([_reference([96])])
    project = _project([_note(3360, 0)], signatures=signatures)
    violations = [
        v for v in check.check_project(project, thresholds) if v.kind == "zero_duration"
    ]
    assert len(violations) == 1
    assert violations[0].measure == 4


def test_continuation_ratio_over_limit_is_reported():
    # 正解の継続割合 1/3 の 1.5 倍 = 50% を超える 60% は違反。
    ref = _reference([96, 96], cont_durs=[96])
    thresholds = check.compute_thresholds([ref])
    notes = []
    pos = 0
    for lyric in ["あ", "-", "-", "い", "-"]:  # 継続 3/5 = 60%
        notes.append(_note(pos, 96, lyric=lyric))
        pos += 96 + 960
    violations = [
        v
        for v in check.check_project(_project(notes), thresholds)
        if v.kind == "continuation_ratio"
    ]
    assert len(violations) == 1
    # 割合は曲全体の性質なので、特定の小節番号を持たない。
    assert violations[0].measure is None


def test_continuation_ratio_within_limit_passes():
    ref = _reference([96, 96], cont_durs=[96])  # 上限 50%
    thresholds = check.compute_thresholds([ref])
    notes = [_note(0, 96), _note(2000, 96, lyric="-"), _note(4000, 96)]  # 継続 1/3
    assert [
        v
        for v in check.check_project(_project(notes), thresholds)
        if v.kind == "continuation_ratio"
    ] == []


def test_continuation_ratio_equal_to_limit_passes():
    # 上限とちょうど等しい割合は「超えない」ので合格。
    ref = _reference([96, 96], cont_durs=[96])  # 上限 = 1/3 の 1.5 倍 = 50%
    thresholds = check.compute_thresholds([ref])
    notes = [_note(0, 96), _note(2000, 96, lyric="-")]  # 継続 1/2 = 50%
    assert [
        v
        for v in check.check_project(_project(notes), thresholds)
        if v.kind == "continuation_ratio"
    ] == []


def test_clean_project_has_no_violations():
    thresholds = check.compute_thresholds([_reference([96, 192], cont_durs=[192])])
    notes = [_note(0, 192), _note(192, 192, lyric="-"), _note(384, 192), _note(2000, 240)]
    assert check.check_project(_project(notes), thresholds) == []
