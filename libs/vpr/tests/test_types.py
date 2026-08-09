"""vpr データモデル型のテスト。

型定義(フィールド・既定値)のみを検証する。read/write の振る舞いはこのファイルでは扱わない。
"""

import pytest


def test_note_holds_documented_fields():
    from vpr import Note

    note = Note(
        start_tick=480,
        duration_tick=240,
        pitch=60,
        lyric="ら",
        phonemes=["r", "a"],
        velocity=64,
    )
    assert note.start_tick == 480
    assert note.duration_tick == 240
    assert note.pitch == 60
    assert note.lyric == "ら"
    assert note.phonemes == ["r", "a"]
    assert note.velocity == 64


def test_note_phonemes_default_empty_and_independent():
    from vpr import Note

    a = Note(start_tick=0, duration_tick=1, pitch=60, lyric="あ", velocity=0)
    b = Note(start_tick=0, duration_tick=1, pitch=60, lyric="い", velocity=0)
    # 音素列は空可。既定の可変リストがインスタンス間で共有されないこと。
    assert a.phonemes == []
    a.phonemes.append("a")
    assert b.phonemes == []


@pytest.mark.xfail(reason="impl pending: Note.is_protected", strict=True)
def test_note_is_protected_defaults_to_false():
    from vpr import Note

    # 音素の保護は既定で偽。値の意味付け・選別はフォーマット層が持たず、器だけを持つ。
    note = Note(start_tick=0, duration_tick=1, pitch=60, lyric="あ", velocity=0)
    assert note.is_protected is False


@pytest.mark.xfail(reason="impl pending: Note.is_protected", strict=True)
def test_note_is_protected_holds_true():
    from vpr import Note

    note = Note(start_tick=0, duration_tick=1, pitch=60, lyric="か", velocity=0,
                is_protected=True)
    assert note.is_protected is True


def test_note_velocity_upper_bound():
    from vpr import Note

    # ベロシティは 0〜127 の生値。上限 127 を保持できること。
    note = Note(start_tick=0, duration_tick=1, pitch=60, lyric="は", velocity=127)
    assert note.velocity == 127


def test_tempo_event_fields():
    from vpr import TempoEvent

    tempo = TempoEvent(tick=0, bpm=120.0)
    assert tempo.tick == 0
    assert tempo.bpm == pytest.approx(120.0)


def test_time_signature_fields():
    from vpr import TimeSignature

    ts = TimeSignature(tick=0, numerator=3, denominator=4)
    assert ts.tick == 0
    assert ts.numerator == 3
    assert ts.denominator == 4


def test_part_holds_absolute_start_and_notes():
    from vpr import Note, Part

    note = Note(start_tick=960, duration_tick=480, pitch=62, lyric="そ", velocity=80)
    part = Part(name="part1", start_tick=960, notes=[note])
    assert part.name == "part1"
    assert part.start_tick == 960
    assert part.notes == [note]


def test_part_notes_default_empty_and_independent():
    from vpr import Part

    a = Part(name="a", start_tick=0)
    b = Part(name="b", start_tick=0)
    assert a.notes == []
    a.notes.append(object())
    assert b.notes == []


def test_track_holds_parts():
    from vpr import Part, Track

    part = Part(name="p", start_tick=0)
    track = Track(name="vocal", parts=[part])
    assert track.name == "vocal"
    assert track.parts == [part]


def test_track_parts_default_empty():
    from vpr import Track

    assert Track(name="vocal").parts == []


def test_project_holds_resolution_tempos_signatures_tracks():
    from vpr import TempoEvent, TimeSignature, Track, VprProject

    project = VprProject(
        resolution=480,
        tempos=[TempoEvent(tick=0, bpm=120.0)],
        time_signatures=[TimeSignature(tick=0, numerator=4, denominator=4)],
        tracks=[Track(name="vocal")],
    )
    assert project.resolution == 480
    assert project.tempos[0].bpm == pytest.approx(120.0)
    assert project.time_signatures[0].numerator == 4
    assert project.tracks[0].name == "vocal"


def test_project_collection_fields_default_empty():
    from vpr import VprProject

    project = VprProject(resolution=480)
    assert project.tempos == []
    assert project.time_signatures == []
    assert project.tracks == []


def test_vpr_format_error_is_exception():
    from vpr import VprFormatError

    assert issubclass(VprFormatError, Exception)


def test_vpr_warning_holds_code_and_message():
    from vpr import VprWarning

    warning = VprWarning(code="overlap", message="発音区間が重なっています")
    assert warning.code == "overlap"
    assert warning.message == "発音区間が重なっています"
