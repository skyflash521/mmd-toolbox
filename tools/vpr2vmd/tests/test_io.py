import pytest

from vpr import (
    Note,
    Part,
    TempoEvent,
    Track,
    VprProject,
)
from vpr2vmd import io as vio

_SINGING_TRACK_TYPE = 2


def _note(start, dur, *, vel=64, phonemes=None, lyric="ら"):
    return Note(
        start_tick=start, duration_tick=dur, pitch=60, lyric=lyric,
        velocity=vel, phonemes=phonemes if phonemes is not None else [],
    )


def _part(notes, *, name="p", start=0):
    return Part(name=name, start_tick=start, notes=notes)


def _track(parts, *, name="vocal"):
    return Track(name=name, parts=parts)


def _project(tracks, *, tempos=None):
    return VprProject(
        resolution=480,
        tempos=tempos if tempos is not None else [TempoEvent(0, 120.0)],
        tracks=tracks,
    )


def test_select_track_defaults_to_first():
    project = _project([_track([], name="a"), _track([], name="b")])
    assert vio.select_track(project, None).name == "a"


def test_select_track_by_index():
    project = _project([_track([], name="a"), _track([], name="b")])
    assert vio.select_track(project, "1").name == "b"


def test_select_track_full_width_digit_selects_by_name():
    project = _project([_track([], name="１"), _track([], name="b")])
    assert vio.select_track(project, "１").name == "１"


def test_select_track_spaced_digit_is_treated_as_name():
    project = _project([_track([], name="a"), _track([], name="b")])
    with pytest.raises(vio.TrackSelectionError):
        vio.select_track(project, " 1")


def test_select_track_index_out_of_range_errors():
    project = _project([_track([], name="a")])
    with pytest.raises(vio.TrackSelectionError):
        vio.select_track(project, "5")


def test_select_track_by_name():
    project = _project([_track([], name="inst"), _track([], name="lead")])
    assert vio.select_track(project, "lead").name == "lead"


def test_select_track_name_no_match_errors():
    project = _project([_track([], name="lead")])
    with pytest.raises(vio.TrackSelectionError):
        vio.select_track(project, "nope")


def test_select_track_ambiguous_name_errors():
    project = _project([_track([], name="dup"), _track([], name="dup")])
    with pytest.raises(vio.TrackSelectionError):
        vio.select_track(project, "dup")


def test_select_track_no_tracks_errors():
    project = _project([])
    with pytest.raises(vio.TrackSelectionError):
        vio.select_track(project, None)


def test_collect_notes_merges_parts_in_order():
    track = _track([
        _part([_note(0, 100)], name="p0"),
        _part([_note(200, 100)], name="p1"),
    ])
    notes = vio.collect_notes(track)
    assert [n.start_tick for n in notes] == [0, 200]


def test_collect_notes_sorts_by_start_then_longer_duration_first():
    track = _track([_part([_note(0, 50), _note(0, 120)])])
    notes = vio.collect_notes(track)
    assert [n.duration_tick for n in notes] == [120, 50]


def test_collect_notes_tiebreak_by_part_order():
    track = _track([
        _part([_note(0, 100, lyric="first")], name="p0"),
        _part([_note(0, 100, lyric="second")], name="p1"),
    ])
    notes = vio.collect_notes(track)
    assert [n.lyric for n in notes] == ["first", "second"]


def test_collect_notes_is_stable_within_part():
    track = _track([_part([_note(0, 100, lyric="x"), _note(0, 100, lyric="y")])])
    notes = vio.collect_notes(track)
    assert [n.lyric for n in notes] == ["x", "y"]


def test_collect_notes_empty_track():
    assert vio.collect_notes(_track([])) == []
