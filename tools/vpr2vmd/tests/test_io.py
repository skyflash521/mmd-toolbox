import io
import json
import zipfile

import pytest

from vpr import (
    Note,
    Part,
    TempoEvent,
    Track,
    VprProject,
    rest_intervals,
)
from vpr import read as vpr_read
from vpr2vmd import io as vio

_SINGING_TRACK_TYPE = 2


def _make_vpr(sequence: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Project/sequence.json", json.dumps(sequence, ensure_ascii=False))
    return buf.getvalue()


def _sequence(tracks, tempo_events=None):
    return {
        "version": {"major": 6, "minor": 5, "revision": 1},
        "vender": "Yamaha Corporation",
        "title": "test",
        "masterTrack": {
            "samplingRate": 44100,
            "tempo": {"events": tempo_events or [{"pos": 0, "value": 12000}]},
            "timeSig": {"events": [{"bar": 0, "numer": 4, "denom": 4}]},
        },
        "voices": [],
        "tracks": tracks,
    }


def _seq_note(pos, duration, number=60, lyric="ら", phoneme="4 a", velocity=64):
    return {
        "pos": pos, "duration": duration, "number": number,
        "lyric": lyric, "phoneme": phoneme, "velocity": velocity,
    }


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


def test_select_track_negative_index_errors():
    project = _project([_track([], name="a")])
    with pytest.raises(vio.TrackSelectionError):
        vio.select_track(project, "-1")


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


def test_extracts_notes_tempo_rests_from_representative_vpr():
    vpr = _make_vpr(_sequence(
        [{
            "type": _SINGING_TRACK_TYPE, "name": "vocal",
            "parts": [{"name": "p", "pos": 0, "duration": 1920, "notes": [
                _seq_note(0, 240, number=60, lyric="ら", phoneme="4 a", velocity=64),
                _seq_note(480, 360, number=62, lyric="り", phoneme="4 i", velocity=100),
            ]}],
        }],
        tempo_events=[{"pos": 0, "value": 12000}],
    ))
    project, _warnings = vpr_read(vpr)

    assert [(t.tick, t.bpm) for t in project.tempos] == [(0, 120.0)]

    track = vio.select_track(project, None)
    notes = vio.collect_notes(track)
    extracted = [
        (n.start_tick, n.duration_tick, n.pitch, n.lyric, n.velocity, n.phonemes)
        for n in notes
    ]
    assert extracted == [
        (0, 240, 60, "ら", 64, ["4", "a"]),
        (480, 360, 62, "り", 100, ["4", "i"]),
    ]

    end_tick = max(n.start_tick + n.duration_tick for n in notes)
    assert rest_intervals(notes, end_tick) == [(240, 480)]


def test_non_vpr_bytes_raise_vpr_format_error():
    from vpr import VprFormatError

    with pytest.raises(VprFormatError):
        vpr_read(b"not a zip")
