import io
import json
import math
import zipfile

import pytest


def _make_vpr(sequence: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Project/sequence.json", json.dumps(sequence, ensure_ascii=False))
    return buf.getvalue()


def _sequence(tracks, tempo_events=None, timesig_events=None) -> dict:
    return {
        "version": {"major": 6, "minor": 5, "revision": 1},
        "vender": "Yamaha Corporation",
        "title": "test",
        "masterTrack": {
            "samplingRate": 44100,
            "tempo": {"events": tempo_events if tempo_events is not None else [{"pos": 0, "value": 12000}]},
            "timeSig": {
                "events": timesig_events if timesig_events is not None
                else [{"bar": 0, "numer": 4, "denom": 4}]
            },
        },
        "voices": [],
        "tracks": tracks,
    }


def _singing_track(notes, name="vocal", part_pos=0, part_duration=1920):
    return {
        "type": 2,
        "name": name,
        "parts": [{"name": "p", "pos": part_pos, "duration": part_duration, "notes": notes}],
    }


def _note(pos, duration, number, lyric, phoneme, velocity):
    return {
        "pos": pos,
        "duration": duration,
        "number": number,
        "lyric": lyric,
        "phoneme": phoneme,
        "velocity": velocity,
    }


def test_read_returns_project_and_warning_list():
    from vpr import VprProject, read

    project, warnings = read(_make_vpr(_sequence([_singing_track([])])))
    assert isinstance(project, VprProject)
    assert isinstance(warnings, list)


def test_read_resolution_is_480():
    from vpr import read

    project, _ = read(_make_vpr(_sequence([_singing_track([])])))
    assert project.resolution == 480


def test_read_tempo_value_is_bpm_times_100():
    from vpr import read

    project, _ = read(_make_vpr(_sequence([_singing_track([])], tempo_events=[{"pos": 0, "value": 13600}])))
    assert len(project.tempos) == 1
    assert project.tempos[0].tick == 0
    assert project.tempos[0].bpm == pytest.approx(136.0)


def test_read_maps_multiple_tempo_events():
    from vpr import read

    events = [{"pos": 0, "value": 12000}, {"pos": 1920, "value": 14400}]
    project, _ = read(_make_vpr(_sequence([_singing_track([])], tempo_events=events)))
    assert [t.tick for t in project.tempos] == [0, 1920]
    assert project.tempos[0].bpm == pytest.approx(120.0)
    assert project.tempos[1].bpm == pytest.approx(144.0)


def test_read_timesig_bar_zero_maps_to_tick_zero():
    from vpr import read

    project, _ = read(_make_vpr(_sequence([_singing_track([])], timesig_events=[{"bar": 0, "numer": 3, "denom": 4}])))
    assert len(project.time_signatures) == 1
    ts = project.time_signatures[0]
    assert ts.tick == 0
    assert ts.numerator == 3
    assert ts.denominator == 4


def test_read_timesig_nonzero_bar_maps_to_tick():
    from vpr import read

    events = [{"bar": 0, "numer": 4, "denom": 4}, {"bar": 1, "numer": 3, "denom": 4}]
    project, _ = read(_make_vpr(_sequence([_singing_track([])], timesig_events=events)))
    mapped = [(t.tick, t.numerator, t.denominator) for t in project.time_signatures]
    assert mapped == [(0, 4, 4), (1920, 3, 4)]


def test_read_timesig_first_event_after_bar_zero_uses_default_4_4():
    from vpr import read

    events = [{"bar": 2, "numer": 4, "denom": 4}]
    project, _ = read(_make_vpr(_sequence([_singing_track([])], timesig_events=events)))
    assert [(t.tick, t.numerator, t.denominator) for t in project.time_signatures] == [(3840, 4, 4)]


def test_read_timesig_accumulates_nonuniform_bar_lengths():
    from vpr import read

    events = [{"bar": 0, "numer": 3, "denom": 4}, {"bar": 2, "numer": 4, "denom": 4}]
    project, _ = read(_make_vpr(_sequence([_singing_track([])], timesig_events=events)))
    mapped = [(t.tick, t.numerator, t.denominator) for t in project.time_signatures]
    assert mapped == [(0, 3, 4), (2880, 4, 4)]


def test_read_maps_note_fields_and_splits_phonemes():
    from vpr import read

    notes = [_note(pos=480, duration=240, number=60, lyric="ら", phoneme="r a", velocity=64)]
    project, _ = read(_make_vpr(_sequence([_singing_track(notes)])))
    track = project.tracks[0]
    assert track.name == "vocal"
    assert track.parts[0].name == "p"
    note = track.parts[0].notes[0]
    assert note.duration_tick == 240
    assert note.pitch == 60
    assert note.lyric == "ら"
    assert note.velocity == 64
    assert note.phonemes == ["r", "a"]


def test_read_absolutizes_note_start_with_part_pos():
    from vpr import read

    notes = [_note(pos=480, duration=240, number=60, lyric="ら", phoneme="r a", velocity=64)]
    track = _singing_track(notes, part_pos=1920, part_duration=2400)
    project, _ = read(_make_vpr(_sequence([track])))
    part = project.tracks[0].parts[0]
    assert part.start_tick == 1920
    assert part.notes[0].start_tick == 2400


def test_read_empty_phoneme_yields_empty_list():
    from vpr import read

    notes = [_note(pos=0, duration=240, number=60, lyric="あ", phoneme="", velocity=64)]
    project, _ = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert project.tracks[0].parts[0].notes[0].phonemes == []


def test_read_splits_multi_token_phoneme():
    from vpr import read

    notes = [_note(pos=0, duration=480, number=62, lyric="Tell", phoneme="t th e l", velocity=64)]
    project, _ = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert project.tracks[0].parts[0].notes[0].phonemes == ["t", "th", "e", "l"]


@pytest.mark.parametrize(("stored", "expected"), [(True, True), (False, False)])
def test_read_maps_is_protected(stored, expected):
    from vpr import read

    note = _note(pos=0, duration=240, number=60, lyric="か", phoneme="k a", velocity=64)
    note["isProtected"] = stored
    project, _ = read(_make_vpr(_sequence([_singing_track([note])])))
    assert project.tracks[0].parts[0].notes[0].is_protected is expected


def test_read_missing_is_protected_is_false():
    from vpr import read

    notes = [_note(pos=0, duration=240, number=60, lyric="あ", phoneme="a", velocity=64)]
    project, _ = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert project.tracks[0].parts[0].notes[0].is_protected is False


@pytest.mark.parametrize("stored", ["true", 1, None, {}])
def test_read_non_boolean_is_protected_is_false_without_error(stored):
    from vpr import read

    note = _note(pos=0, duration=240, number=60, lyric="あ", phoneme="a", velocity=64)
    note["isProtected"] = stored
    project, warnings = read(_make_vpr(_sequence([_singing_track([note])])))
    assert project.tracks[0].parts[0].notes[0].is_protected is False
    assert [w.code for w in warnings] == []


def test_read_notes_sorted_by_start_tick():
    from vpr import read

    notes = [
        _note(pos=960, duration=240, number=62, lyric="そ", phoneme="s o", velocity=64),
        _note(pos=480, duration=240, number=60, lyric="ら", phoneme="r a", velocity=64),
    ]
    project, _ = read(_make_vpr(_sequence([_singing_track(notes)])))
    starts = [n.start_tick for n in project.tracks[0].parts[0].notes]
    assert starts == [480, 960]


def test_read_excludes_audio_track():
    from vpr import read

    audio_track = {"type": 1, "name": "audio", "parts": [{"name": "a", "pos": 0, "wav": {}, "region": {}}]}
    seq = _sequence([_singing_track([]), audio_track])
    project, _ = read(_make_vpr(seq))
    assert [t.name for t in project.tracks] == ["vocal"]


def test_read_extracts_all_singing_tracks():
    from vpr import read

    seq = _sequence([_singing_track([], name="vocal1"), _singing_track([], name="vocal2")])
    project, _ = read(_make_vpr(seq))
    assert [t.name for t in project.tracks] == ["vocal1", "vocal2"]


def test_read_extracts_all_parts_in_track():
    from vpr import read

    track = {
        "type": 2,
        "name": "vocal",
        "parts": [
            {"name": "p1", "pos": 0, "duration": 1920, "notes": []},
            {"name": "p2", "pos": 1920, "duration": 1920, "notes": []},
        ],
    }
    project, _ = read(_make_vpr(_sequence([track])))
    assert [p.name for p in project.tracks[0].parts] == ["p1", "p2"]
    assert [p.start_tick for p in project.tracks[0].parts] == [0, 1920]


def test_read_accepts_bytes_str_and_path(tmp_path):
    from vpr import read

    data = _make_vpr(_sequence([_singing_track([])]))
    from_bytes, _ = read(data)
    path = tmp_path / "sample.vpr"
    path.write_bytes(data)
    from_path, _ = read(path)
    from_str, _ = read(str(path))
    assert from_bytes.resolution == from_path.resolution == from_str.resolution == 480


def _zip_with(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in entries.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_read_raises_format_error_on_non_zip():
    from vpr import VprFormatError, read

    with pytest.raises(VprFormatError):
        read(b"this is not a zip archive")


def test_read_raises_format_error_on_missing_sequence_json():
    from vpr import VprFormatError, read

    with pytest.raises(VprFormatError):
        read(_zip_with({"Project/other.txt": "x"}))


def test_read_raises_format_error_on_invalid_json():
    from vpr import VprFormatError, read

    with pytest.raises(VprFormatError):
        read(_zip_with({"Project/sequence.json": "{ not valid json "}))


def test_read_raises_format_error_on_invalid_utf8_sequence_json():
    from vpr import VprFormatError, read

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Project/sequence.json", b"\x80\x81\x82\xff")
    with pytest.raises(VprFormatError):
        read(buf.getvalue())


def test_read_raises_format_error_on_missing_master_track():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])])
    del seq["masterTrack"]
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_missing_tracks():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])])
    del seq["tracks"]
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_malformed_tempo_event():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])], tempo_events=[{"pos": 0}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_tempo_event_type_error():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])], tempo_events=[{"pos": 0, "value": "fast"}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_malformed_timesig_event():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])], timesig_events=[{"bar": 0, "denom": 4}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_timesig_event_type_error():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])], timesig_events=[{"bar": 0, "numer": 4, "denom": "x"}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_missing_required_note_field():
    from vpr import VprFormatError, read

    note = _note(pos=0, duration=240, number=60, lyric="あ", phoneme="a", velocity=64)
    del note["number"]
    with pytest.raises(VprFormatError):
        read(_make_vpr(_sequence([_singing_track([note])])))


def test_read_format_error_carries_locator():
    from vpr import VprFormatError, read

    note = _note(pos=0, duration=240, number=60, lyric="あ", phoneme="a", velocity=64)
    del note["number"]
    with pytest.raises(VprFormatError) as exc:
        read(_make_vpr(_sequence([_singing_track([note])])))
    assert exc.value.path is not None
    assert exc.value.key == "number"


def test_read_raises_format_error_on_missing_track_name():
    from vpr import VprFormatError, read

    track = {"type": 2, "parts": [{"name": "p", "pos": 0, "duration": 1920, "notes": []}]}
    with pytest.raises(VprFormatError):
        read(_make_vpr(_sequence([track])))


def test_read_raises_format_error_on_missing_part_pos():
    from vpr import VprFormatError, read

    track = {"type": 2, "name": "vocal", "parts": [{"name": "p", "duration": 1920, "notes": []}]}
    with pytest.raises(VprFormatError):
        read(_make_vpr(_sequence([track])))


def test_read_raises_format_error_on_note_field_type_error():
    from vpr import VprFormatError, read

    note = _note(pos=0, duration=240, number="x", lyric="あ", phoneme="a", velocity=64)
    with pytest.raises(VprFormatError) as exc:
        read(_make_vpr(_sequence([_singing_track([note])])))
    assert exc.value.key == "number"


def test_read_raises_format_error_on_bool_as_int_field():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])], timesig_events=[{"bar": 0, "numer": 4, "denom": False}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_tempo_pos_type_error():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])], tempo_events=[{"pos": "zero", "value": 12000}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_non_list_tempo_events():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])])
    seq["masterTrack"]["tempo"]["events"] = None
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_non_list_parts():
    from vpr import VprFormatError, read

    track = {"type": 2, "name": "vocal", "parts": None}
    with pytest.raises(VprFormatError):
        read(_make_vpr(_sequence([track])))


def test_read_tolerates_missing_notes_as_empty_part():
    from vpr import read

    track = {"type": 2, "name": "vocal", "parts": [{"name": "p", "pos": 0, "duration": 1920}]}
    project, _ = read(_make_vpr(_sequence([track])))
    assert project.tracks[0].parts[0].notes == []


def test_read_tolerates_unknown_top_level_keys():
    from vpr import read

    seq = _sequence([_singing_track([])])
    seq["someUnknownKey"] = {"x": 1}
    project, _ = read(_make_vpr(seq))
    assert project.resolution == 480


def test_read_warns_on_overlapping_notes_in_part():
    from vpr import read

    notes = [
        _note(pos=0, duration=480, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=240, duration=480, number=62, lyric="い", phoneme="i", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert [w.code for w in warnings] == ["overlapping_notes"]


def test_read_overlap_warning_locates_the_pair_and_the_overlap_start():
    from vpr import read

    notes = [
        _note(pos=0, duration=480, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=240, duration=480, number=62, lyric="い", phoneme="i", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    w = next(w for w in warnings if w.code == "overlapping_notes")
    assert w.track_index == 0
    assert w.part_index == 0
    assert w.note_index == 0
    assert w.related_note_index == 1
    assert w.tick == 240


def test_read_warns_once_per_overlapping_pair():
    from vpr import read

    notes = [
        _note(pos=0, duration=480, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=240, duration=260, number=62, lyric="い", phoneme="i", velocity=64),
        _note(pos=1000, duration=480, number=64, lyric="う", phoneme="u", velocity=64),
        _note(pos=1200, duration=300, number=65, lyric="え", phoneme="e", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert [w.code for w in warnings] == ["overlapping_notes", "overlapping_notes"]


def test_read_warns_for_all_overlapping_pairs():
    from vpr import read

    notes = [
        _note(pos=0, duration=1000, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=240, duration=240, number=62, lyric="い", phoneme="i", velocity=64),
        _note(pos=300, duration=300, number=64, lyric="う", phoneme="u", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    pairs = [(w.note_index, w.related_note_index) for w in warnings if w.code == "overlapping_notes"]
    assert pairs == [(0, 1), (0, 2), (1, 2)]


def test_read_no_warning_for_non_overlapping_notes():
    from vpr import read

    notes = [
        _note(pos=0, duration=240, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=480, duration=240, number=62, lyric="い", phoneme="i", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert warnings == []


def test_read_no_warning_for_adjacent_notes():
    from vpr import read

    notes = [
        _note(pos=0, duration=480, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=480, duration=480, number=62, lyric="い", phoneme="i", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert warnings == []


def test_read_no_overlap_warning_across_parts():
    from vpr import read

    track = {
        "type": 2,
        "name": "vocal",
        "parts": [
            {"name": "p1", "pos": 0, "duration": 1920, "notes": [_note(0, 960, 60, "あ", "a", 64)]},
            {"name": "p2", "pos": 480, "duration": 1920, "notes": [_note(0, 960, 62, "い", "i", 64)]},
        ],
    }
    _, warnings = read(_make_vpr(_sequence([track])))
    assert warnings == []


def test_read_retains_raw_sequence():
    from vpr import read

    seq = _sequence([_singing_track([])])
    project, _ = read(_make_vpr(seq))
    assert project.raw_sequence == seq


def test_raw_sequence_retains_uninterpreted_data():
    from vpr import read

    seq = _sequence([_singing_track([])])
    seq["customField"] = {"keep": [1, 2, 3]}
    project, _ = read(_make_vpr(seq))
    assert project.raw_sequence["customField"] == {"keep": [1, 2, 3]}


def test_raw_sequence_retains_audio_track_excluded_from_model():
    from vpr import read

    audio = {"type": 1, "name": "audio", "parts": [{"name": "a", "pos": 0, "wav": {}, "region": {}}]}
    seq = _sequence([_singing_track([], name="vocal"), audio])
    project, _ = read(_make_vpr(seq))
    assert [t.name for t in project.tracks] == ["vocal"]
    assert project.raw_sequence["tracks"][1]["name"] == "audio"


def test_vprproject_raw_sequence_defaults_to_none():
    from vpr import VprProject

    project = VprProject(resolution=480)
    assert project.raw_sequence is None


def _singing_track_with_controllers(notes, controllers, part_pos=0, part_duration=1920):
    return {
        "type": 2,
        "name": "vocal",
        "parts": [
            {
                "name": "p",
                "pos": part_pos,
                "duration": part_duration,
                "notes": notes,
                "controllers": controllers,
            }
        ],
    }


def test_controllers_extracted_with_absolute_tick():
    from vpr import read

    controllers = [{"name": "dynamics", "events": [{"pos": 100, "value": 64}, {"pos": 300, "value": 90}]}]
    seq = _sequence([_singing_track_with_controllers([], controllers, part_pos=480)])
    project, _ = read(_make_vpr(seq))
    curves = project.tracks[0].parts[0].controllers
    assert len(curves) == 1
    assert curves[0].name == "dynamics"
    assert [(e.tick, e.value) for e in curves[0].events] == [(580, 64), (780, 90)]


def test_controllers_default_empty_when_absent():
    from vpr import read

    project, _ = read(_make_vpr(_sequence([_singing_track([])])))
    assert project.tracks[0].parts[0].controllers == []


def test_controller_events_sorted_by_tick():
    from vpr import read

    controllers = [{"name": "dynamics", "events": [{"pos": 300, "value": 90}, {"pos": 100, "value": 64}]}]
    project, _ = read(_make_vpr(_sequence([_singing_track_with_controllers([], controllers)])))
    ticks = [e.tick for e in project.tracks[0].parts[0].controllers[0].events]
    assert ticks == sorted(ticks)


def test_multiple_controllers_all_preserved_in_order():
    from vpr import read

    controllers = [
        {"name": "dynamics", "events": [{"pos": 0, "value": 64}]},
        {"name": "s5Expression", "events": [{"pos": 0, "value": 30}]},
        {"name": "brightness", "events": [{"pos": 0, "value": 64}]},
    ]
    project, _ = read(_make_vpr(_sequence([_singing_track_with_controllers([], controllers)])))
    names = [c.name for c in project.tracks[0].parts[0].controllers]
    assert names == ["dynamics", "s5Expression", "brightness"]


def test_controller_event_missing_value_is_format_error():
    from vpr import VprFormatError, read

    controllers = [{"name": "dynamics", "events": [{"pos": 0}]}]
    with pytest.raises(VprFormatError):
        read(_make_vpr(_sequence([_singing_track_with_controllers([], controllers)])))


def test_read_raises_format_error_on_empty_tempo_events():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])], tempo_events=[])
    with pytest.raises(VprFormatError) as exc:
        read(_make_vpr(seq))
    e = exc.value
    assert e.path == "masterTrack.tempo.events"
    assert e.key == "events"
    assert e.value == []


@pytest.mark.parametrize("raw", [0, -12000, float("nan"), float("inf"), float("-inf")])
def test_read_raises_format_error_on_bpm_that_is_not_positive_and_finite(raw):
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])],
                    tempo_events=[{"pos": 0, "value": 12000}, {"pos": 960, "value": raw}])
    with pytest.raises(VprFormatError) as exc:
        read(_make_vpr(seq))
    e = exc.value
    assert e.path == "masterTrack.tempo.events[1]"
    assert e.key == "value"
    if math.isnan(raw):
        assert math.isnan(e.value)
    else:
        assert e.value == raw


def test_read_accepts_fractional_tempo_value():
    from vpr import read

    seq = _sequence([_singing_track([])], tempo_events=[{"pos": 0, "value": 12050.5}])
    project, _ = read(_make_vpr(seq))
    assert project.tempos[0].bpm == 120.505


def _vibrato_note(vibrato, pos=0, duration=480):
    note = _note(pos, duration, 60, "あ", "a", 64)
    note["vibrato"] = vibrato
    return note


def test_read_maps_vibrato_points_to_absolute_ticks():
    from vpr import read

    seq = _sequence([_singing_track([_vibrato_note(
        {"type": 0, "duration": 240, "depths": [{"pos": 0, "value": 64}],
         "rates": [{"pos": 120, "value": 70}]}, pos=960, duration=480)], part_pos=480)])
    project, _ = read(_make_vpr(seq))
    vibrato = project.tracks[0].parts[0].notes[0].vibrato
    span_start = 480 + 960 + 480 - 240
    assert [(p.pos, p.value) for p in vibrato.depths] == [(span_start, 64)]
    assert [(p.pos, p.value) for p in vibrato.rates] == [(span_start + 120, 70)]


@pytest.mark.parametrize("vibrato", [
    pytest.param({"type": 0, "duration": 0}, id="zero_duration_means_no_vibrato"),
    pytest.param({"type": 0, "duration": -240}, id="negative_duration"),
    pytest.param({"type": 0}, id="missing_duration"),
    pytest.param({"type": 0, "duration": "240"}, id="non_integer_duration"),
    pytest.param({"type": 0, "duration": 240, "depths": [{"pos": 0}]}, id="point_without_value"),
    pytest.param("vibrato", id="not_an_object"),
])
def test_read_maps_unusable_vibrato_to_none(vibrato):
    from vpr import read

    project, _ = read(_make_vpr(_sequence([_singing_track([_vibrato_note(vibrato)])])))
    assert project.tracks[0].parts[0].notes[0].vibrato is None


@pytest.mark.parametrize("expression", [
    pytest.param({"vibratoLeadingDepth": 0.25}, id="only_one_depth"),
    pytest.param({"vibratoLeadingDepth": 0.25, "vibratoFollowingDepth": True}, id="boolean_depth"),
    pytest.param({}, id="no_depths"),
    pytest.param("aiExp", id="not_an_object"),
])
def test_read_maps_unusable_depth_envelope_to_none(expression):
    from vpr import read

    note = _note(0, 480, 60, "あ", "a", 64)
    note["aiExp"] = expression
    project, _ = read(_make_vpr(_sequence([_singing_track([note])])))
    assert project.tracks[0].parts[0].notes[0].ai_expression is None


def test_read_maps_missing_part_duration_and_title_to_defaults():
    from vpr import read

    seq = _sequence([{"type": 2, "name": "vocal", "parts": [{"name": "p", "pos": 0, "notes": []}]}])
    del seq["title"]
    project, _ = read(_make_vpr(seq))
    assert project.title == ""
    assert project.tracks[0].parts[0].duration_tick == 0


def test_read_raises_format_error_on_zero_time_signature_denominator():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])], timesig_events=[{"bar": 0, "numer": 4, "denom": 0}])
    with pytest.raises(VprFormatError) as exc:
        read(_make_vpr(seq))
    assert exc.value.key == "denom"
    assert exc.value.path == "masterTrack.timeSig.events[0]"


@pytest.mark.parametrize("field", ["pos", "duration", "number", "lyric", "phoneme", "velocity"])
def test_read_raises_format_error_on_each_missing_required_note_field(field):
    from vpr import VprFormatError, read

    note = _note(pos=0, duration=240, number=60, lyric="あ", phoneme="a", velocity=64)
    del note[field]
    with pytest.raises(VprFormatError) as exc:
        read(_make_vpr(_sequence([_singing_track([note])])))
    assert exc.value.key == field


def test_read_maps_non_string_title_to_empty():
    from vpr import read

    seq = _sequence([_singing_track([])])
    seq["title"] = 1
    project, _ = read(_make_vpr(seq))
    assert project.title == ""


def test_read_resolves_part_voice_from_voice_definitions():
    from vpr import VoiceBank, read

    seq = _sequence([_singing_track([])])
    seq["voices"] = [{"compID": "TESTCOMPID000001", "name": "TEST_VOICE"}]
    seq["tracks"][0]["parts"][0]["aiVoice"] = {"compID": "TESTCOMPID000001"}
    project, _ = read(_make_vpr(seq))
    assert project.tracks[0].parts[0].voice == VoiceBank(comp_id="TESTCOMPID000001",
                                                         name="TEST_VOICE")


def test_read_maps_voice_reference_without_definition_to_none():
    from vpr import read

    seq = _sequence([_singing_track([])])
    seq["tracks"][0]["parts"][0]["aiVoice"] = {"compID": "TESTCOMPID000001"}
    project, _ = read(_make_vpr(seq))
    assert project.tracks[0].parts[0].voice is None


def test_read_raises_format_error_on_voices_that_is_not_a_list():
    from vpr import VprFormatError, read

    seq = _sequence([_singing_track([])])
    seq["voices"] = 5
    seq["tracks"][0]["parts"][0]["aiVoice"] = {"compID": "TESTCOMPID000001"}
    with pytest.raises(VprFormatError) as exc:
        read(_make_vpr(seq))
    assert exc.value.key == "voices"


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param(None, id="missing"),
        pytest.param("x", id="string"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
    ],
)
def test_read_maps_part_duration_that_is_missing_or_not_an_integer_to_zero(stored):
    from vpr import read

    seq = _sequence([_singing_track([])])
    part = seq["tracks"][0]["parts"][0]
    if stored is None:
        del part["duration"]
    else:
        part["duration"] = stored
    project, _ = read(_make_vpr(seq))
    assert project.tracks[0].parts[0].duration_tick == 0


def test_read_keeps_zip_entries_other_than_sequence_json():
    from vpr import read

    data = _zip_with({"Project/sequence.json": json.dumps(_sequence([_singing_track([])])),
                      "Project/Audio/x.wav": "RIFF"})
    project, _ = read(data)
    assert project.entries == {"Project/Audio/x.wav": b"RIFF"}
