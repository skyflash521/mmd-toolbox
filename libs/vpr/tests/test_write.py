import io
import json
import zipfile

import pytest

from vpr import (
    ControllerCurve,
    ControllerEvent,
    Note,
    NoteAiExpression,
    NoteVibrato,
    Part,
    TempoEvent,
    TimeSignature,
    Track,
    VibratoPoint,
    VoiceBank,
    VprFormatError,
    VprProject,
    read,
    write,
    write_file,
)


def _make_vpr(sequence, entries=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Project/sequence.json", json.dumps(sequence, ensure_ascii=False))
        for name, data in (entries or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


def _sequence(tracks, voices=None, title="test") -> dict:
    return {
        "version": {"major": 6, "minor": 5, "revision": 1},
        "vender": "Yamaha Corporation",
        "title": title,
        "masterTrack": {
            "samplingRate": 44100,
            "tempo": {"events": [{"pos": 0, "value": 12000}]},
            "timeSig": {"events": [{"bar": 0, "numer": 4, "denom": 4}]},
        },
        "voices": voices if voices is not None else [],
        "tracks": tracks,
    }


def _singing_track(notes, name="vocal", part_pos=0, part_duration=1920, comp_id=None):
    part = {"name": "p", "pos": part_pos, "duration": part_duration, "notes": notes}
    if comp_id is not None:
        part["aiVoice"] = {"compID": comp_id, "langIDs": [{"langID": 0}]}
    return {"type": 2, "name": name, "parts": [part]}


def _raw_note(pos, duration, number, lyric, phoneme, velocity):
    return {"pos": pos, "duration": duration, "number": number, "lyric": lyric,
            "phoneme": phoneme, "velocity": velocity}


def _project(notes=None, *, title="song", voice=None, parts=None):
    part = Part(name="p", start_tick=0, duration_tick=1920, voice=voice,
                notes=notes if notes is not None else [])
    return VprProject(
        resolution=480,
        tempos=[TempoEvent(tick=0, bpm=120.0)],
        time_signatures=[TimeSignature(tick=0, numerator=4, denominator=4)],
        tracks=[Track(name="vocal", parts=parts if parts is not None else [part])],
        title=title,
    )


def _note(start=0, duration=480, pitch=60, lyric="あ", velocity=64, **kwargs):
    return Note(start_tick=start, duration_tick=duration, pitch=pitch, lyric=lyric,
                velocity=velocity, **kwargs)


def _read_back(data):
    project, _warnings = read(data)
    return project


def _sequence_of(data) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return json.loads(z.read("Project/sequence.json"))


def test_round_trip_keeps_the_public_model():
    source = _project([_note(0, 480, 60, "あ"), _note(480, 240, 62, "い")])
    result = _read_back(write(source))

    assert result.title == source.title
    assert [(t.tick, t.bpm) for t in result.tempos] == [(0, 120.0)]
    assert [(s.tick, s.numerator, s.denominator) for s in result.time_signatures] == [(0, 4, 4)]
    assert len(result.tracks) == 1
    assert result.tracks[0].name == "vocal"
    part = result.tracks[0].parts[0]
    assert (part.name, part.start_tick, part.duration_tick) == ("p", 0, 1920)
    assert [(n.start_tick, n.duration_tick, n.pitch, n.lyric, n.velocity) for n in part.notes] == \
           [(0, 480, 60, "あ", 64), (480, 240, 62, "い", 64)]


def test_round_trip_keeps_the_phonemes():
    source = _project([_note(phonemes=["k", "a"])])
    assert _read_back(write(source)).tracks[0].parts[0].notes[0].phonemes == ["k", "a"]


def test_round_trip_keeps_an_empty_phoneme_list():
    source = _project([_note(phonemes=[])])
    assert _read_back(write(source)).tracks[0].parts[0].notes[0].phonemes == []


@pytest.mark.parametrize("is_protected", [True, False])
def test_round_trip_keeps_is_protected(is_protected):
    source = _project([_note(is_protected=is_protected)])
    written = _read_back(write(source)).tracks[0].parts[0].notes[0]
    assert written.is_protected is is_protected


@pytest.mark.parametrize("is_protected", [True, False])
def test_is_protected_is_written_as_the_format_field(is_protected):
    written = _sequence_of(write(_project([_note(is_protected=is_protected)])))
    assert written["tracks"][0]["parts"][0]["notes"][0]["isProtected"] is is_protected


def test_is_protected_cleared_by_the_caller_overwrites_the_raw_value():
    raw = _raw_note(0, 480, 60, "か", "k a", 64)
    raw["isProtected"] = True
    project = _read_back(_make_vpr(_sequence([_singing_track([raw])])))
    project.tracks[0].parts[0].notes[0].is_protected = False

    assert _raw_note_of(write(project))["isProtected"] is False


def test_round_trip_keeps_the_controllers():
    part = Part(name="p", start_tick=0, duration_tick=1920, notes=[_note()],
                controllers=[ControllerCurve(name="dynamics",
                                             events=[ControllerEvent(tick=0, value=64),
                                                     ControllerEvent(tick=240, value=100)])])
    result = _read_back(write(_project(parts=[part])))
    curve = result.tracks[0].parts[0].controllers[0]
    assert curve.name == "dynamics"
    assert [(e.tick, e.value) for e in curve.events] == [(0, 64), (240, 100)]


@pytest.mark.parametrize("vibrato", [
    pytest.param(None, id="none"),
    pytest.param(NoteVibrato(type=0, duration=240), id="without_curves"),
    pytest.param(NoteVibrato(type=0, duration=240, depths=[VibratoPoint(pos=240, value=64)],
                             rates=[VibratoPoint(pos=240, value=70)]), id="with_curves"),
])
def test_round_trip_keeps_the_vibrato(vibrato):
    result = _read_back(write(_project([_note(vibrato=vibrato)])))
    assert result.tracks[0].parts[0].notes[0].vibrato == vibrato


def test_vibrato_points_are_written_relative_to_the_span_start():
    note = _note(960, 480, vibrato=NoteVibrato(
        type=0, duration=240, depths=[VibratoPoint(pos=1200, value=64)],
        rates=[VibratoPoint(pos=1320, value=70)]))
    raw = _sequence_of(write(_project([note])))["tracks"][0]["parts"][0]["notes"][0]["vibrato"]
    assert [p["pos"] for p in raw["depths"]] == [0]
    assert [p["pos"] for p in raw["rates"]] == [120]


@pytest.mark.parametrize("expression", [
    None,
    NoteAiExpression(vibrato_leading_depth=0.25, vibrato_following_depth=0.75),
])
def test_round_trip_keeps_the_depth_envelope(expression):
    result = _read_back(write(_project([_note(ai_expression=expression)])))
    assert result.tracks[0].parts[0].notes[0].ai_expression == expression


def test_round_trip_keeps_the_voice_bank():
    voice = VoiceBank(comp_id="TESTCOMPID000001", name="TEST_VOICE")
    assert _read_back(write(_project([_note()], voice=voice))).tracks[0].parts[0].voice == voice


def test_unparsed_keys_survive_a_round_trip():
    sequence = _sequence([_singing_track([_raw_note(0, 480, 60, "あ", "a", 64)])])
    sequence["masterTrack"]["loop"] = {"isEnabled": False, "begin": 0, "end": 7680}
    sequence["tracks"][0]["color"] = 7

    written = write(_read_back(_make_vpr(sequence)))
    result = _sequence_of(written)
    assert result["masterTrack"]["loop"] == {"isEnabled": False, "begin": 0, "end": 7680}
    assert result["tracks"][0]["color"] == 7


def test_audio_tracks_survive_a_round_trip():
    audio = {"type": 1, "name": "audio",
             "parts": [{"name": "a", "pos": 0, "wav": {"name": "x.wav"}, "region": {}}]}
    sequence = _sequence([audio, _singing_track([_raw_note(0, 480, 60, "あ", "a", 64)])])

    written = write(_read_back(_make_vpr(sequence)))
    result = _sequence_of(written)
    assert [t["type"] for t in result["tracks"]] == [1, 2]
    assert result["tracks"][0]["parts"][0]["wav"] == {"name": "x.wav"}


def test_other_zip_entries_survive_a_round_trip():
    sequence = _sequence([_singing_track([])])
    written = write(_read_back(_make_vpr(sequence, entries={"Project/Audio/x.wav": b"RIFF...."})))
    with zipfile.ZipFile(io.BytesIO(written)) as z:
        assert z.read("Project/Audio/x.wav") == b"RIFF...."


def test_note_added_to_a_read_project_is_written():
    sequence = _sequence([_singing_track([_raw_note(0, 480, 60, "あ", "a", 64)])])
    project = _read_back(_make_vpr(sequence))
    project.tracks[0].parts[0].notes.append(_note(480, 240, 62, "い"))

    result = _read_back(write(project))
    assert [(n.start_tick, n.pitch) for n in result.tracks[0].parts[0].notes] == [(0, 60), (480, 62)]


def test_note_removed_from_a_read_project_is_dropped():
    sequence = _sequence([_singing_track([_raw_note(0, 480, 60, "あ", "a", 64),
                                          _raw_note(480, 240, 62, "い", "i", 64)])])
    project = _read_back(_make_vpr(sequence))
    del project.tracks[0].parts[0].notes[1]

    result = _read_back(write(project))
    assert [(n.start_tick, n.pitch) for n in result.tracks[0].parts[0].notes] == [(0, 60)]


def test_voice_bank_definition_is_added_when_missing():
    voice = VoiceBank(comp_id="TESTCOMPID000001", name="TEST_VOICE")
    written = write(_project([_note()], voice=voice))
    result = _sequence_of(written)
    assert {"compID": "TESTCOMPID000001", "name": "TEST_VOICE"} in result["voices"]
    assert result["tracks"][0]["parts"][0]["aiVoice"]["compID"] == "TESTCOMPID000001"


def test_raw_voices_that_is_not_a_list_is_replaced_by_the_used_definitions():
    voice = VoiceBank(comp_id="TESTCOMPID000001", name="TEST_VOICE")
    project = _project([_note()], voice=voice)
    project.raw_sequence = {"voices": 5}
    result = _sequence_of(write(project))
    assert result["voices"] == [{"compID": "TESTCOMPID000001", "name": "TEST_VOICE"}]


def test_existing_voice_bank_definition_is_not_replaced():
    voices = [{"compID": "TESTCOMPID000001", "name": "ORIGINAL_NAME", "extra": 1}]
    sequence = _sequence([_singing_track([], comp_id="TESTCOMPID000001")], voices=voices)
    project = _read_back(_make_vpr(sequence))

    result = _sequence_of(write(project))
    assert result["voices"] == voices


def test_every_part_voice_reference_resolves_when_a_singing_track_is_added():
    audio = {"type": 1, "name": "audio", "parts": [{"name": "a", "pos": 0, "wav": {}, "region": {}}]}
    sequence = _sequence([audio])
    project = _read_back(_make_vpr(sequence))
    project.tracks.append(Track(name="vocal", parts=[
        Part(name="p", start_tick=0, duration_tick=1920,
             voice=VoiceBank(comp_id="TESTCOMPID000001", name="TEST_VOICE"), notes=[_note()])]))

    result = _sequence_of(write(project))
    defined = {v["compID"] for v in result["voices"]}
    for track in result["tracks"]:
        for part in track["parts"]:
            if "aiVoice" in part:
                assert part["aiVoice"]["compID"] in defined


def _raw_note_of(data):
    return _sequence_of(data)["tracks"][0]["parts"][0]["notes"][0]


def test_vibrato_cleared_by_the_caller_is_written_as_a_zero_length_span_keeping_unparsed_keys():
    raw = _raw_note(0, 480, 60, "あ", "a", 64)
    raw["vibrato"] = {"type": 0, "duration": 240, "depths": [{"pos": 0, "value": 64}], "hint": 1}
    project = _read_back(_make_vpr(_sequence([_singing_track([raw])])))
    project.tracks[0].parts[0].notes[0].vibrato = None

    written = _raw_note_of(write(project))["vibrato"]
    assert written["duration"] == 0
    assert written["hint"] == 1
    assert _read_back(write(project)).tracks[0].parts[0].notes[0].vibrato is None


@pytest.mark.parametrize("vibrato", [{"type": 0}, {"type": 0, "duration": "240"}])
def test_vibrato_the_reader_could_not_map_is_left_untouched(vibrato):
    raw = _raw_note(0, 480, 60, "あ", "a", 64)
    raw["vibrato"] = vibrato
    project = _read_back(_make_vpr(_sequence([_singing_track([raw])])))
    assert project.tracks[0].parts[0].notes[0].vibrato is None
    assert _raw_note_of(write(project))["vibrato"] == vibrato


def test_depth_envelope_with_only_one_depth_is_left_untouched():
    raw = _raw_note(0, 480, 60, "あ", "a", 64)
    raw["aiExp"] = {"vibratoLeadingDepth": 0.25, "pitchFine": 3}
    project = _read_back(_make_vpr(_sequence([_singing_track([raw])])))
    assert project.tracks[0].parts[0].notes[0].ai_expression is None
    assert _raw_note_of(write(project))["aiExp"] == {"vibratoLeadingDepth": 0.25, "pitchFine": 3}


def test_depth_envelope_cleared_by_the_caller_is_removed():
    raw = _raw_note(0, 480, 60, "あ", "a", 64)
    raw["aiExp"] = {"vibratoLeadingDepth": 0.25, "vibratoFollowingDepth": 0.75, "pitchFine": 3}
    project = _read_back(_make_vpr(_sequence([_singing_track([raw])])))
    project.tracks[0].parts[0].notes[0].ai_expression = None
    assert _raw_note_of(write(project))["aiExp"] == {"pitchFine": 3}


@pytest.mark.parametrize("note", [
    pytest.param(_note(pitch=-1), id="pitch_below_midi"),
    pytest.param(_note(pitch=128), id="pitch_above_midi"),
    pytest.param(_note(duration=0), id="zero_duration"),
    pytest.param(_note(duration=-1), id="negative_duration"),
    pytest.param(_note(velocity=-1), id="velocity_below_0"),
    pytest.param(_note(velocity=128), id="velocity_above_127"),
    pytest.param(_note(phonemes=["k a"]), id="phoneme_with_space"),
    pytest.param(_note(phonemes=[""]), id="empty_phoneme"),
    pytest.param(_note(vibrato=NoteVibrato(type=0, duration=0)), id="zero_vibrato_duration"),
    pytest.param(_note(duration=480, vibrato=NoteVibrato(type=0, duration=481)),
                 id="vibrato_longer_than_note"),
    pytest.param(_note(duration=480, vibrato=NoteVibrato(
        type=0, duration=240, depths=[VibratoPoint(pos=120, value=64)])),
                 id="depth_point_before_vibrato_span"),
    pytest.param(_note(duration=480, vibrato=NoteVibrato(
        type=0, duration=240, rates=[VibratoPoint(pos=481, value=64)])),
                 id="rate_point_after_vibrato_span"),
])
def test_values_the_format_cannot_hold_are_rejected(note):
    with pytest.raises(VprFormatError):
        write(_project([note]))


def test_resolution_the_format_cannot_hold_is_rejected():
    project = _project([_note()])
    project.resolution = 960
    with pytest.raises(VprFormatError):
        write(project)


@pytest.mark.parametrize("signature", [
    pytest.param(TimeSignature(tick=0, numerator=0, denominator=4), id="zero_numerator"),
    pytest.param(TimeSignature(tick=0, numerator=4, denominator=0), id="zero_denominator"),
])
def test_time_signature_without_a_bar_length_is_rejected(signature):
    project = _project([_note()])
    project.time_signatures = [signature]
    with pytest.raises(VprFormatError):
        write(project)


def test_time_signature_with_an_unusual_denominator_is_written():
    project = _project([_note()])
    project.time_signatures = [TimeSignature(tick=0, numerator=5, denominator=6)]
    assert _read_back(write(project)).time_signatures[0].denominator == 6


def test_tempo_that_the_format_cannot_hold_is_rejected():
    project = _project([_note()])
    project.tempos = [TempoEvent(tick=0, bpm=0.0)]
    with pytest.raises(VprFormatError):
        write(project)


def test_time_signature_off_a_bar_boundary_is_rejected():
    project = _project([_note()])
    project.time_signatures = [TimeSignature(tick=0, numerator=4, denominator=4),
                               TimeSignature(tick=100, numerator=3, denominator=4)]
    with pytest.raises(VprFormatError):
        write(project)


def test_tempo_is_rounded_to_the_storable_grain():
    project = _project([_note()])
    project.tempos = [TempoEvent(tick=0, bpm=120.005)]
    result = _sequence_of(write(project))
    value = result["masterTrack"]["tempo"]["events"][0]["value"]
    assert value == round(value)


def test_notes_are_written_in_position_order():
    project = _project([_note(480, 240, 62, "い"), _note(0, 480, 60, "あ")])
    result = _sequence_of(write(project))
    assert [n["pos"] for n in result["tracks"][0]["parts"][0]["notes"]] == [0, 480]


def test_tempos_are_written_in_position_order():
    project = _project([_note()])
    project.tempos = [TempoEvent(tick=960, bpm=140.0), TempoEvent(tick=0, bpm=120.0)]
    result = _sequence_of(write(project))
    assert [e["pos"] for e in result["masterTrack"]["tempo"]["events"]] == [0, 960]


def test_write_file_produces_a_readable_vpr(tmp_path):
    path = tmp_path / "out.vpr"
    write_file(_project([_note()]), path)
    assert _read_back(path.read_bytes()).tracks[0].parts[0].notes[0].pitch == 60


def test_write_file_rejected_by_validation_keeps_the_existing_output_and_no_temp_file(tmp_path):
    path = tmp_path / "out.vpr"
    write_file(_project([_note()]), path)
    original = path.read_bytes()

    broken = _project([_note(pitch=200)])
    with pytest.raises(VprFormatError):
        write_file(broken, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_same_model_gives_the_same_sequence_json():
    source = _project([_note(0, 480, 60, "あ"), _note(480, 240, 62, "い")])
    assert _sequence_of(write(source)) == _sequence_of(write(source))


def test_project_without_tempos_is_rejected():
    project = _project([_note()])
    project.tempos = []
    with pytest.raises(VprFormatError):
        write(project)


def test_project_without_time_signatures_is_rejected():
    project = _project([_note()])
    project.time_signatures = []
    with pytest.raises(VprFormatError):
        write(project)


def test_time_signatures_are_written_in_position_order_as_bars():
    project = _project([_note()])
    project.time_signatures = [TimeSignature(tick=1920, numerator=3, denominator=4),
                               TimeSignature(tick=0, numerator=4, denominator=4)]
    events = _sequence_of(write(project))["masterTrack"]["timeSig"]["events"]
    assert [(e["bar"], e["numer"], e["denom"]) for e in events] == [(0, 4, 4), (1, 3, 4)]


def test_controller_events_are_written_in_position_order():
    part = Part(name="p", start_tick=0, duration_tick=1920, notes=[_note()],
                controllers=[ControllerCurve(name="dynamics",
                                             events=[ControllerEvent(tick=240, value=100),
                                                     ControllerEvent(tick=0, value=64)])])
    raw = _sequence_of(write(_project(parts=[part])))["tracks"][0]["parts"][0]["controllers"][0]
    assert [e["pos"] for e in raw["events"]] == [0, 240]


def test_vibrato_without_curves_is_written_without_curve_keys():
    note = _note(vibrato=NoteVibrato(type=0, duration=240))
    written = _raw_note_of(write(_project([note])))["vibrato"]
    assert "depths" not in written
    assert "rates" not in written


def test_empty_curve_keys_in_the_raw_vibrato_survive_a_round_trip():
    raw = _raw_note(0, 480, 60, "あ", "a", 64)
    raw["vibrato"] = {"type": 0, "duration": 240, "depths": [], "rates": []}
    project = _read_back(_make_vpr(_sequence([_singing_track([raw])])))
    written = _raw_note_of(write(project))["vibrato"]
    assert written["depths"] == []
    assert written["rates"] == []


def test_added_singing_track_is_placed_after_the_last_singing_track():
    audio = {"type": 1, "name": "audio", "parts": []}
    sequence = _sequence([_singing_track([], name="vocal1"), audio])
    project = _read_back(_make_vpr(sequence))
    project.tracks.append(Track(name="vocal2"))

    result = _sequence_of(write(project))
    assert [(t["type"], t["name"]) for t in result["tracks"]] == [
        (2, "vocal1"),
        (2, "vocal2"),
        (1, "audio"),
    ]


def test_fewer_singing_tracks_drop_the_trailing_raw_singing_tracks():
    audio = {"type": 1, "name": "audio", "parts": []}
    sequence = _sequence([_singing_track([], name="vocal1"), audio,
                          _singing_track([], name="vocal2")])
    sequence["tracks"][0]["color"] = 7
    project = _read_back(_make_vpr(sequence))
    del project.tracks[0]

    result = _sequence_of(write(project))
    assert [(t["type"], t["name"]) for t in result["tracks"]] == [(2, "vocal2"), (1, "audio")]
    assert result["tracks"][0]["color"] == 7
