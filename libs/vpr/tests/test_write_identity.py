import io
import json
import zipfile

from vpr import Note, Part, Track, read, write

from .test_write import _make_vpr, _raw_note, _sequence, _singing_track


def _raw_with(pos, number, **extra):
    raw = _raw_note(pos, 240, number, "あ", "a", 64)
    raw.update(extra)
    return raw


def _read_write(raw_notes, mutate=None):
    project, _ = read(_make_vpr(_sequence([_singing_track(raw_notes)])))
    if mutate is not None:
        mutate(project)
    data = write(project)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return json.loads(archive.read("Project/sequence.json"))


def _written_notes(sequence):
    return sequence["tracks"][0]["parts"][0]["notes"]


def test_unknown_note_keys_stay_with_their_note_when_raw_notes_are_not_ascending():
    raw_notes = [
        _raw_with(480, 62, exp={"opening": 100}),
        _raw_with(0, 60, phonemePositions=[{"pos": 1}]),
        _raw_with(240, 61, extra="mid"),
    ]

    written = _written_notes(_read_write(raw_notes))

    by_pos = {note["pos"]: note for note in written}
    assert by_pos[480]["exp"] == {"opening": 100}
    assert "phonemePositions" not in by_pos[480]
    assert by_pos[0]["phonemePositions"] == [{"pos": 1}]
    assert "exp" not in by_pos[0]
    assert by_pos[240]["extra"] == "mid"


def test_inserted_note_does_not_inherit_unknown_keys_and_later_notes_keep_theirs():
    raw_notes = [
        _raw_with(0, 60, exp={"a": 1}),
        _raw_with(480, 62, exp={"b": 2}),
    ]

    def insert(project):
        project.tracks[0].parts[0].notes.insert(
            1, Note(start_tick=240, duration_tick=240, pitch=61, lyric="い", velocity=64))

    written = _written_notes(_read_write(raw_notes, insert))

    by_pos = {note["pos"]: note for note in written}
    assert by_pos[0]["exp"] == {"a": 1}
    assert by_pos[480]["exp"] == {"b": 2}
    assert "exp" not in by_pos[240]
    assert by_pos[240]["number"] == 61


def test_removed_note_takes_its_unknown_keys_with_it():
    raw_notes = [
        _raw_with(0, 60, exp={"a": 1}),
        _raw_with(240, 61, exp={"b": 2}),
        _raw_with(480, 62, exp={"c": 3}),
    ]

    def remove(project):
        del project.tracks[0].parts[0].notes[1]

    written = _written_notes(_read_write(raw_notes, remove))

    assert [(note["pos"], note["exp"]) for note in written] == [(0, {"a": 1}), (480, {"c": 3})]


def test_added_note_is_not_a_copy_of_the_first_raw_note():
    raw_notes = [_raw_with(0, 60, exp={"a": 1}, isProtected=True)]

    def add(project):
        project.tracks[0].parts[0].notes.append(
            Note(start_tick=480, duration_tick=240, pitch=64, lyric="う", velocity=50))

    written = _written_notes(_read_write(raw_notes, add))

    added = next(note for note in written if note["pos"] == 480)
    assert "exp" not in added
    assert added["isProtected"] is False


def test_added_singing_track_is_not_a_copy_of_the_first_singing_track():
    sequence = _sequence([_singing_track([_raw_with(0, 60)])])
    sequence["tracks"][0]["color"] = 7
    sequence["tracks"][0]["customKey"] = "first"
    project, _ = read(_make_vpr(sequence))
    project.tracks.append(Track(name="second", parts=[Part(name="q", start_tick=0)]))

    data = write(project)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        written = json.loads(archive.read("Project/sequence.json"))

    first, second = written["tracks"]
    assert first["customKey"] == "first"
    assert second["name"] == "second"
    assert "customKey" not in second
    assert second["color"] != 7


def test_unmodified_read_write_does_not_change_unknown_note_keys():
    raw_notes = [_raw_with(480, 62, exp={"x": 1}), _raw_with(0, 60, exp={"y": 2})]

    written = _written_notes(_read_write(raw_notes))

    assert sorted((note["pos"], note["exp"]) for note in written) == [(0, {"y": 2}), (480, {"x": 1})]
