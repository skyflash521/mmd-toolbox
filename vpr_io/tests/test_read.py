"""vpr read のテスト(vpr_io.md §3、docs/specs/vpr/VPR_file_format.md)。

テスト用 vpr は最小の sequence.json を zip 化してテスト内で合成する(実素材に依存しない)。
"""

import io
import json
import zipfile

import pytest


def _make_vpr(sequence: dict) -> bytes:
    """sequence.json を Project/sequence.json として持つ最小の vpr(zip)を組み立てる。"""
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
            "timeSig": {"events": timesig_events if timesig_events is not None else [{"bar": 0, "numer": 4, "denom": 4}]},
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
    from vpr_io import VprProject, read

    project, warnings = read(_make_vpr(_sequence([_singing_track([])])))
    assert isinstance(project, VprProject)
    assert isinstance(warnings, list)


def test_read_resolution_is_480():
    from vpr_io import read

    project, _ = read(_make_vpr(_sequence([_singing_track([])])))
    assert project.resolution == 480


def test_read_tempo_value_is_bpm_times_100():
    from vpr_io import read

    project, _ = read(_make_vpr(_sequence([_singing_track([])], tempo_events=[{"pos": 0, "value": 13600}])))
    assert len(project.tempos) == 1
    assert project.tempos[0].tick == 0
    assert project.tempos[0].bpm == pytest.approx(136.0)


def test_read_maps_multiple_tempo_events():
    from vpr_io import read

    events = [{"pos": 0, "value": 12000}, {"pos": 1920, "value": 14400}]
    project, _ = read(_make_vpr(_sequence([_singing_track([])], tempo_events=events)))
    assert [t.tick for t in project.tempos] == [0, 1920]
    assert project.tempos[0].bpm == pytest.approx(120.0)
    assert project.tempos[1].bpm == pytest.approx(144.0)


def test_read_timesig_bar_zero_maps_to_tick_zero():
    from vpr_io import read

    project, _ = read(_make_vpr(_sequence([_singing_track([])], timesig_events=[{"bar": 0, "numer": 3, "denom": 4}])))
    assert len(project.time_signatures) == 1
    ts = project.time_signatures[0]
    assert ts.tick == 0
    assert ts.numerator == 3
    assert ts.denominator == 4


def test_read_timesig_nonzero_bar_maps_to_tick():
    from vpr_io import read

    # bar 0 は 4/4(1小節 = 4 * 480 = 1920 tick)。bar 1 の拍子変更は tick 1920 に写る。
    events = [{"bar": 0, "numer": 4, "denom": 4}, {"bar": 1, "numer": 3, "denom": 4}]
    project, _ = read(_make_vpr(_sequence([_singing_track([])], timesig_events=events)))
    mapped = [(t.tick, t.numerator, t.denominator) for t in project.time_signatures]
    assert mapped == [(0, 4, 4), (1920, 3, 4)]


def test_read_timesig_first_event_after_bar_zero_uses_default_4_4():
    from vpr_io import read

    # 最初の拍子イベントが bar>0 のとき、その前の小節は既定 4/4(1小節=1920 tick)で積算する。
    events = [{"bar": 2, "numer": 4, "denom": 4}]
    project, _ = read(_make_vpr(_sequence([_singing_track([])], timesig_events=events)))
    assert [(t.tick, t.numerator, t.denominator) for t in project.time_signatures] == [(3840, 4, 4)]


def test_read_timesig_accumulates_nonuniform_bar_lengths():
    from vpr_io import read

    # bar 0・1 は 3/4(1小節 = 3 * 480 = 1440 tick)。bar 2 の拍子変更は tick 2880(= 1440 * 2)に写る。
    # 先行小節長を積算するので「常に bar * 1920」では誤りになる。
    events = [{"bar": 0, "numer": 3, "denom": 4}, {"bar": 2, "numer": 4, "denom": 4}]
    project, _ = read(_make_vpr(_sequence([_singing_track([])], timesig_events=events)))
    mapped = [(t.tick, t.numerator, t.denominator) for t in project.time_signatures]
    assert mapped == [(0, 3, 4), (2880, 4, 4)]


def test_read_maps_note_fields_and_splits_phonemes():
    from vpr_io import read

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
    from vpr_io import read

    notes = [_note(pos=480, duration=240, number=60, lyric="ら", phoneme="r a", velocity=64)]
    # パート相対の note.pos を part.pos 加算で絶対化する(1920 + 480 = 2400)。
    track = _singing_track(notes, part_pos=1920, part_duration=2400)
    project, _ = read(_make_vpr(_sequence([track])))
    part = project.tracks[0].parts[0]
    assert part.start_tick == 1920
    assert part.notes[0].start_tick == 2400


def test_read_empty_phoneme_yields_empty_list():
    from vpr_io import read

    notes = [_note(pos=0, duration=240, number=60, lyric="あ", phoneme="", velocity=64)]
    project, _ = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert project.tracks[0].parts[0].notes[0].phonemes == []


def test_read_splits_multi_token_phoneme():
    from vpr_io import read

    notes = [_note(pos=0, duration=480, number=62, lyric="Tell", phoneme="t th e l", velocity=64)]
    project, _ = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert project.tracks[0].parts[0].notes[0].phonemes == ["t", "th", "e", "l"]


def test_read_notes_sorted_by_start_tick():
    from vpr_io import read

    # 入力順が start_tick 昇順でなくても、モデルは昇順で返す(vpr_io.md §2.1)。
    notes = [
        _note(pos=960, duration=240, number=62, lyric="そ", phoneme="s o", velocity=64),
        _note(pos=480, duration=240, number=60, lyric="ら", phoneme="r a", velocity=64),
    ]
    project, _ = read(_make_vpr(_sequence([_singing_track(notes)])))
    starts = [n.start_tick for n in project.tracks[0].parts[0].notes]
    assert starts == [480, 960]


def test_read_excludes_audio_track():
    from vpr_io import read

    audio_track = {"type": 1, "name": "audio", "parts": [{"name": "a", "pos": 0, "wav": {}, "region": {}}]}
    seq = _sequence([_singing_track([]), audio_track])
    project, _ = read(_make_vpr(seq))
    # 歌唱トラック(type 2)のみがデータモデルに現れる。
    assert [t.name for t in project.tracks] == ["vocal"]


def test_read_extracts_all_singing_tracks():
    from vpr_io import read

    seq = _sequence([_singing_track([], name="vocal1"), _singing_track([], name="vocal2")])
    project, _ = read(_make_vpr(seq))
    # type=2 のトラックは全件抽出する。
    assert [t.name for t in project.tracks] == ["vocal1", "vocal2"]


def test_read_extracts_all_parts_in_track():
    from vpr_io import read

    track = {
        "type": 2,
        "name": "vocal",
        "parts": [
            {"name": "p1", "pos": 0, "duration": 1920, "notes": []},
            {"name": "p2", "pos": 1920, "duration": 1920, "notes": []},
        ],
    }
    project, _ = read(_make_vpr(_sequence([track])))
    # 1トラック内のパートは全件抽出する。
    assert [p.name for p in project.tracks[0].parts] == ["p1", "p2"]
    assert [p.start_tick for p in project.tracks[0].parts] == [0, 1920]


def test_read_accepts_bytes_str_and_path(tmp_path):
    from vpr_io import read

    data = _make_vpr(_sequence([_singing_track([])]))
    from_bytes, _ = read(data)
    path = tmp_path / "sample.vpr"
    path.write_bytes(data)
    from_path, _ = read(path)
    from_str, _ = read(str(path))
    assert from_bytes.resolution == from_path.resolution == from_str.resolution == 480
