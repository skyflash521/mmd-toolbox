"""vpr2vmd の vpr 抽出層のテスト(vpr2vmd.md §3・§4)。

vpr の読み込み(vpr_io へ委譲)・対象トラック選択・音符収集(全パートの統合と安定整列)を、
合成した `VprProject`(vpr_io データモデル)を入力に決定論的に検証する。重なり解決・フレーム変換・
口形写像は後続の口形イベント確定で扱うため、ここでは生の抽出のみを対象にする。
"""

import io
import json
import zipfile

import pytest
from vpr_io import (
    Note,
    Part,
    TempoEvent,
    Track,
    VprProject,
    rest_intervals,
)
from vpr_io import read as vpr_read

from vpr2vmd import io as vio


def _make_vpr(sequence: dict) -> bytes:
    """sequence.json を Project/sequence.json として持つ最小の vpr(zip)を組み立てる。"""
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


# --- 対象トラック選択(vpr2vmd.md §4.2、整数=0-based INDEX / 非整数=Track.name) ---

def test_select_track_defaults_to_first():
    project = _project([_track([], name="a"), _track([], name="b")])
    assert vio.select_track(project, None).name == "a"


def test_select_track_by_index():
    project = _project([_track([], name="a"), _track([], name="b")])
    assert vio.select_track(project, "1").name == "b"


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


# --- 音符収集(全パート統合 + 安定整列: start昇順, duration降順, パート出現順, 音符索引昇順) ---

def test_collect_notes_merges_parts_in_order():
    track = _track([
        _part([_note(0, 100)], name="p0"),
        _part([_note(200, 100)], name="p1"),
    ])
    notes = vio.collect_notes(track)
    assert [n.start_tick for n in notes] == [0, 200]


def test_collect_notes_sorts_by_start_then_longer_duration_first():
    # 同一 start は duration 降順(長い方が先)。
    track = _track([_part([_note(0, 50), _note(0, 120)])])
    notes = vio.collect_notes(track)
    assert [n.duration_tick for n in notes] == [120, 50]


def test_collect_notes_tiebreak_by_part_order():
    # 同一 start・同一 duration はパート出現順。
    track = _track([
        _part([_note(0, 100, lyric="first")], name="p0"),
        _part([_note(0, 100, lyric="second")], name="p1"),
    ])
    notes = vio.collect_notes(track)
    assert [n.lyric for n in notes] == ["first", "second"]


def test_collect_notes_is_stable_within_part():
    # 同一 start・同一 duration の同一パート内は音符索引昇順(入力順)を保つ。
    track = _track([_part([_note(0, 100, lyric="x"), _note(0, 100, lyric="y")])])
    notes = vio.collect_notes(track)
    assert [n.lyric for n in notes] == ["x", "y"]


def test_collect_notes_empty_track():
    assert vio.collect_notes(_track([])) == []


# --- 代表 vpr からの抽出(読み込みは vpr_io、選択・収集は vpr2vmd)---

def test_extracts_notes_tempo_rests_from_representative_vpr():
    """代表 vpr から音符・休符・テンポが取り出せる(vpr2vmd.md §3、実装計画の受入条件)。"""
    vpr = _make_vpr(_sequence(
        [{
            "type": 2, "name": "vocal",
            "parts": [{"name": "p", "pos": 0, "duration": 1920, "notes": [
                _seq_note(0, 240, number=60, lyric="ら", phoneme="4 a", velocity=64),
                _seq_note(480, 360, number=62, lyric="り", phoneme="4 i", velocity=100),
            ]}],
        }],
        tempo_events=[{"pos": 0, "value": 12000}],
    ))
    project, _warnings = vpr_read(vpr)

    # テンポが取り出せる(value/100 = bpm)。
    assert [(t.tick, t.bpm) for t in project.tempos] == [(0, 120.0)]

    # 音符が取り出せる(時刻・長さ・ピッチ・歌詞・強弱・音素を全て保持)。
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

    # 休符が取り出せる(発音区間の補集合。240..480 が休符)。
    end_tick = max(n.start_tick + n.duration_tick for n in notes)
    assert rest_intervals(notes, end_tick) == [(240, 480)]


def test_non_vpr_bytes_raise_vpr_format_error():
    """非vpr(壊れた zip)は vpr_io が VprFormatError を送出する(入力不正の根拠)。"""
    from vpr_io import VprFormatError

    with pytest.raises(VprFormatError):
        vpr_read(b"not a zip")
