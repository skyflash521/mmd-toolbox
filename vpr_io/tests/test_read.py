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


# --- 構造異常(VprFormatError)と許容入力(vpr_io.md §3.1) ---


def _zip_with(entries: dict) -> bytes:
    """{エントリ名: 文字列} を持つ最小の zip を組み立てる(壊れた sequence.json 等の合成用)。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in entries.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_read_raises_format_error_on_non_zip():
    from vpr_io import VprFormatError, read

    with pytest.raises(VprFormatError):
        read(b"this is not a zip archive")


def test_read_raises_format_error_on_missing_sequence_json():
    from vpr_io import VprFormatError, read

    with pytest.raises(VprFormatError):
        read(_zip_with({"Project/other.txt": "x"}))


def test_read_raises_format_error_on_invalid_json():
    from vpr_io import VprFormatError, read

    with pytest.raises(VprFormatError):
        read(_zip_with({"Project/sequence.json": "{ not valid json "}))


def test_read_raises_format_error_on_invalid_utf8_sequence_json():
    from vpr_io import VprFormatError, read

    # sequence.json が UTF-8 として復号できない(vpr_io.md §3.1)。
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Project/sequence.json", b"\x80\x81\x82\xff")
    with pytest.raises(VprFormatError):
        read(buf.getvalue())


def test_read_raises_format_error_on_missing_master_track():
    from vpr_io import VprFormatError, read

    seq = _sequence([_singing_track([])])
    del seq["masterTrack"]
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_missing_tracks():
    from vpr_io import VprFormatError, read

    seq = _sequence([_singing_track([])])
    del seq["tracks"]
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_malformed_tempo_event():
    from vpr_io import VprFormatError, read

    # TempoEvent を構築するための必須キー(value)が無い(vpr_io.md §3.1)。
    seq = _sequence([_singing_track([])], tempo_events=[{"pos": 0}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_tempo_event_type_error():
    from vpr_io import VprFormatError, read

    # value が数値でない(bpm 構築で型不正。vpr_io.md §3.1)。
    seq = _sequence([_singing_track([])], tempo_events=[{"pos": 0, "value": "fast"}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_malformed_timesig_event():
    from vpr_io import VprFormatError, read

    # TimeSignature を構築するための必須キー(numer)が無い(vpr_io.md §3.1)。
    seq = _sequence([_singing_track([])], timesig_events=[{"bar": 0, "denom": 4}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_timesig_event_type_error():
    from vpr_io import VprFormatError, read

    # denom が数値でない(小節長 tick 構築で型不正。vpr_io.md §3.1)。
    seq = _sequence([_singing_track([])], timesig_events=[{"bar": 0, "numer": 4, "denom": "x"}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_missing_required_note_field():
    from vpr_io import VprFormatError, read

    note = _note(pos=0, duration=240, number=60, lyric="あ", phoneme="a", velocity=64)
    del note["number"]
    with pytest.raises(VprFormatError):
        read(_make_vpr(_sequence([_singing_track([note])])))


def test_read_format_error_carries_locator():
    from vpr_io import VprFormatError, read

    note = _note(pos=0, duration=240, number=60, lyric="あ", phoneme="a", velocity=64)
    del note["number"]
    with pytest.raises(VprFormatError) as exc:
        read(_make_vpr(_sequence([_singing_track([note])])))
    # 原因特定のためのロケータ(JSON パスと欠落キー)を持つ(vpr_io.md §3.1)。
    assert exc.value.path is not None
    assert exc.value.key == "number"


def test_read_raises_format_error_on_missing_track_name():
    from vpr_io import VprFormatError, read

    # 歌唱トラックの公開モデル対象フィールド(name)が欠落(vpr_io.md §3.1)。
    track = {"type": 2, "parts": [{"name": "p", "pos": 0, "duration": 1920, "notes": []}]}
    with pytest.raises(VprFormatError):
        read(_make_vpr(_sequence([track])))


def test_read_raises_format_error_on_missing_part_pos():
    from vpr_io import VprFormatError, read

    # パートの公開モデル対象フィールド(pos)が欠落(vpr_io.md §3.1)。
    track = {"type": 2, "name": "vocal", "parts": [{"name": "p", "duration": 1920, "notes": []}]}
    with pytest.raises(VprFormatError):
        read(_make_vpr(_sequence([track])))


def test_read_raises_format_error_on_note_field_type_error():
    from vpr_io import VprFormatError, read

    # 音符の公開モデル対象フィールド(number)の型が不正(vpr_io.md §3.1)。ロケータは欠落キーを指す。
    note = _note(pos=0, duration=240, number="x", lyric="あ", phoneme="a", velocity=64)
    with pytest.raises(VprFormatError) as exc:
        read(_make_vpr(_sequence([_singing_track([note])])))
    assert exc.value.key == "number"


def test_read_raises_format_error_on_bool_as_int_field():
    from vpr_io import VprFormatError, read

    # JSON の bool は int フィールドの型不正(bool は int のサブクラスだが値として不正。vpr_io.md §3.1)。
    seq = _sequence([_singing_track([])], timesig_events=[{"bar": 0, "numer": 4, "denom": False}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_tempo_pos_type_error():
    from vpr_io import VprFormatError, read

    # tempo event の pos が整数でない(vpr_io.md §3.1)。
    seq = _sequence([_singing_track([])], tempo_events=[{"pos": "zero", "value": 12000}])
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_non_list_tempo_events():
    from vpr_io import VprFormatError, read

    # tempo.events が配列でない(null)場合、生の TypeError を漏らさず VprFormatError(vpr_io.md §3.1)。
    seq = _sequence([_singing_track([])])
    seq["masterTrack"]["tempo"]["events"] = None
    with pytest.raises(VprFormatError):
        read(_make_vpr(seq))


def test_read_raises_format_error_on_non_list_parts():
    from vpr_io import VprFormatError, read

    # 歌唱トラックの parts が配列でない(null)場合も VprFormatError(vpr_io.md §3.1)。
    track = {"type": 2, "name": "vocal", "parts": None}
    with pytest.raises(VprFormatError):
        read(_make_vpr(_sequence([track])))


def test_read_tolerates_missing_notes_as_empty_part():
    from vpr_io import read

    # 歌唱パートに notes が無い場合は空の音符列として扱う(エラーにしない)。
    track = {"type": 2, "name": "vocal", "parts": [{"name": "p", "pos": 0, "duration": 1920}]}
    project, _ = read(_make_vpr(_sequence([track])))
    assert project.tracks[0].parts[0].notes == []


def test_read_tolerates_unknown_top_level_keys():
    from vpr_io import read

    # 公開データモデル対象外の未知キーが存在しても read は失敗しない。
    seq = _sequence([_singing_track([])])
    seq["someUnknownKey"] = {"x": 1}
    project, _ = read(_make_vpr(seq))
    assert project.resolution == 480


# --- 発音区間の重なり警告(VprWarning。vpr_io.md §3.2) ---


def test_read_warns_on_overlapping_notes_in_part():
    from vpr_io import read

    # 同一パート内で発音区間が重なる2音符([0,480) と [240,720))。
    notes = [
        _note(pos=0, duration=480, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=240, duration=480, number=62, lyric="い", phoneme="i", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert [w.code for w in warnings] == ["overlapping_notes"]


def test_read_overlap_warning_carries_locator():
    from vpr_io import read

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
    assert w.tick == 240  # 重なり開始位置(後続音符の開始)


def test_read_warns_once_per_overlapping_pair():
    from vpr_io import read

    # 独立した2組の重なり([0,480)&[240,500)、[1000,1480)&[1200,1500))→ ペアごとに1件。
    notes = [
        _note(pos=0, duration=480, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=240, duration=260, number=62, lyric="い", phoneme="i", velocity=64),
        _note(pos=1000, duration=480, number=64, lyric="う", phoneme="u", velocity=64),
        _note(pos=1200, duration=300, number=65, lyric="え", phoneme="e", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert [w.code for w in warnings] == ["overlapping_notes", "overlapping_notes"]


def test_read_warns_for_all_overlapping_pairs():
    from vpr_io import read

    # 3音符が相互に重なる([0,1000)・[240,480)・[300,600))→ ペア (0,1)(0,2)(1,2) の3件。
    # 最大終端の先行音符だけを見ると (1,2) を取りこぼすため、全ペアを報告することを検証する。
    notes = [
        _note(pos=0, duration=1000, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=240, duration=240, number=62, lyric="い", phoneme="i", velocity=64),
        _note(pos=300, duration=300, number=64, lyric="う", phoneme="u", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    pairs = [(w.note_index, w.related_note_index) for w in warnings if w.code == "overlapping_notes"]
    assert pairs == [(0, 1), (0, 2), (1, 2)]


def test_read_no_warning_for_non_overlapping_notes():
    from vpr_io import read

    notes = [
        _note(pos=0, duration=240, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=480, duration=240, number=62, lyric="い", phoneme="i", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert warnings == []


def test_read_no_warning_for_adjacent_notes():
    from vpr_io import read

    # 隣接(前音符の終端 == 次音符の開始)は半開区間 [start, start+dur) では重ならない。
    notes = [
        _note(pos=0, duration=480, number=60, lyric="あ", phoneme="a", velocity=64),
        _note(pos=480, duration=480, number=62, lyric="い", phoneme="i", velocity=64),
    ]
    _, warnings = read(_make_vpr(_sequence([_singing_track(notes)])))
    assert warnings == []


def test_read_no_overlap_warning_across_parts():
    from vpr_io import read

    # クロスパートの重なりは初期スコープ外(検出は同一パート内のみ。vpr_io.md §3.2)。
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
