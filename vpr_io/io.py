"""vpr 読み込み(vpr_io.md §3)。

形式レイアウトの正: docs/specs/vpr/VPR_file_format.md
"""

import io
import json
import zipfile

from .types import (
    Note,
    Part,
    TempoEvent,
    TimeSignature,
    Track,
    VprFormatError,
    VprProject,
    VprWarning,
)

_RESOLUTION = 480  # tick/四分音符(vpr に格納されない固定値)
_SEQUENCE_PATH = "Project/sequence.json"
_SINGING_TRACK_TYPE = 2


def _require(obj, key, path, type_=None):
    """obj[key] を返す。欠落・型不一致は構造異常 VprFormatError(§3.1)。

    type_ を与えると値の型を検証する。原因特定のため path・key・value を付与する。
    """
    loc = f"{path}.{key}" if path else key
    try:
        value = obj[key]
    except (KeyError, TypeError) as e:
        raise VprFormatError(f"必須キー {key!r} がありません", path=loc, key=key) from e
    # bool は int のサブクラスなので、JSON の true/false が int フィールドを素通りしないよう除外する。
    if type_ is not None and (not isinstance(value, type_) or isinstance(value, bool)):
        raise VprFormatError(f"{key!r} の型が不正です", path=loc, key=key, value=value)
    return value


def _optional(obj, key, default):
    """obj[key] を返す。obj が辞書でない/キーが無ければ default(許容入力。§3.1)。"""
    return obj.get(key, default) if isinstance(obj, dict) else default


def _optional_list(obj, key, path):
    """obj[key] を返す。キーが無ければ []。値が配列でなければ VprFormatError(§3.1)。"""
    if not isinstance(obj, dict) or key not in obj:
        return []
    value = obj[key]
    if not isinstance(value, list):
        loc = f"{path}.{key}" if path else key
        raise VprFormatError(f"{key!r} は配列でなければなりません", path=loc, key=key, value=value)
    return value


def _load_sequence(src):
    """vpr(ZIP)から Project/sequence.json を読み JSON として返す。構造異常は VprFormatError。"""
    source = io.BytesIO(src) if isinstance(src, bytes) else src
    try:
        with zipfile.ZipFile(source) as archive:
            try:
                data = archive.read(_SEQUENCE_PATH)
            except KeyError as e:
                raise VprFormatError(
                    f"{_SEQUENCE_PATH} がありません", path=_SEQUENCE_PATH
                ) from e
    except zipfile.BadZipFile as e:
        raise VprFormatError("ZIP アーカイブとして読み込めません") from e
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise VprFormatError(
            f"{_SEQUENCE_PATH} を UTF-8 JSON として解析できません", path=_SEQUENCE_PATH
        ) from e


def _ticks_per_bar(numerator: int, denominator: int) -> int:
    """拍子の1小節あたりの tick 長。"""
    return numerator * _RESOLUTION * 4 // denominator


def _tempos(master) -> list[TempoEvent]:
    """テンポイベントを TempoEvent(bpm = value/100)へ写像する。"""
    tempo = _require(master, "tempo", "masterTrack", dict)
    events = _require(tempo, "events", "masterTrack.tempo", list)
    result: list[TempoEvent] = []
    for i, event in enumerate(events):
        path = f"masterTrack.tempo.events[{i}]"
        pos = _require(event, "pos", path, int)
        value = _require(event, "value", path, (int, float))
        result.append(TempoEvent(tick=pos, bpm=value / 100))
    return result


def _time_signatures(master) -> list[TimeSignature]:
    """拍子イベント(小節番号 bar 基準)を、先行小節長を積算して tick へ変換する。"""
    timesig = _require(master, "timeSig", "masterTrack", dict)
    events = _require(timesig, "events", "masterTrack.timeSig", list)
    indexed = []
    for i, event in enumerate(events):
        path = f"masterTrack.timeSig.events[{i}]"
        indexed.append(
            (
                _require(event, "bar", path, int),
                _require(event, "numer", path, int),
                _require(event, "denom", path, int),
            )
        )

    result: list[TimeSignature] = []
    tick = 0
    prev_bar = 0
    # 最初の明示イベントより前の小節は VOCALOID 既定の 4/4 とみなして積算する。
    prev_ticks_per_bar = _ticks_per_bar(4, 4)
    for bar, numerator, denominator in sorted(indexed, key=lambda x: x[0]):
        tick += (bar - prev_bar) * prev_ticks_per_bar
        result.append(TimeSignature(tick=tick, numerator=numerator, denominator=denominator))
        prev_ticks_per_bar = _ticks_per_bar(numerator, denominator)
        prev_bar = bar
    return result


def _notes(raw_notes, part_pos, part_path) -> list[Note]:
    """音符を絶対 tick 化(part 開始位置を加算)し、start_tick 昇順で返す。"""
    notes: list[Note] = []
    for i, note in enumerate(raw_notes):
        path = f"{part_path}.notes[{i}]"
        notes.append(
            Note(
                start_tick=part_pos + _require(note, "pos", path, int),
                duration_tick=_require(note, "duration", path, int),
                pitch=_require(note, "number", path, int),
                lyric=_require(note, "lyric", path, str),
                velocity=_require(note, "velocity", path, int),
                phonemes=_require(note, "phoneme", path, str).split(),
            )
        )
    notes.sort(key=lambda note: note.start_tick)
    return notes


def _tracks(raw_tracks) -> list[Track]:
    """歌唱トラック(type==2)のみをデータモデルへ写像する。"""
    tracks: list[Track] = []
    for i, track in enumerate(raw_tracks):
        if _optional(track, "type", None) != _SINGING_TRACK_TYPE:
            continue
        path = f"tracks[{i}]"
        parts: list[Part] = []
        for j, part in enumerate(_optional_list(track, "parts", path)):
            part_path = f"{path}.parts[{j}]"
            part_pos = _require(part, "pos", part_path, int)
            parts.append(
                Part(
                    name=_require(part, "name", part_path, str),
                    start_tick=part_pos,
                    notes=_notes(_optional_list(part, "notes", part_path), part_pos, part_path),
                )
            )
        tracks.append(Track(name=_require(track, "name", path, str), parts=parts))
    return tracks


def _overlap_warnings(tracks) -> list[VprWarning]:
    """同一パート内で発音区間が重なる音符ペアを警告する(単音想定違反。vpr_io.md §3.2)。

    notes は start_tick 昇順。各音符について、まだ終端に達していない先行音符(active)を残し、その
    全てと重なるとみなして音符ペアごとに1件報告する。半開区間 [start, start+duration) なので終端 ==
    開始は重ならない。重なりが無い通常入力では active は短く保たれる。
    """
    warnings: list[VprWarning] = []
    for track_index, track in enumerate(tracks):
        for part_index, part in enumerate(track.parts):
            active: list[tuple[int, int]] = []  # (発音終端, note 添字)
            for note_index, note in enumerate(part.notes):
                active = [(end, idx) for end, idx in active if end > note.start_tick]
                for _end, idx in active:
                    warnings.append(
                        VprWarning(
                            code="overlapping_notes",
                            message="同一パート内で発音区間が重なっています",
                            track_index=track_index,
                            part_index=part_index,
                            note_index=idx,
                            related_note_index=note_index,
                            tick=note.start_tick,
                        )
                    )
                active.append((note.start_tick + note.duration_tick, note_index))
    return warnings


def read(src) -> tuple[VprProject, list[VprWarning]]:
    """vpr を読み、データモデル(vpr_io.md §2)と警告を返す。"""
    sequence = _load_sequence(src)
    master = _require(sequence, "masterTrack", "", dict)
    raw_tracks = _require(sequence, "tracks", "", list)

    project = VprProject(
        resolution=_RESOLUTION,
        tempos=_tempos(master),
        time_signatures=_time_signatures(master),
        tracks=_tracks(raw_tracks),
    )
    return project, _overlap_warnings(project.tracks)
