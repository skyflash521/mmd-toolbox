import io
import json
import math
import zipfile
from pathlib import Path

from .constants import DEFAULT_TIME_SIGNATURE, RESOLUTION, SEQUENCE_PATH, SINGING_TRACK_TYPE
from .types import (
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
    VprWarning,
)


def _require(obj, key, path, type_=None):
    loc = f"{path}.{key}" if path else key
    try:
        value = obj[key]
    except (KeyError, TypeError) as e:
        raise VprFormatError(f"必須キー {key!r} がありません", path=loc, key=key) from e
    # Python の bool は int のサブクラスなので、isinstance だけでは JSON の true/false を弾けない。
    if type_ is not None and (not isinstance(value, type_) or isinstance(value, bool)):
        raise VprFormatError(f"{key!r} の型が不正です", path=loc, key=key, value=value)
    return value


def _optional(obj, key, default):
    return obj.get(key, default) if isinstance(obj, dict) else default


def _optional_list(obj, key, path):
    if not isinstance(obj, dict) or key not in obj:
        return []
    value = obj[key]
    if not isinstance(value, list):
        loc = f"{path}.{key}" if path else key
        raise VprFormatError(f"{key!r} は配列でなければなりません", path=loc, key=key, value=value)
    return value


def _load_sequence(src):
    source = io.BytesIO(src) if isinstance(src, bytes) else src
    try:
        with zipfile.ZipFile(source) as archive:
            try:
                data = archive.read(SEQUENCE_PATH)
            except KeyError as e:
                raise VprFormatError(
                    f"{SEQUENCE_PATH} がありません", path=SEQUENCE_PATH
                ) from e
            entries = {name: archive.read(name) for name in archive.namelist()
                       if name != SEQUENCE_PATH}
    except zipfile.BadZipFile as e:
        raise VprFormatError("ZIP アーカイブとして読み込めません") from e
    try:
        return json.loads(data), entries
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise VprFormatError(
            f"{SEQUENCE_PATH} を UTF-8 JSON として解析できません", path=SEQUENCE_PATH
        ) from e


def _ticks_per_bar(numerator: int, denominator: int, path=None) -> int:
    if denominator < 1:
        raise VprFormatError("拍子の分母は 1 以上でなければなりません",
                             path=path, key="denom", value=denominator)
    return numerator * RESOLUTION * 4 // denominator


def _tempos(master) -> list[TempoEvent]:
    tempo = _require(master, "tempo", "masterTrack", dict)
    events = _require(tempo, "events", "masterTrack.tempo", list)
    if not events:
        raise VprFormatError("テンポイベントが 1 件もありません",
                             path="masterTrack.tempo.events", key="events", value=events)
    result: list[TempoEvent] = []
    for i, event in enumerate(events):
        path = f"masterTrack.tempo.events[{i}]"
        pos = _require(event, "pos", path, int)
        value = _require(event, "value", path, (int, float))
        bpm = value / 100
        if not math.isfinite(bpm) or bpm <= 0.0:
            raise VprFormatError("BPM は正の有限値でなければなりません",
                                 path=path, key="value", value=value)
        result.append(TempoEvent(tick=pos, bpm=bpm))
    return result


def _time_signatures(master) -> list[TimeSignature]:
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
                path,
            )
        )

    result: list[TimeSignature] = []
    tick = 0
    prev_bar = 0
    prev_ticks_per_bar = _ticks_per_bar(*DEFAULT_TIME_SIGNATURE)
    for bar, numerator, denominator, path in sorted(indexed, key=lambda x: x[0]):
        tick += (bar - prev_bar) * prev_ticks_per_bar
        result.append(TimeSignature(tick=tick, numerator=numerator, denominator=denominator))
        prev_ticks_per_bar = _ticks_per_bar(numerator, denominator, path)
        prev_bar = bar
    return result


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def accepts_vibrato(raw) -> bool:
    if not isinstance(raw, dict):
        return False
    if not _is_int(raw.get("type")) or not _is_int(raw.get("duration")):
        return False
    for key in ("depths", "rates"):
        points = raw.get(key, [])
        if not isinstance(points, list):
            return False
        for point in points:
            if not isinstance(point, dict) or not _is_int(point.get("pos")) \
                    or not _is_int(point.get("value")):
                return False
    return True


def accepts_depth_envelope(raw) -> bool:
    if not isinstance(raw, dict):
        return False
    return all(isinstance(value, (int, float)) and not isinstance(value, bool)
               for value in (raw.get("vibratoLeadingDepth"), raw.get("vibratoFollowingDepth")))


def _vibrato_points(raw_points, span_start) -> list[VibratoPoint]:
    return [VibratoPoint(pos=span_start + point["pos"], value=point["value"])
            for point in raw_points]


def _vibrato(note, note_start, note_duration) -> NoteVibrato | None:
    raw = _optional(note, "vibrato", None)
    if not accepts_vibrato(raw) or raw["duration"] <= 0:
        return None
    span_start = note_start + note_duration - raw["duration"]
    return NoteVibrato(
        type=raw["type"],
        duration=raw["duration"],
        depths=_vibrato_points(raw.get("depths", []), span_start),
        rates=_vibrato_points(raw.get("rates", []), span_start),
    )


def _ai_expression(note) -> NoteAiExpression | None:
    raw = _optional(note, "aiExp", None)
    if not accepts_depth_envelope(raw):
        return None
    return NoteAiExpression(vibrato_leading_depth=float(raw["vibratoLeadingDepth"]),
                            vibrato_following_depth=float(raw["vibratoFollowingDepth"]))


def _notes(raw_notes, part_pos, part_path) -> list[Note]:
    notes: list[Note] = []
    for i, note in enumerate(raw_notes):
        path = f"{part_path}.notes[{i}]"
        start_tick = part_pos + _require(note, "pos", path, int)
        duration_tick = _require(note, "duration", path, int)
        notes.append(
            Note(
                start_tick=start_tick,
                duration_tick=duration_tick,
                pitch=_require(note, "number", path, int),
                lyric=_require(note, "lyric", path, str),
                velocity=_require(note, "velocity", path, int),
                phonemes=_require(note, "phoneme", path, str).split(),
                is_protected=_optional(note, "isProtected", False) is True,
                vibrato=_vibrato(note, start_tick, duration_tick),
                ai_expression=_ai_expression(note),
            )
        )
    notes.sort(key=lambda note: note.start_tick)
    return notes


def _controllers(raw_controllers, part_pos, part_path) -> list[ControllerCurve]:
    curves: list[ControllerCurve] = []
    for i, controller in enumerate(raw_controllers):
        path = f"{part_path}.controllers[{i}]"
        name = _require(controller, "name", path, str)
        events: list[ControllerEvent] = []
        for j, event in enumerate(_optional_list(controller, "events", path)):
            event_path = f"{path}.events[{j}]"
            events.append(
                ControllerEvent(
                    tick=part_pos + _require(event, "pos", event_path, int),
                    value=_require(event, "value", event_path, int),
                )
            )
        events.sort(key=lambda e: e.tick)
        curves.append(ControllerCurve(name=name, events=events))
    return curves


def _voice(part, voices) -> VoiceBank | None:
    reference = _optional(part, "aiVoice", None)
    if not isinstance(reference, dict):
        return None
    comp_id = _optional(reference, "compID", None)
    if not isinstance(comp_id, str):
        return None
    for definition in voices:
        if isinstance(definition, dict) and definition.get("compID") == comp_id:
            name = definition.get("name")
            return VoiceBank(comp_id=comp_id, name=name if isinstance(name, str) else "")
    return None


def _tracks(raw_tracks, voices) -> list[Track]:
    tracks: list[Track] = []
    for i, track in enumerate(raw_tracks):
        if _optional(track, "type", None) != SINGING_TRACK_TYPE:
            continue
        path = f"tracks[{i}]"
        parts: list[Part] = []
        for j, part in enumerate(_optional_list(track, "parts", path)):
            part_path = f"{path}.parts[{j}]"
            part_pos = _require(part, "pos", part_path, int)
            raw_duration = _optional(part, "duration", 0)
            parts.append(
                Part(
                    name=_require(part, "name", part_path, str),
                    start_tick=part_pos,
                    duration_tick=raw_duration if _is_int(raw_duration) else 0,
                    voice=_voice(part, voices),
                    notes=_notes(_optional_list(part, "notes", part_path), part_pos, part_path),
                    controllers=_controllers(
                        _optional_list(part, "controllers", part_path), part_pos, part_path
                    ),
                )
            )
        tracks.append(Track(name=_require(track, "name", path, str), parts=parts))
    return tracks


def _overlap_warnings(tracks) -> list[VprWarning]:
    warnings: list[VprWarning] = []
    for track_index, track in enumerate(tracks):
        for part_index, part in enumerate(track.parts):
            sounding: list[tuple[int, int]] = []
            for note_index, note in enumerate(part.notes):
                sounding = [(end, idx) for end, idx in sounding if end > note.start_tick]
                for _end, idx in sounding:
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
                sounding.append((note.start_tick + note.duration_tick, note_index))
    return warnings


def read(src: str | Path | bytes) -> tuple[VprProject, list[VprWarning]]:
    sequence, entries = _load_sequence(src)
    master = _require(sequence, "masterTrack", "", dict)
    raw_tracks = _require(sequence, "tracks", "", list)
    title = _optional(sequence, "title", "")

    project = VprProject(
        resolution=RESOLUTION,
        tempos=_tempos(master),
        time_signatures=_time_signatures(master),
        tracks=_tracks(raw_tracks, _optional_list(sequence, "voices", "")),
        title=title if isinstance(title, str) else "",
        raw_sequence=sequence,
        entries=entries,
    )
    return project, _overlap_warnings(project.tracks)
