import copy
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

from . import template
from .constants import DEFAULT_TIME_SIGNATURE, RESOLUTION, SEQUENCE_PATH, SINGING_TRACK_TYPE
from .io import accepts_depth_envelope, accepts_vibrato
from .types import VprFormatError, VprProject

_MIDI_RANGE = (0, 127)
_VELOCITY_RANGE = (0, 127)


def _ticks_per_bar(numerator: int, denominator: int) -> int:
    return numerator * RESOLUTION * 4 // denominator


def _pair_up(raw_items, count, fallback):
    items = [copy.deepcopy(item) if isinstance(item, dict) else fallback()
             for item in raw_items[:count]]
    source = raw_items[0] if raw_items and isinstance(raw_items[0], dict) else None
    while len(items) < count:
        items.append(copy.deepcopy(source) if source is not None else fallback())
    return items


def _check_note(note, path):
    if not _MIDI_RANGE[0] <= note.pitch <= _MIDI_RANGE[1]:
        raise VprFormatError("音高が MIDI の範囲を外れています",
                             path=path, key="pitch", value=note.pitch)
    if note.duration_tick <= 0:
        raise VprFormatError("音符の長さは正でなければなりません",
                             path=path, key="duration_tick", value=note.duration_tick)
    if not _VELOCITY_RANGE[0] <= note.velocity <= _VELOCITY_RANGE[1]:
        raise VprFormatError("ベロシティが格納できる範囲を外れています",
                             path=path, key="velocity", value=note.velocity)
    for phoneme in note.phonemes:
        if not phoneme or phoneme.split() != [phoneme]:
            raise VprFormatError("音素は空白を含まない非空の文字列でなければなりません",
                                 path=path, key="phonemes", value=phoneme)
    if note.vibrato is None:
        return
    if note.vibrato.duration <= 0:
        raise VprFormatError("ビブラートの区間長は正でなければなりません",
                             path=f"{path}.vibrato", key="duration", value=note.vibrato.duration)
    if note.vibrato.duration > note.duration_tick:
        raise VprFormatError("ビブラートの区間長が音符の長さを超えています",
                             path=f"{path}.vibrato", key="duration", value=note.vibrato.duration)
    span_start = note.start_tick + note.duration_tick - note.vibrato.duration
    for key, points in (("depths", note.vibrato.depths), ("rates", note.vibrato.rates)):
        for point in points:
            if not 0 <= point.pos - span_start <= note.vibrato.duration:
                raise VprFormatError("ビブラートの制御点が区間の外にあります",
                                     path=f"{path}.vibrato.{key}", key="pos", value=point.pos)


def _vibrato_points(points, span_start):
    return [{"pos": point.pos - span_start, "value": point.value} for point in points]


def _write_vibrato(raw, note):
    previous = raw.get("vibrato")
    if note.vibrato is None:
        if accepts_vibrato(previous):
            previous["duration"] = 0
        return
    span_start = note.start_tick + note.duration_tick - note.vibrato.duration
    written = {"type": note.vibrato.type, "duration": note.vibrato.duration}
    for key, points in (("depths", note.vibrato.depths), ("rates", note.vibrato.rates)):
        if points or (isinstance(previous, dict) and key in previous):
            written[key] = _vibrato_points(points, span_start)
    raw["vibrato"] = written


def _write_ai_expression(raw, note):
    expression = raw.get("aiExp")
    if note.ai_expression is None:
        if accepts_depth_envelope(expression):
            del expression["vibratoLeadingDepth"]
            del expression["vibratoFollowingDepth"]
            if not expression:
                del raw["aiExp"]
        return
    if not isinstance(expression, dict):
        expression = {}
        raw["aiExp"] = expression
    expression["vibratoLeadingDepth"] = note.ai_expression.vibrato_leading_depth
    expression["vibratoFollowingDepth"] = note.ai_expression.vibrato_following_depth


def _write_note(raw, note, part_start, path):
    _check_note(note, path)
    raw["pos"] = note.start_tick - part_start
    raw["duration"] = note.duration_tick
    raw["number"] = note.pitch
    raw["lyric"] = note.lyric
    raw["phoneme"] = " ".join(note.phonemes)
    raw["velocity"] = note.velocity
    raw["isProtected"] = note.is_protected
    _write_vibrato(raw, note)
    _write_ai_expression(raw, note)


def _write_controllers(raw_controllers, curves, part_start):
    result = _pair_up(raw_controllers, len(curves), template.controller)
    for raw, curve in zip(result, curves, strict=True):
        raw["name"] = curve.name
        events = _pair_up(raw.get("events") or [], len(curve.events), lambda: {"pos": 0, "value": 0})
        for raw_event, event in zip(events, sorted(curve.events, key=lambda e: e.tick),
                                    strict=True):
            raw_event["pos"] = event.tick - part_start
            raw_event["value"] = event.value
        raw["events"] = events
    return result


def _write_part(raw, part, path):
    raw["name"] = part.name
    raw["pos"] = part.start_tick
    raw["duration"] = part.duration_tick

    notes = _pair_up(raw.get("notes") or [], len(part.notes), template.note)
    for i, (raw_note, note) in enumerate(
            zip(notes, sorted(part.notes, key=lambda n: n.start_tick), strict=True)):
        _write_note(raw_note, note, part.start_tick, f"{path}.notes[{i}]")
    raw["notes"] = notes
    raw["controllers"] = _write_controllers(raw.get("controllers") or [], part.controllers,
                                            part.start_tick)

    if part.voice is not None:
        reference = raw.get("aiVoice")
        if not isinstance(reference, dict):
            reference = {"langIDs": template.lang_ids()}
            raw["aiVoice"] = reference
        reference["compID"] = part.voice.comp_id
    return raw


def _write_tracks(raw_tracks, tracks):
    singing = [i for i, track in enumerate(raw_tracks)
               if isinstance(track, dict) and track.get("type") == SINGING_TRACK_TYPE]
    source = copy.deepcopy(raw_tracks[singing[0]]) if singing else template.singing_track()

    result = [copy.deepcopy(track) for track in raw_tracks]
    for index in reversed(singing[len(tracks):]):
        del result[index]
    kept = [i for i, track in enumerate(result)
            if isinstance(track, dict) and track.get("type") == SINGING_TRACK_TYPE]
    insert_at = kept[-1] + 1 if kept else len(result)
    while len(kept) < len(tracks):
        addition = copy.deepcopy(source)
        addition["type"] = SINGING_TRACK_TYPE
        result.insert(insert_at, addition)
        insert_at += 1
        kept = [i for i, track in enumerate(result)
                if isinstance(track, dict) and track.get("type") == SINGING_TRACK_TYPE]

    for position, (index, track) in enumerate(zip(kept, tracks, strict=True)):
        raw = result[index]
        raw["name"] = track.name
        parts = _pair_up(raw.get("parts") or [], len(track.parts), template.part)
        for i, (raw_part, part) in enumerate(zip(parts, track.parts, strict=True)):
            _write_part(raw_part, part, f"tracks[{position}].parts[{i}]")
        raw["parts"] = parts
    return result


def _write_tempos(raw_events, tempos):
    if not tempos:
        raise VprFormatError("テンポイベントが 1 件もありません", path="tempos")
    events = _pair_up(raw_events, len(tempos), template.tempo_event)
    for raw, tempo in zip(events, sorted(tempos, key=lambda t: t.tick), strict=True):
        if not math.isfinite(tempo.bpm) or tempo.bpm <= 0.0:
            raise VprFormatError("BPM は正の有限値でなければなりません",
                                 path="tempos", key="bpm", value=tempo.bpm)
        raw["pos"] = tempo.tick
        raw["value"] = round(tempo.bpm * 100)
    return events


def _write_time_signatures(raw_events, signatures):
    if not signatures:
        raise VprFormatError("拍子イベントが 1 件もありません", path="time_signatures")
    ordered = sorted(signatures, key=lambda s: s.tick)
    events = _pair_up(raw_events, len(ordered), template.time_signature_event)

    tick = 0
    bar = 0
    ticks_per_bar = _ticks_per_bar(*DEFAULT_TIME_SIGNATURE)
    for raw, signature in zip(events, ordered, strict=True):
        if signature.numerator < 1 or signature.denominator < 1:
            raise VprFormatError("拍子の分子・分母は 1 以上でなければなりません",
                                 path="time_signatures", key="numerator",
                                 value=(signature.numerator, signature.denominator))
        if signature.tick != tick:
            span = signature.tick - tick
            if span < 0 or span % ticks_per_bar:
                raise VprFormatError("拍子の位置が小節境界に一致しません",
                                     path="time_signatures", key="tick", value=signature.tick)
            bar += span // ticks_per_bar
            tick = signature.tick
        raw["bar"] = bar
        raw["numer"] = signature.numerator
        raw["denom"] = signature.denominator
        ticks_per_bar = _ticks_per_bar(signature.numerator, signature.denominator)
    return events


def _resolve_voices(raw_voices, tracks):
    voices = [copy.deepcopy(voice) for voice in raw_voices if isinstance(voice, dict)]
    defined = {voice.get("compID") for voice in voices}
    for track in tracks:
        for part in track.parts:
            if part.voice is not None and part.voice.comp_id not in defined:
                voices.append({"compID": part.voice.comp_id, "name": part.voice.name})
                defined.add(part.voice.comp_id)
    return voices


def write(project: VprProject) -> bytes:
    if project.resolution != RESOLUTION:
        raise VprFormatError("分解能が形式の固定値と異なります",
                             path="resolution", key="resolution", value=project.resolution)
    base = copy.deepcopy(project.raw_sequence) if isinstance(project.raw_sequence, dict) \
        else template.sequence()
    master = base.setdefault("masterTrack", template.sequence()["masterTrack"])
    tempo = master.setdefault("tempo", {"events": []})
    timesig = master.setdefault("timeSig", {"events": []})

    base["title"] = project.title
    tempo["events"] = _write_tempos(tempo.get("events") or [], project.tempos)
    timesig["events"] = _write_time_signatures(timesig.get("events") or [],
                                               project.time_signatures)
    base["tracks"] = _write_tracks(base.get("tracks") or [], project.tracks)
    raw_voices = base.get("voices")
    base["voices"] = _resolve_voices(raw_voices if isinstance(raw_voices, list) else [], project.tracks)

    return _archive(base, project.entries or {})


def _archive(sequence, entries) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(SEQUENCE_PATH, json.dumps(sequence, ensure_ascii=False))
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def write_file(project: VprProject, path: str | Path) -> None:
    """出力先を原子置換で書き換える。途中で失敗しても既存の出力先は壊れない。"""
    path = Path(path)
    data = write(project)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
