"""vpr 書き出し。

読んだプロジェクトは生の JSON(`raw_sequence`)を基礎にし、手組みのプロジェクトは骨組みを基礎に
する。基礎の各要素へ公開モデルの値を上書きする形にして、公開モデルへ写さないキー・トラックを
落とさない。

公開モデルと生 JSON の対応付けは、読みが使う並べ替えを再現して一意に定める。読みは音符を位置で
安定ソートし、拍子を小節番号で安定ソートし、それ以外は出現順のまま公開するので、書き出しも同じ
順序で突き合わせれば同じ置換になる。

「読みが写せた形か」の判定は読み側(`io`)にだけ置き、ここはそれを使う。公開モデルが空のときに
生の構造を消してよいかは読みの受理条件そのものなので、判定を写すと対称性が静かに崩れる。
"""

import copy
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

from . import template
from .constants import RESOLUTION, SEQUENCE_PATH, SINGING_TRACK_TYPE
from .io import accepts_depth_envelope, accepts_vibrato
from .types import VprFormatError

_MIDI_RANGE = (0, 127)
_VELOCITY_RANGE = (0, 127)


def _ticks_per_bar(numerator: int, denominator: int) -> int:
    return numerator * RESOLUTION * 4 // denominator


def _pair_up(raw_items, count, fallback):
    """生の配列を公開モデルの要素数へ合わせる。

    余った公開モデルの要素は生の先頭要素(生が空なら骨組み)を複製して作り、余った生の要素は捨てる。
    配列ごと作り直さないのは、要素別の未解釈キーを保つため。
    """
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
    # 形式は音素列を空白区切りの1つの文字列として持つので、要素自体が空白を含むと別の列として
    # 読み戻される(空要素も同じく列から消える)。
    for phoneme in note.phonemes:
        if not phoneme or phoneme.split() != [phoneme]:
            raise VprFormatError("音素は空白を含まない非空の文字列でなければなりません",
                                 path=path, key="phonemes", value=phoneme)
    if note.vibrato is None:
        return
    if note.vibrato.duration <= 0:
        # 形式では区間長 0 がビブラート無しを表し、読みはそれを None として写す(往復しない)。
        raise VprFormatError("ビブラートの区間長は正でなければなりません",
                             path=f"{path}.vibrato", key="duration", value=note.vibrato.duration)
    if note.vibrato.duration > note.duration_tick:
        # 区間始端は「音符終端 − 区間長」なので、音符長を超える区間は音符の開始前へはみ出す。
        raise VprFormatError("ビブラートの区間長が音符の長さを超えています",
                             path=f"{path}.vibrato", key="duration", value=note.vibrato.duration)
    # 制御点は区間始端からの相対位置で格納するので、区間の外にある点は書けない(音符や区間長だけを
    # 動かすと絶対 tick の点が区間から外れるため、区間長の検査と対で見る)。
    span_start = note.start_tick + note.duration_tick - note.vibrato.duration
    for key, points in (("depths", note.vibrato.depths), ("rates", note.vibrato.rates)):
        for point in points:
            if not 0 <= point.pos - span_start <= note.vibrato.duration:
                raise VprFormatError("ビブラートの制御点が区間の外にあります",
                                     path=f"{path}.vibrato.{key}", key="pos", value=point.pos)


def _vibrato_points(points, span_start):
    """絶対 tick の制御点を、格納表現(ビブラート区間始端からの相対位置)へ戻す。"""
    return [{"pos": point.pos - span_start, "value": point.value} for point in points]


def _write_vibrato(raw, note):
    """ビブラートを書く。

    公開値が None のときは、読みが写せた形の生の値だけを区間長 0(=ビブラート無し)で無効化し、
    欠落・型不正で写せなかった生の構造には手を触れない(読み取り時に写せなかった値を消さない)。
    """
    previous = raw.get("vibrato")
    if note.vibrato is None:
        if accepts_vibrato(previous):
            previous["duration"] = 0
        return
    span_start = note.start_tick + note.duration_tick - note.vibrato.duration
    written = {"type": note.vibrato.type, "duration": note.vibrato.duration}
    # 形式は自動化曲線のキー自体を持たない音符を許し、読みは欠落を空として写す。空のまま書き足すと
    # 無加工の書き戻しで生の形が変わるので、値があるか元から持っていたキーだけを書く。
    for key, points in (("depths", note.vibrato.depths), ("rates", note.vibrato.rates)):
        if points or (isinstance(previous, dict) and key in previous):
            written[key] = _vibrato_points(points, span_start)
    raw["vibrato"] = written


def _write_ai_expression(raw, note):
    """深さ包絡を書く。読みが None とした音符へ値を作らず、他の表現パラメータには触らない。

    公開値が None のとき2キーを消すのは、読みがその2つを写せた場合だけにする(片方だけ・型不正で
    写せなかった生の値は、公開モデルの None が「元から無い」を意味するので残す)。
    """
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
    """骨組みまたは生の音符へ公開モデルの値を上書きする。"""
    _check_note(note, path)
    raw["pos"] = note.start_tick - part_start
    raw["duration"] = note.duration_tick
    raw["number"] = note.pitch
    raw["lyric"] = note.lyric
    raw["phoneme"] = " ".join(note.phonemes)
    raw["velocity"] = note.velocity
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
    """歌唱トラックだけを公開モデルと対応させ、それ以外は変更せず相対順序を保つ。"""
    singing = [i for i, track in enumerate(raw_tracks)
               if isinstance(track, dict) and track.get("type") == SINGING_TRACK_TYPE]
    source = copy.deepcopy(raw_tracks[singing[0]]) if singing else template.singing_track()

    result = [copy.deepcopy(track) for track in raw_tracks]
    # 公開モデルより多い歌唱トラックは捨て、少なければ最後の歌唱トラックの直後(無ければ末尾)へ足す。
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
    """公開モデルの tick を小節番号へ逆変換して書く。"""
    if not signatures:
        raise VprFormatError("拍子イベントが 1 件もありません", path="time_signatures")
    ordered = sorted(signatures, key=lambda s: s.tick)
    events = _pair_up(raw_events, len(ordered), template.time_signature_event)

    tick = 0
    bar = 0
    ticks_per_bar = _ticks_per_bar(4, 4)
    for raw, signature in zip(events, ordered, strict=True):
        # 分母は2の冪へ限定しない(形式は整数としか定めず、読みも任意の整数分母で小節長を出す)。
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
    """パートが参照するボイスバンクの定義を、欠けていれば足す(既存は置き換えない)。"""
    voices = [copy.deepcopy(voice) for voice in raw_voices if isinstance(voice, dict)]
    defined = {voice.get("compID") for voice in voices}
    for track in tracks:
        for part in track.parts:
            if part.voice is not None and part.voice.comp_id not in defined:
                voices.append({"compID": part.voice.comp_id, "name": part.voice.name})
                defined.add(part.voice.comp_id)
    return voices


def write(project) -> bytes:
    """データモデルを vpr(ZIP)へ直列化する。"""
    if project.resolution != RESOLUTION:
        # 形式は分解能をファイルへ格納せず固定値とするため、他の分解能は表現できない。
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
    base["voices"] = _resolve_voices(base.get("voices") or [], project.tracks)

    buffer = _archive(base, project.entries or {})
    return buffer


def _archive(sequence, entries) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(SEQUENCE_PATH, json.dumps(sequence, ensure_ascii=False))
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def write_file(project, path) -> None:
    """原子的に書き出す。

    同ディレクトリの一時ファイルへ書いて fsync し、os.replace で原子置換する。書き込み途中の中断・
    ディスクフルでも、既存の出力先(入力と同一パスへの上書きを含む)を破損させない。
    """
    path = Path(path)
    data = write(project)  # 検査で拒否される場合は一時ファイルを作る前に落とす
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
