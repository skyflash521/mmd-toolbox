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
    VprProject,
    VprWarning,
)

_RESOLUTION = 480  # tick/四分音符(vpr に格納されない固定値)
_SEQUENCE_PATH = "Project/sequence.json"
_SINGING_TRACK_TYPE = 2


def _ticks_per_bar(numerator: int, denominator: int) -> int:
    """拍子の1小節あたりの tick 長。"""
    return numerator * _RESOLUTION * 4 // denominator


def _time_signatures(events: list[dict]) -> list[TimeSignature]:
    """拍子イベント(小節番号 bar 基準)を、先行小節長を積算して tick へ変換する。"""
    result: list[TimeSignature] = []
    tick = 0
    prev_bar = 0
    # 最初の明示イベントより前の小節は VOCALOID 既定の 4/4 とみなして積算する。
    prev_ticks_per_bar = _ticks_per_bar(4, 4)
    for event in sorted(events, key=lambda e: e["bar"]):
        tick += (event["bar"] - prev_bar) * prev_ticks_per_bar
        result.append(
            TimeSignature(tick=tick, numerator=event["numer"], denominator=event["denom"])
        )
        prev_bar = event["bar"]
        prev_ticks_per_bar = _ticks_per_bar(event["numer"], event["denom"])
    return result


def _notes(raw_notes: list[dict], part_pos: int) -> list[Note]:
    """音符を絶対 tick 化(part 開始位置を加算)し、start_tick 昇順で返す。"""
    notes = [
        Note(
            start_tick=part_pos + note["pos"],
            duration_tick=note["duration"],
            pitch=note["number"],
            lyric=note["lyric"],
            velocity=note["velocity"],
            phonemes=note["phoneme"].split(),
        )
        for note in raw_notes
    ]
    notes.sort(key=lambda note: note.start_tick)
    return notes


def read(src) -> tuple[VprProject, list[VprWarning]]:
    """vpr を読み、データモデル(vpr_io.md §2)と警告を返す。"""
    source = io.BytesIO(src) if isinstance(src, bytes) else src
    with zipfile.ZipFile(source) as archive:
        with archive.open(_SEQUENCE_PATH) as f:
            sequence = json.load(f)

    master = sequence["masterTrack"]
    tempos = [
        TempoEvent(tick=event["pos"], bpm=event["value"] / 100)
        for event in master["tempo"]["events"]
    ]
    time_signatures = _time_signatures(master["timeSig"]["events"])

    tracks: list[Track] = []
    for track in sequence["tracks"]:
        if track.get("type") != _SINGING_TRACK_TYPE:
            continue
        parts = [
            Part(
                name=part["name"],
                start_tick=part["pos"],
                notes=_notes(part.get("notes", []), part["pos"]),
            )
            for part in track.get("parts", [])
        ]
        tracks.append(Track(name=track["name"], parts=parts))

    project = VprProject(
        resolution=_RESOLUTION,
        tempos=tempos,
        time_signatures=time_signatures,
        tracks=tracks,
    )
    warnings: list[VprWarning] = []
    return project, warnings
