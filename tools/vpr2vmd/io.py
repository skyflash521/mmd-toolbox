import re

from vpr import Note, Track, VprProject


class TrackSelectionError(Exception):
    pass


_TRACK_INDEX_PATTERN = re.compile(r"[0-9]+")


def select_track(project: VprProject, track: str | None) -> Track:
    """対象トラックを一意に選べなければ TrackSelectionError を送出する。"""
    tracks = project.tracks
    if not tracks:
        raise TrackSelectionError("対象トラックがありません")
    if track is None:
        return tracks[0]
    # int() は全角数字・前後の空白・符号も受理する。
    if _TRACK_INDEX_PATTERN.fullmatch(track):
        index = int(track)
        if not 0 <= index < len(tracks):
            raise TrackSelectionError(f"トラック INDEX が範囲外: {track!r}")
        return tracks[index]
    matches = [t for t in tracks if t.name == track]
    if not matches:
        raise TrackSelectionError(f"トラック名が一致しません: {track!r}")
    if len(matches) > 1:
        raise TrackSelectionError(f"トラック名が複数一致します: {track!r}")
    return matches[0]


def collect_notes(track: Track) -> list[Note]:
    keyed = [
        ((note.start_tick, -note.duration_tick, part_index, note_index), note)
        for part_index, part in enumerate(track.parts)
        for note_index, note in enumerate(part.notes)
    ]
    keyed.sort(key=lambda item: item[0])
    return [note for _, note in keyed]
