"""vpr の抽出層。

vpr の読み込みは形式I/Oモジュール `vpr` に委譲し、本モジュールは vpr2vmd 固有の入口処理の
うち、対象とする歌唱トラックの選択と、選択トラックの全パート音符の統合・安定整列を担う。
重なり解決・時刻⇔フレーム変換・口形写像は口形イベント確定で扱う。
"""

import re

from vpr import Note, Track, VprProject


class TrackSelectionError(Exception):
    """`--track` が対象トラックを一意に選べない(範囲外・不一致・複数一致・トラック無し)。"""


_INDEX_PATTERN = re.compile(r"[0-9]+")  # INDEX と解釈する指定(半角数字だけ・符号なし)


def select_track(project: VprProject, track: str | None) -> Track:
    """対象の歌唱トラックを選ぶ。

    `track` が None なら先頭トラック。半角数字だけからなる指定は 0-based の INDEX、それ以外は
    `Track.name` と解釈する。名前の複数一致・不一致・INDEX 範囲外・トラック無しは
    `TrackSelectionError`。
    """
    tracks = project.tracks
    if not tracks:
        raise TrackSelectionError("対象トラックがありません")
    if track is None:
        return tracks[0]
    # int() は全角数字・前後空白・符号も受理するので、それらを INDEX と解釈しないよう
    # 半角数字だけの指定に絞る(全角数字だけの名前を持つトラックを名前で選べるようにする)。
    if _INDEX_PATTERN.fullmatch(track):
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
    """選択トラックの全パートの音符を統合し安定整列する。

    整列順は (start_tick 昇順, duration_tick 降順, パート出現順, パート内の音符索引昇順)。
    duration_tick の降順は符号反転で表す。整列キーが各音符を一意に定めるため結果は決定的。
    """
    indexed = [
        (note.start_tick, -note.duration_tick, part_index, note_index, note)
        for part_index, part in enumerate(track.parts)
        for note_index, note in enumerate(part.notes)
    ]
    indexed.sort(key=lambda item: item[:4])
    return [item[4] for item in indexed]
