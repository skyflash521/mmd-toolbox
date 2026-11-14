"""実素材を使う人手検査で起動するコマンドラインスクリプト。

使い方: python scripts/check_vpr_output.py <検査対象.vpr> <正解.vpr> [<正解.vpr> ...]

違反は1件1行で標準出力へ出し、音符・並びに対する違反の行は「小節<N>: 」で始まる。違反があれば
終了コード1、無ければ何も出力せず0、引数が足りなければ2で終わる。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from vpr import TimeSignature, read
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libs"))
    from vpr import TimeSignature, read

_EPSILON = 1e-9


@dataclass
class Thresholds:
    min_lyric_note_sec: float
    min_mora_sec: float
    max_continuation_ratio: float


@dataclass
class Violation:
    kind: str
    measure: int | None
    message: str


def seconds_at(tick, tempos, resolution):
    events = sorted(tempos, key=lambda e: e.tick)
    if not events:
        raise ValueError("テンポイベントがありません")
    bpm = events[0].bpm
    sec = 0.0
    pos = 0
    for event in events:
        t = max(event.tick, 0)
        if t >= tick:
            break
        if t > pos:
            sec += (t - pos) * 60.0 / (bpm * resolution)
            pos = t
        bpm = event.bpm
    sec += (tick - pos) * 60.0 / (bpm * resolution)
    return sec


def measure_number(tick, time_signatures, resolution):
    events = sorted(time_signatures, key=lambda e: e.tick)
    if not events:
        events = [TimeSignature(tick=0, numerator=4, denominator=4)]
    segments = []
    for event in events:
        t = max(event.tick, 0)
        if segments and segments[-1][0] == t:
            segments[-1] = (t, event.numerator, event.denominator)
        else:
            segments.append((t, event.numerator, event.denominator))
    if segments[0][0] > 0:
        first = segments[0]
        segments.insert(0, (0, first[1], first[2]))
    measures = 0
    for i, (start, numerator, denominator) in enumerate(segments):
        ticks_per_measure = resolution * 4 * numerator / denominator
        end = segments[i + 1][0] if i + 1 < len(segments) else None
        if end is not None and end <= tick:
            # 編集器は拍子の変わり目から新しい小節を始める。
            measures += max(1, math.ceil((end - start) / ticks_per_measure - _EPSILON))
            continue
        return measures + int((tick - start) // ticks_per_measure) + 1
    raise AssertionError("unreachable")


def _iter_parts(project):
    for track in project.tracks:
        for part in track.parts:
            yield part


def _has_lyric(note):
    return note.lyric not in ("", "-")


def _runs(notes):
    runs = []
    for note in sorted(notes, key=lambda n: n.start_tick):
        if runs and runs[-1][-1].start_tick + runs[-1][-1].duration_tick == note.start_tick:
            runs[-1].append(note)
        else:
            runs.append([note])
    return runs


def _lyric_note_secs(project):
    values = []
    for part in _iter_parts(project):
        for note in part.notes:
            if not _has_lyric(note):
                continue
            sec = seconds_at(
                note.start_tick + note.duration_tick, project.tempos, project.resolution
            ) - seconds_at(note.start_tick, project.tempos, project.resolution)
            values.append((sec, note.start_tick))
    return values


def _mora_run_secs(project):
    values = []
    for part in _iter_parts(project):
        for run in _runs(part.notes):
            lyric_count = sum(1 for note in run if _has_lyric(note))
            if lyric_count == 0:
                continue
            total = seconds_at(
                run[-1].start_tick + run[-1].duration_tick, project.tempos, project.resolution
            ) - seconds_at(run[0].start_tick, project.tempos, project.resolution)
            values.append((total / lyric_count, run[0].start_tick))
    return values


def _continuation_ratio(project):
    total = 0
    continuation = 0
    for part in _iter_parts(project):
        for note in part.notes:
            total += 1
            if note.lyric == "-":
                continuation += 1
    return continuation / total if total else 0.0


def compute_thresholds(references):
    lyric_mins = []
    mora_mins = []
    ratios = []
    for ref in references:
        lyric_secs = [sec for sec, _ in _lyric_note_secs(ref)]
        if lyric_secs:
            lyric_mins.append(min(lyric_secs))
        mora_secs = [sec for sec, _ in _mora_run_secs(ref)]
        if mora_secs:
            mora_mins.append(min(mora_secs))
        ratios.append(_continuation_ratio(ref))
    if not lyric_mins or not mora_mins:
        raise ValueError("正解 vpr に表示歌詞を持つ音符がありません")
    return Thresholds(
        min_lyric_note_sec=min(lyric_mins),
        min_mora_sec=min(mora_mins),
        max_continuation_ratio=1.5 * max(ratios),
    )


def check_project(project, thresholds):
    violations = []

    def measure_of(tick):
        return measure_number(tick, project.time_signatures, project.resolution)

    for part in _iter_parts(project):
        notes = sorted(part.notes, key=lambda n: n.start_tick)
        furthest_end = 0
        for i, note in enumerate(notes):
            if i > 0 and note.duration_tick > 0 and note.start_tick < furthest_end:
                violations.append(
                    Violation(
                        kind="overlap",
                        measure=measure_of(note.start_tick),
                        message="音符が重なっている",
                    )
                )
            furthest_end = max(furthest_end, note.start_tick + note.duration_tick)
            if note.duration_tick <= 0:
                violations.append(
                    Violation(
                        kind="zero_duration",
                        measure=measure_of(note.start_tick),
                        message="長さ0の音符",
                    )
                )

    for sec, tick in _lyric_note_secs(project):
        if sec < thresholds.min_lyric_note_sec - _EPSILON:
            violations.append(
                Violation(
                    kind="short_lyric_note",
                    measure=measure_of(tick),
                    message=(
                        f"表示歌詞を持つ音符が短すぎる"
                        f"({sec * 1000:.0f}ミリ秒 < 基準{thresholds.min_lyric_note_sec * 1000:.0f}ミリ秒)"
                    ),
                )
            )

    for sec, tick in _mora_run_secs(project):
        if sec < thresholds.min_mora_sec - _EPSILON:
            violations.append(
                Violation(
                    kind="short_mora_run",
                    measure=measure_of(tick),
                    message=(
                        f"切れ目なく続く並びの1モーラあたりが短すぎる"
                        f"({sec * 1000:.0f}ミリ秒 < 基準{thresholds.min_mora_sec * 1000:.0f}ミリ秒)"
                    ),
                )
            )

    ratio = _continuation_ratio(project)
    if ratio > thresholds.max_continuation_ratio + _EPSILON:
        violations.append(
            Violation(
                kind="continuation_ratio",
                measure=None,
                message=(
                    f"継続の音符の割合が基準を超えている"
                    f"({ratio:.0%} > 上限{thresholds.max_continuation_ratio:.0%})"
                ),
            )
        )
    return violations


def main(argv):
    if len(argv) < 3:
        print(
            "使い方: python scripts/check_vpr_output.py <検査対象.vpr> <正解.vpr> [<正解.vpr> ...]",
            file=sys.stderr,
        )
        return 2
    target, _ = read(argv[1])
    references = [read(path)[0] for path in argv[2:]]
    violations = check_project(target, compute_thresholds(references))
    for violation in violations:
        prefix = f"小節{violation.measure}: " if violation.measure is not None else ""
        print(f"{prefix}{violation.message}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
