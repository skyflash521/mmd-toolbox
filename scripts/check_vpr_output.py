"""song2vpr が出力した vpr を、人が作った正解 vpr 由来の基準で機械検査する。

使い方: python scripts/check_vpr_output.py <検査対象.vpr> <正解.vpr> [<正解.vpr> ...]

検査項目は5つ。音符が重ならない・長さ0の音符が無い(いずれも song2vpr 仕様の不変条件)、
表示歌詞を持つ音符の長さが正解の最短を下回らない・切れ目なく続く並びの1モーラあたりの長さが
正解の最短を下回らない・継続の音符の割合が正解の最大値の1.5倍を超えない(基準は引数の正解から
毎回算出し、固定値をここに書かない)。違反は編集器の表示に合わせた1始まりの小節番号とともに
列挙して終了コード1で終わる。違反が無ければ何も出力しない。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from vpr import TimeSignature, read
except ImportError:  # 単体起動ではリポジトリの libs を import パスへ足す
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libs"))
    from vpr import TimeSignature, read

# 「等しい値は合格」を浮動小数点の丸めで壊さないための許容差。
_EPSILON = 1e-9


@dataclass
class Thresholds:
    """正解 vpr から算出した基準。"""

    min_lyric_note_sec: float  # 表示歌詞を持つ音符の長さの下限(秒)
    min_mora_sec: float  # 切れ目なく続く並びの1モーラあたりの長さの下限(秒)
    max_continuation_ratio: float  # 継続の音符の割合の上限(正解の最大値の1.5倍)


@dataclass
class Violation:
    kind: str
    measure: int | None  # 1始まりの小節番号。曲全体の性質の違反では None
    message: str


def seconds_at(tick, tempos, resolution):
    """テンポイベント列に沿って tick を秒へ区分的に換算する。

    先頭イベントより前の tick は先頭イベントのテンポで換算する。
    """
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
    """拍子イベント列に沿って、1始まりの小節番号(編集器の表示)を求める。"""
    events = sorted(time_signatures, key=lambda e: e.tick)
    if not events:
        events = [TimeSignature(tick=0, numerator=4, denominator=4)]
    segments = []  # (開始tick, 分子, 分母)。同一 tick は後のイベントで上書き
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
            # 拍子の変わり目までの小節数。端数の小節は1小節として数える
            # (編集器は変わり目から新しい小節を始める)。
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
    """前の音符の終端と次の音符の開始が一致して続く並びへ分ける。"""
    runs = []
    for note in sorted(notes, key=lambda n: n.start_tick):
        if runs and runs[-1][-1].start_tick + runs[-1][-1].duration_tick == note.start_tick:
            runs[-1].append(note)
        else:
            runs.append([note])
    return runs


def _lyric_note_secs(project):
    """表示歌詞を持つ各音符の長さ(秒)と開始 tick。"""
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
    """切れ目なく続く各並びの1モーラあたりの長さ(秒)と開始 tick。

    1モーラあたりの長さは、並びの全長を並びの中の表示歌詞を持つ音符の数で割った値。
    表示歌詞を持つ音符の無い並びは対象にしない。
    """
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
    """継続の音符(表示歌詞が「-」)の全音符に対する割合。音符が無ければ 0。"""
    total = 0
    continuation = 0
    for part in _iter_parts(project):
        for note in part.notes:
            total += 1
            if note.lyric == "-":
                continuation += 1
    return continuation / total if total else 0.0


def compute_thresholds(references):
    """正解 vpr の列から基準を算出する。長さ系は正解ごとの最短の最小、割合は最大の1.5倍。"""
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
    """検査対象の vpr を基準と突き合わせ、違反を列挙する。"""
    violations = []

    def measure_of(tick):
        return measure_number(tick, project.time_signatures, project.resolution)

    for part in _iter_parts(project):
        notes = sorted(part.notes, key=lambda n: n.start_tick)
        reach = 0  # ここまでに見た発音区間の終端の最大
        for i, note in enumerate(notes):
            # 長さ0の音符の発音区間は空なので重なりにならない(長さ0は別の違反として報告する)。
            if i > 0 and note.duration_tick > 0 and note.start_tick < reach:
                violations.append(
                    Violation(
                        kind="overlap",
                        measure=measure_of(note.start_tick),
                        message="音符が重なっている",
                    )
                )
            reach = max(reach, note.start_tick + note.duration_tick)
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
