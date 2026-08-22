"""song2vpr が出力する vpr の組み立て。

秒で確定した音符列とテンポ推定から、vpr の公開データモデルを作る。受け取る音符列は互いに重ならない
(前段が満たす)ものとし、ここでは重なりを正さない。形式への直列化は vpr が持つので、
ここが決めるのは「どんなモデルを組み立てるか」——単一のトラックとパート、先頭のテンポと拍子、
秒から tick への写しと、写した結果が音符の要件(重ねない・長さ 0 を出さない)を保つための後始末。

音符の端点は独立に量子化する。秒の長さを直接丸めると、隣接する音符の境界が同じ tick に落ちず、
丸めによる隙間や重なりが出るため。
"""

import math
from dataclasses import dataclass, field

from vpr import Note, Part, TempoEvent, TimeSignature, Track, VoiceBank, VprProject

from .lyrics import CONTINUATION

# 音高の格納範囲。外れた値は端へ丸めて件数を診断へ出す。
_MIDI_RANGE = (0, 127)

# 歌える長さの下限(秒)。表示歌詞を持つ音符がこの長さを下回ると VOCALOID が子音を発音し
# きれないため、tick へ写した後に切れ目なく続く並びの中で時間を配り直して確保する。
# CLI へは公開しない内部の値。
_MIN_SINGABLE_SEC = 0.028

# 形式が要求するボイスバンクの指定。利用者の環境を調べて選び分けることはせず固定値を与える
# (特定の歌手を推すものではなく、利用者は出力した vpr を VOCALOID で開いて差し替える)。
_VOICE = VoiceBank(comp_id="BHKCEKSYNBXG3HB2", name="HATSUNE_MIKU_V6_ORIGINAL")


@dataclass
class Diagnostics:
    """組み立ての過程で数えた件数。"""

    note_count: int = 0  # tick へ写した後の音符数(まとめ・引き伸ばしを済ませた最終の数)
    pitch_clamped_notes: int = 0
    quantized_merged_notes: int = 0
    quantized_stretched_notes: int = 0
    short_notes: int = 0  # 配り直しでも歌える長さの下限に届かないまま残った音符数


@dataclass
class BuildResult:
    project: VprProject
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


def _ticks_per_bar(tempo) -> int:
    return tempo.numerator * tempo.resolution * 4 // tempo.denominator


def _clamp_pitch(midi, diagnostics):
    clamped = max(_MIDI_RANGE[0], min(_MIDI_RANGE[1], midi))
    if clamped != midi:
        diagnostics.pitch_clamped_notes += 1
    return clamped


def _merge_collapsed(quantized, diagnostics):
    """同じ tick へ潰れた音符を1つにまとめる。

    音高と強弱は秒の区間が最も長い音符のもの(同じ長さなら先の音符)。同じ位置に潰れた音符は
    最後の1つ以外は必ず tick 長が 0 になるので、tick で比べると区間の長短を反映しない。
    表示歌詞・音素列・音素の保護は、まとめた中に音節の先頭の音符があればその音符(複数あれば先の
    音符)のものにする。継続の音符は自分より前に同じ音節の音符があることを表す表記なので、先頭を
    飲み込んだ結果としてそれだけが残ると、音素列を持つ音符が1つも無い音節ができてしまう。
    まとめた結果を (開始 tick, 終端 tick, 最も長い音符, 音節の先頭の音符) で返す。
    """
    merged = []
    for start_tick, end_tick, note in quantized:
        head = None if note.lyric == CONTINUATION else note
        if merged and merged[-1][0] == start_tick:
            kept_start, kept_end, longest, kept_head = merged[-1]
            if note.end_sec - note.start_sec > longest.end_sec - longest.start_sec:
                longest = note
            merged[-1] = (kept_start, max(kept_end, end_tick), longest,
                          kept_head if kept_head is not None else head)
            diagnostics.quantized_merged_notes += 1
        else:
            merged.append((start_tick, end_tick, note, head))
    return merged


def _stretch_empty(merged, diagnostics):
    """長さ 0 になった音符へ 1 tick を与える。

    まとめを済ませた後なので開始 tick は互いに異なり、次の音符は必ず 1 tick 以上先にある。
    """
    spans = []
    for start_tick, end_tick, longest, head in merged:
        if end_tick <= start_tick:
            end_tick = start_tick + 1
            diagnostics.quantized_stretched_notes += 1
        spans.append((start_tick, end_tick, longest, head))
    return spans


def _min_singable_ticks(tempo) -> int:
    """歌える長さの下限を tick で表した値(下限以上を保証する最小の整数 tick)。"""
    return math.ceil(_MIN_SINGABLE_SEC * tempo.bpm * tempo.resolution / 60.0 - 1e-9)


def _redistribute_run(run, floor):
    """切れ目なく続く並びの中で、全長と順序を保ったまま各音符へ下限以上の長さを配り直す。

    まず全音符へ下限を確保し、残り(全長 − 音符数×下限)を元の長さが下限を超える音符へ
    (元の長さ − 下限)の比で配る。整数 tick は下限+比例配分の整数部分を与え、残った tick を
    端数の大きい音符から順に(端数が同じなら先の音符から)1 tick ずつ配って全長へ合わせる。
    """
    total = run[-1].start_tick + run[-1].duration_tick - run[0].start_tick
    surplus = total - len(run) * floor
    weights = [max(0, note.duration_tick - floor) for note in run]
    weight_total = sum(weights)
    # 端数の比較は整数の剰余で行う(分母が共通なので剰余の大小が端数の大小と一致し、
    # 浮動小数点の丸めで数学的に同じ端数の順位が崩れない)。
    quotients = [divmod(surplus * weight, weight_total) for weight in weights]
    durations = [floor + quotient for quotient, _ in quotients]
    order = sorted(range(len(run)), key=lambda i: (-quotients[i][1], i))
    for i in order[:total - sum(durations)]:
        durations[i] += 1

    redistributed = []
    position = run[0].start_tick
    for note, duration in zip(run, durations, strict=True):
        redistributed.append(Note(
            start_tick=position, duration_tick=duration, pitch=note.pitch,
            lyric=note.lyric, velocity=note.velocity, phonemes=note.phonemes,
            is_protected=note.is_protected))
        position += duration
    return redistributed


def _ensure_singable(notes, floor, diagnostics):
    """表示歌詞を持つ音符が歌える長さの下限を満たすよう、並びの中で時間を配り直す。

    下限を下回る表示歌詞音符を含む並びだけを対象にする。単独の音符は次の音符へ食い込まない
    範囲で下限まで伸ばす。全長が音符数×下限に満たない並びは変えず、下限に届かないまま残った
    音符数を診断へ数える。
    """
    runs = []
    for note in notes:
        if runs and runs[-1][-1].start_tick + runs[-1][-1].duration_tick == note.start_tick:
            runs[-1].append(note)
        else:
            runs.append([note])

    result = []
    for index, run in enumerate(runs):
        if not any(note.lyric != CONTINUATION and note.duration_tick < floor for note in run):
            result.extend(run)
            continue
        if len(run) == 1:
            note = run[0]
            limit = floor
            if index + 1 < len(runs):
                limit = min(floor, runs[index + 1][0].start_tick - note.start_tick)
            duration = max(note.duration_tick, limit)
            if duration < floor:
                diagnostics.short_notes += 1
            result.append(Note(start_tick=note.start_tick, duration_tick=duration,
                               pitch=note.pitch, lyric=note.lyric, velocity=note.velocity,
                               phonemes=note.phonemes, is_protected=note.is_protected))
            continue
        total = run[-1].start_tick + run[-1].duration_tick - run[0].start_tick
        if total < len(run) * floor:
            diagnostics.short_notes += sum(1 for note in run if note.duration_tick < floor)
            result.extend(run)
            continue
        result.extend(_redistribute_run(run, floor))
    return result


def build(notes, tempo, *, name: str) -> BuildResult:
    """秒の音符列とテンポ推定から、出力する vpr のデータモデルを組み立てる。

    name は出力ファイルの基底名で、曲名・トラック名・パート名に入れる。
    """
    diagnostics = Diagnostics()
    quantized = [(tempo.to_tick(note.start_sec), tempo.to_tick(note.end_sec), note)
                 for note in sorted(notes, key=lambda note: note.start_sec)]
    spans = _stretch_empty(_merge_collapsed(quantized, diagnostics), diagnostics)

    written = [Note(start_tick=start_tick, duration_tick=end_tick - start_tick,
                    pitch=_clamp_pitch(longest.midi, diagnostics), velocity=longest.velocity,
                    lyric=(head or longest).lyric,
                    phonemes=list((head or longest).phonemes),
                    is_protected=(head or longest).is_protected)
               for start_tick, end_tick, longest, head in spans]
    written = _ensure_singable(written, _min_singable_ticks(tempo), diagnostics)

    diagnostics.note_count = len(written)

    # 音符が無いときも長さ 0 のパートを作らないので、1小節分を与える。パート長は配り直しを
    # 済ませた最終の終端に合わせる(末尾の音符が伸びた場合に追随する)。
    duration_tick = (written[-1].start_tick + written[-1].duration_tick
                     if written else _ticks_per_bar(tempo))
    part = Part(name=name, start_tick=0, duration_tick=duration_tick, voice=_VOICE, notes=written)

    project = VprProject(
        resolution=tempo.resolution,
        tempos=[TempoEvent(tick=0, bpm=tempo.bpm)],
        time_signatures=[TimeSignature(tick=0, numerator=tempo.numerator,
                                       denominator=tempo.denominator)],
        tracks=[Track(name=name, parts=[part])],
        title=name,
    )
    return BuildResult(project=project, diagnostics=diagnostics)
