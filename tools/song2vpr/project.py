"""song2vpr が出力する vpr の組み立て。

秒で確定した音符列とテンポ推定から、vpr の公開データモデルを作る。受け取る音符列は互いに重ならない
(前段が満たす)ものとし、ここでは重なりを正さない。形式への直列化は vpr が持つので、
ここが決めるのは「どんなモデルを組み立てるか」——単一のトラックとパート、先頭のテンポと拍子、
秒から tick への写しと、写した結果が音符の要件(重ねない・長さ 0 を出さない)を保つための後始末。

音符の端点は独立に量子化する。秒の長さを直接丸めると、隣接する音符の境界が同じ tick に落ちず、
丸めによる隙間や重なりが出るため。
"""

from dataclasses import dataclass, field

from vpr import Note, Part, TempoEvent, TimeSignature, Track, VoiceBank, VprProject

# 音高の格納範囲。外れた値は端へ丸めて件数を診断へ出す。
_MIDI_RANGE = (0, 127)

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

    残す値は秒の区間が最も長い音符のもの(同じ長さなら先の音符)。同じ位置に潰れた音符は最後の1つ
    以外は必ず tick 長が 0 になるので、tick で比べると区間の長短を反映しない。
    """
    merged = []
    for start_tick, end_tick, note in quantized:
        if merged and merged[-1][0] == start_tick:
            kept_start, kept_end, kept = merged[-1]
            if note.end_sec - note.start_sec > kept.end_sec - kept.start_sec:
                kept = note
            merged[-1] = (kept_start, max(kept_end, end_tick), kept)
            diagnostics.quantized_merged_notes += 1
        else:
            merged.append((start_tick, end_tick, note))
    return merged


def _stretch_empty(merged, diagnostics):
    """長さ 0 になった音符へ 1 tick を与える。

    まとめを済ませた後なので開始 tick は互いに異なり、次の音符は必ず 1 tick 以上先にある。
    """
    spans = []
    for start_tick, end_tick, note in merged:
        if end_tick <= start_tick:
            end_tick = start_tick + 1
            diagnostics.quantized_stretched_notes += 1
        spans.append((start_tick, end_tick, note))
    return spans


def build(notes, tempo, *, name: str) -> BuildResult:
    """秒の音符列とテンポ推定から、出力する vpr のデータモデルを組み立てる。

    name は出力ファイルの基底名で、曲名・トラック名・パート名に入れる。
    """
    diagnostics = Diagnostics()
    quantized = [(tempo.to_tick(note.start_sec), tempo.to_tick(note.end_sec), note)
                 for note in sorted(notes, key=lambda note: note.start_sec)]
    spans = _stretch_empty(_merge_collapsed(quantized, diagnostics), diagnostics)

    written = [Note(start_tick=start_tick, duration_tick=end_tick - start_tick,
                    pitch=_clamp_pitch(note.midi, diagnostics), lyric=note.lyric,
                    velocity=note.velocity, phonemes=list(note.phonemes),
                    is_protected=note.is_protected)
               for start_tick, end_tick, note in spans]

    diagnostics.note_count = len(written)

    # 音符が無いときも長さ 0 のパートを作らないので、1小節分を与える。
    duration_tick = spans[-1][1] if spans else _ticks_per_bar(tempo)
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
