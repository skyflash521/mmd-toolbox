"""出力する vpr の組み立てのテスト。

入力の音符・テンポは合成し、結果は公開データモデルと書き出した vpr の読み戻しで観測する。
"""

import pytest

from song2vpr import project
from song2vpr.lyrics import SungNote
from song2vpr.tempo import TempoEstimate
from vpr import read, write_file


def _tempo(bpm=120.0, numerator=4, denominator=4, first_bar_sec=0.0):
    return TempoEstimate(bpm=bpm, numerator=numerator, denominator=denominator,
                         beat_offset_sec=first_bar_sec, first_bar_sec=first_bar_sec,
                         tempo_source="estimated", time_signature_source="estimated")


def _sung(start_sec, end_sec, midi=60, lyric="あ", phonemes=None, velocity=64):
    return SungNote(start_sec=start_sec, end_sec=end_sec, midi=midi, lyric=lyric,
                    phonemes=list(phonemes) if phonemes is not None else ["a"], velocity=velocity)


@pytest.mark.parametrize("is_protected", [True, False])
def test_the_phoneme_protection_reaches_the_written_note(is_protected):
    """音素の保護は、組み立てた vpr の音符まで運ぶ(付与の段だけで持っていても書き出されない)。"""
    note = SungNote(start_sec=0.0, end_sec=0.5, midi=60, lyric="か", phonemes=["k", "a"],
                    velocity=64, is_protected=is_protected)
    result = project.build([note], _tempo(), name="song")
    assert result.project.tracks[0].parts[0].notes[0].is_protected is is_protected


def test_project_has_one_singing_track_with_one_part():
    result = project.build([_sung(0.0, 0.5)], _tempo(), name="song")
    assert len(result.project.tracks) == 1
    assert len(result.project.tracks[0].parts) == 1


def test_names_use_the_output_base_name():
    """曲名・トラック名・パート名には出力ファイルの基底名を入れる。"""
    result = project.build([_sung(0.0, 0.5)], _tempo(), name="my_song")
    track = result.project.tracks[0]
    assert (result.project.title, track.name, track.parts[0].name) == \
           ("my_song", "my_song", "my_song")


def test_part_starts_at_zero_and_ends_at_the_last_note():
    tempo = _tempo()
    result = project.build([_sung(0.0, 0.5), _sung(0.5, 1.0)], tempo, name="song")
    part = result.project.tracks[0].parts[0]
    assert part.start_tick == 0
    assert part.duration_tick == tempo.to_tick(1.0)


def test_part_of_a_project_without_notes_is_one_bar_long():
    """長さ 0 のパートを作らない。"""
    result = project.build([], _tempo(bpm=120.0, numerator=3, denominator=8), name="song")
    part = result.project.tracks[0].parts[0]
    assert part.notes == []
    assert part.duration_tick == 3 * 480 * 4 // 8


def test_tempo_and_time_signature_are_single_events_at_the_head():
    result = project.build([_sung(0.0, 0.5)], _tempo(bpm=136.0, numerator=3, denominator=8),
                           name="song")
    assert [(t.tick, t.bpm) for t in result.project.tempos] == [(0, 136.0)]
    assert [(s.tick, s.numerator, s.denominator) for s in result.project.time_signatures] == \
           [(0, 3, 8)]


def test_part_has_no_controller_curves():
    """曲線は利用者が VOCALOID で付けるので、下書きには載せない。"""
    result = project.build([_sung(0.0, 0.5)], _tempo(), name="song")
    assert result.project.tracks[0].parts[0].controllers == []


def test_part_specifies_a_voice_bank():
    """形式が要求するボイスバンクの指定はツール側が与える。"""
    voice = project.build([_sung(0.0, 0.5)], _tempo(), name="song").project.tracks[0].parts[0].voice
    assert voice is not None
    assert voice.comp_id and voice.name


def test_note_fields_are_carried_over():
    tempo = _tempo()
    note = _sung(0.5, 1.0, midi=62, lyric="い", phonemes=["i"], velocity=100)
    written = project.build([note], tempo, name="song").project.tracks[0].parts[0].notes[0]
    assert (written.start_tick, written.pitch, written.lyric, written.velocity) == \
           (tempo.to_tick(0.5), 62, "い", 100)
    assert written.phonemes == ["i"]
    assert written.duration_tick == tempo.to_tick(1.0) - tempo.to_tick(0.5)


def test_notes_are_not_shifted_onto_the_bar_line():
    """曲の頭が小節線に揃っていなくても、音符は入力の時刻をそのまま写した位置に置く。"""
    tempo = _tempo(first_bar_sec=0.37)
    result = project.build([_sung(0.5, 1.0)], tempo, name="song")
    part = result.project.tracks[0].parts[0]
    assert part.start_tick == 0
    assert part.notes[0].start_tick == tempo.to_tick(0.5)
    assert [t.tick for t in result.project.tempos] == [0]
    assert [s.tick for s in result.project.time_signatures] == [0]


def test_adjacent_notes_share_the_boundary_tick():
    """秒の時点で切れ目なく続いていた音符は tick でも切れ目なく続く。"""
    notes = project.build([_sung(0.0, 0.31), _sung(0.31, 0.62)], _tempo(),
                          name="song").project.tracks[0].parts[0].notes
    assert notes[0].start_tick + notes[0].duration_tick == notes[1].start_tick


def test_note_that_rounds_to_zero_length_is_stretched_to_one_tick():
    result = project.build([_sung(0.0, 0.0001), _sung(0.5, 1.0)], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert notes[0].duration_tick == 1
    assert result.diagnostics.quantized_stretched_notes == 1


def test_notes_that_collapse_to_the_same_tick_are_merged():
    """同じ位置に潰れた音符は1つにまとめ、値は最も長い音符から採り、区間は範囲の終端まで伸ばす。"""
    # 120 BPM では 1 tick = 1/960 秒。最も長い音符を中間に置いて、選択が並び順(先頭・最後の
    # どちらでも)でなく長さで決まることを見る。値は4つとも同じ音符から採る。
    result = project.build([_sung(0.0, 0.0001, midi=60, lyric="あ", phonemes=["a"], velocity=10),
                            _sung(0.0001, 0.0005, midi=62, lyric="い", phonemes=["i"], velocity=20),
                            _sung(0.0005, 0.0006, midi=64, lyric="う", phonemes=["M"], velocity=30)],
                           _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert len(notes) == 1
    assert (notes[0].pitch, notes[0].lyric, notes[0].phonemes, notes[0].velocity) == \
           (62, "い", ["i"], 20)
    assert notes[0].start_tick + notes[0].duration_tick == _tempo().to_tick(0.0006)
    assert result.diagnostics.quantized_merged_notes == 2


def test_notes_of_the_same_length_are_merged_into_the_earlier_one():
    notes = project.build([_sung(0.0, 0.0005, midi=60), _sung(0.0005, 0.001, midi=62)],
                          _tempo(), name="song").project.tracks[0].parts[0].notes
    assert [n.pitch for n in notes] == [60]


@pytest.mark.parametrize(("midi", "expected"), [(-3, 0), (200, 127)])
def test_pitch_outside_the_storable_range_is_clamped(midi, expected):
    result = project.build([_sung(0.0, 0.5, midi=midi)], _tempo(), name="song")
    assert result.project.tracks[0].parts[0].notes[0].pitch == expected
    assert result.diagnostics.pitch_clamped_notes == 1


def test_notes_do_not_overlap_after_merging_and_stretching():
    """まとめと引き伸ばしを経ても、重ならず長さ 0 も残らない。

    刻みより短い音符を続けて、丸めの衝突と 1 tick の付与が同じ列で起きる入力にする。
    """
    # 先頭2つは同じ tick へ潰れてまとまり、休符で隔てられた3つ目は単独で長さ 0 になって伸びる。
    result = project.build([_sung(0.0, 0.0005), _sung(0.0005, 0.0011),
                            _sung(0.01, 0.0105), _sung(0.0115, 0.5)], _tempo(), name="song")
    notes = result.project.tracks[0].parts[0].notes
    assert (result.diagnostics.quantized_merged_notes,
            result.diagnostics.quantized_stretched_notes) == (1, 1)
    ends = [n.start_tick + n.duration_tick for n in notes]
    assert all(end <= nxt.start_tick for end, nxt in zip(ends, notes[1:], strict=False))
    assert all(n.duration_tick > 0 for n in notes)


def test_built_project_can_be_written_and_read_back(tmp_path):
    path = tmp_path / "song.vpr"
    result = project.build([_sung(0.0, 0.5, lyric="あ", phonemes=["a"])], _tempo(), name="song")
    write_file(result.project, path)

    restored, _warnings = read(path.read_bytes())
    note = restored.tracks[0].parts[0].notes[0]
    assert (restored.title, note.lyric, note.phonemes) == ("song", "あ", ["a"])


def test_same_input_gives_the_same_project():
    notes = [_sung(0.0, 0.5), _sung(0.5, 1.1)]
    assert project.build(notes, _tempo(), name="song") == project.build(notes, _tempo(),
                                                                        name="song")
