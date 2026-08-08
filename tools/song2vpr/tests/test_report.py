"""song2vpr のレポート/診断の構造化のテスト。

各段が返した診断から result のフィールドを組み立てる規則を検証する。音声処理は行わないので、
入力はすべて合成した診断データにする。
"""



from song2vpr import lyrics, notes, project, report
from song2vpr.tempo import TempoEstimate


def _tempo(bpm=120.0, numerator=4, denominator=4, tempo_source="estimated",
           time_signature_source="estimated"):
    return TempoEstimate(bpm=bpm, numerator=numerator, denominator=denominator,
                         beat_offset_sec=0.0, first_bar_sec=0.0, tempo_source=tempo_source,
                         time_signature_source=time_signature_source)


def _fields(**overrides):
    given = dict(
        output="out.vpr", tempo=_tempo(), duration_sec=12.5, separated=True,
        backends={"separator": "demucs", "recognizer": None, "recognizer_revision": None,
                  "forced_aligner": None, "english_katakana_method": None},
        split=notes.Diagnostics(short_notes=1),
        annotation=lyrics.Diagnostics(undetermined_vowel_notes=2, moraic_nasal_notes=3,
                                      no_phoneme_notes=4, notes_beyond_morae=5,
                                      discarded_morae=6),
        build=project.Diagnostics(note_count=3, pitch_clamped_notes=7,
                                  quantized_merged_notes=8, quantized_stretched_notes=9),
    )
    given.update(overrides)
    return report.run_fields(**given)


def test_run_fields_carry_every_key_the_contract_names():
    """result(mode:"run")のキーは常に全部載せる。"""
    assert set(_fields()) == {
        "output", "notes", "tempo_bpm", "tempo_source", "time_signature",
        "time_signature_source", "resolution", "duration_sec", "separated", "backends",
        "vowel_undetermined_notes", "moraic_nasal_notes", "fallback_lyric_notes", "dropped_morae",
        "short_notes", "pitch_clamped_notes", "quantized_merged_notes",
        "quantized_stretched_notes", "no_phoneme_notes",
    }


def test_counts_come_from_the_stage_that_judged_them():
    """各件数は、それを判定した段の診断から取る。"""
    fields = _fields()
    assert fields["notes"] == 3
    assert fields["short_notes"] == 1
    assert (fields["vowel_undetermined_notes"], fields["moraic_nasal_notes"]) == (2, 3)
    assert fields["no_phoneme_notes"] == 4
    assert (fields["fallback_lyric_notes"], fields["dropped_morae"]) == (5, 6)
    assert (fields["pitch_clamped_notes"], fields["quantized_merged_notes"],
            fields["quantized_stretched_notes"]) == (7, 8, 9)


def test_tempo_and_time_signature_are_reported_with_their_sources():
    fields = _fields(tempo=_tempo(bpm=96.5, numerator=3, denominator=8, tempo_source="option",
                                  time_signature_source="default"))
    assert (fields["tempo_bpm"], fields["tempo_source"]) == (96.5, "option")
    assert (fields["time_signature"], fields["time_signature_source"]) == ("3/8", "default")
    assert fields["resolution"] == 480


def test_inspect_fields_add_the_input_metadata_and_drop_the_output():
    """入力検査は run の全キーに入力のメタ情報を足し、vpr を書かないので出力先は null。"""
    fields = report.inspect_fields(sample_rate=44100, channels=2, tempo=_tempo(),
                                   duration_sec=12.5, separated=True, backends={},
                                   split=notes.Diagnostics(), annotation=lyrics.Diagnostics(),
                                   build=project.Diagnostics())
    assert set(fields) - set(_fields()) == {"input_kind", "sample_rate", "channels"}
    assert (fields["input_kind"], fields["sample_rate"], fields["channels"]) == ("audio", 44100, 2)
    assert fields["output"] is None


# --- 人間向けの表示 ------------------------------------------------------------


def test_report_text_lists_the_same_values_as_the_result():
    """人間向けの表示は、機械モードの result と同じ値を読める形で並べる。"""
    text = report.report_text(_fields())
    for expected in ("音符数: 3", "テンポ: 120.0", "拍子: 4/4",
                     "分解能(tick/四分音符): 480", "入力の尺(秒): 12.5", "短いまま残った音符: 1"):
        assert expected in text


def test_report_text_is_written_in_japanese():
    """人間向けなので、契約の語や Python の値の表記をそのまま出さない。"""
    separated = report.report_text(_fields(separated=True))
    assert "ボーカル分離の実施: あり" in separated
    assert "ボーカル分離の実施: なし" in report.report_text(_fields(separated=False))
    assert "True" not in separated and "False" not in separated


def test_report_text_omits_the_output_path():
    """書き出し先は出さない(書く実行では完了行が示し、書かない実行では示すものが無い)。"""
    assert "out.vpr" not in report.report_text(_fields(output="out.vpr"))


def test_moraic_counts_appear_only_when_the_lyrics_were_given():
    """モーラ由来の2件は --lyrics 指定時だけ出す(未指定の実行では常に 0 になるため)。"""
    fields = _fields()
    assert "表示歌詞を音声から決めた音符: 5" not in report.report_text(fields)
    assert "破棄した余剰モーラ: 6" not in report.report_text(fields)

    given = report.report_text(fields, lyrics_given=True)
    assert "表示歌詞を音声から決めた音符: 5" in given
    assert "破棄した余剰モーラ: 6" in given


def test_report_text_omits_the_backends_that_were_not_given():
    """渡さなかったバックエンド選択の設定は出さない。"""
    text = report.report_text(_fields(backends={"separator": "demucs", "recognizer": None}))
    assert "ボーカル分離: demucs" in text
    assert "内容認識モデル" not in text


def test_report_text_tells_where_the_tempo_came_from():
    """テンポと拍子は、指定値・推定値・仮置きの別を含める(人間向けなので日本語で出す)。"""
    text = report.report_text(_fields(tempo=_tempo(tempo_source="option",
                                                   time_signature_source="default")))
    assert "テンポ: 120.0 (指定値)" in text
    assert "拍子: 4/4 (仮置き)" in text
