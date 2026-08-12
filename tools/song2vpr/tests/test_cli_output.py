"""song2vpr CLI が vpr を書き出すところのテスト。

音声前段は差し替え、合成した共有出力から後段(F0・音符・歌詞・テンポ・組み立て・書き出し)を
実際に走らせる。後段まで差し替えると結線を検証できないため、差し替えるのは前段だけにする。
"""

import dataclasses
import json
import types
from pathlib import Path

import numpy as np
import pytest

from song2vpr import cli
from song2vpr.pipeline import PipelineResult
from vocal_analysis import AudioPcm, Segment
from vocal_analysis.recognizer import RecognitionError
from vocal_analysis.rms import compute_rms
from vpr import read

_RATE = 22050


def _wave(freq_hz, seconds, amplitude):
    """倍音を持つ合成音(正弦波だけだと推定器が基本波を取り違えやすい)。"""
    times = np.arange(int(seconds * _RATE)) / _RATE
    return sum(amplitude / (k + 1) * np.sin(2 * np.pi * freq_hz * (k + 1) * times)
               for k in range(4)).astype(np.float32)


def _pcm(*waves):
    return AudioPcm(samples=np.stack([np.concatenate(waves)], axis=1), sample_rate=_RATE)


def _front_stage(pcm=None, segments=None, duration_sec=1.0):
    """前段の共有出力。既定は 440 Hz を1秒伸ばした疑似歌声。"""
    pcm = pcm if pcm is not None else _pcm(_wave(440.0, duration_sec, 0.5))
    if segments is None:
        segments = [Segment(type="vowel", start_sec=0.0, end_sec=duration_sec, phoneme="a",
                            confidence=1.0)]
    return PipelineResult(vocal_wav=Path("vocal.wav"), vocal_pcm=pcm, segments=segments,
                          rms=compute_rms(pcm), pcm=pcm, duration_sec=duration_sec,
                          forced_split=False, backends={})


def _stub_pipeline(monkeypatch, front=None, calls=None):
    front = front if front is not None else _front_stage()

    def run(*args, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return front

    monkeypatch.setattr(cli, "_pipeline", types.SimpleNamespace(run=run))


class _SpyProgressRouter:
    """進捗の振り分けの差し替え。完了行はライブ表示が有効なときだけ書かれるので、標準エラーの
    捕捉では観測できない(捕捉した標準エラーは TTY でない)。呼び出しを記録して見る。
    """

    calls = None  # クラス変数: 差し替え先のコンストラクタから書けるよう、テストごとに入れ替える

    def __init__(self, **kwargs):
        pass

    def stage(self, *args, **kwargs):
        pass

    def close(self):
        _SpyProgressRouter.calls.append("close")

    def summary(self, message):
        _SpyProgressRouter.calls.append(("summary", message))


@pytest.fixture
def progress_calls(monkeypatch):
    calls = []
    _SpyProgressRouter.calls = calls
    monkeypatch.setattr(cli._progress, "build_router", lambda **kwargs: _SpyProgressRouter())
    return calls


def _source(tmp_path):
    path = tmp_path / "in.wav"
    path.write_bytes(b"")
    return str(path)


def _events(capsysbinary):
    return [json.loads(line) for line
            in capsysbinary.readouterr().out.decode("utf-8").splitlines() if line]


def _notes_of(path):
    project, _warnings = read(path.read_bytes())
    return project, project.tracks[0].parts[0].notes


def _longest(notes):
    """代表として最も長い音符を採る(端の外れ値が別の音符を作っても揺れない)。"""
    return max(notes, key=lambda note: note.duration_tick)


# --- 書き出し ----------------------------------------------------------------


def test_run_writes_the_notes_the_back_end_produced(tmp_path, monkeypatch):
    """後段が作った音符が実際に載る(空の vpr を書く実装では通らない)。"""
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output)]) == 0

    project, notes = _notes_of(output)
    assert notes
    represented = _longest(notes)
    assert represented.lyric == "あ"  # 母音セグメント a の代表母音
    assert represented.phonemes == ["a"]
    assert represented.pitch == 69  # 440 Hz
    assert project.tracks[0].parts[0].voice is not None


def test_output_defaults_to_the_input_name(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    assert cli.main([_source(tmp_path)]) == 0
    assert (tmp_path / "in.vpr").exists()


def test_names_come_from_the_output_base_name(tmp_path, monkeypatch):
    """曲名・トラック名・パート名は出力ファイルの基底名。"""
    _stub_pipeline(monkeypatch)
    output = tmp_path / "別の名前.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output)]) == 0

    project, _notes = _notes_of(output)
    track = project.tracks[0]
    assert (project.title, track.name, track.parts[0].name) == ("別の名前",) * 3


def test_write_failure_points_at_the_output_option(tmp_path, monkeypatch, capsysbinary):
    _stub_pipeline(monkeypatch)
    output = tmp_path / "存在しない" / "song.vpr"
    assert cli.main(["--machine", _source(tmp_path), "-o", str(output)]) == 3
    event = _events(capsysbinary)[-1]
    assert (event["type"], event["code"], event["field"]) == ("error", "write_failed", "--output")
    assert event["path"] == str(output)


def test_successful_run_leaves_a_completion_line(tmp_path, monkeypatch, progress_calls):
    """書いて正常終了したら、ライブ行を消してから完了行を1行残す。

    先に消さないと、完了行がライブ進捗の行へ連結されて読めなくなる。
    """
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output)]) == 0

    completion = ("summary", f"完了 {output}")
    assert progress_calls.count(completion) == 1
    assert progress_calls.index("close") < progress_calls.index(completion)


def test_dry_run_diagnostics_come_after_the_live_line_is_cleared(tmp_path, monkeypatch,
                                                                 progress_calls):
    """vpr を書かない実行の診断も、ライブ行を消してから標準出力へ書く。

    先に消さないと、診断の1行目がライブ進捗の行へ連結される。書き出す実行の完了行と同じ順序だが、
    ライブ表示は TTY でないと無効なので、捕捉した出力からは観測できない。

    警告も消してから書くので、記録には同じ対が複数並びうる。診断は標準出力へ、警告は標準エラーへ
    書かれる違いで見分け、診断の直前が消去であることを見る(先頭一致で見ると、警告が1件でも出た
    実行では診断の側を戻しても通ってしまう)。
    """
    _stub_pipeline(monkeypatch)
    monkeypatch.setattr("builtins.print",
                        lambda *args, **kwargs: progress_calls.append(("print",
                                                                       kwargs.get("file"))))
    assert cli.main([_source(tmp_path), "-o", str(tmp_path / "song.vpr"), "--dry-run"]) == 0

    diagnostics = ("print", None)  # 標準出力(既定の出力先)へ書かれたもの=診断
    assert progress_calls.count(diagnostics) == 1
    position = progress_calls.index(diagnostics)
    # 位置を先に押さえる(0 のとき position-1 は末尾へ回り込み、必ず消去で終わる記録を拾ってしまう)。
    assert position >= 1
    assert progress_calls[position - 1] == "close"


def test_velocity_does_not_follow_the_singing_volume(tmp_path, monkeypatch):
    """強弱の差はベロシティに出さない(ベロシティは音量の欄ではないため、全音符で同じ値)。"""
    quiet, loud = _wave(440.0, 0.6, 0.05), _wave(523.25, 0.6, 0.45)
    segments = [Segment(type="vowel", start_sec=0.0, end_sec=0.6, phoneme="a", confidence=1.0),
                Segment(type="vowel", start_sec=0.6, end_sec=1.2, phoneme="i", confidence=1.0)]
    _stub_pipeline(monkeypatch, _front_stage(pcm=_pcm(quiet, loud), segments=segments,
                                             duration_sec=1.2))
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output)]) == 0

    _project, notes = _notes_of(output)
    assert len(notes) >= 2
    assert {note.velocity for note in notes} == {64}


# --- 引数の結線 --------------------------------------------------------------


def test_specified_tempo_and_time_signature_are_used(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output),
                     "--tempo", "96.5", "--time-signature", "3/4"]) == 0

    project, _notes = _notes_of(output)
    assert project.tempos[0].bpm == 96.5
    signature = project.time_signatures[0]
    assert (signature.numerator, signature.denominator) == (3, 4)


def test_given_lyrics_replace_the_display_text(tmp_path, monkeypatch):
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("ゆき", encoding="utf-8")
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output), "--lyrics", str(lyrics)]) == 0

    _project, notes = _notes_of(output)
    assert notes[0].lyric == "ゆ"


def test_unreadable_lyrics_fail_before_the_front_stage_runs(tmp_path, monkeypatch, capsysbinary):
    """歌詞ファイルは音声前段を始める前に読む(誤指定が分離・認識を終えてから露見しないため)。"""
    calls = []
    _stub_pipeline(monkeypatch, calls=calls)
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr"),
                     "--lyrics", str(tmp_path / "無い.txt")]) == 1
    assert calls == []
    event = _events(capsysbinary)[-1]
    assert (event["code"], event["field"]) == ("lyrics_unreadable", "--lyrics")


# --- --dry-run ---------------------------------------------------------------


def test_dry_run_does_not_write_the_output(tmp_path, monkeypatch, progress_calls):
    """完了行は出力を書いたときだけ残すので、書かない実行では残らない。"""
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output), "--dry-run"]) == 0
    assert not output.exists()
    assert not [call for call in progress_calls if call != "close"]


def test_dry_run_still_runs_everything_but_the_write(tmp_path, monkeypatch, capsysbinary):
    """診断の件数は実測値なので、書き出し以外は通常実行と同じに走る。"""
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", "--dry-run", _source(tmp_path),
                     "-o", str(tmp_path / "song.vpr")]) == 0
    stages = {e["stage"] for e in _events(capsysbinary) if e["type"] == "progress"}
    assert {"f0", "notes"} <= stages
    assert "write" not in stages


# --- 進捗と警告 --------------------------------------------------------------


def test_progress_reports_the_stages_song2vpr_owns(tmp_path, monkeypatch, capsysbinary):
    """音符化以降の3段(ピッチ推定・音符化・書き出し)は song2vpr 自身が報告する。"""
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    stages = {e["stage"] for e in _events(capsysbinary) if e["type"] == "progress"}
    assert {"f0", "notes", "write"} <= stages


def test_no_notes_warns_and_still_succeeds(tmp_path, monkeypatch, capsysbinary):
    """音符が1つも得られなくても正常終了し、見直しの材料を警告で知らせる。"""
    silence = AudioPcm(samples=np.zeros((_RATE // 2, 1), dtype=np.float32), sample_rate=_RATE)
    _stub_pipeline(monkeypatch, _front_stage(pcm=silence, segments=[], duration_sec=0.5))

    output = tmp_path / "song.vpr"
    assert cli.main(["--machine", _source(tmp_path), "-o", str(output)]) == 0
    assert _notes_of(output)[1] == []
    assert any(e["type"] == "warning" and e["code"] == "no_notes" for e in _events(capsysbinary))


# --- 結線した値から出す警告と失敗 ----------------------------------------------


def test_kana_reading_failure_is_a_stage_failure_of_the_note_stage(tmp_path, monkeypatch,
                                                                   capsysbinary):
    """かな読みは音符へ歌詞を割り当てる段の中で行うので、失敗が指す段は認識でなく音符化。"""
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("ゆき", encoding="utf-8")
    _stub_pipeline(monkeypatch)

    def fail(*_args, **_kwargs):
        raise RecognitionError("読みを取れない")

    monkeypatch.setattr(cli._lyrics, "annotate", fail)

    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr"),
                     "--lyrics", str(lyrics)]) == 4
    event = _events(capsysbinary)[-1]
    assert (event["type"], event["code"], event["stage"]) == ("error", "stage_failed", "notes")


def test_byte_order_mark_is_removed_when_the_lyrics_are_read(tmp_path, monkeypatch):
    """先頭のバイト順マークは読み込み時に取り除く(かな読みへ持ち込まない)。"""
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_bytes("\ufeffゆき".encode())
    _stub_pipeline(monkeypatch)

    seen = {}
    original = cli._lyrics.annotate

    def annotate(*args, **kwargs):
        seen["text"] = kwargs.get("lyrics_text")
        return original(*args, **kwargs)

    monkeypatch.setattr(cli._lyrics, "annotate", annotate)
    assert cli.main([_source(tmp_path), "-o", str(tmp_path / "song.vpr"),
                     "--lyrics", str(lyrics)]) == 0
    assert seen["text"] == "ゆき"


def test_estimation_fallbacks_are_reported(tmp_path, monkeypatch, capsysbinary):
    """推定できず仮置きへ倒したことは、後段の戻り値から警告として知らせる。"""
    silence = AudioPcm(samples=np.zeros((_RATE // 2, 1), dtype=np.float32), sample_rate=_RATE)
    _stub_pipeline(monkeypatch, _front_stage(pcm=silence, segments=[], duration_sec=0.5))
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    assert any(e["type"] == "warning" and e["code"] == "tempo_defaulted"
               for e in _events(capsysbinary))


def test_the_time_signature_is_not_reported_as_a_fallback(tmp_path, monkeypatch, capsysbinary):
    """拍子は推定しないので、推定できなかったことを知らせる警告も出ない。"""
    silence = AudioPcm(samples=np.zeros((_RATE // 2, 1), dtype=np.float32), sample_rate=_RATE)
    _stub_pipeline(monkeypatch, _front_stage(pcm=silence, segments=[], duration_sec=0.5))
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    assert not any(e["type"] == "warning" and e["code"] == "time_signature_defaulted"
                   for e in _events(capsysbinary))


def test_forced_split_is_reported(tmp_path, monkeypatch, capsysbinary):
    """長尺の自動分割で強制分割した事実は、前段が運ぶ値から警告として知らせる。"""
    front = dataclasses.replace(_front_stage(), forced_split=True)
    _stub_pipeline(monkeypatch, front)
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    assert any(e["type"] == "warning" and e["code"] == "forced_split" for e in _events(capsysbinary))


def test_ineffective_kana_reading_reports_the_counts(tmp_path, monkeypatch, capsysbinary):
    """かな読みが効いていないおそれは、判定に使った2つの件数を添えて知らせる。

    判定に効くのは歌詞本文でなく変換後の読みなので、読みを差し替えて固定する(実際の読みは
    G2P の辞書に依存し、契約ではない)。
    """
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("なんでもよい", encoding="utf-8")
    monkeypatch.setattr("vocal_analysis.reading.to_kana_reading", lambda text: "あ漢A")
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr"),
                     "--lyrics", str(lyrics)]) == 0

    warnings = [e for e in _events(capsysbinary)
                if e["type"] == "warning" and e["code"] == "kana_reading_ineffective"]
    assert warnings
    # 「漢」と「A」がかなへ変換されるべきだった文字、判定に使ったのはそれと「あ」の3文字。
    assert (warnings[0]["unconverted_chars"], warnings[0]["counted_chars"]) == (2, 3)


# --- result のペイロード -------------------------------------------------------


def _result_of(capsysbinary):
    events = _events(capsysbinary)
    assert events[-1]["type"] == "result"
    return events[-1]


def test_run_result_reports_the_written_output_and_the_counts(tmp_path, monkeypatch,
                                                              capsysbinary):
    """通常実行の result は、書き出しパスと各段が数えた件数を載せる。"""
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main(["--machine", _source(tmp_path), "-o", str(output),
                     "--tempo", "96.5", "--time-signature", "3/4"]) == 0

    result = _result_of(capsysbinary)
    assert result["mode"] == "run"
    assert result["output"] == str(output)
    _project, notes = _notes_of(output)
    assert result["notes"] == len(notes)
    assert result["separated"] is True
    assert result["backends"] == {}
    assert result["duration_sec"] == 1.0
    assert (result["tempo_bpm"], result["tempo_source"]) == (96.5, "option")
    assert (result["time_signature"], result["time_signature_source"]) == ("3/4", "option")
    assert result["resolution"] == 480


def test_inspect_result_adds_the_input_metadata_and_writes_nothing(tmp_path, monkeypatch,
                                                                   capsysbinary):
    """入力検査は run の全キーに入力のメタ情報を足し、vpr を書かないので出力先は null。"""
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main(["--machine", "--dry-run", _source(tmp_path), "-o", str(output)]) == 0

    result = _result_of(capsysbinary)
    assert result["mode"] == "inspect"
    assert result["output"] is None
    assert (result["input_kind"], result["sample_rate"], result["channels"]) == ("audio", _RATE, 1)
    assert not output.exists()


def test_separation_choice_is_reported(tmp_path, monkeypatch, capsysbinary):
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr"),
                     "--separate-vocals", "never"]) == 0
    assert _result_of(capsysbinary)["separated"] is False


def test_counts_come_from_the_stages_that_judged_them(tmp_path, monkeypatch, capsysbinary):
    """件数は判定した段の値をそのまま載せる(渡し違いを落とす)。"""
    # 核が撥音だけの入力。撥音を入れた音符の件数だけが立ち、他の件数は立たない。
    segments = [Segment(type="consonant", start_sec=0.0, end_sec=1.0, phoneme="ɴ", confidence=1.0)]
    _stub_pipeline(monkeypatch, _front_stage(segments=segments))
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0

    result = _result_of(capsysbinary)
    assert result["moraic_nasal_notes"] == 1
    assert (result["vowel_undetermined_notes"], result["no_phoneme_notes"]) == (0, 0)
    assert (result["fallback_lyric_notes"], result["dropped_morae"]) == (0, 0)


# --- 人間向けの診断表示 --------------------------------------------------------


def test_dry_run_shows_the_diagnostics_on_stdout(tmp_path, monkeypatch, capsys):
    """--dry-run は診断を標準出力へ出す。"""
    _stub_pipeline(monkeypatch)
    assert cli.main([_source(tmp_path), "-o", str(tmp_path / "song.vpr"), "--dry-run"]) == 0
    assert "テンポ" in capsys.readouterr().out


def test_verbose_shows_the_same_diagnostics_after_writing(tmp_path, monkeypatch, capsys):
    """-v は出力 vpr を書いたうえで、同じ診断を実行完了後に表示する。"""
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output), "-v"]) == 0
    assert output.exists()
    assert "テンポ" in capsys.readouterr().out


def test_machine_mode_does_not_print_the_human_report(tmp_path, monkeypatch, capsysbinary):
    """機械モードでは人間向けの表示をしない(標準出力はイベント専用)。"""
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", "--dry-run", _source(tmp_path),
                     "-o", str(tmp_path / "song.vpr")]) == 0
    for line in capsysbinary.readouterr().out.decode("utf-8").splitlines():
        assert line.startswith("{")


def test_normal_run_without_verbose_stays_quiet(tmp_path, monkeypatch, capsys):
    """診断の表示は --dry-run と -v のときだけ。"""
    _stub_pipeline(monkeypatch)
    assert cli.main([_source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    assert capsys.readouterr().out == ""
