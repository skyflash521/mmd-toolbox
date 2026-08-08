"""song2vpr CLI が vpr を書き出すところのテスト。

音声前段は差し替え、合成した共有出力から後段(F0・音符・歌詞・テンポ・組み立て・書き出し)を
実際に走らせる。後段まで差し替えると結線を検証できないため、差し替えるのは前段だけにする。
"""

import json
import types
from pathlib import Path

import numpy as np
import pytest

from song2vpr import cli
from song2vpr.pipeline import PipelineResult
from vocal_analysis import AudioPcm, Segment
from vocal_analysis.rms import compute_rms
from vpr import read

_PENDING = "impl pending: 音符化以降を CLI へ結線していない"
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
                          forced_split=False)


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


@pytest.mark.xfail(reason=_PENDING, strict=True)
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


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_output_defaults_to_the_input_name(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    assert cli.main([_source(tmp_path)]) == 0
    assert (tmp_path / "in.vpr").exists()


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_names_come_from_the_output_base_name(tmp_path, monkeypatch):
    """曲名・トラック名・パート名は出力ファイルの基底名。"""
    _stub_pipeline(monkeypatch)
    output = tmp_path / "別の名前.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output)]) == 0

    project, _notes = _notes_of(output)
    track = project.tracks[0]
    assert (project.title, track.name, track.parts[0].name) == ("別の名前",) * 3


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_write_failure_points_at_the_output_option(tmp_path, monkeypatch, capsysbinary):
    _stub_pipeline(monkeypatch)
    output = tmp_path / "存在しない" / "song.vpr"
    assert cli.main(["--machine", _source(tmp_path), "-o", str(output)]) == 3
    event = _events(capsysbinary)[-1]
    assert (event["type"], event["code"], event["field"]) == ("error", "write_failed", "--output")
    assert event["path"] == str(output)


@pytest.mark.xfail(reason=_PENDING, strict=True)
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


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_louder_singing_gets_a_larger_velocity(tmp_path, monkeypatch):
    """発声している区間どうしの強弱の差が、ベロシティの差として現れる。"""
    quiet, loud = _wave(440.0, 0.6, 0.05), _wave(523.25, 0.6, 0.45)
    segments = [Segment(type="vowel", start_sec=0.0, end_sec=0.6, phoneme="a", confidence=1.0),
                Segment(type="vowel", start_sec=0.6, end_sec=1.2, phoneme="i", confidence=1.0)]
    _stub_pipeline(monkeypatch, _front_stage(pcm=_pcm(quiet, loud), segments=segments,
                                             duration_sec=1.2))
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output)]) == 0

    _project, notes = _notes_of(output)
    assert len(notes) >= 2
    assert notes[0].velocity < notes[-1].velocity


# --- 引数の結線 --------------------------------------------------------------


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_specified_tempo_and_time_signature_are_used(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output),
                     "--tempo", "96.5", "--time-signature", "3/4"]) == 0

    project, _notes = _notes_of(output)
    assert project.tempos[0].bpm == 96.5
    signature = project.time_signatures[0]
    assert (signature.numerator, signature.denominator) == (3, 4)


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_given_lyrics_replace_the_display_text(tmp_path, monkeypatch):
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("ゆき", encoding="utf-8")
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output), "--lyrics", str(lyrics)]) == 0

    _project, notes = _notes_of(output)
    assert notes[0].lyric == "ゆ"


@pytest.mark.xfail(reason=_PENDING, strict=True)
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


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_dry_run_still_runs_everything_but_the_write(tmp_path, monkeypatch, capsysbinary):
    """診断の件数は実測値なので、書き出し以外は通常実行と同じに走る。"""
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", "--dry-run", _source(tmp_path),
                     "-o", str(tmp_path / "song.vpr")]) == 0
    stages = {e["stage"] for e in _events(capsysbinary) if e["type"] == "progress"}
    assert {"f0", "notes"} <= stages
    assert "write" not in stages


# --- 進捗と警告 --------------------------------------------------------------


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_progress_reports_the_stages_song2vpr_owns(tmp_path, monkeypatch, capsysbinary):
    """音符化以降の3段(ピッチ推定・音符化・書き出し)は song2vpr 自身が報告する。"""
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    stages = {e["stage"] for e in _events(capsysbinary) if e["type"] == "progress"}
    assert {"f0", "notes", "write"} <= stages


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_no_notes_warns_and_still_succeeds(tmp_path, monkeypatch, capsysbinary):
    """音符が1つも得られなくても正常終了し、見直しの材料を警告で知らせる。"""
    silence = AudioPcm(samples=np.zeros((_RATE // 2, 1), dtype=np.float32), sample_rate=_RATE)
    _stub_pipeline(monkeypatch, _front_stage(pcm=silence, segments=[], duration_sec=0.5))

    output = tmp_path / "song.vpr"
    assert cli.main(["--machine", _source(tmp_path), "-o", str(output)]) == 0
    assert _notes_of(output)[1] == []
    assert any(e["type"] == "warning" and e["code"] == "no_notes" for e in _events(capsysbinary))
