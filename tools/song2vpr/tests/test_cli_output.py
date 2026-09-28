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


def _harmonic_wave(freq_hz, seconds, amplitude):
    times = np.arange(int(seconds * _RATE)) / _RATE
    return sum(amplitude / (k + 1) * np.sin(2 * np.pi * freq_hz * (k + 1) * times)
               for k in range(4)).astype(np.float32)


def _pcm(*waves):
    return AudioPcm(samples=np.stack([np.concatenate(waves)], axis=1), sample_rate=_RATE)


def _front_stage(pcm=None, segments=None, duration_sec=1.0):
    pcm = pcm if pcm is not None else _pcm(_harmonic_wave(440.0, duration_sec, 0.5))
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
    calls = None

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


def _longest_note(notes):
    return max(notes, key=lambda note: note.duration_tick)


def test_run_writes_the_back_end_notes_with_the_vowel_lyric_and_the_440hz_pitch(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output)]) == 0

    project, notes = _notes_of(output)
    assert notes
    represented = _longest_note(notes)
    assert represented.lyric == "あ"
    assert represented.phonemes == ["a"]
    assert represented.pitch == 69
    assert project.tracks[0].parts[0].voice is not None


def test_output_defaults_to_the_input_name(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    assert cli.main([_source(tmp_path)]) == 0
    assert (tmp_path / "in.vpr").exists()


def test_title_track_and_part_names_come_from_the_output_base_name(tmp_path, monkeypatch):
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


def test_successful_run_leaves_one_completion_line_after_closing_the_live_line(tmp_path, monkeypatch,
                                                                              progress_calls):
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output)]) == 0

    completion = ("summary", f"完了 {output}")
    assert progress_calls.count(completion) == 1
    assert progress_calls.index("close") < progress_calls.index(completion)


def test_dry_run_diagnostics_on_stdout_come_right_after_closing_the_live_line(tmp_path, monkeypatch,
                                                                              progress_calls):
    _stub_pipeline(monkeypatch)
    monkeypatch.setattr("builtins.print",
                        lambda *args, **kwargs: progress_calls.append(("print",
                                                                       kwargs.get("file"))))
    assert cli.main([_source(tmp_path), "-o", str(tmp_path / "song.vpr"), "--dry-run"]) == 0

    print_to_stdout = ("print", None)
    assert progress_calls.count(print_to_stdout) == 1
    position = progress_calls.index(print_to_stdout)
    assert position >= 1
    assert progress_calls[position - 1] == "close"


def test_velocity_is_the_same_for_all_notes_regardless_of_singing_volume(tmp_path, monkeypatch):
    quiet, loud = _harmonic_wave(440.0, 0.6, 0.05), _harmonic_wave(523.25, 0.6, 0.45)
    segments = [Segment(type="vowel", start_sec=0.0, end_sec=0.6, phoneme="a", confidence=1.0),
                Segment(type="vowel", start_sec=0.6, end_sec=1.2, phoneme="i", confidence=1.0)]
    _stub_pipeline(monkeypatch, _front_stage(pcm=_pcm(quiet, loud), segments=segments,
                                             duration_sec=1.2))
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output)]) == 0

    _project, notes = _notes_of(output)
    assert len(notes) >= 2
    assert {note.velocity for note in notes} == {64}


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
    calls = []
    _stub_pipeline(monkeypatch, calls=calls)
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr"),
                     "--lyrics", str(tmp_path / "無い.txt")]) == 1
    assert calls == []
    event = _events(capsysbinary)[-1]
    assert (event["code"], event["field"]) == ("lyrics_unreadable", "--lyrics")


def test_lyrics_that_do_not_decode_as_utf8_are_unreadable(tmp_path, monkeypatch, capsysbinary):
    _stub_pipeline(monkeypatch)
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_bytes("ゆき".encode("cp932"))
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr"),
                     "--lyrics", str(lyrics)]) == 1
    event = _events(capsysbinary)[-1]
    assert (event["code"], event["field"], event["exit_code"]) == (
        "lyrics_unreadable", "--lyrics", 1)


def test_the_lyrics_are_read_as_one_text(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    readings = []

    def to_kana_reading(text):
        readings.append(text)
        return "ゆき"

    monkeypatch.setattr("vocal_analysis.reading.to_kana_reading", to_kana_reading)
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("雪\n降る\n", encoding="utf-8")
    assert cli.main([_source(tmp_path), "-o", str(tmp_path / "song.vpr"),
                     "--lyrics", str(lyrics)]) == 0
    assert readings == ["雪\n降る\n"]


def test_dry_run_writes_neither_the_output_nor_a_completion_line(tmp_path, monkeypatch, progress_calls):
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output), "--dry-run"]) == 0
    assert not output.exists()
    assert not [call for call in progress_calls if call != "close"]


def test_dry_run_still_runs_everything_but_the_write(tmp_path, monkeypatch, capsysbinary):
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", "--dry-run", _source(tmp_path),
                     "-o", str(tmp_path / "song.vpr")]) == 0
    stages = {e["stage"] for e in _events(capsysbinary) if e["type"] == "progress"}
    assert {"f0", "notes"} <= stages
    assert "write" not in stages


def test_progress_reports_the_f0_notes_and_write_stages(tmp_path, monkeypatch, capsysbinary):
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    stages = {e["stage"] for e in _events(capsysbinary) if e["type"] == "progress"}
    assert {"f0", "notes", "write"} <= stages


def test_no_notes_warns_and_still_succeeds(tmp_path, monkeypatch, capsysbinary):
    silence = AudioPcm(samples=np.zeros((_RATE // 2, 1), dtype=np.float32), sample_rate=_RATE)
    _stub_pipeline(monkeypatch, _front_stage(pcm=silence, segments=[], duration_sec=0.5))

    output = tmp_path / "song.vpr"
    assert cli.main(["--machine", _source(tmp_path), "-o", str(output)]) == 0
    assert _notes_of(output)[1] == []
    assert any(e["type"] == "warning" and e["code"] == "no_notes" for e in _events(capsysbinary))


def test_no_notes_warns_in_dry_run_too(tmp_path, monkeypatch, capsysbinary):
    silence = AudioPcm(samples=np.zeros((_RATE // 2, 1), dtype=np.float32), sample_rate=_RATE)
    _stub_pipeline(monkeypatch, _front_stage(pcm=silence, segments=[], duration_sec=0.5))

    assert cli.main(["--machine", "--dry-run", _source(tmp_path),
                     "-o", str(tmp_path / "song.vpr")]) == 0
    assert any(e["type"] == "warning" and e["code"] == "no_notes" for e in _events(capsysbinary))


@pytest.mark.parametrize("quiet", [False, True])
def test_quiet_reaches_the_progress_display(tmp_path, monkeypatch, quiet):
    _stub_pipeline(monkeypatch)
    received = {}

    def build_router(**kwargs):
        received.update(kwargs)
        return _SpyProgressRouter()

    _SpyProgressRouter.calls = []
    monkeypatch.setattr(cli._progress, "build_router", build_router)
    argv = [_source(tmp_path), "-o", str(tmp_path / "song.vpr")] + (["--quiet"] if quiet else [])
    assert cli.main(argv) == 0
    assert received["quiet"] is quiet


def test_kana_reading_failure_is_a_stage_failure_of_the_notes_stage(tmp_path, monkeypatch,
                                                                    capsysbinary):
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


def test_byte_order_mark_is_removed_before_the_lyrics_reach_annotation(tmp_path, monkeypatch):
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


def test_tempo_estimation_fallback_is_reported_as_a_warning(tmp_path, monkeypatch, capsysbinary):
    silence = AudioPcm(samples=np.zeros((_RATE // 2, 1), dtype=np.float32), sample_rate=_RATE)
    _stub_pipeline(monkeypatch, _front_stage(pcm=silence, segments=[], duration_sec=0.5))
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    assert any(e["type"] == "warning" and e["code"] == "tempo_defaulted"
               for e in _events(capsysbinary))


def test_the_time_signature_is_not_reported_as_a_fallback(tmp_path, monkeypatch, capsysbinary):
    silence = AudioPcm(samples=np.zeros((_RATE // 2, 1), dtype=np.float32), sample_rate=_RATE)
    _stub_pipeline(monkeypatch, _front_stage(pcm=silence, segments=[], duration_sec=0.5))
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    assert not any(e["type"] == "warning" and e["code"] == "time_signature_defaulted"
                   for e in _events(capsysbinary))


def test_forced_split_from_the_front_stage_is_reported(tmp_path, monkeypatch, capsysbinary):
    front = dataclasses.replace(_front_stage(), forced_split=True)
    _stub_pipeline(monkeypatch, front)
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    assert any(e["type"] == "warning" and e["code"] == "forced_split" for e in _events(capsysbinary))


def test_ineffective_kana_reading_reports_the_counts_taken_from_the_reading(tmp_path, monkeypatch,
                                                                            capsysbinary):
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("なんでもよい", encoding="utf-8")
    monkeypatch.setattr("vocal_analysis.reading.to_kana_reading", lambda text: "あ漢A")
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr"),
                     "--lyrics", str(lyrics)]) == 0

    warnings = [e for e in _events(capsysbinary)
                if e["type"] == "warning" and e["code"] == "kana_reading_ineffective"]
    assert warnings
    assert (warnings[0]["unconverted_chars"], warnings[0]["counted_chars"]) == (2, 3)


def _result_of(capsysbinary):
    events = _events(capsysbinary)
    assert events[-1]["type"] == "result"
    return events[-1]


def test_run_result_reports_the_written_output_and_the_counts(tmp_path, monkeypatch,
                                                              capsysbinary):
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


def test_inspect_result_adds_the_input_metadata_with_null_output_and_writes_nothing(
        tmp_path, monkeypatch, capsysbinary):
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


def test_moraic_nasal_only_input_raises_only_the_moraic_nasal_count(tmp_path, monkeypatch, capsysbinary):
    segments = [Segment(type="consonant", start_sec=0.0, end_sec=1.0, phoneme="ɴ", confidence=1.0)]
    _stub_pipeline(monkeypatch, _front_stage(segments=segments))
    assert cli.main(["--machine", _source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0

    result = _result_of(capsysbinary)
    assert result["moraic_nasal_notes"] == 1
    assert (result["vowel_undetermined_notes"], result["no_phoneme_notes"]) == (0, 0)
    assert (result["fallback_lyric_notes"], result["dropped_morae"]) == (0, 0)


def test_dry_run_shows_the_diagnostics_on_stdout(tmp_path, monkeypatch, capsys):
    _stub_pipeline(monkeypatch)
    assert cli.main([_source(tmp_path), "-o", str(tmp_path / "song.vpr"), "--dry-run"]) == 0
    assert "テンポ" in capsys.readouterr().out


def test_verbose_writes_the_output_and_shows_the_diagnostics(tmp_path, monkeypatch, capsys):
    _stub_pipeline(monkeypatch)
    output = tmp_path / "song.vpr"
    assert cli.main([_source(tmp_path), "-o", str(output), "-v"]) == 0
    assert output.exists()
    assert "テンポ" in capsys.readouterr().out


def test_machine_mode_stdout_has_only_json_events_without_the_human_report(tmp_path, monkeypatch,
                                                                           capsysbinary):
    _stub_pipeline(monkeypatch)
    assert cli.main(["--machine", "--dry-run", _source(tmp_path),
                     "-o", str(tmp_path / "song.vpr")]) == 0
    for line in capsysbinary.readouterr().out.decode("utf-8").splitlines():
        assert line.startswith("{")


def test_normal_run_without_verbose_or_dry_run_prints_nothing_to_stdout(tmp_path, monkeypatch, capsys):
    _stub_pipeline(monkeypatch)
    assert cli.main([_source(tmp_path), "-o", str(tmp_path / "song.vpr")]) == 0
    assert capsys.readouterr().out == ""
