import json
import sys

import pytest

from song2vpr import cli


def _touch(path):
    path.write_bytes(b"")
    return str(path)


class _UnwritableStdout:
    class _Buffer:
        def write(self, data):
            raise OSError("stdout is closed")

        def flush(self):
            raise OSError("stdout is closed")

    buffer = _Buffer()

    def write(self, text):
        raise OSError("stdout is closed")

    def flush(self):
        raise OSError("stdout is closed")


def machine_events(capsysbinary):
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln]


def sole_terminal_error_event(capsysbinary):
    events = machine_events(capsysbinary)
    assert events, "標準出力に少なくとも1イベントが要る"
    assert events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    return events[-1]


def test_version_prints_and_exits_zero(capsys):
    import song2vpr

    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "song2vpr" in out and song2vpr.__version__ in out


def test_machine_version_stays_human(capsys):
    assert cli.main(["--machine", "--version"]) == 0
    out = capsys.readouterr().out
    assert "song2vpr" in out and not out.lstrip().startswith("{")


def test_machine_help_stays_human(capsys):
    assert cli.main(["--machine", "--help"]) == 0
    out = capsys.readouterr().out
    assert "song2vpr" in out and not out.lstrip().startswith("{")


def test_machine_processing_path_ends_with_exactly_one_terminal_event_regardless_of_exit_code(
        tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr"), "--dry-run"])
    events = machine_events(capsysbinary)
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1


def test_machine_error_bad_argument_unknown_option(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    assert cli.main(["--machine", src, "--bogus"]) == 2
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "--bogus"
    assert event["exit_code"] == 2
    assert isinstance(event["message"], str) and event["message"]


def test_machine_error_bad_argument_missing_input(capsysbinary):
    assert cli.main(["--machine"]) == 2
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "input"
    assert event["exit_code"] == 2


def test_machine_error_bad_argument_value_error_names_the_option(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    assert cli.main(["--machine", src, "--max-duration", "abc"]) == 2
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "--max-duration"


def test_machine_error_time_signature_denominator_names_the_option(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    assert cli.main(["--machine", src, "--time-signature", "4/3"]) == 2
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "--time-signature"


def test_machine_error_combination_check_names_the_option(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    assert cli.main(["--machine", src, "--recognizer-model-revision", "abc123"]) == 2
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "--recognizer-model-revision"
    assert event["exit_code"] == 2


def test_machine_error_sofa_without_required_args_names_sofa_python_first(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    assert cli.main(["--machine", src, "--forced-aligner", "sofa-forcedalign"]) == 2
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "--sofa-python"
    assert event["exit_code"] == 2


def test_machine_error_output_exists_has_null_path(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main(["--machine", src, "-o", out, "--dry-run"]) == 2
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "output_exists"
    assert event["field"] == "--output"
    assert event["path"] is None
    assert event["exit_code"] == 2


@pytest.mark.parametrize("extra", [[], ["--overwrite"]])
def test_machine_error_output_is_directory_regardless_of_overwrite(tmp_path, capsysbinary, extra):
    src = _touch(tmp_path / "in.wav")
    outdir = tmp_path / "dir"
    outdir.mkdir()
    assert cli.main(["--machine", src, "-o", str(outdir), *extra, "--dry-run"]) == 2
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "output_is_directory"
    assert event["field"] == "--output"
    assert event["path"] == str(outdir)
    assert event["exit_code"] == 2


def test_machine_error_stdout_is_valid_json_lines_lf_only(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main(["--machine", src, "-o", out, "--dry-run"]) == 2
    raw = capsysbinary.readouterr().out
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    for line in raw.decode("utf-8").split("\n")[:-1]:
        json.loads(line)


def test_broken_stdout_in_machine_mode_reports_the_original_reason_in_one_stderr_line(
        tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main([src, "-o", out]) == 2
    reason_with_working_stdout = capsys.readouterr().err.splitlines()
    assert len(reason_with_working_stdout) == 1 and reason_with_working_stdout[0].startswith("error: ")

    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", _UnwritableStdout())
        rc = cli.main([src, "-o", out, "--machine"])
    assert rc == 2
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1
    assert err[0] == reason_with_working_stdout[0]


def _raise(exc):
    def _f(*_args, **_kwargs):
        raise exc

    return _f


def test_interrupt_is_reported_as_cancelled_without_field(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "_build_parser", _raise(KeyboardInterrupt()))
    assert cli.main(["--machine", src]) == 130
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "cancelled"
    assert event["field"] is None
    assert event["exit_code"] == 130


def test_interrupt_in_non_machine_mode_prints_single_line(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "_build_parser", _raise(KeyboardInterrupt()))
    assert cli.main([src]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")


def test_unexpected_exception_is_reported_as_internal_error_without_field(tmp_path, monkeypatch,
                                                                          capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "_build_parser", _raise(RuntimeError("boom")))
    assert cli.main(["--machine", src]) == 1
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "internal_error"
    assert event["field"] is None
    assert event["exit_code"] == 1


def test_unexpected_exception_in_non_machine_mode_omits_traceback(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "_build_parser", _raise(RuntimeError("boom")))
    assert cli.main([src]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert len(captured.err.splitlines()) == 1


def test_non_machine_overwrite_guard_prints_reason_to_stderr(tmp_path, capsys):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main([src, "-o", out, "--dry-run"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")


def test_non_machine_usage_error_is_single_error_line_without_usage_block(tmp_path, capsys):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--bogus"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "usage:" not in captured.err


def test_non_machine_missing_input_is_single_error_line(capsys):
    assert cli.main([]) == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
