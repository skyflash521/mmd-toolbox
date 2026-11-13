import json

from song2vpr import cli


def _touch(path):
    path.write_bytes(b"")
    return str(path)


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


def test_machine_processing_path_ends_with_exactly_one_terminal_event_regardless_of_exit_code(
        tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr"), "--dry-run"])
    events = machine_events(capsysbinary)
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1


def test_machine_error_bad_argument_missing_input(capsysbinary):
    assert cli.main(["--machine"]) == 2
    event = sole_terminal_error_event(capsysbinary)
    assert event["code"] == "bad_argument"
    assert event["field"] == "input"
    assert event["exit_code"] == 2


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


