import json
import sys

from song2vmd import cli


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def machine_events(capsysbinary):
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln]


def single_terminal_error(capsysbinary):
    events = machine_events(capsysbinary)
    assert events, "stdout に少なくとも1イベントが要る"
    assert events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    return events[-1]


def test_machine_version_stays_human(capsys):
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "song2vmd" in out and not out.lstrip().startswith("{")


def test_machine_help_stays_human(capsys):
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


def test_machine_error_bad_argument_unknown_option(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([src, "--machine", "--bogus"])
    assert rc == 2
    e = single_terminal_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2
    assert e["field"] == "--bogus"
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_bad_argument_missing_input(capsysbinary):
    rc = cli.main(["--machine"])
    assert rc == 2
    e = single_terminal_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "input" and e["exit_code"] == 2


def test_machine_error_bad_argument_type_error_field(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([src, "--machine", "--max-duration", "abc"])
    assert rc == 2
    e = single_terminal_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--max-duration" and e["exit_code"] == 2


def test_machine_error_bad_argument_unknown_style(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([src, "--machine", "--style", "nope"])
    assert rc == 2
    e = single_terminal_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--style" and e["exit_code"] == 2


def test_machine_error_output_exists(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([str(src), "-o", str(src), "--machine"])
    assert rc == 2
    e = single_terminal_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output" and e["exit_code"] == 2


def test_machine_error_output_exists_distinct_path(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vmd")
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 2
    e = single_terminal_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output" and e["exit_code"] == 2


def test_machine_error_stdout_is_valid_json_lines_lf_only(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([src, "-o", str(src), "--machine"])
    assert rc == 2
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n") and b"\r" not in raw
    for ln in raw.decode("utf-8").split("\n"):
        if ln:
            obj = json.loads(ln)
            assert "type" in obj


def test_non_machine_missing_input_is_arg_error(capsys):
    rc = cli.main([])
    assert rc == 2
    assert "error:" in capsys.readouterr().err.lower()


def test_non_machine_overwrite_guard_prints_reason_to_stderr(tmp_path, capsys):
    src = _touch(tmp_path / "in.wav")
    rc = cli.main([src, "-o", src])
    assert rc == 2
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")


class _ClosedPipeStdout:
    class _Buffer:
        def write(self, _data):
            raise OSError("broken pipe")

    def __init__(self):
        self.buffer = self._Buffer()


def test_closed_stdout_in_machine_mode_reports_original_reason_as_one_stderr_line(
        tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", _ClosedPipeStdout())
        rc = cli.main([src, "-o", src, "--machine"])
    assert rc == 2
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1
    assert err[0].startswith("error: ") and "出力先に既存ファイルがあります" in err[0]


def test_machine_error_output_is_directory_with_or_without_overwrite(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    for extra in ([], ["--overwrite"]):
        rc = cli.main([src, "-o", str(outdir), "--machine", *extra])
        assert rc == 2
        e = single_terminal_error(capsysbinary)
        assert e["code"] == "output_is_directory" and e["field"] == "--output"
        assert e["exit_code"] == 2 and e["path"] == str(outdir)


def test_non_machine_usage_error_is_single_error_line_without_usage_text(capsys):
    rc = cli.main(["in.wav", "--bogus"])
    assert rc == 2
    assert capsys.readouterr().err == "error: unrecognized arguments: --bogus\n"
