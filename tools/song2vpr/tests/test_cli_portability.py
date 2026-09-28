import io
import json
import sys
import types

import pytest

from song2vpr import cli

from .support import front_stage_result


@pytest.fixture(autouse=True)
def _stub_front_stage_with_short_silence(monkeypatch):
    monkeypatch.setattr(cli, "_pipeline",
                        types.SimpleNamespace(run=lambda *a, **k: front_stage_result()))

CHAR_OUTSIDE_CP932 = "\U0001f3a5"


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _strict_cp932_stderr(monkeypatch):
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


def test_non_ascii_input_default_output_resolves_to_stem_vpr_and_trips_overwrite_guard(tmp_path):
    src = _touch(tmp_path / "ボーカル.wav")
    _touch(tmp_path / "ボーカル.vpr")
    assert cli.main([src, "--dry-run"]) == 2


def test_non_ascii_input_does_not_collide_with_other_extension(tmp_path):
    src = _touch(tmp_path / "ボーカル.wav")
    _touch(tmp_path / "ボーカル.vmd")
    assert cli.main([src, "--dry-run"]) == 0


def test_non_ascii_output_path_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "-o", str(tmp_path / "歌声パート.vpr"), "--dry-run"]) == 0


def test_machine_stdout_keeps_non_ascii_as_utf8_without_escaping(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    outdir = tmp_path / "出力先ディレクトリ"
    outdir.mkdir()
    assert cli.main(["--machine", src, "-o", str(outdir), "--dry-run"]) == 2
    raw = capsysbinary.readouterr().out
    assert "出力先ディレクトリ".encode() in raw
    events = [json.loads(ln) for ln in raw.decode("utf-8").split("\n") if ln]
    assert events[-1]["path"] == str(outdir)


def test_stderr_under_strict_cp932_still_writes_a_reason_containing_an_unencodable_char(
        tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    stderr = _strict_cp932_stderr(monkeypatch)
    assert cli.main([src, "--separator", CHAR_OUTSIDE_CP932, "--dry-run"]) == 2
    stderr.flush()
    written = stderr.buffer.getvalue().decode("cp932")
    assert written


def test_relative_input_is_resolved_against_cwd(tmp_path, monkeypatch):
    _touch(tmp_path / "song.wav")
    _touch(tmp_path / "song.vpr")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["song.wav", "--dry-run"]) == 2


def test_default_output_is_next_to_the_input_not_in_cwd(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(sub / "song.wav")
    monkeypatch.chdir(tmp_path)
    _touch(sub / "song.vpr")
    assert cli.main(["sub/song.wav", "--dry-run"]) == 2


def test_default_output_ignores_same_name_in_cwd(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(sub / "song.wav")
    _touch(tmp_path / "song.vpr")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["sub/song.wav", "--dry-run"]) == 0


def test_relative_output_is_resolved_against_cwd(tmp_path, monkeypatch):
    _touch(tmp_path / "in.wav")
    _touch(tmp_path / "out.vpr")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["in.wav", "-o", "out.vpr", "--dry-run"]) == 2
    assert cli.main(["in.wav", "-o", "other.vpr", "--dry-run"]) == 0
