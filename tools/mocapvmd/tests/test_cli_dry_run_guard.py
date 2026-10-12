import json

from mocapvmd import cli

from .helpers import bone, write_vmd


def _ramp(path):
    write_vmd(path, bone=[bone("センター", f, pos=(float(f), 0.0, 0.0)) for f in range(11)])


def test_dry_run_with_existing_output_and_no_overwrite_exits_2(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp(src)
    out.write_bytes(b"old content")
    assert cli.main([str(src), "-o", str(out), "--dry-run"]) == 2
    assert out.read_bytes() == b"old content"


def test_dry_run_with_existing_default_output_and_no_overwrite_exits_2(tmp_path):
    src = tmp_path / "in.vmd"
    _ramp(src)
    (tmp_path / "in_mocap.vmd").write_bytes(b"old content")
    assert cli.main([str(src), "--dry-run"]) == 2


def test_dry_run_machine_reports_output_exists(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp(src)
    out.write_bytes(b"old content")
    rc = cli.main([str(src), "-o", str(out), "--dry-run", "--machine"])
    assert rc == 2
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]
    last = events[-1]
    assert last["type"] == "error"
    assert last["code"] == "output_exists" and last["field"] == "--output" and last["exit_code"] == 2


def test_dry_run_with_existing_output_and_overwrite_passes_the_guard(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp(src)
    out.write_bytes(b"old content")
    assert cli.main([str(src), "-o", str(out), "--dry-run", "--overwrite"]) == 0
    assert out.read_bytes() == b"old content"
