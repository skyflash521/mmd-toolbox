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
