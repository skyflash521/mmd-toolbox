import io
import json
import sys

from shakevmd import cli
from vmd import io as vmd_io
from vmd.types import CameraKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, LINEAR, fov, persp)


KEYS = [
    cam(0, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, dist=-25.0, center=(10.0, 5.0, 2.0), rot=(0.2, 0.1, 0.0), persp=1),
    cam(60, dist=-20.0, center=(20.0, 0.0, -3.0), rot=(-0.1, 0.3, 0.05), persp=1),
]


def write_input(path, keys=KEYS):
    vmd_io.write_file(VmdDocument(camera=list(keys)), str(path))
    return str(path)


class _TTYStringIO(io.StringIO):
    def isatty(self):
        return True


def test_version_shows_version_and_exit0(capsys):
    from shakevmd import __version__
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "shakevmd" in out
    assert __version__ in out


def test_progress_noop_when_not_tty(tmp_path, capsys):
    inp = write_input(tmp_path / "in.vmd")
    assert cli.main([inp, "-o", str(tmp_path / "out.vmd")]) == 0
    assert capsys.readouterr().err == ""


def test_progress_side_effect_only_bytes_identical(tmp_path, monkeypatch):
    inp = write_input(tmp_path / "in.vmd")
    out_quiet = tmp_path / "quiet.vmd"
    assert cli.main([inp, "-o", str(out_quiet), "--quiet"]) == 0
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    out_live = tmp_path / "live.vmd"
    assert cli.main([inp, "-o", str(out_live)]) == 0
    assert out_live.read_bytes() == out_quiet.read_bytes()


def test_quiet_suppresses_progress_even_on_tty(tmp_path, monkeypatch):
    inp = write_input(tmp_path / "in.vmd")
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--quiet"]) == 0
    assert sys.stderr.getvalue() == ""


def test_quiet_does_not_change_nonprogress_output(tmp_path, capsys):
    dup = [cam(0), cam(30), cam(30, center=(9.0, 9.0, 9.0)), cam(60)]
    inp = write_input(tmp_path / "dup.vmd", dup)
    rc_plain = cli.main([inp, "-o", str(tmp_path / "a.vmd"), "--dry-run"])
    plain = capsys.readouterr()
    rc_quiet = cli.main([inp, "-o", str(tmp_path / "b.vmd"), "--quiet", "--dry-run"])
    quiet = capsys.readouterr()
    assert rc_plain == rc_quiet == 0
    assert (quiet.out, quiet.err) == (plain.out, plain.err)
    assert "warning:" in (plain.out + plain.err).lower()


def test_quiet_does_not_change_exit_code(tmp_path):
    inp = write_input(tmp_path / "in.vmd")
    assert cli.main([inp, "-o", inp]) == cli.main([inp, "-o", inp, "--quiet"]) == 2


def test_no_smooth_skips_smoothing_stage(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    assert cli.main([inp, "-o", str(tmp_path / "a.vmd"), "--machine", "--no-smooth"]) == 0
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").splitlines() if ln]
    stages = {e["stage"] for e in events if e["type"] == "progress"}
    assert "bake" in stages
    assert "smooth" not in stages


def test_smooth_reports_reduce_progress_in_machine_events(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine"]) == 0
    events = [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").splitlines() if ln]
    smooth = [e for e in events if e["type"] == "progress" and e["stage"] == "smooth"]
    assert any(e["total"] is not None and 0 < e["done"] <= e["total"] for e in smooth)


def test_summary_on_success(tmp_path, monkeypatch):
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    assert cli.main([inp, "-o", str(out), "--no-smooth"]) == 0
    assert f"完了 {out}" in sys.stderr.getvalue().splitlines()


def test_no_summary_on_dry_run(tmp_path, monkeypatch):
    inp = write_input(tmp_path / "in.vmd")
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    assert cli.main([inp, "--dry-run"]) == 0
    assert "完了" not in sys.stderr.getvalue()
