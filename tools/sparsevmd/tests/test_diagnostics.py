from sparsevmd import report
from vmd.types import CameraKey

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, interp_block=None):
    return CameraKey(frame, dist, center, rot, interp_block or CAM_LINEAR, fov, 0)

def _camera_jumping_at_frame_15():
    return [
        cam(f, center=((float(f) if f < 15 else float(f) + 50.0), 0.0, 0.0))
        for f in range(31)
    ]

def _diag():
    return {
        "cuts": [15],
        "splits": [{"frame": 7, "channel": "position", "norm_error": 2.0}],
        "seam_rewrites": [10],
    }


def test_build_report_includes_camera_diagnostics():
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_diag=_diag(),
    )
    assert rep["camera"]["diagnostics"] == _diag()


def test_build_report_no_diagnostics_when_omitted():
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[],
    )
    assert "diagnostics" not in rep["camera"]


def test_format_dry_run_shows_cut_positions():
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_diag=_diag(),
    )
    text = report.format_dry_run(rep)
    assert "cuts" in text.lower() or "不連続" in text
    assert "15" in text


def test_cli_dry_run_includes_cut_positions(tmp_path, capsys):
    from sparsevmd import cli
    from vmd import io
    from vmd.types import VmdDocument

    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    io.write_file(VmdDocument(camera=_camera_jumping_at_frame_15()), str(src))
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--dry-run"])
    assert code == 0
    text = capsys.readouterr().out
    assert "cuts" in text
    assert "15" in text


def test_cli_dry_run_includes_verify(tmp_path, capsys):
    from sparsevmd import cli
    from vmd import io
    from vmd.types import VmdDocument

    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [cam(f, dist=-30.0 - float(f)) for f in range(31)]
    io.write_file(VmdDocument(camera=keys), str(src))
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--dry-run"])
    assert code == 0
    text = capsys.readouterr().out
    assert "verify" in text
    assert "iterations=" in text


def test_format_dry_run_shows_verify_iterations_and_added_total():
    diag = dict(_diag())
    diag["verify"] = [{"range": [0, 30], "iterations": 3,
                       "bad_counts": [5, 2, 0], "added_counts": [5, 2, 0], "added_total": 7}]
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_diag=diag,
    )
    text = report.format_dry_run(rep)
    assert "verify" in text.lower()
    assert "iterations=3" in text
    assert "added_total=7" in text


def test_log_diagnostics_shows_verify_totals_and_per_iteration_counts_on_stdout(capsys):
    from sparsevmd import cli

    diag = dict(_diag())
    diag["verify"] = [{"range": [0, 30], "iterations": 3,
                       "bad_counts": [5, 2, 0], "added_counts": [5, 2, 0], "added_total": 7}]
    cli._log_diagnostics(diag, None)
    out = capsys.readouterr().out
    assert "出力後検証" in out
    assert "反復3" in out
    assert "追加7" in out
    assert "bad=[5, 2, 0]" in out
    assert "added=[5, 2, 0]" in out


def test_format_dry_run_shows_seam_rewrites():
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_diag=_diag(),
    )
    assert "seam rewrites: [10]" in report.format_dry_run(rep)


def test_log_diagnostics_shows_seam_rewrites(capsys):
    from sparsevmd import cli

    cli._log_diagnostics(_diag(), None)
    assert "継ぎ目書き換え [10]" in capsys.readouterr().out
