from sparsevmd import report
from vmd import interp
from vmd.types import CameraKey

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
EASE = (96, 0, 96, 30)


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, 0)


def _cam_errors():
    return {"pos_x": 0.5, "pos_y": 0.0, "pos_z": 0.0, "distance": 0.1, "fov": 0.0, "rot_deg": 0.3}


def test_build_report_includes_camera_errors():
    rep = report.build_report(
        target="camera",
        camera=(31, 2),
        bones=None,
        selected_bones=set(),
        ranges=[(0, 30)],
        keep_frames=[],
        camera_errors=_cam_errors(),
    )
    assert rep["camera"]["errors"] == _cam_errors()


def test_build_report_includes_bone_errors_per_name():
    berr = {"センター": {"pos_x": 0.2, "pos_y": 0.0, "pos_z": 0.0, "rot_deg": 0.1}}
    rep = report.build_report(
        target="bone",
        camera=None,
        bones={"センター": (31, 3)},
        selected_bones={"センター"},
        ranges=[(0, 30)],
        keep_frames=[],
        bone_errors=berr,
    )
    entry = rep["bones"][0]
    assert entry["errors"] == berr["センター"]


def test_format_dry_run_shows_camera_max_error():
    rep = report.build_report(
        target="camera", camera=(31, 2), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_errors=_cam_errors(),
    )
    text = report.format_dry_run(rep)
    assert "max error" in text or "最大誤差" in text
    assert "0.5" in text


def test_cli_dry_run_includes_camera_errors(tmp_path, capsys):
    from sparsevmd import cli
    from vmd import io
    from vmd.types import VmdDocument

    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [
        cam(f, dist=-10.0 - 100.0 * interp._solve_factor(*EASE, f / 30.0))
        for f in range(31)
    ]
    io.write_file(VmdDocument(camera=keys), str(src))
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--dry-run"])
    assert code == 0
    text = capsys.readouterr().out
    assert "max error" in text
    for k in ("pos_x", "pos_y", "pos_z", "distance", "fov", "rot_deg"):
        assert k in text


def test_build_report_no_errors_when_omitted():
    rep = report.build_report(
        target="camera", camera=(31, 2), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[],
    )
    assert "errors" not in rep["camera"]
