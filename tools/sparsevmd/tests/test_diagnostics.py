from sparsevmd import presets, report
from sparsevmd.reduce import (
    BONE_LINEAR_INTERP,
    camera_interp_bytes,
    reduce_bone_track,
    reduce_camera_track,
)
from vmd import interp
from vmd.types import BoneKey, CameraKey

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
EASE = (96, 0, 96, 30)
LIN = (20, 20, 107, 107)
TOLS = presets.resolve_tolerances("balanced")
CAM_CUT = (5.0, 20.0, 5.0)


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, interp_block=None):
    return CameraKey(frame, dist, center, rot, interp_block or CAM_LINEAR, fov, 0)


def _eased(v0, v1, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*EASE, f / span) for f in range(n)]


def _camera_jumping_at_frame_15():
    return [
        cam(f, center=((float(f) if f < 15 else float(f) + 50.0), 0.0, 0.0))
        for f in range(31)
    ]


def camera_track(source, ranges, **kw):
    opts = dict(
        cut_thresholds=CAM_CUT,
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
    )
    opts.update(kw)
    return reduce_camera_track(source, ranges, TOLS, **opts)


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, rot, BONE_LINEAR_INTERP)


def bone_track(source, ranges, **kw):
    opts = dict(
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
    )
    opts.update(kw)
    return reduce_bone_track(source, ranges, TOLS, **opts)


def test_diagnostics_captures_cut_positions():
    d = {}
    camera_track(_camera_jumping_at_frame_15(), [(0, 30)], diagnostics=d, no_cut_detect=False)
    assert 15 in d["cuts"]


def test_diagnostics_captures_split_with_position_channel_in_linear_mode():
    ys = _eased(0.0, 100.0)
    src = [cam(f, center=(0.0, ys[f], 0.0)) for f in range(11)]
    d = {}
    camera_track(src, [(0, 10)], diagnostics=d, curve_mode="linear")
    assert d["splits"]
    s = d["splits"][0]
    assert "frame" in s
    assert s["channel"] == "position"
    assert s["norm_error"] > 1.0


def test_diagnostics_captures_seam_rewrite_of_range_start_key():
    blk = camera_interp_bytes(EASE, LIN, LIN, LIN, LIN, LIN)
    src = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(100.0, 0.0, 0.0), interp_block=blk),
           cam(20, center=(100.0, 0.0, 0.0))]
    d = {}
    camera_track(src, [(10, 20)], diagnostics=d, curve_mode="bezier")
    assert 10 in d["seam_rewrites"]


def test_diagnostics_argument_is_optional():
    src = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(11)]
    keys = camera_track(src, [(0, 10)])
    assert [k.frame for k in keys] == [0, 10]


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


def test_build_report_includes_splits_and_seams():
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_diag=_diag(),
    )
    diag = rep["camera"]["diagnostics"]
    assert diag["splits"][0]["channel"] == "position"
    assert diag["seam_rewrites"] == [10]
    assert diag["cuts"] == [15]


def _force_linear_output_curves(monkeypatch):
    import vmd.fit as fit

    linear_cp = (20, 20, 107, 107)
    for cls in (
        fit.LinearScalarChannel,
        fit.FovChannel,
        fit.CameraRotationChannel,
        fit.BoneRotationChannel,
    ):
        monkeypatch.setattr(cls, "curve", lambda self, a, b: linear_cp)
    monkeypatch.setattr(
        fit.EuclideanVectorChannel, "curve", lambda self, a, b: (linear_cp,) * 3
    )


def test_verify_diag_recorded_clean_no_densification():
    xs = _eased(-10.0, -110.0)
    src = [cam(f, dist=xs[f]) for f in range(11)]
    d = {}
    camera_track(src, [(0, 10)], diagnostics=d, curve_mode="bezier")
    assert "verify" in d
    rec = d["verify"][0]
    assert rec["range"] == [0, 10]
    assert rec["iterations"] >= 1
    assert rec["added_total"] == 0
    assert rec["bad_counts"][-1] == 0
    assert len(rec["bad_counts"]) == rec["iterations"]
    assert len(rec["added_counts"]) == rec["iterations"]


def test_verify_diag_records_densification_until_converged(monkeypatch):
    xs = _eased(-10.0, -110.0)
    src = [cam(f, dist=xs[f]) for f in range(11)]
    _force_linear_output_curves(monkeypatch)
    d = {}
    camera_track(src, [(0, 10)], diagnostics=d, curve_mode="bezier")
    rec = d["verify"][0]
    assert rec["added_total"] > 0
    assert rec["added_total"] == sum(rec["added_counts"])
    assert rec["iterations"] >= 2
    assert len(rec["bad_counts"]) == rec["iterations"]
    assert len(rec["added_counts"]) == rec["iterations"]
    assert rec["bad_counts"][-1] == 0
    assert rec["added_counts"][-1] == 0


def test_verify_diag_bone_records_densification(monkeypatch):
    ys = _eased(0.0, 100.0)
    src = [bone("c", f, pos=(0.0, ys[f], 0.0)) for f in range(11)]
    _force_linear_output_curves(monkeypatch)
    d = {}
    bone_track(src, [(0, 10)], diagnostics=d, curve_mode="bezier")
    rec = d["verify"][0]
    assert rec["range"] == [0, 10]
    assert rec["added_total"] > 0
    assert rec["added_total"] == sum(rec["added_counts"])
    assert len(rec["bad_counts"]) == rec["iterations"]
    assert len(rec["added_counts"]) == rec["iterations"]


def test_verify_diag_one_record_per_range():
    src = [cam(f, dist=-30.0 - float(f)) for f in range(31)]
    d = {}
    camera_track(src, [(0, 10), (20, 30)], diagnostics=d, curve_mode="bezier")
    assert [r["range"] for r in d["verify"]] == [[0, 10], [20, 30]]


def test_build_report_preserves_verify_passthrough():
    diag = dict(_diag())
    diag["verify"] = [{"range": [0, 30], "iterations": 1,
                       "bad_counts": [0], "added_counts": [0], "added_total": 0}]
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_diag=diag,
    )
    assert rep["camera"]["diagnostics"]["verify"][0]["added_total"] == 0


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


def test_log_diagnostics_writes_to_given_file(capsys):
    import sys

    from sparsevmd import cli

    diag = dict(_diag())
    cli._log_diagnostics(diag, None, file=sys.stderr)
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "不連続検出位置" in cap.err
