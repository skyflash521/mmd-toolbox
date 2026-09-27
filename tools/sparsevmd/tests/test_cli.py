import builtins
import sys

import pytest

from sparsevmd import cli
from vmd import interp, io
from vmd.reduce import bone_interp_bytes, camera_interp_bytes
from vmd.types import (
    BoneKey,
    CameraKey,
    IkBone,
    IkPropertyKey,
    LightKey,
    MorphKey,
    SelfShadowKey,
    VmdDocument,
)

_LINEAR_CP = (20, 20, 107, 107)
_EASE_CP = (40, 10, 90, 118)
CAM_LINEAR = camera_interp_bytes(*([_LINEAR_CP] * 6))
_CP932_LONE_LEAD_BYTE = b"\x81"
UNDECODABLE_NAME = "\ufffd"


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()


def cam(frame, center=(0.0, 0.0, 0.0)):
    return CameraKey(frame, -30.0, center, (0.0, 0.0, 0.0), CAM_LINEAR, 30, 0)


def bone(name, frame, pos=(0.0, 0.0, 0.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, (0.0, 0.0, 0.0, 1.0), BL)


def write_vmd(path, **sections):
    io.write_file(VmdDocument(**sections), str(path))


def linear_camera_doc():
    return [cam(f, center=(float(f), 0.0, 0.0)) for f in range(31)]


def long_linear_camera_doc(n):
    return [cam(f, center=(float(f), 0.0, 0.0)) for f in range(n + 1)]


def test_camera_reduce_writes_output(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert code == 0
    assert out.exists()
    doc, _ = io.read(str(out))
    assert [k.frame for k in doc.camera] == [0, 30]


def test_max_segment_frames_default_is_unlimited(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=long_linear_camera_doc(360))
    out = tmp_path / "out.vmd"
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert code == 0
    doc, _ = io.read(str(out))
    assert [k.frame for k in doc.camera] == [0, 360]


def test_max_segment_frames_explicit_caps_span(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=long_linear_camera_doc(360))
    out = tmp_path / "out.vmd"
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear",
                     "--max-segment-frames", "180"])
    assert code == 0
    doc, _ = io.read(str(out))
    frames = [k.frame for k in doc.camera]
    assert frames[0] == 0 and frames[-1] == 360
    assert len(frames) > 2


def test_max_segment_frames_zero_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "out.vmd"
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--max-segment-frames", "0"])
    assert code == 2


EASE = (96, 0, 96, 30)


def eased_camera_doc():
    return [
        cam(f, center=(30.0 * interp._solve_factor(*EASE, f / 30.0), 0.0, 0.0))
        for f in range(31)
    ]


def test_curve_mode_bezier_reduces_curved_motion(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=eased_camera_doc())

    bez_out = tmp_path / "bez.vmd"
    assert cli.main([str(src), "-o", str(bez_out), "--target", "camera", "--no-cut-detect"]) == 0
    bez, _ = io.read(str(bez_out))

    lin_out = tmp_path / "lin.vmd"
    assert cli.main(
        [str(src), "-o", str(lin_out), "--target", "camera", "--curve-mode", "linear",
         "--no-cut-detect"]
    ) == 0
    lin, _ = io.read(str(lin_out))

    assert [k.frame for k in bez.camera] == [0, 30]
    assert len(lin.camera) > 2
    assert bez.camera[-1].interpolation[0:4] != CAM_LINEAR[0:4]


def test_default_output_path(tmp_path):
    src = tmp_path / "motion.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--curve-mode", "linear"])
    assert code == 0
    assert (tmp_path / "motion_sparse.vmd").exists()


def test_non_target_section_and_model_name_pass_through(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    cam_keys = [cam(0), cam(30)]
    bone_keys = [bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
    model_name_raw = b"TestModel".ljust(20, b"\x00")
    write_vmd(src, camera=cam_keys, bone=bone_keys, model_name_raw=model_name_raw)
    code = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear"])
    assert code == 0
    doc, _ = io.read(str(out))
    assert doc.camera == cam_keys
    assert len(doc.bone) < 31
    assert doc.model_name_raw == model_name_raw


def test_duplicate_frame_camera_last_wins_and_single_key_per_frame(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [cam(0, center=(0.0, 0.0, 0.0)), cam(0, center=(5.0, 0.0, 0.0)),
            cam(10, center=(5.0, 0.0, 0.0))]
    write_vmd(src, camera=keys)
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert code == 0
    doc, _ = io.read(str(out))
    frames = [k.frame for k in doc.camera]
    assert len(frames) == len(set(frames))
    assert interp.sample(doc.camera, "pos_x", 0) == pytest.approx(5.0)


def test_single_key_camera_preserved_verbatim(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    ease_blk = camera_interp_bytes(_EASE_CP, *([_LINEAR_CP] * 5))
    key = CameraKey(7, -30.0, (3.0, 0.0, 0.0), (0.0, 0.0, 0.0), ease_blk, 30, 0)
    write_vmd(src, camera=[key])
    code = cli.main([str(src), "-o", str(out), "--target", "camera"])
    assert code == 0
    doc, _ = io.read(str(out))
    assert doc.camera == [key]


def test_single_key_bone_preserved_verbatim(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    nonlinear = bone_interp_bytes(_EASE_CP, _LINEAR_CP, _LINEAR_CP, _LINEAR_CP)
    key = BoneKey("センター".encode("cp932").ljust(15, b"\x00"), 5, (1.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0, 1.0), nonlinear)
    write_vmd(src, bone=[key])
    code = cli.main([str(src), "-o", str(out), "--target", "bone"])
    assert code == 0
    doc, _ = io.read(str(out))
    assert doc.bone == [key]


def test_dry_run_records_no_reduction_target(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=[cam(7, center=(3.0, 0.0, 0.0))])
    code = cli.main([str(src), "--target", "camera", "--dry-run"])
    assert code == 0
    assert "削減対象なし" in capsys.readouterr().out


def test_verbose_logs_cut_frame_to_stdout(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [cam(f, center=((float(f) if f < 15 else float(f) + 50.0), 0.0, 0.0))
            for f in range(31)]
    write_vmd(src, camera=keys)
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "-v"])
    assert code == 0
    cap = capsys.readouterr()
    assert "15" in cap.out
    assert "15" not in cap.err


def test_bone_file_decode_error_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(11)])
    bf = tmp_path / "bones.txt"
    bf.write_bytes(b"\xff\xfe\x00 invalid utf8")
    code = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "bone",
                     "--bone-file", str(bf)])
    assert code == 1


def test_keep_frame_out_of_range_warns(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear",
                     "--range", "0:10", "--keep-frame", "50"])
    assert code == 0
    err = capsys.readouterr().err
    assert "50" in err and "warning: keep_frame_ignored: " in err


def test_list_bones_without_bone_keys_warns_selection_unresolved(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=[cam(0), cam(30)])
    code = cli.main([str(src), "--list-bones", "--bone", "存在しない"])
    assert code == 0
    err = capsys.readouterr().err
    assert "warning: selection_unresolved: " in err


def test_list_bones_shows_name_count_and_selection(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30), bone("頭", 0)])
    code = cli.main([str(src), "--bone", "センター", "--list-bones"])
    assert code == 0
    out = capsys.readouterr().out
    assert "センター" in out and "頭" in out
    assert "2" in out
    low = out.lower()
    assert "select" in low or "選択" in out


def test_list_bones_does_not_write_output_and_exits_zero(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0)])
    code = cli.main([str(src), "-o", str(out), "--list-bones"])
    assert code == 0
    assert not out.exists()


def test_missing_input_is_input_error(tmp_path):
    code = cli.main([str(tmp_path / "nope.vmd"), "--target", "camera"])
    assert code == 1


def test_input_directory_is_input_error(tmp_path):
    d = tmp_path / "indir"
    d.mkdir()
    code = cli.main([str(d), "--target", "camera"])
    assert code == 1


def test_bad_range_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--range", "20:10"])
    assert code == 2


def test_target_camera_with_bone_selection_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=[bone("センター", 0)])
    code = cli.main([str(src), "--target", "camera", "--bone", "センター"])
    assert code == 2


def test_fov_tol_below_half_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--camera-fov-tol", "0.4"])
    assert code == 2


def test_same_path_without_overwrite_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(src), "--target", "camera"])
    assert code == 2


def test_same_path_with_overwrite_ok(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(src), "--overwrite", "--target", "camera", "--curve-mode", "linear"])
    assert code == 0


def test_target_camera_no_camera_keys_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30)])
    code = cli.main([str(src), "--target", "camera"])
    assert code == 1


def test_bone_file_missing_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30)])
    code = cli.main([str(src), "--target", "bone", "--bone-file", str(tmp_path / "nope.txt")])
    assert code == 1


def test_keep_frame_negative_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--keep-frame", "-3"])
    assert code == 2


def test_target_all_explicit_bone_absent_section_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "all", "--bone", "存在しない"])
    assert code == 1


def test_target_bone_explicit_missing_name_is_input_error_when_section_empty(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "bone", "--bone", "存在しない"])
    assert code == 1


def test_unmatched_glob_warns_in_reduce_path(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    bones = [bone("頭", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
    write_vmd(src, bone=bones)
    code = cli.main(
        [str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear",
         "--bone", "頭", "--bone-glob", "幻*"]
    )
    assert code == 0
    assert "幻*" in capsys.readouterr().err


def test_sole_unmatched_glob_warns_before_exit2(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("頭", 0), bone("頭", 30)])
    code = cli.main([str(src), "--target", "bone", "--bone-glob", "幻*"])
    assert code == 2
    assert "幻*" in capsys.readouterr().err


def test_range_reduces_inside_and_keeps_keys_outside(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    cam_keys = [cam(f, center=(float(f), 0.0, 0.0)) for f in (0, 5, 10, 15, 20, 25, 30)]
    write_vmd(src, camera=cam_keys)
    code = cli.main(
        [str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "--range", "0:10"]
    )
    assert code == 0
    doc, _ = io.read(str(out))
    fr = [k.frame for k in doc.camera]
    assert 0 in fr and 10 in fr
    assert 5 not in fr
    assert [k for k in doc.camera if k.frame > 10] == [k for k in cam_keys if k.frame > 10]


def test_range_start_beyond_track_end_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--range", "999:"])
    assert code == 2


def test_dry_run_prints_stats_and_writes_no_output(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "--dry-run"])
    assert code == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert "camera" in text and "31" in text


def test_report_json_option_is_removed(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--report-json", str(tmp_path / "report.json")])
    assert code == 2


def test_preview_csv_option_is_removed(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--preview-csv", str(tmp_path / "preview.csv")])
    assert code == 2


def test_output_parent_missing_is_write_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "nodir" / "out.vmd"
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert code == 3


def test_strict_unsatisfiable_by_min_segment_is_exit_4(tmp_path):
    src = tmp_path / "in.vmd"
    cam_keys = [cam(f, center=(0.0, 0.0 if f % 2 == 0 else 5.0, 0.0)) for f in range(9)]
    write_vmd(src, camera=cam_keys)
    out = tmp_path / "out.vmd"
    code = cli.main(
        [
            str(src),
            "-o",
            str(out),
            "--target",
            "camera",
            "--curve-mode",
            "linear",
            "--strict",
            "--min-segment-frames",
            "8",
            "--max-segment-frames",
            "180",
        ]
    )
    assert code == 4


def undecodable_bone(frame, pos=(0.0, 0.0, 0.0)):
    return BoneKey(_CP932_LONE_LEAD_BYTE.ljust(15, b"\x00"), frame, pos, (0.0, 0.0, 0.0, 1.0), BL)


def test_undecodable_bone_name_warns_and_reduces(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [undecodable_bone(0), undecodable_bone(10, (1.0, 0.0, 0.0)), undecodable_bone(20, (2.0, 0.0, 0.0))]
    write_vmd(src, bone=keys)
    code = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear"])
    assert code == 0
    err = capsys.readouterr().err
    assert "デコード" in err or "Shift-JIS" in err
    assert out.exists()
    out_doc, _ = io.read(str(out))
    assert [k.frame for k in out_doc.bone] == [0, 20]


def test_undecodable_name_not_targetable_by_bone(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [undecodable_bone(0), undecodable_bone(10), undecodable_bone(20)]
    write_vmd(src, bone=keys)
    code = cli.main([str(src), "-o", str(out), "--target", "bone", "--bone", UNDECODABLE_NAME])
    assert code == 2


def _single_stderr_line(err):
    lines = err.splitlines()
    assert len(lines) == 1
    return lines[0]


def test_read_warning_line_uses_common_format(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [undecodable_bone(0), undecodable_bone(10, (1.0, 0.0, 0.0)), undecodable_bone(20, (2.0, 0.0, 0.0))]
    write_vmd(src, bone=keys)
    rc = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear"])
    assert rc == 0
    out_text, err = capsys.readouterr()
    assert out_text == ""
    line = _single_stderr_line(err)
    prefix = "warning: decode-error: "
    assert line.startswith(prefix)
    body = line[len(prefix):]
    assert body.strip()
    assert body.lstrip() == body
    assert "警告:" not in err


def test_selector_unmatched_warning_line_uses_common_format(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("頭", f) for f in range(31)])
    rc = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear",
                   "--bone", "頭", "--bone-glob", "幻*"])
    assert rc == 0
    out_text, err = capsys.readouterr()
    assert out_text == ""
    line = _single_stderr_line(err)
    prefix = "warning: selector_unmatched: "
    assert line.startswith(prefix)
    body = line[len(prefix):]
    assert body.strip()
    assert body.lstrip() == body
    assert "警告:" not in err


def test_keep_frame_ignored_warning_line_uses_common_format(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear",
                   "--range", "0:10", "--keep-frame", "50"])
    assert rc == 0
    out_text, err = capsys.readouterr()
    assert out_text == ""
    line = _single_stderr_line(err)
    prefix = "warning: keep_frame_ignored: "
    assert line.startswith(prefix)
    body = line[len(prefix):]
    assert body.strip()
    assert body.lstrip() == body
    assert "警告:" not in err


def test_selection_unresolved_warning_line_uses_common_format(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("頭", 0), bone("頭", 30)])
    rc = cli.main([str(src), "--list-bones", "--bone-glob", "幻*"])
    assert rc == 0
    out_text, err = capsys.readouterr()
    assert "warning" not in out_text
    assert "警告:" not in out_text
    lines = err.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("warning: selector_unmatched: ")
    assert lines[1].startswith("warning: selection_unresolved: ")
    for line in lines:
        label, code, body = line.split(": ", 2)
        assert label == "warning" and code
        assert body.strip()
        assert body.lstrip() == body
    assert "警告:" not in err


class _SpyProgressReporter:
    calls = None

    def __init__(self, *args, **kwargs):
        pass

    def stage(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        _SpyProgressReporter.calls.append("close")

    def summary(self, message):
        _SpyProgressReporter.calls.append(("summary", message))


@pytest.fixture
def spy_progress(monkeypatch):
    calls = []
    _SpyProgressReporter.calls = calls
    monkeypatch.setattr(cli.progress, "ProgressReporter", _SpyProgressReporter)

    real_print = print

    def spy_print(*args, **kwargs):
        if kwargs.get("file") is sys.stderr and args:
            calls.append(("stderr_print", args[0]))
        real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", spy_print)
    return calls


def test_normal_run_closes_progress_then_shows_completion(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    assert spy_progress[:2] == ["close", ("summary", f"完了 {out}")]
    assert all(call == "close" for call in spy_progress[2:])


def test_dry_run_closes_progress_without_completion_line(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "--target", "camera", "--curve-mode", "linear", "--dry-run"])
    assert rc == 0
    assert spy_progress[0] == "close"
    assert not any(isinstance(c, tuple) and c[0] == "summary" for c in spy_progress)


def test_strict_failure_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    cam_keys = [cam(f, center=(0.0, 0.0 if f % 2 == 0 else 5.0, 0.0)) for f in range(9)]
    write_vmd(src, camera=cam_keys)
    out = tmp_path / "out.vmd"
    rc = cli.main([
        str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear",
        "--strict", "--min-segment-frames", "8", "--max-segment-frames", "8",
    ])
    assert rc == 4
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))
    assert not any(isinstance(c, tuple) and c[0] == "summary" for c in spy_progress)


def test_write_failure_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "nodir" / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 3
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))
    assert not any(isinstance(c, tuple) and c[0] == "summary" for c in spy_progress)


def test_keyboard_interrupt_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())

    def raise_interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "reduce_camera_track", raise_interrupt)
    rc = cli.main([str(src), "--target", "camera", "--curve-mode", "linear", "--dry-run"])
    assert rc == 130
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))
    assert not any(isinstance(c, tuple) and c[0] == "summary" for c in spy_progress)


def test_verbose_closes_progress_before_diagnostics(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    monkeypatch.setattr(cli, "_log_diagnostics", lambda *a, **k: spy_progress.append("log_diagnostics"))
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "-v"])
    assert rc == 0
    assert "close" in spy_progress and "log_diagnostics" in spy_progress
    assert spy_progress.index("close") < spy_progress.index("log_diagnostics")


class _LabelSpyProgressReporter:
    calls = None

    def __init__(self, *args, **kwargs):
        pass

    def stage(self, label):
        _LabelSpyProgressReporter.calls.append(("stage", label))

    def update(self, done, total, note=""):
        _LabelSpyProgressReporter.calls.append(("update", done, total, note))

    def close(self):
        pass

    def summary(self, message):
        pass


def test_progress_label_is_keyframe_reduction_with_target_note(tmp_path, monkeypatch):
    calls = []
    _LabelSpyProgressReporter.calls = calls
    monkeypatch.setattr(cli.progress, "ProgressReporter", _LabelSpyProgressReporter)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=[bone("センター", f, pos=(0.0, float(f), 0.0))
                                                      for f in range(31)])
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "all", "--curve-mode", "linear"])
    assert rc == 0
    stages = [c for c in calls if c[0] == "stage"]
    updates = [c for c in calls if c[0] == "update"]
    assert stages and all(label == "キーフレーム圧縮" for _, label in stages)
    assert any(note == "カメラ" for _, _done, _total, note in updates)
    assert any(note == "センター" for _, _done, _total, note in updates)


def test_progress_camera_note_appends_reducer_note(tmp_path, monkeypatch):
    calls = []
    _LabelSpyProgressReporter.calls = calls
    monkeypatch.setattr(cli.progress, "ProgressReporter", _LabelSpyProgressReporter)

    def fake_reduce_camera_track(cam, cam_ranges, tols, cut_thresholds=None, diagnostics=None,
                                  progress=None, **kw):
        if progress is not None:
            progress(1, 2, "出力後検証")
        return cam

    monkeypatch.setattr(cli, "reduce_camera_track", fake_reduce_camera_track)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    updates = [c for c in calls if c[0] == "update"]
    assert any(note == "カメラ 出力後検証" for _, _done, _total, note in updates)


def test_progress_bone_note_shows_target_before_processing(tmp_path, monkeypatch):
    calls = []
    _LabelSpyProgressReporter.calls = calls
    monkeypatch.setattr(cli.progress, "ProgressReporter", _LabelSpyProgressReporter)

    seen_notes_at_call = []

    def fake_reduce_bone_track(keys, track_ranges, tols, cut_thresholds=None, diagnostics=None, **kw):
        updates = [c for c in calls if c[0] == "update"]
        seen_notes_at_call.append(updates[-1][3] if updates else None)
        return keys

    monkeypatch.setattr(cli, "reduce_bone_track", fake_reduce_bone_track)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=(
        [bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
        + [bone("上半身", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
    ))
    rc = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear"])
    assert rc == 0
    assert seen_notes_at_call == ["センター", "上半身"]


class _EnabledCapturingReporter:
    captured_enabled = None

    def __init__(self, *args, **kwargs):
        _EnabledCapturingReporter.captured_enabled = kwargs.get("enabled")

    def stage(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def summary(self, message):
        pass


class _TTYWrapper:
    def __init__(self, stream):
        self._stream = stream

    def isatty(self):
        return True

    def __getattr__(self, name):
        return getattr(self._stream, name)


def test_progress_disabled_when_not_tty(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.progress, "ProgressReporter", _EnabledCapturingReporter)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    assert _EnabledCapturingReporter.captured_enabled is False


def test_progress_enabled_when_tty_and_not_quiet_and_not_machine(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stderr", _TTYWrapper(sys.stderr))
    monkeypatch.setattr(cli.progress, "ProgressReporter", _EnabledCapturingReporter)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    assert _EnabledCapturingReporter.captured_enabled is True


def test_progress_disabled_with_quiet_flag_even_when_tty(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stderr", _TTYWrapper(sys.stderr))
    monkeypatch.setattr(cli.progress, "ProgressReporter", _EnabledCapturingReporter)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "--quiet"])
    assert rc == 0
    assert _EnabledCapturingReporter.captured_enabled is False


def test_progress_disabled_in_machine_mode_even_when_tty(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stderr", _TTYWrapper(sys.stderr))
    monkeypatch.setattr(cli.progress, "ProgressReporter", _EnabledCapturingReporter)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "--machine"])
    assert rc == 0
    assert _EnabledCapturingReporter.captured_enabled is False


def ramp_bone_doc(name="センター", last_frame=30):
    return [bone(name, f, pos=(0.0, float(f), 0.0)) for f in range(last_frame + 1)]


def test_morph_light_self_shadow_ik_sections_pass_through_unchanged(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    morphs = [
        MorphKey("あ".encode("cp932").ljust(15, b"\x00"), 30, 1.0),
        MorphKey("あ".encode("cp932").ljust(15, b"\x00"), 0, 0.0),
    ]
    lights = [LightKey(0, (0.6, 0.6, 0.6), (-0.5, -1.0, 0.5))]
    shadows = [SelfShadowKey(0, 1, 0.1)]
    iks = [IkPropertyKey(0, 1, [IkBone("左足ＩＫ".encode("cp932").ljust(20, b"\x00"), 1)])]
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc(), morph=morphs, light=lights,
              self_shadow=shadows, ik_property=iks)
    assert cli.main([str(src), "-o", str(out), "--curve-mode", "linear"]) == 0
    before, _ = io.read(str(src))
    after, _ = io.read(str(out))
    assert [k.frame for k in after.morph] == [30, 0]
    assert after.morph == before.morph
    assert after.light == before.light
    assert after.self_shadow == before.self_shadow
    assert after.ik_property == before.ik_property
    assert _non_target_section_bytes(out) == _non_target_section_bytes(src)


_HEADER_AND_MODEL_NAME_SIZE = 50
_BONE_RECORD_SIZE = 111
_MORPH_RECORD_SIZE = 23
_CAMERA_RECORD_SIZE = 61


def _non_target_section_bytes(path):
    data = path.read_bytes()
    pos = _HEADER_AND_MODEL_NAME_SIZE
    pos += 4 + int.from_bytes(data[pos:pos + 4], "little") * _BONE_RECORD_SIZE
    morph_end = pos + 4 + int.from_bytes(data[pos:pos + 4], "little") * _MORPH_RECORD_SIZE
    morph_section = data[pos:morph_end]
    camera_end = morph_end + 4 + int.from_bytes(data[morph_end:morph_end + 4], "little") * _CAMERA_RECORD_SIZE
    return morph_section, data[camera_end:]


def test_same_input_and_arguments_give_byte_identical_output(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=eased_camera_doc(), bone=ramp_bone_doc())
    out = tmp_path / "out.vmd"
    args = [str(src), "-o", str(out), "--overwrite"]
    assert cli.main(args) == 0
    first = out.read_bytes()
    assert cli.main(args) == 0
    assert out.read_bytes() == first


def test_name_selector_is_case_sensitive(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=ramp_bone_doc("Head"))
    assert cli.main([str(src), "--target", "bone", "--bone", "head", "--dry-run"]) == 2


def test_target_bone_without_bone_keys_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    assert cli.main([str(src), "--target", "bone"]) == 1


def test_target_all_without_any_camera_or_bone_keys_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, morph=[MorphKey("あ".encode("cp932").ljust(15, b"\x00"), 0, 0.0)])
    assert cli.main([str(src), "--target", "all"]) == 1


def test_target_all_without_bone_keys_reduces_camera_only(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    assert cli.main([str(src), "-o", str(out), "--target", "all", "--curve-mode", "linear"]) == 0
    doc, _ = io.read(str(out))
    assert [k.frame for k in doc.camera] == [0, 30]


@pytest.mark.parametrize(
    "selection",
    [pytest.param([], id="no_selection"), pytest.param(["--bone", "センター"], id="explicit_selection")],
)
def test_target_all_without_camera_keys_reduces_bone_only(tmp_path, selection):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=ramp_bone_doc())
    assert cli.main([str(src), "-o", str(out), "--target", "all", "--curve-mode", "linear", *selection]) == 0
    doc, _ = io.read(str(out))
    assert [k.frame for k in doc.bone] == [0, 30]


def test_target_all_bone_selection_leaves_camera_reduced_and_other_bones_verbatim(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    head_keys = ramp_bone_doc("頭")
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc() + head_keys)
    assert cli.main([str(src), "-o", str(out), "--target", "all", "--curve-mode", "linear",
                     "--bone", "センター"]) == 0
    doc, _ = io.read(str(out))
    assert [k.frame for k in doc.camera] == [0, 30]
    assert [k.frame for k in doc.bone if k.name == "センター"] == [0, 30]
    assert [k for k in doc.bone if k.name == "頭"] == head_keys


def test_list_bones_allows_bone_selection_with_target_camera(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    assert cli.main([str(src), "--target", "camera", "--bone", "センター", "--list-bones"]) == 0


def test_list_bones_ignores_existing_output_file(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    out.write_bytes(b"old")
    write_vmd(src, bone=ramp_bone_doc())
    assert cli.main([str(src), "-o", str(out), "--list-bones"]) == 0
    assert out.read_bytes() == b"old"


def test_range_outside_every_track_keeps_keys_and_reports_no_reduction(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    cam_keys = linear_camera_doc()
    write_vmd(src, camera=cam_keys)
    args = [str(src), "--target", "camera", "--curve-mode", "linear", "--range", "100:200"]
    assert cli.main([*args, "-o", str(out)]) == 0
    doc, _ = io.read(str(out))
    assert doc.camera == cam_keys
    capsys.readouterr()
    assert cli.main([*args, "--dry-run"]) == 0
    assert "削減対象なし" in capsys.readouterr().out


def test_keep_frame_outside_one_track_range_is_ignored_only_for_that_track(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=ramp_bone_doc("センター") + ramp_bone_doc("頭", last_frame=10))
    assert cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear",
                     "--keep-frame", "20"]) == 0
    assert "keep_frame_ignored" not in capsys.readouterr().err
    doc, _ = io.read(str(out))
    assert [k.frame for k in doc.bone if k.name == "センター"] == [0, 20, 30]
    assert [k.frame for k in doc.bone if k.name == "頭"] == [0, 10]


def _camera_with_distance_jump_at_frame_15():
    return [
        CameraKey(f, -30.0 if f < 15 else -60.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), CAM_LINEAR, 30, 0)
        for f in range(31)
    ]


@pytest.mark.parametrize(
    ("thresholds", "expect_cut"),
    [
        pytest.param([], True, id="default_distance_threshold"),
        pytest.param(["--cut-threshold-camera", "5.0,20.0,1000.0"], False, id="raised_distance_threshold"),
    ],
)
def test_camera_distance_jump_is_cut_by_dist_threshold(tmp_path, capsys, thresholds, expect_cut):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=_camera_with_distance_jump_at_frame_15())
    assert cli.main([str(src), "--target", "camera", "--dry-run", *thresholds]) == 0
    assert ("cuts: [15]" in capsys.readouterr().out) is expect_cut


def test_repeated_flag_behaves_as_single(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    assert cli.main([str(src), "-o", str(out), "--target", "camera", "--dry-run", "--dry-run"]) == 0
    assert not out.exists()


def test_main_installs_sigbreak_handler(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "install_sigbreak_handler", lambda: calls.append(True))
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    assert cli.main([str(src), "--target", "camera", "--dry-run"]) == 0
    assert calls == [True]
