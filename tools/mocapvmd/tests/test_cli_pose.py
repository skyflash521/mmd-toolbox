from mocapvmd import cli
from mocapvmd.model_profile import STANDARD_BONE_NAMES
from vmd import io

from .helpers import bone, build_standard_pmx, write_vmd

_PMX_ENCODING_BYTE_OFFSET = len(b"PMX ") + 4 + 1
_UNSUPPORTED_PMX_ENCODING = 5
_INPUT_FRAME_RANGE = list(range(0, 11))


def _write_input(path):
    write_vmd(
        path,
        bone=[
            bone("センター", 0),
            bone("センター", 10, pos=(0.3, 0.0, 0.0)),
            bone("頭", 0),
            bone("頭", 10, rot=(0.0, 0.0, 0.05, 0.99875)),
        ],
    )


def test_bone_mode_is_default(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_input(src)
    code = cli.main([str(src), "-o", str(out), "--no-reduce"])
    assert code == 0
    assert out.is_file()


def test_pose_mode_without_pmx_outputs_dense_center_keys(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_input(src)
    code = cli.main([str(src), "-o", str(out), "--denoise-mode", "pose", "--no-reduce"])
    assert code == 0
    doc, _ = io.read(str(out))
    names = {k.name for k in doc.bone}
    assert "センター" in names and "頭" in names
    center_frames = sorted(k.frame for k in doc.bone if k.name == "センター")
    assert center_frames == _INPUT_FRAME_RANGE


def test_pose_mode_with_pmx_outputs_dense_center_keys(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    pmx = tmp_path / "model.pmx"
    _write_input(src)
    pmx.write_bytes(build_standard_pmx(list(STANDARD_BONE_NAMES.values())))
    code = cli.main(
        [str(src), "-o", str(out), "--denoise-mode", "pose", "--pmx", str(pmx), "--no-reduce"]
    )
    assert code == 0
    doc, _ = io.read(str(out))
    center_frames = sorted(k.frame for k in doc.bone if k.name == "センター")
    assert center_frames == _INPUT_FRAME_RANGE


def test_pose_pmx_missing_path_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    _write_input(src)
    code = cli.main(
        [
            str(src),
            "-o",
            str(tmp_path / "out.vmd"),
            "--denoise-mode",
            "pose",
            "--pmx",
            str(tmp_path / "nope.pmx"),
        ]
    )
    assert code == 1


def test_pose_pmx_directory_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    _write_input(src)
    d = tmp_path / "pmxdir"
    d.mkdir()
    code = cli.main(
        [str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose", "--pmx", str(d)]
    )
    assert code == 1


def test_pose_pmx_unsupported_encoding_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    pmx = tmp_path / "model.pmx"
    _write_input(src)
    data = bytearray(build_standard_pmx(list(STANDARD_BONE_NAMES.values())))
    data[_PMX_ENCODING_BYTE_OFFSET] = _UNSUPPORTED_PMX_ENCODING
    pmx.write_bytes(bytes(data))
    code = cli.main(
        [str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose", "--pmx", str(pmx)]
    )
    assert code == 1


def test_pose_pmx_missing_required_role_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    pmx = tmp_path / "model.pmx"
    _write_input(src)
    names_without_right_wrist = [
        n for n in STANDARD_BONE_NAMES.values() if n != STANDARD_BONE_NAMES["wrist_r"]
    ]
    pmx.write_bytes(build_standard_pmx(names_without_right_wrist))
    code = cli.main(
        [str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose", "--pmx", str(pmx)]
    )
    assert code == 1


def test_pose_pmx_format_error_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    pmx = tmp_path / "bad.pmx"
    _write_input(src)
    pmx.write_bytes(b"NOTPMX")
    code = cli.main(
        [str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose", "--pmx", str(pmx)]
    )
    assert code == 1


def test_no_denoise_pose_mode_needs_no_pmx(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_input(src)
    code = cli.main(
        [str(src), "-o", str(out), "--denoise-mode", "pose", "--no-denoise", "--no-reduce"]
    )
    assert code == 0
    assert out.is_file()


def test_pose_dry_run_reports_marker_check_displacement_and_fit(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    _write_input(src)
    code = cli.main([str(src), "--dry-run", "--denoise-mode", "pose"])
    assert code == 0
    out = capsys.readouterr().out
    assert "pose_denoise" in out
    assert "既定モデルプロファイル" in out
    assert "available=" in out
    assert "required_bones_ok=True" in out
    assert "marker_disp:" in out
    assert "fit:" in out


def test_pose_dry_run_without_pmx_reports_default_profile_and_fallback(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    _write_input(src)
    code = cli.main([str(src), "--dry-run", "--denoise-mode", "pose"])
    assert code == 0
    out = capsys.readouterr().out
    assert "pose_denoise" in out
    assert "既定モデルプロファイル" in out
    assert "available=" in out
    assert "fallback=" in out


def test_bone_mode_dry_run_omits_pose_denoise(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    _write_input(src)
    code = cli.main([str(src), "--dry-run"])
    assert code == 0
    assert "pose_denoise" not in capsys.readouterr().out
