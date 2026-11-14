from mocapvmd import cli

from .helpers import bone, write_vmd

_FRAME_COUNT = 21
_WOBBLE_X = (0.0, 0.02, -0.01, 0.03, 0.0)
_BONE_MODE_ONLY_TOKENS = ("clean_strength:", "clean_pos=", "clean_rot=", "win=[", "spikes=", "protected=")


def _write_input(path):
    keys = []
    for f in range(_FRAME_COUNT):
        wobble = _WOBBLE_X[f % len(_WOBBLE_X)]
        keys.append(bone("センター", f, pos=(0.05 * f + wobble, 0.0, 0.0)))
        keys.append(bone("右足ＩＫ", f, pos=(0.05 * f + wobble, 0.0, 0.0)))
        keys.append(bone("頭", f, rot=(0.0, 0.0, 0.01 * wobble, 1.0)))
    write_vmd(path, bone=keys)


def _report_text(tmp_path, capsys, *args):
    src = tmp_path / "in.vmd"
    _write_input(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), *args])
    assert rc == 0
    return capsys.readouterr().out


def test_bone_mode_dry_run_report_lists_denoise_strength_rows(tmp_path, capsys):
    text = _report_text(tmp_path, capsys, "--dry-run", "--denoise-mode", "bone")
    for token in _BONE_MODE_ONLY_TOKENS:
        assert token in text


def test_pose_mode_dry_run_report_has_no_bone_mode_denoise_rows(tmp_path, capsys):
    text = _report_text(tmp_path, capsys, "--dry-run", "--denoise-mode", "pose")
    assert "pose_denoise:" in text
    assert "clean_strength:" not in text
    assert "clean_pos=" not in text
