import pytest

from mocapvmd import cli, denoise, footik

from .helpers import bone, write_vmd

_FRAME_COUNT = 21
_WOBBLE_X = (0.0, 0.02, -0.01, 0.03, 0.0)
_BONE_MODE_ONLY_TOKENS = ("clean_strength:", "clean_pos=", "clean_rot=", "win=[", "spikes=", "protected=")
_REPORT_FLAGS = (["--dry-run"], ["--verbose"], ["--dry-run", "--verbose"])
_REPORT_FLAG_IDS = ("dry-run", "verbose", "dry-run-verbose")


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


@pytest.mark.parametrize("token", _BONE_MODE_ONLY_TOKENS)
def test_pose_mode_dry_run_report_has_no_bone_mode_denoise_rows(tmp_path, capsys, token):
    text = _report_text(tmp_path, capsys, "--dry-run", "--denoise-mode", "pose")
    assert "pose_denoise:" in text
    assert token not in text


@pytest.mark.parametrize("token", _BONE_MODE_ONLY_TOKENS)
def test_pose_mode_verbose_report_has_no_bone_mode_denoise_rows(tmp_path, capsys, token):
    text = _report_text(tmp_path, capsys, "--verbose", "--denoise-mode", "pose")
    assert "pose_denoise:" in text
    assert token not in text


class _CallCounter:
    def __init__(self, monkeypatch, module, name):
        self.calls = 0
        original = getattr(module, name)

        def counted(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)


def _stage_call_counts(workdir, monkeypatch, capsys, *args):
    workdir.mkdir()
    counters = {
        "_clean_bones": _CallCounter(monkeypatch, cli, "_clean_bones"),
        "_stabilize_bones": _CallCounter(monkeypatch, cli, "_stabilize_bones"),
        "apply_denoise": _CallCounter(monkeypatch, denoise, "apply_denoise"),
        "detect_noise_events": _CallCounter(monkeypatch, denoise, "detect_noise_events"),
        "stabilize_foot_ik": _CallCounter(monkeypatch, footik, "stabilize_foot_ik"),
    }
    src = workdir / "in.vmd"
    _write_input(src)
    rc = cli.main([str(src), "-o", str(workdir / "out.vmd"), *args])
    assert rc == 0
    capsys.readouterr()
    return {name: c.calls for name, c in counters.items()}


def _plain_and_with_report_counts(tmp_path, monkeypatch, capsys, report_flags, *args):
    plain = _stage_call_counts(tmp_path / "plain", monkeypatch, capsys, *args)
    with_report = _stage_call_counts(tmp_path / "report", monkeypatch, capsys, *report_flags, *args)
    return plain, with_report


def test_each_stage_function_runs_once_per_plain_bone_mode_run(tmp_path, monkeypatch, capsys):
    counts = _stage_call_counts(tmp_path / "plain", monkeypatch, capsys)
    assert counts["_clean_bones"] == 1
    assert counts["_stabilize_bones"] == 1
    assert counts["stabilize_foot_ik"] == 1


@pytest.mark.parametrize("report_flags", _REPORT_FLAGS, ids=_REPORT_FLAG_IDS)
def test_report_does_not_rerun_denoise_or_stabilization_in_bone_mode(tmp_path, monkeypatch, capsys, report_flags):
    plain, with_report = _plain_and_with_report_counts(tmp_path, monkeypatch, capsys, report_flags)
    assert with_report["_clean_bones"] == plain["_clean_bones"] == 1
    assert with_report["_stabilize_bones"] == plain["_stabilize_bones"] == 1
    assert with_report["apply_denoise"] == plain["apply_denoise"]
    assert with_report["stabilize_foot_ik"] == plain["stabilize_foot_ik"] == 1


@pytest.mark.parametrize("report_flags", _REPORT_FLAGS, ids=_REPORT_FLAG_IDS)
def test_report_does_not_rerun_stabilization_in_pose_mode(tmp_path, monkeypatch, capsys, report_flags):
    plain, with_report = _plain_and_with_report_counts(
        tmp_path, monkeypatch, capsys, report_flags, "--denoise-mode", "pose")
    assert with_report["_stabilize_bones"] == plain["_stabilize_bones"] == 1
    assert with_report["stabilize_foot_ik"] == plain["stabilize_foot_ik"] == 1


@pytest.mark.parametrize("report_flags", _REPORT_FLAGS, ids=_REPORT_FLAG_IDS)
def test_report_does_not_recompute_noise_detection_in_bone_mode(tmp_path, monkeypatch, capsys, report_flags):
    plain, with_report = _plain_and_with_report_counts(tmp_path, monkeypatch, capsys, report_flags)
    assert with_report["detect_noise_events"] == plain["detect_noise_events"]
