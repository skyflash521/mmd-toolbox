import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _make_config(tmp_path, *, timeout_sec=None):
    from vocal_analysis import SofaAlignerConfig

    sofa_python = tmp_path / "sofa-venv" / "python"
    sofa_root = tmp_path / "SOFA"
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    sofa_python.parent.mkdir(parents=True, exist_ok=True)
    sofa_python.write_text("", encoding="utf-8")
    sofa_root.mkdir(exist_ok=True)
    (sofa_root / "infer.py").write_text("", encoding="utf-8")
    checkpoint_path.write_text("", encoding="utf-8")
    overrides = {} if timeout_sec is None else {"timeout_sec": timeout_sec}
    return SofaAlignerConfig(
        sofa_python=sofa_python,
        sofa_root=sofa_root,
        checkpoint_path=checkpoint_path,
        **overrides,
    )


class _FakeCompletedPopen:
    def __init__(self, cmd, *, on_communicate=None, returncode=0, pid=4242, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = pid
        self.returncode = returncode
        self._on_communicate = on_communicate

    def kill(self):
        pass

    def communicate(self, timeout=None):
        if self._on_communicate is not None:
            self._on_communicate()
        return ("", "")


class _FakeTimeoutPopen:
    def __init__(self, cmd, *, pid=4242, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = pid
        self.returncode = None
        self.killed = False

    def kill(self):
        self.killed = True

    def communicate(self, timeout=None):
        if self.killed:
            return ("", "")
        raise subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout)


class _FakeInterruptedPopen:
    def __init__(self, cmd, *, pid=4242, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = pid
        self.returncode = None
        self.killed = False
        self._raised = False

    def kill(self):
        self.killed = True

    def communicate(self, timeout=None):
        if self.killed:
            return ("", "")
        if not self._raised:
            self._raised = True
            raise KeyboardInterrupt()
        return ("", "")


class _FakeSlowThenCompletePopen:
    def __init__(self, cmd, *, stall_calls, on_communicate=None, pid=4242, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = pid
        self.returncode = 0
        self._remaining_stalls = stall_calls
        self._on_communicate = on_communicate
        self.communicate_call_count = 0

    def kill(self):
        pass

    def communicate(self, timeout=None):
        self.communicate_call_count += 1
        if self._remaining_stalls > 0:
            self._remaining_stalls -= 1
            raise subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout)
        if self._on_communicate is not None:
            self._on_communicate()
        return ("", "")


def _folder_arg(cmd):
    return Path(cmd[cmd.index("--folder") + 1])


def _write_htk_label(folder, basename, rows):
    phones_dir = folder / "htk" / "phones"
    phones_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{start} {end} {label}" for start, end, label in rows]
    (phones_dir / f"{basename}.lab").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_platform(monkeypatch, sofa_align, platform):
    # sys.platform を書き換えると soundfile も参照し、win32 以外の OS では存在しない関数を引いて失敗する。
    monkeypatch.setattr(sofa_align, "sys", SimpleNamespace(platform=platform))


def _fake_windows(monkeypatch, sofa_align):
    _fake_platform(monkeypatch, sofa_align, "win32")
    creationflags = object()
    # subprocess.CREATE_NEW_PROCESS_GROUP は POSIX の標準ライブラリに無い。
    monkeypatch.setattr(
        sofa_align.subprocess, "CREATE_NEW_PROCESS_GROUP", creationflags, raising=False
    )
    return creationflags


def test_align_batch_happy_path_parses_htk_output_as_seconds(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path)

    def fake_popen(cmd, **kwargs):
        folder = _folder_arg(cmd)

        def write_output():
            _write_htk_label(folder, "segment_0000", [(0, 5000000, "pau"), (5000000, 10000000, "a")])

        return _FakeCompletedPopen(cmd, on_communicate=write_output)

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    result = sofa_align._align_batch([(samples, 16000, ["a"])], config)

    assert result == {"segment_0000": [(0.0, 0.5, "pau"), (0.5, 1.0, "a")]}


def test_align_batch_absolutizes_relative_config_paths(tmp_path, monkeypatch):
    from vocal_analysis import SofaAlignerConfig, sofa_align

    _make_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = SofaAlignerConfig(
        sofa_python=Path("sofa-venv") / "python",
        sofa_root=Path("SOFA"),
        checkpoint_path=Path("checkpoint.ckpt"),
    )

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        folder = _folder_arg(cmd)

        def write_output():
            _write_htk_label(folder, "segment_0000", [(0, 10000000, "a")])

        return _FakeCompletedPopen(cmd, on_communicate=write_output)

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    sofa_align._align_batch([(samples, 16000, ["a"])], config)

    python_arg = Path(captured["cmd"][0])
    ckpt_arg = Path(captured["cmd"][captured["cmd"].index("--ckpt") + 1])
    cwd_arg = Path(captured["cwd"])
    assert python_arg.is_absolute() and python_arg == tmp_path / "sofa-venv" / "python"
    assert ckpt_arg.is_absolute() and ckpt_arg == tmp_path / "checkpoint.ckpt"
    assert cwd_arg.is_absolute() and cwd_arg == tmp_path / "SOFA"


def test_align_batch_writes_ascii_fixed_width_basenames_for_multiple_targets(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path)
    seen_basenames = []

    def fake_popen(cmd, **kwargs):
        folder = _folder_arg(cmd)
        wav_files = sorted(p.stem for p in folder.glob("*.wav"))
        seen_basenames.extend(wav_files)

        def write_output():
            for basename in wav_files:
                _write_htk_label(folder, basename, [(0, 10000000, "pau")])

        return _FakeCompletedPopen(cmd, on_communicate=write_output)

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    targets = [(samples, 16000, ["a"]), (samples, 16000, ["i"]), (samples, 16000, ["u"])]
    result = sofa_align._align_batch(targets, config)

    assert seen_basenames == ["segment_0000", "segment_0001", "segment_0002"]
    assert set(result.keys()) == {"segment_0000", "segment_0001", "segment_0002"}


def test_align_batch_writes_space_separated_phonemes_to_lab_input(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path)
    written_lab_texts = {}

    def fake_popen(cmd, **kwargs):
        folder = _folder_arg(cmd)
        for lab_path in folder.glob("*.lab"):
            written_lab_texts[lab_path.stem] = lab_path.read_text(encoding="utf-8")

        def write_output():
            _write_htk_label(folder, "segment_0000", [(0, 10000000, "pau")])
            _write_htk_label(folder, "segment_0001", [(0, 10000000, "pau")])

        return _FakeCompletedPopen(cmd, on_communicate=write_output)

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    targets = [(samples, 16000, ["k", "a", "sh", "i"]), (samples, 16000, ["pau"])]
    sofa_align._align_batch(targets, config)

    assert written_lab_texts["segment_0000"].strip() == "k a sh i"
    assert written_lab_texts["segment_0001"].strip() == "pau"


def test_align_batch_normalizes_devoiced_vowels_for_sofa_vocab(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path)
    written_lab_texts = {}

    def fake_popen(cmd, **kwargs):
        folder = _folder_arg(cmd)
        for lab_path in folder.glob("*.lab"):
            written_lab_texts[lab_path.stem] = lab_path.read_text(encoding="utf-8")

        def write_output():
            _write_htk_label(folder, "segment_0000", [(0, 10000000, "pau")])

        return _FakeCompletedPopen(cmd, on_communicate=write_output)

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    targets = [(samples, 16000, ["k", "I", "sh", "U"])]
    sofa_align._align_batch(targets, config)

    assert written_lab_texts["segment_0000"].strip() == "k i sh u"


def test_align_batch_empty_targets_does_not_start_subprocess(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path)

    def fail_popen(cmd, **kwargs):
        raise AssertionError("SOFA対象が0件のときサブプロセスを起動してはならない")

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fail_popen)

    assert sofa_align._align_batch([], config) == {}


@pytest.mark.parametrize(
    "remove, expected_label",
    [
        ("sofa_python", "sofa_python"),
        ("sofa_root", "sofa_root"),
        ("infer_py", "sofa_root配下のinfer.py"),
        ("checkpoint", "checkpoint_path"),
    ],
)
def test_align_batch_missing_environment_path_raises_recognition_error_with_path(
    tmp_path, monkeypatch, remove, expected_label
):
    import shutil

    from vocal_analysis import sofa_align
    from vocal_analysis.recognizer import RecognitionError

    config = _make_config(tmp_path)
    if remove == "sofa_python":
        config.sofa_python.unlink()
    elif remove == "sofa_root":
        shutil.rmtree(config.sofa_root)
    elif remove == "infer_py":
        (config.sofa_root / "infer.py").unlink()
    else:
        config.checkpoint_path.unlink()

    def fail_popen(cmd, **kwargs):
        raise AssertionError("実在検証で弾かれるべき構成でサブプロセスを起動してはならない")

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fail_popen)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError, match="SOFAの実行環境パスが不正です") as exc_info:
        sofa_align._align_batch([(samples, 16000, ["a"])], config)
    assert expected_label in str(exc_info.value)
    assert "存在しません" in str(exc_info.value)


@pytest.mark.parametrize(
    "target, expected_reason",
    [
        ("sofa_python_is_dir", "ファイルではありません"),
        ("sofa_root_is_file", "ディレクトリではありません"),
    ],
)
def test_align_batch_wrong_kind_environment_path_raises_recognition_error_with_reason(
    tmp_path, monkeypatch, target, expected_reason
):
    import shutil

    from vocal_analysis import sofa_align
    from vocal_analysis.recognizer import RecognitionError

    config = _make_config(tmp_path)
    if target == "sofa_python_is_dir":
        config.sofa_python.unlink()
        config.sofa_python.mkdir()
    else:
        shutil.rmtree(config.sofa_root)
        config.sofa_root.write_text("", encoding="utf-8")

    def fail_popen(cmd, **kwargs):
        raise AssertionError("種別検証で弾かれるべき構成でサブプロセスを起動してはならない")

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fail_popen)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError, match="SOFAの実行環境パスが不正です") as exc_info:
        sofa_align._align_batch([(samples, 16000, ["a"])], config)
    assert expected_reason in str(exc_info.value)


def test_align_batch_popen_file_not_found_raises_recognition_error_naming_paths(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align
    from vocal_analysis.recognizer import RecognitionError

    config = _make_config(tmp_path)

    def raising_popen(cmd, **kwargs):
        raise FileNotFoundError(2, "指定されたファイルが見つかりません")

    monkeypatch.setattr(sofa_align.subprocess, "Popen", raising_popen)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError, match="SOFAサブプロセスを起動できませんでした") as exc_info:
        sofa_align._align_batch([(samples, 16000, ["a"])], config)
    assert str(config.sofa_python) in str(exc_info.value)


def test_align_batch_nonzero_exit_code_raises_recognition_error(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align
    from vocal_analysis.phonemes import RecognitionError

    config = _make_config(tmp_path)

    def fake_popen(cmd, **kwargs):
        return _FakeCompletedPopen(cmd, returncode=1)

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)


def test_align_batch_missing_output_file_after_exit_code_zero_raises_recognition_error(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align
    from vocal_analysis.phonemes import RecognitionError

    config = _make_config(tmp_path)

    def fake_popen(cmd, **kwargs):
        return _FakeCompletedPopen(cmd)

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)


def test_align_batch_empty_output_file_raises_recognition_error(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align
    from vocal_analysis.phonemes import RecognitionError

    config = _make_config(tmp_path)

    def fake_popen(cmd, **kwargs):
        folder = _folder_arg(cmd)

        def write_empty_output():
            phones_dir = folder / "htk" / "phones"
            phones_dir.mkdir(parents=True, exist_ok=True)
            (phones_dir / "segment_0000.lab").write_text("", encoding="utf-8")

        return _FakeCompletedPopen(cmd, on_communicate=write_empty_output)

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)


def test_align_batch_starts_new_process_group_on_windows(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path)
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs)

        def write_output():
            folder = _folder_arg(cmd)
            _write_htk_label(folder, "segment_0000", [(0, 10000000, "pau")])

        return _FakeCompletedPopen(cmd, on_communicate=write_output)

    fake_creationflags = _fake_windows(monkeypatch, sofa_align)
    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    sofa_align._align_batch([(samples, 16000, ["a"])], config)

    assert captured.get("creationflags") is fake_creationflags


def test_align_batch_starts_new_session_on_posix(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path)
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs)

        def write_output():
            folder = _folder_arg(cmd)
            _write_htk_label(folder, "segment_0000", [(0, 10000000, "pau")])

        return _FakeCompletedPopen(cmd, on_communicate=write_output)

    _fake_platform(monkeypatch, sofa_align, "linux")
    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    sofa_align._align_batch([(samples, 16000, ["a"])], config)

    assert captured.get("start_new_session") is True


def test_align_batch_keyboard_interrupt_kills_process_tree_and_reraises(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path, timeout_sec=10.0)
    killed_pids = []

    _fake_windows(monkeypatch, sofa_align)
    monkeypatch.setattr(sofa_align.subprocess, "Popen", lambda cmd, **kwargs: _FakeInterruptedPopen(cmd))

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "taskkill"
        killed_pids.append(cmd[cmd.index("/PID") + 1])

    monkeypatch.setattr(sofa_align.subprocess, "run", fake_run)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(KeyboardInterrupt):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)

    assert killed_pids == ["4242"]


def test_align_batch_timeout_kills_process_tree_on_windows(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align
    from vocal_analysis.phonemes import RecognitionError

    config = _make_config(tmp_path, timeout_sec=0.05)
    killed_pids = []

    _fake_windows(monkeypatch, sofa_align)
    monkeypatch.setattr(sofa_align.subprocess, "Popen", lambda cmd, **kwargs: _FakeTimeoutPopen(cmd))

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "taskkill"
        assert "/F" in cmd and "/T" in cmd and "/PID" in cmd
        killed_pids.append(cmd[cmd.index("/PID") + 1])

    monkeypatch.setattr(sofa_align.subprocess, "run", fake_run)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)

    assert killed_pids == ["4242"]


def test_align_batch_survives_transient_timeout_within_deadline(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path, timeout_sec=10.0)
    fake = {}
    stall_calls = 2

    def fake_popen(cmd, **kwargs):
        folder = _folder_arg(cmd)

        def write_output():
            _write_htk_label(folder, "segment_0000", [(0, 10000000, "pau")])

        p = _FakeSlowThenCompletePopen(cmd, stall_calls=stall_calls, on_communicate=write_output)
        fake["proc"] = p
        return p

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    sofa_align._align_batch([(samples, 16000, ["a"])], config)

    assert fake["proc"].communicate_call_count == stall_calls + 1


def test_align_batch_timeout_kills_process_tree_on_posix(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align
    from vocal_analysis.phonemes import RecognitionError

    config = _make_config(tmp_path, timeout_sec=0.05)
    killed = []
    fake_sigkill = object()

    _fake_platform(monkeypatch, sofa_align, "linux")
    monkeypatch.setattr(sofa_align.subprocess, "Popen", lambda cmd, **kwargs: _FakeTimeoutPopen(cmd))
    # os.getpgid・os.killpg・signal.SIGKILL は Windows の標準ライブラリに無い。
    monkeypatch.setattr(sofa_align.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(sofa_align.signal, "SIGKILL", fake_sigkill, raising=False)

    def fake_killpg(pgid, sig):
        killed.append((pgid, sig))

    monkeypatch.setattr(sofa_align.os, "killpg", fake_killpg, raising=False)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)

    assert killed == [(4242, fake_sigkill)]


def test_parse_htk_label_file_converts_100ns_units_to_seconds(tmp_path):
    from vocal_analysis.sofa_align import _parse_htk_label_file

    path = tmp_path / "segment_0000.lab"
    path.write_text("0 5000000 pau\n5000000 12345000 a\n", encoding="utf-8")

    assert _parse_htk_label_file(path) == [(0.0, 0.5, "pau"), (0.5, 1.2345, "a")]


def test_segment_contract_tolerance_is_50ms():
    from vocal_analysis.sofa_align import _SEGMENT_CONTRACT_TOLERANCE_SEC

    assert _SEGMENT_CONTRACT_TOLERANCE_SEC == 0.05


def test_validate_and_normalize_segments_passes_through_exact_input():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    segments = [(0.0, 0.5, "pau"), (0.5, 1.0, "a")]
    assert _validate_and_normalize_segments(segments, trim_duration_sec=1.0) == segments


def test_validate_and_normalize_segments_snaps_within_tolerance_using_preceding_end_as_boundary():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    segments = [(0.01, 0.49, "pau"), (0.52, 0.98, "a")]
    result = _validate_and_normalize_segments(segments, trim_duration_sec=1.0)

    assert result == [(0.0, 0.49, "pau"), (0.49, 1.0, "a")]


def test_validate_and_normalize_segments_rejects_start_after_end_even_when_boundaries_are_within_tolerance():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    segments = [(0.0, 0.5, "pau"), (0.53, 0.51, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=0.51)


def test_validate_and_normalize_segments_rejects_gap_between_segments_beyond_tolerance():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    segments = [(0.0, 0.5, "pau"), (0.56, 1.0, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_overlap_between_segments_beyond_tolerance():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    segments = [(0.0, 0.5, "pau"), (0.44, 1.0, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_first_start_far_from_zero():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    segments = [(0.06, 0.5, "pau"), (0.5, 1.0, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_last_end_short_of_trim_duration():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    segments = [(0.0, 0.5, "pau"), (0.5, 0.93, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_last_end_exceeding_trim_duration():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    segments = [(0.0, 0.5, "pau"), (0.5, 1.07, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_empty_list():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments([], trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_new_violation_created_by_snapping():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    segments = [(0.0, 1.04, "pau"), (1.00, 1.02, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.02)


def _make_ascii_config():
    from vocal_analysis import SofaAlignerConfig

    return SofaAlignerConfig(
        sofa_python=Path("ascii_root") / "sofa-venv" / "python",
        sofa_root=Path("ascii_root") / "SOFA",
        checkpoint_path=Path("ascii_root") / "checkpoint.ckpt",
    )


def test_check_ascii_paths_accepts_all_ascii_paths():
    from vocal_analysis.sofa_align import _check_ascii_paths

    config = _make_ascii_config()
    _check_ascii_paths(Path("ascii_root") / "work_dir", config)


def test_check_ascii_paths_rejects_non_ascii_work_dir():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _check_ascii_paths

    config = _make_ascii_config()
    with pytest.raises(RecognitionError):
        _check_ascii_paths(Path("ascii_root") / "作業ディレクトリ", config)


def test_check_ascii_paths_rejects_non_ascii_sofa_root():
    from vocal_analysis import SofaAlignerConfig
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _check_ascii_paths

    config = SofaAlignerConfig(
        sofa_python=Path("ascii_root") / "sofa-venv" / "python",
        sofa_root=Path("ascii_root") / "SOFAリポジトリ",
        checkpoint_path=Path("ascii_root") / "checkpoint.ckpt",
    )
    with pytest.raises(RecognitionError):
        _check_ascii_paths(Path("ascii_root") / "work_dir", config)


def test_check_ascii_paths_rejects_non_ascii_checkpoint_path():
    from vocal_analysis import SofaAlignerConfig
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _check_ascii_paths

    config = SofaAlignerConfig(
        sofa_python=Path("ascii_root") / "sofa-venv" / "python",
        sofa_root=Path("ascii_root") / "SOFA",
        checkpoint_path=Path("ascii_root") / "チェックポイント.ckpt",
    )
    with pytest.raises(RecognitionError):
        _check_ascii_paths(Path("ascii_root") / "work_dir", config)


def test_check_ascii_paths_rejects_non_ascii_in_intermediate_component():
    from vocal_analysis import SofaAlignerConfig
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _check_ascii_paths

    config = SofaAlignerConfig(
        sofa_python=Path("ascii_root") / "sofa-venv" / "python",
        sofa_root=Path("ascii_root") / "SOFA",
        checkpoint_path=Path("ascii_root") / "日本語" / "checkpoint.ckpt",
    )
    with pytest.raises(RecognitionError):
        _check_ascii_paths(Path("ascii_root") / "work_dir", config)


def test_check_ascii_paths_does_not_check_sofa_python():
    from vocal_analysis import SofaAlignerConfig
    from vocal_analysis.sofa_align import _check_ascii_paths

    config = SofaAlignerConfig(
        sofa_python=Path("ascii_root") / "専用venv" / "python",
        sofa_root=Path("ascii_root") / "SOFA",
        checkpoint_path=Path("ascii_root") / "checkpoint.ckpt",
    )
    _check_ascii_paths(Path("ascii_root") / "work_dir", config)


class _FakeTemporaryDirectory:
    def __init__(self, path, **kwargs):
        self._path = path

    def __enter__(self):
        return self._path

    def __exit__(self, *exc_info):
        return False


def test_align_batch_non_ascii_checkpoint_path_does_not_start_subprocess(monkeypatch):
    from vocal_analysis import SofaAlignerConfig, sofa_align
    from vocal_analysis.phonemes import RecognitionError

    ascii_config = _make_ascii_config()
    config = SofaAlignerConfig(
        sofa_python=ascii_config.sofa_python,
        sofa_root=ascii_config.sofa_root,
        checkpoint_path=Path("ascii_root") / "チェックポイント.ckpt",
    )

    def fail_popen(cmd, **kwargs):
        raise AssertionError("非ASCIIパスが含まれる場合はSOFAを起動してはならない")

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(
        sofa_align.tempfile, "TemporaryDirectory",
        lambda **kwargs: _FakeTemporaryDirectory("ascii_root/work_dir"),
    )

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)


def test_align_batch_non_ascii_work_dir_does_not_start_subprocess(monkeypatch):
    from vocal_analysis import sofa_align
    from vocal_analysis.phonemes import RecognitionError

    config = _make_ascii_config()

    def fail_popen(cmd, **kwargs):
        raise AssertionError("非ASCIIパスが含まれる場合はSOFAを起動してはならない")

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(
        sofa_align.tempfile, "TemporaryDirectory",
        lambda **kwargs: _FakeTemporaryDirectory("ascii_root/一時ディレクトリ"),
    )

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)


def test_clamp_words_to_valid_list_passes_through_non_overlapping_words():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    words = [(["k", "a"], 0.5, 1.0), (["i"], 1.0, 1.5)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == words


def test_clamp_words_to_valid_list_clamps_end_to_trim_duration():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    words = [(["a"], 0.5, 1.5)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=1.0) == [(["a"], 0.5, 1.0)]


def test_clamp_words_to_valid_list_clamps_overlap_to_cursor():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    words = [(["a"], 0.5, 1.0), (["i"], 0.8, 1.2)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == [
        (["a"], 0.5, 1.0),
        (["i"], 1.0, 1.2),
    ]


def test_clamp_words_to_valid_list_invalidates_words_fully_covered_by_cursor():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    words = [(["a"], 0.5, 2.0), (["i"], 0.8, 0.9), (["u"], 1.0, 1.2)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == [(["a"], 0.5, 2.0)]


def test_clamp_words_to_valid_list_invalidates_word_already_shorter_than_minimum():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    words = [(["a"], 0.5, 1.0), (["i"], 0.98, 1.02), (["u"], 1.05, 1.3)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == [
        (["a"], 0.5, 1.0),
        (["u"], 1.05, 1.3),
    ]


def test_clamp_words_to_valid_list_invalidates_word_shrunk_below_minimum_by_clamp():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    words = [(["a"], 0.5, 1.0), (["i"], 0.97, 1.03)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == [(["a"], 0.5, 1.0)]


def test_clamp_words_to_valid_list_cursor_persists_through_consecutive_invalid_words():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    words = [
        (["a"], 0.5, 1.0),
        (["i"], 0.9, 1.02),
        (["u"], 0.95, 1.01),
        (["e"], 0.9, 1.4),
    ]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == [
        (["a"], 0.5, 1.0),
        (["e"], 1.0, 1.4),
    ]


def test_clamp_words_to_valid_list_invalidates_empty_phoneme_symbols_without_moving_cursor():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    words = [(["a"], 0.5, 1.0), ([], 1.0, 1.5), (["i"], 1.2, 2.0)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == [
        (["a"], 0.5, 1.0),
        (["i"], 1.2, 2.0),
    ]


def test_clamp_words_to_valid_list_all_invalid_returns_empty():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    words = [([], 0.0, 0.01)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=1.0) == []


def test_clamp_words_to_valid_list_empty_input_returns_empty():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    assert _clamp_words_to_valid_list([], trim_duration_sec=1.0) == []


def test_determine_word_gaps_no_valid_words_covers_whole_trim_duration():
    from vocal_analysis.sofa_align import _determine_word_gaps

    assert _determine_word_gaps([], trim_duration_sec=1.5) == [(0.0, 1.5)]


def test_covered_invalid_words_produce_no_gap_and_intervals_tile_whole_duration():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list, _determine_word_gaps

    words = [(["a"], 0.5, 2.0), (["i"], 0.8, 0.9), (["u"], 1.0, 1.2)]
    trim_duration_sec = 2.5

    valid_words = _clamp_words_to_valid_list(words, trim_duration_sec=trim_duration_sec)
    gaps = _determine_word_gaps(valid_words, trim_duration_sec=trim_duration_sec)

    assert valid_words == [(["a"], 0.5, 2.0)]
    assert gaps == [(0.0, 0.5), (2.0, 2.5)]

    intervals = sorted(
        [(start, end) for _, start, end in valid_words] + gaps, key=lambda iv: iv[0])
    assert intervals[0][0] == 0.0
    assert intervals[-1][1] == trim_duration_sec
    for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:], strict=False):
        assert prev_end == next_start


def test_all_words_invalidated_confirms_whole_duration_as_single_gap():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list, _determine_word_gaps

    words = [(["a"], 0.00, 0.04), (["i"], 0.02, 0.05), (["u"], 0.04, 0.07)]
    trim_duration_sec = 1.0

    valid_words = _clamp_words_to_valid_list(words, trim_duration_sec=trim_duration_sec)
    gaps = _determine_word_gaps(valid_words, trim_duration_sec=trim_duration_sec)

    assert valid_words == []
    assert gaps == [(0.0, 1.0)]


def test_determine_word_gaps_no_gaps_when_words_cover_whole_duration():
    from vocal_analysis.sofa_align import _determine_word_gaps

    words = [(["a"], 0.0, 0.5), (["i"], 0.5, 1.0)]
    assert _determine_word_gaps(words, trim_duration_sec=1.0) == []


def test_determine_word_gaps_before_between_and_after():
    from vocal_analysis.sofa_align import _determine_word_gaps

    words = [(["a"], 0.2, 0.5), (["i"], 0.8, 1.0)]
    assert _determine_word_gaps(words, trim_duration_sec=1.5) == [(0.0, 0.2), (0.5, 0.8), (1.0, 1.5)]


def test_determine_word_gaps_does_not_emit_zero_length_gap():
    from vocal_analysis.sofa_align import _determine_word_gaps

    words = [(["a"], 0.0, 1.0)]
    assert _determine_word_gaps(words, trim_duration_sec=1.0) == []


def test_map_symbol_to_segment_fields_vowel():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("a") == ("vowel", "a")


def test_map_symbol_to_segment_fields_consonant():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("k") == ("consonant", "k")


def test_map_symbol_to_segment_fields_classifies_mapped_symbol_y_to_j_as_consonant():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("y") == ("consonant", "j")


def test_map_symbol_to_segment_fields_classifies_mapped_symbol_devoiced_i_as_vowel():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("I") == ("vowel", "i")


def test_map_symbol_to_segment_fields_vowel_with_ipa_conversion():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("u") == ("vowel", "ɯ")


def test_map_symbol_to_segment_fields_pau_is_gap():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("pau") == ("gap", None)


def test_map_symbol_to_segment_fields_cl_is_gap():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("cl") == ("gap", None)


def test_map_symbol_to_segment_fields_breath_ap_is_gap():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("AP") == ("gap", None)


def test_map_symbol_to_segment_fields_sp_is_gap():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("SP") == ("gap", None)


def test_map_symbol_to_segment_fields_unmapped_symbol_raises():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    with pytest.raises(RecognitionError):
        _map_symbol_to_segment_fields("xyz_unknown")


def test_segments_from_raw_converts_all_fields():
    from vocal_analysis.sofa_align import _segments_from_raw
    from vocal_analysis.types import Segment

    raw = [(0.0, 0.5, "pau"), (0.5, 0.7, "k"), (0.7, 1.0, "a")]
    assert _segments_from_raw(raw) == [
        Segment(type="gap", start_sec=0.0, end_sec=0.5, phoneme=None, confidence=None),
        Segment(type="consonant", start_sec=0.5, end_sec=0.7, phoneme="k", confidence=None),
        Segment(type="vowel", start_sec=0.7, end_sec=1.0, phoneme="a", confidence=None),
    ]


def test_segments_from_raw_empty_input_returns_empty():
    from vocal_analysis.sofa_align import _segments_from_raw

    assert _segments_from_raw([]) == []


def test_segments_from_raw_propagates_unmapped_symbol_error():
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.sofa_align import _segments_from_raw

    with pytest.raises(RecognitionError):
        _segments_from_raw([(0.0, 0.5, "xyz_unknown")])
