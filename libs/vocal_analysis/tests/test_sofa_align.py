"""SOFAサブプロセス呼び出しコアのテスト(vocal_analysis.md §5.3・§8.3)。

SOFA(Singing-Oriented Forced Aligner)は実インストール・専用venvを要するため、通常のpytestスイート
では subprocess をモックした決定論的単体テストで検証する。ここでは入力の書き出し・サブプロセス起動・
終了コード/出力検証・タイムアウト時のプロセスツリーkill・HTK出力(100ナノ秒単位)の秒への変換を扱う。
Segment契約(隙間なく連続・非重複)の検証・非ASCIIパス拒否・IPA写像は別ファイルで扱う。
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

_XFAIL = pytest.mark.xfail(reason="impl pending: sofa_align subprocess core", strict=True)


def _make_config(tmp_path):
    from vocal_analysis import SofaAlignerConfig

    return SofaAlignerConfig(
        sofa_python=tmp_path / "sofa-venv" / "python",
        sofa_root=tmp_path / "SOFA",
        checkpoint_path=tmp_path / "checkpoint.ckpt",
    )


class _FakeCompletedPopen:
    """正常終了する subprocess.Popen の代替。communicate()呼び出し時に指定コールバックを実行する。"""

    def __init__(self, cmd, *, on_communicate=None, returncode=0, pid=4242, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = pid
        self.returncode = returncode
        self._on_communicate = on_communicate

    def communicate(self, timeout=None):
        if self._on_communicate is not None:
            self._on_communicate()
        return ("", "")


class _FakeTimeoutPopen:
    """communicate() の初回呼び出しで必ず TimeoutExpired を送出する Popen の代替。"""

    def __init__(self, cmd, *, pid=4242, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = pid
        self.returncode = None
        self._raised = False

    def communicate(self, timeout=None):
        if not self._raised:
            self._raised = True
            raise subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout)
        return ("", "")


def _folder_arg(cmd):
    """cmd(リスト形式の引数)から --folder の値を取り出す(モックはこの値へHTK出力を書く)。"""
    return Path(cmd[cmd.index("--folder") + 1])


def _write_htk_label(folder, basename, rows):
    """(start_100ns, end_100ns, label) のタプル列をHTKラベル形式で書き出す。"""
    phones_dir = folder / "htk" / "phones"
    phones_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{start} {end} {label}" for start, end, label in rows]
    (phones_dir / f"{basename}.lab").write_text("\n".join(lines) + "\n", encoding="utf-8")


@_XFAIL
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

    # 100ナノ秒単位 → 秒(vocal_analysis.md §5.1「フレーム時間」・§5.3「時刻の変換」):
    # 5000000 / 1e7 = 0.5秒、10000000 / 1e7 = 1.0秒。
    assert result == {"segment_0000": [(0.0, 0.5, "pau"), (0.5, 1.0, "a")]}


@_XFAIL
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

    # ASCII固定名 segment_%04d(0始まりの4桁ゼロ埋め連番)を書き出す。
    assert seen_basenames == ["segment_0000", "segment_0001", "segment_0002"]
    assert set(result.keys()) == {"segment_0000", "segment_0001", "segment_0002"}


@_XFAIL
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

    # 各対象のG2P音素記号列を、SOFAへ渡す.labへ空白区切りでそのまま書き出す(テキスト→音素変換を経ない)。
    assert written_lab_texts["segment_0000"].strip() == "k a sh i"
    assert written_lab_texts["segment_0001"].strip() == "pau"


@_XFAIL
def test_align_batch_empty_targets_does_not_start_subprocess(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path)

    def fail_popen(cmd, **kwargs):
        raise AssertionError("SOFA対象が0件のときサブプロセスを起動してはならない")

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fail_popen)

    assert sofa_align._align_batch([], config) == {}


@_XFAIL
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


@_XFAIL
def test_align_batch_missing_output_file_raises_recognition_error(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align
    from vocal_analysis.phonemes import RecognitionError

    config = _make_config(tmp_path)

    def fake_popen(cmd, **kwargs):
        # 終了コード0だが、htk/phones/*.lab を一切書き出さない(出力欠落)。
        return _FakeCompletedPopen(cmd)

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)


@_XFAIL
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


@_XFAIL
def test_align_batch_starts_new_process_group_on_windows(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path)
    captured = {}
    fake_creationflags = object()

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs)

        def write_output():
            folder = _folder_arg(cmd)
            _write_htk_label(folder, "segment_0000", [(0, 10000000, "pau")])

        return _FakeCompletedPopen(cmd, on_communicate=write_output)

    monkeypatch.setattr(sofa_align.sys, "platform", "win32")
    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)
    # CREATE_NEW_PROCESS_GROUP はPOSIX標準ライブラリに存在しないため、実属性の有無に依らず動作する
    # よう raising=False で差し替える(この環境非依存性はテスト側の都合であり、実装はWindows分岐の
    # 中でのみこれを参照する)。
    monkeypatch.setattr(sofa_align.subprocess, "CREATE_NEW_PROCESS_GROUP", fake_creationflags, raising=False)

    samples = np.zeros(16000, dtype=np.float32)
    sofa_align._align_batch([(samples, 16000, ["a"])], config)

    # プロセスツリーkill(taskkill /T)が効くには、子孫プロセスの受け皿として新しいプロセスグループで
    # 起動しておく必要がある。
    assert captured.get("creationflags") is fake_creationflags


@_XFAIL
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

    monkeypatch.setattr(sofa_align.sys, "platform", "linux")
    monkeypatch.setattr(sofa_align.subprocess, "Popen", fake_popen)

    samples = np.zeros(16000, dtype=np.float32)
    sofa_align._align_batch([(samples, 16000, ["a"])], config)

    # プロセスツリーkill(os.killpgでのプロセスグループ一括終了)が効くには、子孫プロセスの受け皿として
    # 新しいセッション・プロセスグループで起動しておく必要がある。
    assert captured.get("start_new_session") is True


@_XFAIL
def test_align_batch_timeout_kills_process_tree_on_windows(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align
    from vocal_analysis.phonemes import RecognitionError

    config = _make_config(tmp_path)
    killed_pids = []

    monkeypatch.setattr(sofa_align.sys, "platform", "win32")
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


@_XFAIL
def test_align_batch_timeout_kills_process_tree_on_posix(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align
    from vocal_analysis.phonemes import RecognitionError

    config = _make_config(tmp_path)
    killed = []
    fake_sigkill = object()

    monkeypatch.setattr(sofa_align.sys, "platform", "linux")
    monkeypatch.setattr(sofa_align.subprocess, "Popen", lambda cmd, **kwargs: _FakeTimeoutPopen(cmd))
    # os.getpgid・signal.SIGKILL はWindows開発環境の標準ライブラリに存在しないため、実属性の有無に
    # 依らず動作するよう raising=False で差し替える(この環境非依存性はテスト側の都合であり、実装は
    # POSIX分岐の中でのみこれらを参照する)。
    monkeypatch.setattr(sofa_align.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(sofa_align.signal, "SIGKILL", fake_sigkill, raising=False)

    def fake_killpg(pgid, sig):
        killed.append((pgid, sig))

    monkeypatch.setattr(sofa_align.os, "killpg", fake_killpg, raising=False)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)

    assert killed == [(4242, fake_sigkill)]


@_XFAIL
def test_parse_htk_label_file_converts_100ns_units_to_seconds(tmp_path):
    from vocal_analysis.sofa_align import _parse_htk_label_file

    path = tmp_path / "segment_0000.lab"
    path.write_text("0 5000000 pau\n5000000 12345000 a\n", encoding="utf-8")

    assert _parse_htk_label_file(path) == [(0.0, 0.5, "pau"), (0.5, 1.2345, "a")]
