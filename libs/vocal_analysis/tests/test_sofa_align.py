"""SOFAサブプロセス呼び出しコアのテスト(vocal_analysis.md §5.3・§8.3)。

SOFA(Singing-Oriented Forced Aligner)は実インストール・専用venvを要するため、通常のpytestスイート
では subprocess をモックした決定論的単体テストで検証する。ここでは入力の書き出し・サブプロセス起動・
終了コード/出力検証・タイムアウト時のプロセスツリーkill・HTK出力(100ナノ秒単位)の秒への変換・
Segment契約(隙間なく連続・非重複で全時間軸を被覆)の検証と許容誤差スナップ・非ASCIIパスの拒否・
単語単位分割(有効な単語列の確定・gapの確定)・IPA写像(SOFA出力記号のSegment化)を扱う。
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest


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


def test_align_batch_empty_targets_does_not_start_subprocess(tmp_path, monkeypatch):
    from vocal_analysis import sofa_align

    config = _make_config(tmp_path)

    def fail_popen(cmd, **kwargs):
        raise AssertionError("SOFA対象が0件のときサブプロセスを起動してはならない")

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fail_popen)

    assert sofa_align._align_batch([], config) == {}


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


def test_parse_htk_label_file_converts_100ns_units_to_seconds(tmp_path):
    from vocal_analysis.sofa_align import _parse_htk_label_file

    path = tmp_path / "segment_0000.lab"
    path.write_text("0 5000000 pau\n5000000 12345000 a\n", encoding="utf-8")

    assert _parse_htk_label_file(path) == [(0.0, 0.5, "pau"), (0.5, 1.2345, "a")]


def test_validate_and_normalize_segments_passes_through_exact_input():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    segments = [(0.0, 0.5, "pau"), (0.5, 1.0, "a")]
    assert _validate_and_normalize_segments(segments, trim_duration_sec=1.0) == segments


def test_validate_and_normalize_segments_snaps_within_tolerance_to_exact_values():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments

    # 先頭・末尾・境界ともに1ミリ秒以内のずれ(SOFAの100ナノ秒単位からの変換誤差を模す)。
    # 正規化は先頭のstart→0.0、末尾のend→trim_duration_sec、後続の各startを先行のend(元の値)へ
    # それぞれ上書きする。境界の基準は先行セグメントのend側(0.4995)であり、後続のstart側(0.5003)
    # ではない。
    segments = [(0.0002, 0.4995, "pau"), (0.5003, 0.9997, "a")]
    result = _validate_and_normalize_segments(segments, trim_duration_sec=1.0)

    assert result == [(0.0, 0.4995, "pau"), (0.4995, 1.0, "a")]


def test_validate_and_normalize_segments_rejects_start_after_end():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments
    from vocal_analysis.phonemes import RecognitionError

    # (b)(c)(d)はいずれも許容誤差1ミリ秒以内で通過し、2件目のみ(a) start<=end に単独で違反する
    # (0.5008 > 0.5001)。(b)(c)(d)しか検証しない誤実装でもこの入力を通してしまわないことを確認する。
    segments = [(0.0, 0.5, "pau"), (0.5008, 0.5001, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=0.5001)


def test_validate_and_normalize_segments_rejects_non_adjacent_boundary():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments
    from vocal_analysis.phonemes import RecognitionError

    # 境界の差が許容誤差1ミリ秒を超える(隙間: 次の開始が先行の終了より後ろに離れている)。
    segments = [(0.0, 0.5, "pau"), (0.503, 1.0, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_overlapping_boundary():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments
    from vocal_analysis.phonemes import RecognitionError

    # 境界の差が許容誤差1ミリ秒を超える(重複: 次の開始が先行の終了より前にある)。絶対差での判定
    # なので、隙間方向だけでなく重複方向も同じしきい値で拒否されることを確認する。
    segments = [(0.0, 0.5, "pau"), (0.495, 1.0, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_first_start_far_from_zero():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments
    from vocal_analysis.phonemes import RecognitionError

    segments = [(0.05, 0.5, "pau"), (0.5, 1.0, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_last_end_far_from_trim_duration():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments
    from vocal_analysis.phonemes import RecognitionError

    # 全長に不足する方向(0.9 < 1.0)。
    segments = [(0.0, 0.5, "pau"), (0.5, 0.9, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_last_end_exceeding_trim_duration():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments
    from vocal_analysis.phonemes import RecognitionError

    # 全長を超過する方向(1.1 > 1.0)。絶対差での判定なので、不足方向だけでなく超過方向も同じ
    # しきい値で拒否されることを確認する。
    segments = [(0.0, 0.5, "pau"), (0.5, 1.1, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_empty_list():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments
    from vocal_analysis.phonemes import RecognitionError

    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments([], trim_duration_sec=1.0)


def test_validate_and_normalize_segments_rejects_new_violation_created_by_snapping():
    from vocal_analysis.sofa_align import _validate_and_normalize_segments
    from vocal_analysis.phonemes import RecognitionError

    # 一次検証(許容誤差1ミリ秒)は通るが、正規化(後続の開始時刻を先行の終了時刻へ上書き)により
    # 新たな逆順(start > end)を生む例。極端に短い2件目のセグメント(終了時刻1.0001)へ、1件目の
    # 終了時刻1.0009が上書きされ、上書き後は1.0009 > 1.0001になる
    # (vocal_analysis.md §5.3「Segment契約の検証」が現象として述べる「極端に短い隣接セグメントが
    # 正規化の上書きにより新たな逆順を生む場合がある」の具体例。数値そのものは仕様書には無くこの
    # テストが独自に構成した)。
    segments = [(0.0, 1.0009, "pau"), (1.0000, 1.0001, "a")]
    with pytest.raises(RecognitionError):
        _validate_and_normalize_segments(segments, trim_duration_sec=1.0001)


def _make_ascii_config():
    """`_check_ascii_paths`はパス文字列を判定するだけでファイルへアクセスしないため、実在しない
    固定のASCII専用パスで足りる(pytestの`tmp_path`自体が非ASCIIユーザー名配下になりうる環境依存を
    避ける)。
    """
    from vocal_analysis import SofaAlignerConfig

    return SofaAlignerConfig(
        sofa_python=Path("ascii_root") / "sofa-venv" / "python",
        sofa_root=Path("ascii_root") / "SOFA",
        checkpoint_path=Path("ascii_root") / "checkpoint.ckpt",
    )


def test_check_ascii_paths_accepts_all_ascii_paths():
    from vocal_analysis.sofa_align import _check_ascii_paths

    config = _make_ascii_config()
    _check_ascii_paths(Path("ascii_root") / "work_dir", config)  # 例外が出なければ合格


def test_check_ascii_paths_rejects_non_ascii_work_dir():
    from vocal_analysis.sofa_align import _check_ascii_paths
    from vocal_analysis.phonemes import RecognitionError

    config = _make_ascii_config()
    with pytest.raises(RecognitionError):
        _check_ascii_paths(Path("ascii_root") / "作業ディレクトリ", config)


def test_check_ascii_paths_rejects_non_ascii_sofa_root():
    from vocal_analysis import SofaAlignerConfig
    from vocal_analysis.sofa_align import _check_ascii_paths
    from vocal_analysis.phonemes import RecognitionError

    config = SofaAlignerConfig(
        sofa_python=Path("ascii_root") / "sofa-venv" / "python",
        sofa_root=Path("ascii_root") / "SOFAリポジトリ",
        checkpoint_path=Path("ascii_root") / "checkpoint.ckpt",
    )
    with pytest.raises(RecognitionError):
        _check_ascii_paths(Path("ascii_root") / "work_dir", config)


def test_check_ascii_paths_rejects_non_ascii_checkpoint_path():
    from vocal_analysis import SofaAlignerConfig
    from vocal_analysis.sofa_align import _check_ascii_paths
    from vocal_analysis.phonemes import RecognitionError

    config = SofaAlignerConfig(
        sofa_python=Path("ascii_root") / "sofa-venv" / "python",
        sofa_root=Path("ascii_root") / "SOFA",
        checkpoint_path=Path("ascii_root") / "チェックポイント.ckpt",
    )
    with pytest.raises(RecognitionError):
        _check_ascii_paths(Path("ascii_root") / "work_dir", config)


def test_check_ascii_paths_rejects_non_ascii_in_intermediate_component():
    """末端要素だけでなく、パス中間の要素の非ASCIIも拒否対象(パス文字列全体を判定する)。"""
    from vocal_analysis import SofaAlignerConfig
    from vocal_analysis.sofa_align import _check_ascii_paths
    from vocal_analysis.phonemes import RecognitionError

    config = SofaAlignerConfig(
        sofa_python=Path("ascii_root") / "sofa-venv" / "python",
        sofa_root=Path("ascii_root") / "SOFA",
        checkpoint_path=Path("ascii_root") / "日本語" / "checkpoint.ckpt",
    )
    with pytest.raises(RecognitionError):
        _check_ascii_paths(Path("ascii_root") / "work_dir", config)


def test_check_ascii_paths_does_not_check_sofa_python():
    """sofa_pythonは非ASCII検証の対象外(確定。一時ディレクトリ・sofa_root・checkpoint_pathの3つのみ)。"""
    from vocal_analysis import SofaAlignerConfig
    from vocal_analysis.sofa_align import _check_ascii_paths

    config = SofaAlignerConfig(
        sofa_python=Path("ascii_root") / "専用venv" / "python",
        sofa_root=Path("ascii_root") / "SOFA",
        checkpoint_path=Path("ascii_root") / "checkpoint.ckpt",
    )
    _check_ascii_paths(Path("ascii_root") / "work_dir", config)  # 例外が出なければ合格


class _FakeTemporaryDirectory:
    """tempfile.TemporaryDirectory()の代替。固定パス(実在しなくてよい。ASCII検証は書き出し前に行う
    設計なので、検証で弾かれるテストでは実際のファイルI/Oへ到達しない)を返す。
    """

    def __init__(self, path):
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
        sofa_align.tempfile, "TemporaryDirectory", lambda: _FakeTemporaryDirectory("ascii_root/work_dir")
    )

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RecognitionError):
        sofa_align._align_batch([(samples, 16000, ["a"])], config)


def test_align_batch_non_ascii_work_dir_does_not_start_subprocess(monkeypatch):
    """SOFA自身が扱えないのは実行時に生成する一時ディレクトリのパスも同様。tempfile.mkdtempが
    非ASCIIパスを返す場合(利用者環境の一時領域自体に非ASCII文字が含まれる場合)を模す。configは
    全フィールドASCIIにし、work_dir単体の非ASCIIが検出されることを固定する。
    """
    from vocal_analysis import sofa_align
    from vocal_analysis.phonemes import RecognitionError

    config = _make_ascii_config()

    def fail_popen(cmd, **kwargs):
        raise AssertionError("非ASCIIパスが含まれる場合はSOFAを起動してはならない")

    monkeypatch.setattr(sofa_align.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(
        sofa_align.tempfile, "TemporaryDirectory", lambda: _FakeTemporaryDirectory("ascii_root/一時ディレクトリ")
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

    # 単語の終了時刻がトリム後区間の全長(1.0)を超えている(手順0の再クランプ)。
    words = [(["a"], 0.5, 1.5)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=1.0) == [(["a"], 0.5, 1.0)]


def test_clamp_words_to_valid_list_clamps_overlap_to_cursor():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    # 単語Bの開始(0.8)が単語Aの終了(1.0)より前で重複する。Bの開始はcursor(=Aの終了1.0)へ
    # クランプされ、[1.0, 1.2)として有効になる(区間長0.2は最小長0.05以上)。
    words = [(["a"], 0.5, 1.0), (["i"], 0.8, 1.2)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == [
        (["a"], 0.5, 1.0),
        (["i"], 1.0, 1.2),
    ]


def test_clamp_words_to_valid_list_invalidates_words_fully_covered_by_cursor():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    # 単語B・Cが単語Aに完全に包含される。Aが有効化されcursor=2.0(Aの終了)になった後、
    # B(クランプ後start=2.0 > end=0.9)・C(クランプ後start=2.0 > end=1.2)とも負長になり
    # 無効化される。無効化された単語を挟んでも、cursorはAの終了(2.0)のまま変化しない。
    words = [(["a"], 0.5, 2.0), (["i"], 0.8, 0.9), (["u"], 1.0, 1.2)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == [(["a"], 0.5, 2.0)]


def test_clamp_words_to_valid_list_invalidates_word_already_shorter_than_minimum():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    # 単語Bは元の区間長(0.04秒)自体が既に最小長(0.05)未満で無効化される(クランプの影響を
    # 受けない単純ケース)。cursorは更新されない(単語Aの終了1.0のまま)。
    words = [(["a"], 0.5, 1.0), (["i"], 0.98, 1.02), (["u"], 1.05, 1.3)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == [
        (["a"], 0.5, 1.0),
        (["u"], 1.05, 1.3),
    ]


def test_clamp_words_to_valid_list_invalidates_word_shrunk_below_minimum_by_clamp():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    # 単語Bは元の区間長(0.06秒。最小長0.05以上)自体は無効化条件を満たさないが、クランプで
    # start が 0.97→1.0(=cursor)へ引き上げられた結果、区間長が0.03秒に縮み無効化される。
    # 「元の区間長」だけを見る誤実装ではこの単語は有効判定されてしまうため、クランプ後の区間長で
    # 判定することを単独で固定する。
    words = [(["a"], 0.5, 1.0), (["i"], 0.97, 1.03)]
    assert _clamp_words_to_valid_list(words, trim_duration_sec=2.0) == [(["a"], 0.5, 1.0)]


def test_clamp_words_to_valid_list_cursor_persists_through_consecutive_invalid_words():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    # A有効(cursor=1.0)→B・C連続無効→D有効、という並び。cursorが「無効化された単語のクランプ後
    # 終了時刻」で誤って更新される実装(例: max(cursor, 無効単語のクランプ後end))だと、Bの無効化時に
    # cursorが1.02へ、Cの無効化時に1.02のまま(1.01<1.02)進み、Dの開始は max(0.9, 1.02)=1.02 と
    # 誤ってクランプされてしまう。正しい実装ではcursorはAの終了(1.0)のまま変化せず、Dの開始は
    # max(0.9, 1.0)=1.0 になる。
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


def test_clamp_words_to_valid_list_invalidates_empty_phoneme_symbols():
    from vocal_analysis.sofa_align import _clamp_words_to_valid_list

    # 空の音素記号列を持つ単語はSOFA対象として意味を成さないため、区間長に関わらず無効とする
    # (vocal_analysis.md §5.3「有効な単語列の確定」に確定)。cursorも他の無効化と同様に更新しない
    # (単語Bの区間長自体は最小長以上だが、空の音素記号列だけを理由に無効化されることを確認する)。
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

    # 単語がトリム後区間の先頭からちょうど始まる場合、先頭側に長さ0のgapを生成しない。
    words = [(["a"], 0.0, 1.0)]
    assert _determine_word_gaps(words, trim_duration_sec=1.0) == []


_IPA_MAPPING_XFAIL = pytest.mark.xfail(reason="impl pending: sofa_align ipa mapping", strict=True)


@_IPA_MAPPING_XFAIL
def test_map_symbol_to_segment_fields_vowel():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    # G2P記号 "a" は音素モデル語彙でも "a"(母音)。
    assert _map_symbol_to_segment_fields("a") == ("vowel", "a")


@_IPA_MAPPING_XFAIL
def test_map_symbol_to_segment_fields_consonant():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    # G2P記号 "k" は音素モデル語彙でも "k"(子音)。
    assert _map_symbol_to_segment_fields("k") == ("consonant", "k")


@_IPA_MAPPING_XFAIL
def test_map_symbol_to_segment_fields_classification_uses_mapped_symbol_not_raw_g2p_symbol_consonant():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    # G2P記号 "y" 自体はIPA母音記号の基準集合に含まれ母音判定になってしまうが、写像先の音素モデル
    # 語彙記号 "j" は子音判定になる。分類が「写像後のIPA記号」に対して行われることを、写像前後で
    # 判定が割れるこの記号で固定する(写像前のG2P記号を誤って分類する実装を検出する)。
    assert _map_symbol_to_segment_fields("y") == ("consonant", "j")


@_IPA_MAPPING_XFAIL
def test_map_symbol_to_segment_fields_classification_uses_mapped_symbol_not_raw_g2p_symbol_vowel():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    # G2P記号 "I"(無声化母音、大文字)自体は母音記号基準集合に無く子音判定になってしまうが、
    # 写像先の音素モデル語彙記号 "i"(小文字)は母音判定になる。上のテストと逆方向(母音→子音では
    # なく子音→母音)で写像順序を固定する。
    assert _map_symbol_to_segment_fields("I") == ("vowel", "i")


@_IPA_MAPPING_XFAIL
def test_map_symbol_to_segment_fields_vowel_with_ipa_conversion():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    # G2P記号 "u" は音素モデル語彙で "ɯ"(母音)。写像を経ることを確認する。
    assert _map_symbol_to_segment_fields("u") == ("vowel", "ɯ")


@_IPA_MAPPING_XFAIL
def test_map_symbol_to_segment_fields_pau_is_gap():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("pau") == ("gap", None)


@_IPA_MAPPING_XFAIL
def test_map_symbol_to_segment_fields_cl_is_gap():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("cl") == ("gap", None)


@_IPA_MAPPING_XFAIL
def test_map_symbol_to_segment_fields_ap_is_gap():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    # AP(吸気音・呼吸音)は無音ではないが、3分類に区分が無いためgapへ倒す。
    assert _map_symbol_to_segment_fields("AP") == ("gap", None)


@_IPA_MAPPING_XFAIL
def test_map_symbol_to_segment_fields_sp_is_gap():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields

    assert _map_symbol_to_segment_fields("SP") == ("gap", None)


@_IPA_MAPPING_XFAIL
def test_map_symbol_to_segment_fields_unmapped_symbol_raises():
    from vocal_analysis.sofa_align import _map_symbol_to_segment_fields
    from vocal_analysis.phonemes import RecognitionError

    with pytest.raises(RecognitionError):
        _map_symbol_to_segment_fields("xyz_unknown")


@_IPA_MAPPING_XFAIL
def test_segments_from_raw_converts_all_fields():
    from vocal_analysis.sofa_align import _segments_from_raw
    from vocal_analysis.types import Segment

    raw = [(0.0, 0.5, "pau"), (0.5, 0.7, "k"), (0.7, 1.0, "a")]
    assert _segments_from_raw(raw) == [
        Segment(type="gap", start_sec=0.0, end_sec=0.5, phoneme=None, confidence=None),
        Segment(type="consonant", start_sec=0.5, end_sec=0.7, phoneme="k", confidence=None),
        Segment(type="vowel", start_sec=0.7, end_sec=1.0, phoneme="a", confidence=None),
    ]


@_IPA_MAPPING_XFAIL
def test_segments_from_raw_empty_input_returns_empty():
    from vocal_analysis.sofa_align import _segments_from_raw

    assert _segments_from_raw([]) == []


@_IPA_MAPPING_XFAIL
def test_segments_from_raw_propagates_unmapped_symbol_error():
    from vocal_analysis.sofa_align import _segments_from_raw
    from vocal_analysis.phonemes import RecognitionError

    with pytest.raises(RecognitionError):
        _segments_from_raw([(0.0, 0.5, "xyz_unknown")])
