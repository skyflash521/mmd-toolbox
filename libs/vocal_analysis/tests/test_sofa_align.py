"""SOFAサブプロセス呼び出しコアのテスト(vocal_analysis.md §5.3・§8.3)。

SOFA(Singing-Oriented Forced Aligner)は実インストール・専用venvを要するため、通常のpytestスイート
では subprocess をモックした決定論的単体テストで検証する。ここでは入力の書き出し・サブプロセス起動・
終了コード/出力検証・タイムアウト時のプロセスツリーkill・HTK出力(100ナノ秒単位)の秒への変換・
Segment契約(隙間なく連続・非重複で全時間軸を被覆)の検証と許容誤差スナップ・非ASCIIパスの拒否を
扱う。単語単位分割・IPA写像は別ファイルで扱う。
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


_ASCII_XFAIL = pytest.mark.xfail(reason="impl pending: sofa_align ascii path validation", strict=True)


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


@_ASCII_XFAIL
def test_check_ascii_paths_accepts_all_ascii_paths():
    from vocal_analysis.sofa_align import _check_ascii_paths

    config = _make_ascii_config()
    _check_ascii_paths(Path("ascii_root") / "work_dir", config)  # 例外が出なければ合格


@_ASCII_XFAIL
def test_check_ascii_paths_rejects_non_ascii_work_dir():
    from vocal_analysis.sofa_align import _check_ascii_paths
    from vocal_analysis.phonemes import RecognitionError

    config = _make_ascii_config()
    with pytest.raises(RecognitionError):
        _check_ascii_paths(Path("ascii_root") / "作業ディレクトリ", config)


@_ASCII_XFAIL
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


@_ASCII_XFAIL
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


@_ASCII_XFAIL
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


@_ASCII_XFAIL
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


@_ASCII_XFAIL
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


@_ASCII_XFAIL
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
