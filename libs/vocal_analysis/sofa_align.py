"""SOFA(Singing-Oriented Forced Aligner)サブプロセス呼び出しコア(vocal_analysis.md §5.3・§8.3)。

利用者提供の専用venv・SOFAリポジトリ・チェックポイントをサブプロセスとして呼び、既知の音素記号列を
音声へ時刻合わせする。SOFA本体のコード・チェックポイントは本モジュールに一切同梱しない。ここでは
入力の書き出し・サブプロセス起動・終了コード/出力検証・タイムアウト時のプロセスツリーkill・HTK出力
(100ナノ秒単位)の秒への変換を扱う。Segment契約の検証・非ASCIIパス拒否・単語単位分割・IPA写像は
別の関数(このモジュールの他の関数、または呼び出し元)が担う。
"""

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SofaAlignerConfig
from .phonemes import RecognitionError


def _write_sofa_inputs(
    work_dir: Path, targets: list[tuple[np.ndarray, int, list[str]]]
) -> list[str]:
    """targets(音声サンプル・サンプルレート・G2P音素記号列の組)をASCII固定名で書き出す。

    basenameは0始まりの4桁ゼロ埋め連番(segment_0000, segment_0001, ...)。音素記号列は空白区切りで
    そのまま.labへ書き出し、SOFA自身のテキスト→音素変換を経由しない。書き出した順のbasename列を返す。
    """
    basenames = []
    for i, (samples, sample_rate, phoneme_symbols) in enumerate(targets):
        basename = f"segment_{i:04d}"
        basenames.append(basename)
        sf.write(work_dir / f"{basename}.wav", samples, sample_rate)
        (work_dir / f"{basename}.lab").write_text(" ".join(phoneme_symbols), encoding="utf-8")
    return basenames


def _build_sofa_command(config: SofaAlignerConfig, work_dir: Path) -> list[str]:
    return [
        str(config.sofa_python),
        "infer.py",
        "--ckpt",
        str(config.checkpoint_path),
        "--folder",
        str(work_dir),
        "--mode",
        "force",
        "--g2p",
        "Phoneme",
        "--out_formats",
        "htk",
    ]


def _popen_kwargs() -> dict:
    """子孫プロセスの受け皿(タイムアウト時にプロセスツリーごと終了するための土台)。"""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_process_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.killpg(os.getpgid(pid), signal.SIGKILL)


def _run_sofa_subprocess(basenames: list[str], config: SofaAlignerConfig, work_dir: Path) -> None:
    """infer.pyを1回呼ぶ。タイムアウト・異常終了・出力欠落/空はいずれも RecognitionError にする。"""
    cmd = _build_sofa_command(config, work_dir)
    proc = subprocess.Popen(
        cmd,
        cwd=str(config.sofa_root),
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_popen_kwargs(),
    )
    try:
        _, stderr = proc.communicate(timeout=config.timeout_sec)
    except subprocess.TimeoutExpired as e:
        _kill_process_tree(proc.pid)
        proc.communicate()
        raise RecognitionError(
            f"SOFA呼び出しが{config.timeout_sec}秒以内に完了しませんでした(タイムアウト)"
        ) from e

    if proc.returncode != 0:
        raise RecognitionError(f"SOFA呼び出しが終了コード{proc.returncode}で失敗しました: {stderr}")

    for basename in basenames:
        lab_path = work_dir / "htk" / "phones" / f"{basename}.lab"
        if not lab_path.exists() or lab_path.stat().st_size == 0:
            raise RecognitionError(f"SOFAの出力ファイルが見つからないか空です: {lab_path}")


def _parse_htk_label_file(path: Path) -> list[tuple[float, float, str]]:
    """HTKラベル形式(100ナノ秒単位の整数)を秒単位の(start_sec, end_sec, label)列へ変換する。"""
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        start_100ns, end_100ns, label = line.split(maxsplit=2)
        segments.append((int(start_100ns) / 1e7, int(end_100ns) / 1e7, label))
    return segments


def _align_batch(
    targets: list[tuple[np.ndarray, int, list[str]]], config: SofaAlignerConfig
) -> dict[str, list[tuple[float, float, str]]]:
    """targetsをまとめて1回のSOFA呼び出しで処理し、basenameごとの生セグメント列を返す。

    Segment契約の検証・IPA写像・非ASCIIパス拒否は含まない(呼び出し元が別途行う)。targetsが空なら
    サブプロセスを起動せず空の結果を返す。
    """
    if not targets:
        return {}

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        basenames = _write_sofa_inputs(work_dir, targets)
        _run_sofa_subprocess(basenames, config, work_dir)
        return {
            basename: _parse_htk_label_file(work_dir / "htk" / "phones" / f"{basename}.lab")
            for basename in basenames
        }
