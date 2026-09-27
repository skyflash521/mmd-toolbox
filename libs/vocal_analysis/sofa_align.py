import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from .config import SofaAlignerConfig
from .phonemes import (
    _BLANK_G2P_SYMBOLS,
    _G2P_TO_VOCAB_SYMBOL,
    _MIN_WORD_DURATION_SEC,
    RecognitionError,
    _classify_symbol,
)
from .types import Segment

# SOFA の日本語チェックポイントの語彙は無声化母音の専用記号を持たない。
_SOFA_VOCAB_SYMBOL_NORMALIZE: dict[str, str] = {"I": "i", "U": "u"}

# OS の待機は完了かタイムアウトまで戻らず、その間に届いた KeyboardInterrupt は待機が終わるまで反映されない。
_COMMUNICATE_POLL_SEC = 1.0

_HTK_100NS_UNITS_PER_SECOND = 1e7

# SOFA は入力をチェックポイントのサンプルレートへリサンプリングするので、出力の末尾時刻が十数ミリ秒ずれうる。
_SEGMENT_CONTRACT_TOLERANCE_SEC = 0.05

_SOFA_BREATH_LABEL = "AP"
_SOFA_SILENCE_LABEL = "SP"


def _normalize_for_sofa_vocab(symbol: str) -> str:
    return _SOFA_VOCAB_SYMBOL_NORMALIZE.get(symbol, symbol)


def _write_sofa_inputs(
    work_dir: Path, targets: list[tuple[np.ndarray, int, list[str]]]
) -> list[str]:
    basenames = []
    for i, (samples, sample_rate, phoneme_symbols) in enumerate(targets):
        basename = f"segment_{i:04d}"
        basenames.append(basename)
        sf.write(work_dir / f"{basename}.wav", samples, sample_rate)
        normalized = " ".join(_normalize_for_sofa_vocab(p) for p in phoneme_symbols)
        (work_dir / f"{basename}.lab").write_text(normalized, encoding="utf-8")
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
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_and_reap(proc: subprocess.Popen) -> None:
    try:
        _kill_process_tree(proc.pid)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.communicate(timeout=_COMMUNICATE_POLL_SEC)
    except Exception:
        pass


def _kill_process_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.killpg(os.getpgid(pid), signal.SIGKILL)


def _check_ascii_paths(work_dir: Path, config: SofaAlignerConfig) -> None:
    # SOFA は非 ASCII 文字を含むパスを扱えない。
    for label, path in (
        ("一時ディレクトリ", work_dir),
        ("sofa_root", config.sofa_root),
        ("checkpoint_path", config.checkpoint_path),
    ):
        if not str(path).isascii():
            raise RecognitionError(f"{label}のパスに非ASCII文字が含まれています: {path}")


def _check_paths_exist(config: SofaAlignerConfig) -> None:
    for label, path, check, kind in (
        ("sofa_python", config.sofa_python, Path.is_file, "ファイル"),
        ("sofa_root", config.sofa_root, Path.is_dir, "ディレクトリ"),
        ("sofa_root配下のinfer.py", config.sofa_root / "infer.py", Path.is_file, "ファイル"),
        ("checkpoint_path", config.checkpoint_path, Path.is_file, "ファイル"),
    ):
        if not check(path):
            reason = "存在しません" if not path.exists() else f"{kind}ではありません"
            raise RecognitionError(f"SOFAの実行環境パスが不正です: {label}={path}({reason})")


def _run_sofa_subprocess(basenames: list[str], config: SofaAlignerConfig, work_dir: Path) -> None:
    cmd = _build_sofa_command(config, work_dir)
    proc = None
    try:
        try:
            # 別のプロセスグループで起動した子には、呼び出し元への Ctrl+C が届かない。
            proc = subprocess.Popen(
                cmd,
                cwd=str(config.sofa_root),
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **_popen_kwargs(),
            )
        except FileNotFoundError as e:
            raise RecognitionError(
                "SOFAサブプロセスを起動できませんでした(実行ファイルまたは作業ディレクトリが"
                f"見つかりません): sofa_python={config.sofa_python}, sofa_root={config.sofa_root}"
            ) from e

        deadline = time.monotonic() + config.timeout_sec
        stderr = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_and_reap(proc)
                raise RecognitionError(
                    f"SOFA呼び出しが{config.timeout_sec}秒以内に完了しませんでした(タイムアウト)"
                )
            # communicate は TimeoutExpired の後に呼び直してよい。
            try:
                _, stderr = proc.communicate(timeout=min(_COMMUNICATE_POLL_SEC, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        if proc is not None:
            _terminate_and_reap(proc)
        raise

    if proc.returncode != 0:
        raise RecognitionError(f"SOFA呼び出しが終了コード{proc.returncode}で失敗しました: {stderr}")

    for basename in basenames:
        lab_path = work_dir / "htk" / "phones" / f"{basename}.lab"
        if not lab_path.exists() or lab_path.stat().st_size == 0:
            raise RecognitionError(f"SOFAの出力ファイルが見つからないか空です: {lab_path}")


def _parse_htk_label_file(path: Path) -> list[tuple[float, float, str]]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        start_100ns, end_100ns, label = line.split(maxsplit=2)
        segments.append((
            int(start_100ns) / _HTK_100NS_UNITS_PER_SECOND,
            int(end_100ns) / _HTK_100NS_UNITS_PER_SECOND,
            label,
        ))
    return segments


def _check_segment_contract(
    segments: list[tuple[float, float, str]], trim_duration_sec: float, tolerance: float
) -> None:
    if not segments:
        raise RecognitionError("SOFA出力のSegment列が空です(全時間軸を被覆できません)")

    for start, end, _ in segments:
        if start > end:
            raise RecognitionError(f"Segmentの開始時刻が終了時刻より後です: start={start}, end={end}")

    for (_, prev_end, _), (next_start, _, _) in zip(segments, segments[1:], strict=False):
        if abs(prev_end - next_start) > tolerance:
            raise RecognitionError(
                f"隣接するSegmentの境界が一致しません: {prev_end} != {next_start}"
            )

    first_start = segments[0][0]
    if abs(first_start - 0.0) > tolerance:
        raise RecognitionError(f"先頭Segmentの開始時刻が0.0と一致しません: {first_start}")

    last_end = segments[-1][1]
    if abs(last_end - trim_duration_sec) > tolerance:
        raise RecognitionError(
            f"末尾Segmentの終了時刻がトリム後区間の全長と一致しません: {last_end} != {trim_duration_sec}"
        )


def _normalize_segments(
    segments: list[tuple[float, float, str]], trim_duration_sec: float
) -> list[tuple[float, float, str]]:
    normalized = list(segments)

    start0, end0, label0 = normalized[0]
    normalized[0] = (0.0, end0, label0)

    for i in range(len(normalized) - 1):
        _, prev_end, _ = normalized[i]
        _, next_end, next_label = normalized[i + 1]
        normalized[i + 1] = (prev_end, next_end, next_label)

    last_start, _, last_label = normalized[-1]
    normalized[-1] = (last_start, trim_duration_sec, last_label)

    return normalized


def _validate_and_normalize_segments(
    segments: list[tuple[float, float, str]], trim_duration_sec: float
) -> list[tuple[float, float, str]]:
    _check_segment_contract(segments, trim_duration_sec, tolerance=_SEGMENT_CONTRACT_TOLERANCE_SEC)
    normalized = _normalize_segments(segments, trim_duration_sec)
    _check_segment_contract(normalized, trim_duration_sec, tolerance=0.0)
    return normalized


def _align_batch(
    targets: list[tuple[np.ndarray, int, list[str]]], config: SofaAlignerConfig
) -> dict[str, list[tuple[float, float, str]]]:
    if not targets:
        return {}

    config = replace(
        config,
        sofa_python=config.sofa_python.absolute(),
        sofa_root=config.sofa_root.absolute(),
        checkpoint_path=config.checkpoint_path.absolute(),
    )

    # Windows では子プロセスを終了してもファイルハンドルの解放が遅れることがある。
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        work_dir = Path(tmp)
        _check_ascii_paths(work_dir, config)
        _check_paths_exist(config)
        basenames = _write_sofa_inputs(work_dir, targets)
        _run_sofa_subprocess(basenames, config, work_dir)
        return {
            basename: _parse_htk_label_file(work_dir / "htk" / "phones" / f"{basename}.lab")
            for basename in basenames
        }


def _clamp_words_to_valid_list(
    words: list[tuple[list[str], float, float]], trim_duration_sec: float
) -> list[tuple[list[str], float, float]]:
    """words の各要素は (音素記号列, 開始秒, 終了秒)。"""
    cursor = 0.0
    valid_words: list[tuple[list[str], float, float]] = []
    for phoneme_symbols, start, end in words:
        clamped_end = min(end, trim_duration_sec)
        clamped_start = max(start, cursor)
        if not phoneme_symbols or (clamped_end - clamped_start) < _MIN_WORD_DURATION_SEC:
            continue
        valid_words.append((phoneme_symbols, clamped_start, clamped_end))
        cursor = clamped_end
    return valid_words


def _determine_word_gaps(
    valid_words: list[tuple[list[str], float, float]], trim_duration_sec: float
) -> list[tuple[float, float]]:
    if not valid_words:
        return [(0.0, trim_duration_sec)]

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for _, start, end in valid_words:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = end
    if cursor < trim_duration_sec:
        gaps.append((cursor, trim_duration_sec))
    return gaps


def _map_symbol_to_segment_fields(symbol: str) -> tuple[Literal["vowel", "consonant", "gap"], str | None]:
    # SOFA は force モードでも、与えた記号列に無い AP(吸気音)・SP(無音)を出力に加えることがある。
    if symbol in _BLANK_G2P_SYMBOLS or symbol in (_SOFA_BREATH_LABEL, _SOFA_SILENCE_LABEL):
        return "gap", None
    vocab_symbol = _G2P_TO_VOCAB_SYMBOL.get(symbol)
    if vocab_symbol is None:
        raise RecognitionError(f"SOFA出力記号 '{symbol}' の音素モデル語彙への写像が未定義です")
    return _classify_symbol(vocab_symbol), vocab_symbol


def _segments_from_raw(raw_segments: list[tuple[float, float, str]]) -> list[Segment]:
    segments = []
    for start, end, symbol in raw_segments:
        seg_type, phoneme = _map_symbol_to_segment_fields(symbol)
        segments.append(
            Segment(type=seg_type, start_sec=start, end_sec=end, phoneme=phoneme, confidence=None)
        )
    return segments
