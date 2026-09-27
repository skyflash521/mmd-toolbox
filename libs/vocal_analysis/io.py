import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from .types import AudioPcm

TARGET_PEAK = 0.95


class AudioLoadError(Exception):
    """reason は "decoder_missing"(ffmpeg が見つからず復号を試みられない)か "not_audio"(ファイルが
    無い、または試みた復号経路がすべて失敗した)。"""

    def __init__(self, message, *, reason: Literal["decoder_missing", "not_audio"]):
        super().__init__(message)
        self.reason = reason


def load_audio(path: Path) -> AudioPcm:
    samples, sample_rate = _read_raw(Path(path))
    return AudioPcm(samples=_normalize_peak(samples), sample_rate=sample_rate)


def _find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _read_raw(path: Path) -> tuple[np.ndarray, int]:
    # soundfile はファイル不在も非対応形式も同じ LibsndfileError で報告する。
    if not path.is_file():
        raise AudioLoadError(f"{path} が見つかりません。", reason="not_audio")
    try:
        return sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        return _read_via_ffmpeg(path)


def _read_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise AudioLoadError(
            f"{path} は soundfile で読み込めない形式です。ffmpeg が見つからないため読み込めません。"
            " ffmpeg を導入してください。",
            reason="decoder_missing",
        )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "decoded.wav"
        try:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), str(tmp_path)],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise AudioLoadError(f"{path} の ffmpeg 変換に失敗しました。", reason="not_audio") from e
        try:
            return sf.read(tmp_path, dtype="float32", always_2d=True)
        except sf.LibsndfileError as e:
            raise AudioLoadError(
                f"{path} は ffmpeg 変換後も読み込めませんでした。", reason="not_audio") from e


def _normalize_peak(samples: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak == 0.0:
        return samples
    return (samples * (TARGET_PEAK / peak)).astype(np.float32)
