"""S0 入力読み込み。

入力音声を復号し、ch/SR を保持したまま入力レベル正規化(ピーク正規化)を適用した AudioPcm を返す。
soundfile(同梱 libsndfile)で読める WAV/FLAC/OGG/mp3 を最低保証の形式として読み、それ以外は
実行時に自動検出した ffmpeg でフォールバックする。ffmpeg はリポジトリに同梱・再配布しない。
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .types import AudioPcm

TARGET_PEAK = 0.95


class AudioLoadError(Exception):
    """S0 の読み込み失敗(soundfile 非対応かつ ffmpeg 未検出など、原因が分かるエラー)。

    reason は失敗原因の分類("decoder_missing": フォールバック復号器(ffmpeg)が未検出のため復号を
    試みられない。"not_audio": 入力ファイルが存在しない、または利用可能な復号経路をいずれも
    実際に試みたが失敗した)。呼び出し側が原因の切り分けに使う。
    """

    def __init__(self, message, *, reason):
        super().__init__(message)
        self.reason = reason


def load_audio(path: Path) -> AudioPcm:
    """入力音声を読み込み、ch/SR を保持したピーク正規化済み AudioPcm を返す。"""
    samples, sample_rate = _read_raw(Path(path))
    return AudioPcm(samples=_normalize_peak(samples), sample_rate=sample_rate)


def _find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _read_raw(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        # soundfile はファイル不在・ディレクトリ指定も非対応フォーマットも同じ LibsndfileError で
        # 報告し、そのまま ffmpeg フォールバックへ進むと ffmpeg も失敗して原因不明の
        # 「ffmpeg 変換に失敗しました」という誤解を招くメッセージになる。ここで先に区別する。
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
    """入力レベル正規化(ピーク正規化・斉次・0除算回避)。"""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak == 0.0:
        return samples
    return (samples * (TARGET_PEAK / peak)).astype(np.float32)
