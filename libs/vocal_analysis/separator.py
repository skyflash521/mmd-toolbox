"""S1 ボーカル抽出(vocal_analysis.md §4・§8.1・§8.3後注)。

S0 の出力(入力レベル正規化済み。ステレオ・元サンプルレート)からボーカルを分離し、ボーカルWAVを得る。
分離は audio-separator 経由で Demucs v4 htdemucs_ft を in-process 実行する。mode(auto/always/never)は
この抽象が解釈する。auto は BGM 有無の自動判定を持たず always と同義(常に分離する)。
"""

import tempfile
from pathlib import Path
from typing import Literal

import soundfile as sf

from .config import SEPARATOR_CONFIG
from .types import AudioPcm


class SeparationError(Exception):
    """S1 の分離失敗(audio-separator 未導入など、原因が分かるエラー)。"""


def separate(pcm: AudioPcm, mode: Literal["auto", "always", "never"]) -> Path:
    """S0出力からボーカルWAVのパスを得る(§4・§8.1)。"""
    if mode not in ("auto", "always", "never"):
        raise ValueError(f"未知の mode です: {mode!r}(auto/always/never のいずれかを指定してください)")

    work_dir = Path(tempfile.mkdtemp(prefix="vocal_analysis_s1_"))
    input_wav = work_dir / "input.wav"
    sf.write(input_wav, pcm.samples, pcm.sample_rate)

    if mode == "never":
        return input_wav

    try:
        separator = _build_separator(work_dir)
    except ImportError as e:
        raise SeparationError(
            "audio-separator が見つかりません。導入してください(pip install audio-separator onnxruntime)。"
        ) from e
    separator.load_model(model_filename=SEPARATOR_CONFIG.model_filename)
    output_files = separator.separate(str(input_wav))
    return Path(output_files[0])


def _build_separator(output_dir: Path):
    """audio-separator の Separator を固定条件(§8.3後注)で構成する。"""
    from audio_separator.separator import Separator

    return Separator(
        output_dir=str(output_dir),
        output_single_stem=SEPARATOR_CONFIG.output_single_stem,
        demucs_params={
            "segment_size": "Default",
            "shifts": SEPARATOR_CONFIG.shifts,
            "overlap": 0.25,
            "segments_enabled": True,
        },
    )
