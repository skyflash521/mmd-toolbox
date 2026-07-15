"""S1 ボーカル抽出。

S0 の出力(入力レベル正規化済み。ステレオ・元サンプルレート)からボーカルを分離し、ボーカルWAVを得る。
分離は audio-separator 経由で Demucs v4 htdemucs_ft を in-process 実行する。mode(auto/always/never)は
この抽象が解釈する。auto は BGM 有無の自動判定を持たず always と同義(常に分離する)。
"""

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import soundfile as sf

from .config import SEPARATOR_CONFIG
from .types import AudioPcm


class SeparationError(Exception):
    """S1 の分離失敗(audio-separator 未導入など、原因が分かるエラー)。"""


def separate(pcm: AudioPcm, mode: Literal["auto", "always", "never"]) -> Path:
    """S0出力からボーカルWAVのパスを得る。

    戻り値のWAVを格納する作業ディレクトリは呼び出し元に公開せず、プロセスの正常終了時に
    削除を試みる(強制終了時や削除失敗時は残置を許容する)。呼び出し元が削除
    タイミングを制御する手段は無い。
    """
    if mode not in ("auto", "always", "never"):
        raise ValueError(f"未知の mode です: {mode!r}(auto/always/never のいずれかを指定してください)")

    work_dir = Path(tempfile.mkdtemp(prefix="vocal_analysis_s1_"))
    atexit.register(shutil.rmtree, work_dir, ignore_errors=True)
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
    output_path = Path(output_files[0])
    # S1のGPUメモリをS2(内容認識・音素モデル)のロード前に返す。保持したままだと、後続の
    # 内容認識パイプラインのロード・推論時にVRAMが逼迫し、処理時間が大きく悪化する。
    del separator
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    # audio-separator は output_dir 相対のファイル名だけを返すことがある(絶対パスの
    # 保証はない)。相対パスの場合は分離器の output_dir(work_dir)を基準に解決する。
    return output_path if output_path.is_absolute() else work_dir / output_path


def _build_separator(output_dir: Path):
    """audio-separator の Separator を固定条件で構成する。"""
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
