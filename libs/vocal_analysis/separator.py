import atexit
import inspect
import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import soundfile as sf

from .config import DEFAULT_SEPARATOR, SEPARATOR_CONFIG, SeparatorId
from .quiet import silence_third_party_output
from .types import AudioPcm

# Demucs の set_progress_bar が受け取る進み具合は 0〜0.8 の範囲で進み、0.8 で推論が終わる。
_DEMUCS_PROGRESS_FRACTION_AT_COMPLETION = 0.8


class SeparationError(Exception):
    pass


def separate(
    pcm: AudioPcm, mode: Literal["always", "never"], *,
    separator: SeparatorId = DEFAULT_SEPARATOR,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """登録の無い separator は mode="never" でも SeparationError。戻り値の WAV は、プロセスの正常終了時に
    削除を試みる非公開の一時ディレクトリに置かれる。"""
    if mode not in ("always", "never"):
        raise ValueError(f"未知の mode です: {mode!r}(always/never のいずれかを指定してください)")
    impl = _SEPARATOR_IMPLS.get(separator)
    if impl is None:
        raise SeparationError(f"未知の separator です: {separator!r}")

    work_dir = Path(tempfile.mkdtemp(prefix="vocal_analysis_s1_"))
    atexit.register(shutil.rmtree, work_dir, ignore_errors=True)
    input_wav = work_dir / "input.wav"
    sf.write(input_wav, pcm.samples, pcm.sample_rate)

    if mode == "never":
        return input_wav
    return impl(work_dir, input_wav, on_progress)


def _separate_audio_separator_htdemucs_ft(work_dir: Path, input_wav: Path, on_progress) -> Path:
    try:
        separator_obj = _build_separator(work_dir)
    except ImportError as e:
        raise SeparationError(
            "audio-separator が見つかりません。導入してください(pip install audio-separator onnxruntime)。"
        ) from e
    downloaded = _load_model_with_progress(separator_obj, on_progress)
    if downloaded:
        on_progress("")
    output_files = _separate_with_progress(separator_obj, input_wav, on_progress)
    output_path = Path(output_files[0])
    del separator_obj
    try:
        import torch

        # torch はキャッシュした GPU メモリを empty_cache まで手放さない。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    # audio-separator は output_dir 相対のファイル名だけを返すことがある。
    return output_path if output_path.is_absolute() else work_dir / output_path


_SEPARATOR_IMPLS = {
    "audio-separator-htdemucs-ft": _separate_audio_separator_htdemucs_ft,
}


def _load_model_with_progress(separator_obj, on_progress) -> bool:
    """戻り値はモデルのダウンロードが実際に起きたか。"""
    if on_progress is None:
        separator_obj.load_model(model_filename=SEPARATOR_CONFIG.model_filename)
        return False

    # 以下で差し替える名前はいずれも audio-separator の公開契約ではない。
    try:
        import audio_separator.separator.separator as _as_separator_module
        from tqdm import tqdm as _base_tqdm

        original_download = _as_separator_module.Separator.download_file_if_not_exists
        original_tqdm = _as_separator_module.tqdm
    except (ImportError, AttributeError):
        on_progress(f"モデル読み込み中: {SEPARATOR_CONFIG.model_filename}")
        separator_obj.load_model(model_filename=SEPARATOR_CONFIG.model_filename)
        return False

    state = {"shown": False, "current_filename": "モデルファイル"}

    class _RelayTqdm(_base_tqdm):
        def __init__(self, *args, **kwargs):
            # tqdm は disable=True のとき self.n を進めない。
            self._relay_n = kwargs.get("initial") or 0
            self._relay_filename = state["current_filename"]
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            result = super().update(n)
            if self.total:
                self._relay_n += n or 0
                state["shown"] = True
                percent = min(100, int(self._relay_n * 100 / self.total))
                on_progress(f"ダウンロード中: {self._relay_filename} {percent}%")
            return result

    # download_file_if_not_exists はモデルを構成するファイルごとに呼ばれ、そのたびに新しい tqdm を作る。
    def _patched_download(self, *args, **kwargs):
        output_path = kwargs.get("output_path") or (args[1] if len(args) > 1 else None)
        if output_path is not None:
            state["current_filename"] = Path(output_path).name
        return original_download(self, *args, **kwargs)

    # audio-separator は tqdm の差し込み口を持たず、モジュール内で import した tqdm を直接生成する。
    _as_separator_module.tqdm = _RelayTqdm
    _as_separator_module.Separator.download_file_if_not_exists = _patched_download
    try:
        on_progress(f"モデル読み込み中: {SEPARATOR_CONFIG.model_filename}")
        separator_obj.load_model(model_filename=SEPARATOR_CONFIG.model_filename)
    finally:
        _as_separator_module.tqdm = original_tqdm
        _as_separator_module.Separator.download_file_if_not_exists = original_download
    return state["shown"]


def _separate_with_progress(separator_obj, input_wav: Path, on_progress):
    if on_progress is None:
        return separator_obj.separate(str(input_wav))

    # audio-separator の非公開の実装: demix_demucs は apply_model を set_progress_bar=None 固定で呼ぶ。
    # audio-separator の非公開の実装: apply_model は受け取った kwargs を再帰呼び出しへそのまま渡す。
    try:
        import audio_separator.separator.architectures.demucs_separator as _demucs_module

        original_apply_model = _demucs_module.apply_model
        injectable = "set_progress_bar" in inspect.signature(original_apply_model).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        injectable = False
    if not injectable:
        on_progress("")
        return separator_obj.separate(str(input_wav))

    def _relay_progress(*args, **kwargs):
        if len(args) < 2:
            return
        fraction = args[1]
        percent = min(100, round(fraction / _DEMUCS_PROGRESS_FRACTION_AT_COMPLETION * 100))
        on_progress(f"分離中: {percent}%")

    def _patched_apply_model(*args, **kwargs):
        kwargs["set_progress_bar"] = _relay_progress
        return original_apply_model(*args, **kwargs)

    _demucs_module.apply_model = _patched_apply_model
    try:
        return separator_obj.separate(str(input_wav))
    finally:
        _demucs_module.apply_model = original_apply_model


def _build_separator(output_dir: Path):
    from audio_separator.separator import Separator

    silence_third_party_output()
    return Separator(
        log_level=logging.CRITICAL,
        output_dir=str(output_dir),
        output_single_stem=SEPARATOR_CONFIG.output_single_stem,
        demucs_params={
            "segment_size": "Default",
            "shifts": SEPARATOR_CONFIG.shifts,
            "overlap": 0.25,
            "segments_enabled": True,
        },
    )
