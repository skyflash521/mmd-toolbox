import gc
from collections.abc import Callable

from .quiet import silenced_third_party_output

# transformers の PyTorch バックエンドは、これらの形式の重みファイルを読まない。
PYTORCH_IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.tflite", "*.onnx", "*.ot", "*.mlmodel"]


def select_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(device: str):
    import torch

    # CPU 実行では fp16 を使えない。
    return torch.float16 if device == "cuda" else torch.float32


def hf_snapshot_download(repo_id, *, revision=None, tqdm_class=None, ignore_patterns=None):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id, revision=revision, tqdm_class=tqdm_class,
        ignore_patterns=PYTORCH_IGNORE_PATTERNS if ignore_patterns is None else ignore_patterns,
    )


def prefetch_with_progress(repo_id: str, revision: str | None, on_progress: Callable[[str], None]) -> bool:
    """戻り値はバイト転送が実際に起きたか。"""
    from huggingface_hub.utils import tqdm as hf_tqdm

    state = {"shown": False}

    class _RelayTqdm(hf_tqdm):
        # 表示を無効化した tqdm は __init__ を途中で抜け、self.unit を持たず self.n も進めない。
        def __init__(self, *args, **kwargs):
            self._relay_unit = kwargs.get("unit")
            self._relay_n = kwargs.get("initial") or 0
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            result = super().update(n)
            # snapshot_download はファイル数を数えるバーと、実際に転送したバイト数を数える unit="B" のバーを作る。
            if self._relay_unit == "B" and self.total:
                self._relay_n += n or 0
                state["shown"] = True
                percent = min(100, int(self._relay_n * 100 / self.total))
                on_progress(f"ダウンロード中: {repo_id} {percent}%")
            return result

    with silenced_third_party_output():
        hf_snapshot_download(repo_id, revision=revision, tqdm_class=_RelayTqdm)
    return state["shown"]


def load_with_progress(
    repo_id: str, revision: str | None, *,
    on_progress: Callable[[str], None] | None, loading_note: str, build: Callable[[], object],
):
    """on_progress があるときは、取得の進捗を中継し、読み込み中の note を出してから build を呼ぶ。"""
    downloaded = False
    if on_progress is not None:
        downloaded = prefetch_with_progress(repo_id, revision, on_progress)
    try:
        if on_progress is not None:
            on_progress(loading_note)
        with silenced_third_party_output():
            return build()
    finally:
        if downloaded:
            on_progress("")


def release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        # torch はキャッシュした GPU メモリを empty_cache まで手放さない。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
