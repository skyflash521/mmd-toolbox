import os
import shutil


def torch_gpu_warning(device_mode: str) -> tuple[str, dict[str, str]] | None:
    if device_mode == "cpu":
        return None
    # nvidia-smi は NVIDIA ドライバと一緒に導入される。
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        import torch
    # 導入が壊れて共有ライブラリを開けないとき、torch の取り込みは OSError を送出する。
    except (ImportError, OSError):
        return None
    version = torch.__version__
    if torch.version.cuda is None:
        return "cpu_only_torch", {"torch_version": version}
    if not torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES で GPU を隠すと torch.cuda.is_available() は False を返す。
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            return None
        return "cuda_unavailable", {"torch_version": version}
    return None
