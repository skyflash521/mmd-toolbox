"""GPU を使えない実行構成の判定。"""

import os
import shutil


def torch_gpu_warning(device_mode):
    """GPU を使えない構成なら (code, fields) を返す。該当しなければ None。

    NVIDIA GPU を積んだ機材で torch の構成が原因で GPU を使えないときだけ返す。GPU の有無は
    NVIDIA ドライバが導入する nvidia-smi が PATH 上にあるかで判定する(存在を確かめるだけで実行は
    しない)。この条件が無いと、GPU を積んでいない機材と macOS で毎回警告が出る。実行デバイスの
    選択が "cpu" なら利用者が CPU 実行を選んでいるので判定しない。torch を読み込めない環境でも
    判定しない。
    """
    if device_mode == "cpu" or shutil.which("nvidia-smi") is None:
        return None
    try:
        import torch
    except (ImportError, OSError):
        # 未導入だけでなく、導入が壊れて共有ライブラリを開けない場合も判定できない
        # (どちらもこの警告のためだけに処理を止める理由にはならない)。
        return None
    version = torch.__version__
    if torch.version.cuda is None:
        return "cpu_only_torch", {"torch_version": version}
    if not torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES が環境から渡っている場合、利用者が見せる GPU を自分で絞った結果
        # として CUDA が使えないことがある。この構成で「別の CUDA のバージョンで入れ直せ」は
        # 効かない対処になるため判定しない(実行デバイスが "cpu" のときは上の早期リターンで到達しない)。
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            return None
        return "cuda_unavailable", {"torch_version": version}
    return None
