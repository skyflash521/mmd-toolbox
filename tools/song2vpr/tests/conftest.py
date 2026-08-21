"""song2vpr のテストが共有する pytest フィクスチャの配線。"""

import os

import pytest

from cli_events.testing import restore_sigbreak_handler as _restore_sigbreak_handler  # noqa: F401
from cli_resource_watch import torch_config as _torch_config


@pytest.fixture(autouse=True)
def _isolate_gpu_environment(monkeypatch):
    """GPU 構成の警告が読む外部環境を各テストから切り離す。

    cli は既定(`--device auto`)の実行で、PATH 上の nvidia-smi と torch の状態から GPU を使えない
    構成を判定して警告を出す。切り離さないと、NVIDIA GPU を積んだ機材に基本手順どおり CPU 専用版の
    torch を入れた開発者(この警告が対象とする構成そのもの)で、イベント数や標準エラーの行数を数える
    テストが落ちる。nvidia-smi を不在に倒しておけばどの機材でも警告は出ず、警告そのものを検証する
    テストは各自で判定を上書きすればよい。

    加えて cli は `--device cpu` の実行で、プロセスから GPU を隠すために環境変数
    CUDA_VISIBLE_DEVICES を直接書き換える(利用できるデバイスの集合はプロセス内の最初の照会以降
    固定されるため、処理の起動前に設定する必要がある)。書き換えたままだと同じプロセスで走る他の
    テストの判定へ漏れるので、その退避と復元もここで行う。
    """
    monkeypatch.setattr(_torch_config.shutil, "which", lambda name: None)
    saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    yield
    if saved is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved
