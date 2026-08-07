"""song2vpr のテストが共有する pytest フィクスチャの配線。"""

import os

import pytest

from cli_events.testing import restore_sigbreak_handler as _restore_sigbreak_handler  # noqa: F401


@pytest.fixture(autouse=True)
def _isolate_cuda_visible_devices():
    """実行デバイスの選択がテストプロセスへ残す環境変数を各テストから切り離す。

    cli は `--device cpu` の実行で、プロセスから GPU を隠すために環境変数 CUDA_VISIBLE_DEVICES を
    直接書き換える(利用できるデバイスの集合はプロセス内の最初の照会以降固定されるため、処理の起動前に
    設定する必要がある)。書き換えたままだと、同じプロセスで走る他のテストの GPU 構成判定へ漏れる。
    """
    saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    yield
    if saved is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved
