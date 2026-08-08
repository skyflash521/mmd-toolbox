"""cli_events の公開 API が及ぼすプロセス状態への副作用を、テスト間で分離するための補助。

install_sigbreak_handler(signals.py)は呼び出されたプロセスの SIGBREAK ハンドラを書き換える。CLI の
main() や並列ワーカ initializer を直接呼ぶテストが同一プロセス内で繰り返し実行されるため、各テストの
前後で保存・復元する pytest フィクスチャを提供する。cli_events 自身のテストと、install_sigbreak_handler
を呼ぶ各 CLI ツールのテストとで共有する(conftest.py から import してそのまま使う)。
"""

import signal
import sys

import pytest


@pytest.fixture(autouse=True)
def restore_sigbreak_handler():
    """Windows の SIGBREAK ハンドラをテストごとに保存・復元する(他 OS・SIGBREAK 非搭載環境では
    何もしない)。"""
    if sys.platform != "win32" or not hasattr(signal, "SIGBREAK"):
        yield
        return
    previous = signal.getsignal(signal.SIGBREAK)
    try:
        yield
    finally:
        signal.signal(signal.SIGBREAK, previous)
