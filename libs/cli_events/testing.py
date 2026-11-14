"""cli_events を使うツールのテストが conftest.py から取り込む pytest フィクスチャ。"""

import contextlib
import signal
import sys

import pytest


@contextlib.contextmanager
def _preserved_sigbreak_handler():
    if sys.platform != "win32" or not hasattr(signal, "SIGBREAK"):
        yield
        return
    previous = signal.getsignal(signal.SIGBREAK)
    try:
        yield
    finally:
        signal.signal(signal.SIGBREAK, previous)


@pytest.fixture(autouse=True)
def restore_sigbreak_handler():
    with _preserved_sigbreak_handler():
        yield
