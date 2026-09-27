import signal
import subprocess
import sys

import pytest

from cli_events import install_sigbreak_handler
from cli_events.testing import _preserved_sigbreak_handler


@pytest.mark.skipif(sys.platform != "win32", reason="SIGBREAK は Windows 固有")
def test_sigbreak_handler_raises_keyboard_interrupt():
    install_sigbreak_handler()
    handler = signal.getsignal(signal.SIGBREAK)
    assert handler is not signal.SIG_DFL
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGBREAK, None)


@pytest.mark.skipif(sys.platform != "win32", reason="SIGBREAK は Windows 固有")
def test_sigbreak_handler_installed_inside_fixture_scope_is_restored_on_exit():
    before = signal.getsignal(signal.SIGBREAK)
    with _preserved_sigbreak_handler():
        install_sigbreak_handler()
        assert signal.getsignal(signal.SIGBREAK) is not before
    assert signal.getsignal(signal.SIGBREAK) is before


@pytest.mark.skipif(sys.platform == "win32", reason="Windows 以外での no-op を検証")
def test_no_op_outside_windows():
    install_sigbreak_handler()


_CHILD_SCRIPT = """
import sys, time
from cli_events import install_sigbreak_handler
install_sigbreak_handler()
print("ready", flush=True)
try:
    end = time.time() + 20
    while time.time() < end:
        pass
except KeyboardInterrupt:
    print("caught", flush=True)
    sys.exit(42)
sys.exit(1)
"""


@pytest.mark.skip(
    reason="OS のコンソール制御イベント配送が確率的に失敗しうるため通常スイートから除外する"
    "(手動診断用。実行するにはこの skip を一時的に外す)。"
)
@pytest.mark.skipif(sys.platform != "win32", reason="CTRL_BREAK_EVENT は Windows 固有")
def test_real_ctrl_break_event_becomes_keyboard_interrupt():
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        ready_line = proc.stdout.readline()
        assert ready_line.strip() == "ready"
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        out, _err = proc.communicate(timeout=15)
        assert proc.returncode == 42
        assert "caught" in out
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
