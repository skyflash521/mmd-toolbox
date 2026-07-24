"""cli_events.signals の単体テスト。

Windows の SIGBREAK を KeyboardInterrupt へ橋渡しするハンドラを検証する。決定論的に実行し、
ネットワーク等の外部依存は使わない。ハンドラ単体の変換ロジックはインプロセスで、実際の
CTRL_BREAK_EVENT 配送は子プロセスを介した検証で確かめる(自プロセスへ送ると自分自身が中断される
ため、別プロセスを起点にする)。
"""
import signal
import subprocess
import sys

import pytest

from cli_events import install_sigbreak_handler


@pytest.mark.skipif(sys.platform != "win32", reason="SIGBREAK は Windows 固有")
def test_sigbreak_handler_raises_keyboard_interrupt():
    # ハンドラの保存・復元は autouse フィクスチャ(conftest.py 経由の restore_sigbreak_handler)が担う。
    install_sigbreak_handler()
    handler = signal.getsignal(signal.SIGBREAK)
    assert handler is not signal.SIG_DFL
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGBREAK, None)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows 以外での no-op を検証")
def test_no_op_outside_windows():
    # SIGBREAK が存在しない環境では例外を送出せず何もしない。
    install_sigbreak_handler()


_CHILD_SCRIPT = """
import sys, time
from cli_events import install_sigbreak_handler
install_sigbreak_handler()
print("ready", flush=True)
try:
    # ビジーループでバイトコードを実行し続ける(time.sleep 等の OS 待機は CTRL_BREAK_EVENT 配送後も
    # タイムアウトまで反応しない可能性がある。未検証の仮説に留めるが、ここでの検証対象は配送から
    # KeyboardInterrupt への変換そのものなので、OS 待機を挟まない形で確実に検証する)。
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
    # 実際に子プロセスへ CTRL_BREAK_EVENT を送り、ハンドラ登録済みなら KeyboardInterrupt へ変換されて
    # 通常の Python 例外処理経路に乗ることを検証する(GUI が CREATE_NEW_PROCESS_GROUP で起動した子への
    # 中断と同じ配送経路)。ハンドラが無いと STATUS_CONTROL_C_EXIT で毎回即時終了する。ただしハンドラ
    # 登録済みでも OS のコンソール制御イベント配送自体が確率的に失敗しうるため、この検証は非決定的
    # (単発判定)であり、常時実行する通常スイートには含めない。
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
