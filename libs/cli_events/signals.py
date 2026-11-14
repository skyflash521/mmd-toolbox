"""CPython は Windows の SIGBREAK(CTRL_BREAK_EVENT)に KeyboardInterrupt を送出する既定ハンドラを持たない。

CREATE_NEW_PROCESS_GROUP で起動された子プロセスへは CTRL_C_EVENT を送れず、CTRL_BREAK_EVENT だけが届く。
"""

import signal
import sys


def install_sigbreak_handler():
    """SIGBREAK の無い環境では何もしない。構造化出力の送出手段を組み上げた後、本体処理より前に呼ぶ。

    プロセスの SIGBREAK ハンドラを書き換えるので、これを呼ぶテストは conftest.py へ
    cli_events.testing.restore_sigbreak_handler を取り込む。
    """
    if sys.platform != "win32" or not hasattr(signal, "SIGBREAK"):
        return

    def _raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGBREAK, _raise_keyboard_interrupt)
