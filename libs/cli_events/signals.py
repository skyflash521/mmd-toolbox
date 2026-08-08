"""OS依存の協調的中断シグナルを KeyboardInterrupt へ橋渡しする。

CPython は SIGINT (Ctrl-C) には既定でハンドラを登録し KeyboardInterrupt を送出するが、
Windows の CTRL_BREAK_EVENT (SIGBREAK) には登録しない。GUI 等の呼び出し側が
CREATE_NEW_PROCESS_GROUP で起動した子プロセスへは CTRL_C_EVENT を送れず CTRL_BREAK_EVENT
のみが使えるため、ハンドラ未登録のツールは中断要求で、Python の例外処理を経ないまま
OS の既定動作(STATUS_CONTROL_C_EXIT)により即座に終了する。
"""

import signal
import sys


def install_sigbreak_handler():
    """Windows の CTRL_BREAK_EVENT (SIGBREAK) を KeyboardInterrupt へ変換する。

    Windows かつ SIGBREAK が存在する環境でのみハンドラを登録する。それ以外の環境では
    何もしないため、プラットフォームを問わず無条件で呼べる。
    """
    if sys.platform != "win32" or not hasattr(signal, "SIGBREAK"):
        return

    def _raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGBREAK, _raise_keyboard_interrupt)
