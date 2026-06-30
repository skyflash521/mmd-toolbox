"""argparse のエラーを機械モードの error イベントへ橋渡しする(cli_events.md §4)。

argparse は既定で使用法エラーを標準エラーへ出して終了する。機械モードではこの経路を error
イベントへ振り替えるため、エラー時に終了する代わりに例外を送出する ArgumentParser と、その例外を
error イベントへ変換するヘルパを提供する。
"""

import argparse

from .events import error_event


class ArgumentParseError(Exception):
    """MachineArgumentParser が argparse の使用法エラー時に送出する(SystemExit の代わり)。"""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class MachineArgumentParser(argparse.ArgumentParser):
    """エラー時に標準エラーへ出して終了する代わりに ArgumentParseError を送出する ArgumentParser。

    機械モードで argparse の使用法エラーを error イベントへ変換できるようにする(cli_events.md §4)。
    --help/--version などのメタ操作は ArgumentParser の既定どおり(error() を経由しないため、
    本クラスの影響を受けない)。
    """

    def error(self, message):
        raise ArgumentParseError(message)


def argparse_error_event(error, *, code, field=None):
    """ArgumentParseError を error イベントへ変換する(cli_events.md §4)。

    argparse のエラーは引数エラーなので exit_code は基底共通の 2 で固定する(規約 §5)。
    安定 code 値・該当 field は呼び出し側が渡し、message は argparse のメッセージを載せる。
    """
    return error_event(code=code, message=error.message, exit_code=2, field=field)
