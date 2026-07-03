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


def argparse_error_field(message):
    """argparse の使用法エラー文言から対象引数名(bad_argument の field)を取り出す(cli_events.md §4)。

    argparse は起因引数を構造化して渡さないので、標準の文言形からベストエフォートで抽出する。
    文言に依存するため、既知のどの形にも当たらなければ None(field なし)へ退避し、詳細は message 側に残す。
    """
    if message.startswith("argument ") and ":" in message:
        # 「argument <引数名>: 」形。コロン前の引数名を採る(コロンが無い文言は本形でないので下へ落とす)。
        name = message[len("argument "):].split(":", 1)[0].strip()
        # 複数のオプション文字列は "-o/--output" のように "/" で連結される。長形式(最後)を採る。
        # 先頭が "-" でない positional 名はそのまま返す。
        return name.split("/")[-1] if name.startswith("-") else name
    if message.startswith("unrecognized arguments:"):
        rest = message[len("unrecognized arguments:"):].split()
        return rest[0] if rest else None
    if message.startswith("the following arguments are required:"):
        rest = message[len("the following arguments are required:"):].strip()
        return rest.split(",")[0].strip() or None
    return None
