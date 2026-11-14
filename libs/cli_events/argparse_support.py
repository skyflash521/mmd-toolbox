import argparse

from .events import error_event

_BAD_ARGUMENT_EXIT_CODE = 2
_ARGUMENT_PREFIX = "argument "
_UNRECOGNIZED_PREFIX = "unrecognized arguments:"
_REQUIRED_PREFIX = "the following arguments are required:"
_OPTION_STRING_SEPARATOR = "/"


class ArgumentParseError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class MachineArgumentParser(argparse.ArgumentParser):
    """使用法エラーでは終了せず ArgumentParseError を送出する。"""

    def error(self, message):
        raise ArgumentParseError(message)


def argparse_error_event(error, *, code, field=None):
    return error_event(code=code, message=error.message, exit_code=_BAD_ARGUMENT_EXIT_CODE, field=field)


def argparse_error_field(message: str) -> str | None:
    if message.startswith(_ARGUMENT_PREFIX) and ":" in message:
        name = message[len(_ARGUMENT_PREFIX):].split(":", 1)[0].strip()
        return name.split(_OPTION_STRING_SEPARATOR)[-1] if name.startswith("-") else name
    if message.startswith(_UNRECOGNIZED_PREFIX):
        rest = message[len(_UNRECOGNIZED_PREFIX):].split()
        return rest[0] if rest else None
    if message.startswith(_REQUIRED_PREFIX):
        rest = message[len(_REQUIRED_PREFIX):].strip()
        return rest.split(",")[0].strip() or None
    return None
