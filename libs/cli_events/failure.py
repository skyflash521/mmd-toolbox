import sys

from .events import error_event

# 呼び出し側がパイプを先に閉じると OSError、標準出力自体が閉じられていると ValueError になる。
_UNUSABLE_STREAM_ERRORS = (OSError, ValueError)


def emit_failure(emitter, *, code, message, exit_code, field=None, path=None, stderr=None,
                 **extra):
    """受け取った exit_code をそのまま返す。extra は error イベントにだけ載り、type を含めてはならない。"""
    if emitter is not None and not emitter.terminated:
        event = error_event(code=code, message=message, exit_code=exit_code, field=field, path=path)
        event.update(extra)
        try:
            emitter.error(**event)
        except _UNUSABLE_STREAM_ERRORS:
            _try_write_error_line(message, stderr)
        return exit_code
    _try_write_error_line(message, stderr)
    return exit_code


def _try_write_error_line(message, stderr):
    try:
        print(f"error: {message}", file=sys.stderr if stderr is None else stderr)
    except Exception:
        pass
