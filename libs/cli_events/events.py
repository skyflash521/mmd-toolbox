import json
import math
from pathlib import Path

import numpy as np

EVENT_TYPES = ("progress", "warning", "result", "error")
_TERMINAL = frozenset({"result", "error"})


class StreamTerminatedError(RuntimeError):
    pass


class EventEmitter:
    """stream は bytes を書き込むバイナリストリーム(sys.stdout.buffer など)。

    result または error の送出後に送出すると StreamTerminatedError を送出する。
    """

    def __init__(self, stream):
        self._stream = stream
        self._terminated = False

    @property
    def terminated(self):
        return self._terminated

    def progress(self, **fields):
        self._emit("progress", fields)

    def warning(self, **fields):
        self._emit("warning", fields)

    def result(self, **fields):
        self._emit("result", fields)

    def error(self, **fields):
        self._emit("error", fields)

    def _emit(self, type_, fields):
        if self._terminated:
            raise StreamTerminatedError(f"ストリームは終端済み(type={type_!r} は送出できない)")
        obj = {"type": type_}
        obj.update(fields)
        line = json.dumps(_to_json_value(obj), ensure_ascii=False, allow_nan=False)
        self._stream.write((line + "\n").encode("utf-8"))
        flush = getattr(self._stream, "flush", None)
        if callable(flush):
            flush()
        if type_ in _TERMINAL:
            self._terminated = True


def _to_json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    return value


def error_event(*, code, message, exit_code, field=None, path=None):
    return {
        "type": "error",
        "code": code,
        "exit_code": exit_code,
        "field": field,
        "path": path,
        "message": message,
    }
