"""機械モードのイベント送出基盤。

JSON Lines のイベントストリームを送出するエミッタと、error イベントの dict ビルダーを提供する。
イベント種別の語彙(progress/warning/result/error)と終端規則だけを共通化し、各イベントの
ペイロード(type 以外のキー)は呼び出し側(各ツール)が決める。
"""

import json

# イベント種別の語彙(この順で公開する)。
EVENT_TYPES = ("progress", "warning", "result", "error")
_TERMINAL = frozenset({"result", "error"})


class StreamTerminatedError(RuntimeError):
    """result/error 送出後の追加送出を拒否する(終端規則)。"""


class EventEmitter:
    """機械モードの JSON Lines イベントストリームを stream へ送出する。

    stream は UTF-8 バイトを書き込めるバイナリストリーム(例 sys.stdout.buffer / io.BytesIO)。
    各イベントを 1 行 1 オブジェクトで UTF-8 符号化して書き、プラットフォームのロケール符号化に
    依存しない。result または error を送出するとストリームは終端し、以後の送出は
    StreamTerminatedError を送出する(1 ストリームにつき終端イベントはちょうど 1 つ)。
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
        # ensure_ascii=False で非 ASCII を UTF-8 のまま出す。1 行 1 オブジェクト + 末尾改行。
        line = json.dumps(obj, ensure_ascii=False)
        self._stream.write((line + "\n").encode("utf-8"))
        flush = getattr(self._stream, "flush", None)
        if callable(flush):
            flush()
        if type_ in _TERMINAL:
            self._terminated = True


def error_event(*, code, message, exit_code, field=None, path=None):
    """error イベントの dict を組む。

    type は "error" 固定。code/exit_code/message は呼び出し側が定める。field/path は対象が
    無ければ None。返り値はそのまま EventEmitter.error(**event) へ展開できる形。
    """
    return {
        "type": "error",
        "code": code,
        "exit_code": exit_code,
        "field": field,
        "path": path,
        "message": message,
    }
