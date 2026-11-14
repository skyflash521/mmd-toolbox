import contextlib
import sys
import threading
import time
import unicodedata
from typing import NamedTuple

# 閉じたストリームへの write は ValueError、端末の符号化で表せない文字の write は UnicodeEncodeError を送出する。
_WRITE_ERRORS = (OSError, ValueError, UnicodeError)


class _Snapshot(NamedTuple):
    label: str
    done: int
    total: int | None
    start: float
    note: str


def _format_line(label, done, total, elapsed, note=""):
    secs = int(elapsed)
    count = "" if total is None else f" {done}/{total}"
    suffix = f" {note}" if note else ""
    return f"[{label}]{count} 経過 {secs // 60}:{secs % 60:02d}{suffix}"


def _display_width(text):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


class ProgressReporter:
    def __init__(self, stream=None, *, enabled=None, now=None, interval=0.15, write_lock=None):
        """enabled が None なら stream.isatty() の結果に従う。write_lock は stream への各書き込みの間保持する。"""
        self._stream = stream if stream is not None else sys.stderr
        if enabled is None:
            isatty = getattr(self._stream, "isatty", None)
            enabled = bool(isatty()) if callable(isatty) else False
        self.enabled = enabled
        self._now = now if now is not None else time.monotonic
        self._interval = interval
        self._write_lock = write_lock if write_lock is not None else contextlib.nullcontext()
        self._snap: _Snapshot | None = None
        self._thread = None
        self._stop = None
        self._last_drawn_width = 0

    def stage(self, label):
        if not self.enabled:
            return
        self._stop_heartbeat()
        self._snap = _Snapshot(label, 0, None, self._now(), "")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def update(self, done, total, note=""):
        snap = self._snap
        if snap is None:
            return
        # CPython では参照の代入が原子的なので、ハートビートスレッドはロックなしで一貫した組を読める。
        self._snap = snap._replace(done=done, total=total, note=note)

    def close(self):
        self._stop_heartbeat()
        if self.enabled and self._last_drawn_width:
            try:
                with self._write_lock:
                    self._stream.write("\r" + " " * self._last_drawn_width + "\r")
                    self._stream.flush()
            except _WRITE_ERRORS:
                self.enabled = False
        self._snap = None
        self._last_drawn_width = 0

    def summary(self, message):
        """close の後に呼ぶ。"""
        if not self.enabled:
            return
        try:
            with self._write_lock:
                self._stream.write(message + "\n")
                self._stream.flush()
        except _WRITE_ERRORS:
            self.enabled = False

    def _heartbeat(self):
        while not self._stop.wait(self._interval):
            self._draw()

    def _stop_heartbeat(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def _draw(self):
        if not self.enabled:
            return
        snap = self._snap
        if snap is None:
            return
        line = _format_line(snap.label, snap.done, snap.total, self._now() - snap.start, snap.note)
        width = _display_width(line)
        residue_padding = " " * max(0, self._last_drawn_width - width)
        try:
            with self._write_lock:
                self._stream.write("\r" + line + residue_padding)
                self._stream.flush()
        except _WRITE_ERRORS:
            self.enabled = False
            return
        self._last_drawn_width = width
